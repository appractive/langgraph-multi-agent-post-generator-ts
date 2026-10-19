import type DatabaseType from "better-sqlite3";
import { RunBudget, type BudgetSnapshot } from "./observability";

// ─── Run context ────────────────────────────────────────────────────────────
//
// Per-thread runtime objects (budget, status) live here, keyed by the LangGraph
// thread_id, so they are never serialised into graph state or sent to clients.
// The store can persist to SQLite, which lets a run paused for human review
// survive a server restart (the checkpoint carries the graph state; this table
// carries the budget and lifecycle status).

export type RunStatus = "running" | "paused" | "done";

export interface RunContext {
  threadId: string;
  budget: RunBudget;
  createdAt: number;
  lastActiveAt: number;
  status: RunStatus;
}

/** Serialised run row (what actually hits the database). */
export interface RunRow {
  threadId: string;
  createdAt: number;
  lastActiveAt: number;
  status: RunStatus;
  budget: BudgetSnapshot;
}

export interface RunStoreBackend {
  loadAll(): RunRow[];
  upsert(row: RunRow): void;
  remove(threadId: string): void;
}

/** SQLite-backed table: one row per live (or paused) run. */
export class SqliteRunBackend implements RunStoreBackend {
  constructor(private readonly db: DatabaseType.Database) {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS runs (
        thread_id      TEXT PRIMARY KEY,
        created_at     INTEGER NOT NULL,
        last_active_at INTEGER NOT NULL,
        status         TEXT NOT NULL,
        budget         TEXT NOT NULL
      )
    `);
  }

  loadAll(): RunRow[] {
    const rows = this.db
      .prepare("SELECT thread_id, created_at, last_active_at, status, budget FROM runs")
      .all() as Array<{
      thread_id: string;
      created_at: number;
      last_active_at: number;
      status: RunStatus;
      budget: string;
    }>;
    return rows.flatMap(row => {
      try {
        return [{
          threadId: row.thread_id,
          createdAt: row.created_at,
          lastActiveAt: row.last_active_at,
          status: row.status,
          budget: JSON.parse(row.budget) as BudgetSnapshot,
        }];
      } catch {
        // A corrupt row must not stop the server from starting.
        return [];
      }
    });
  }

  upsert(row: RunRow): void {
    this.db.prepare(
      `INSERT INTO runs (thread_id, created_at, last_active_at, status, budget)
       VALUES (@threadId, @createdAt, @lastActiveAt, @status, @budget)
       ON CONFLICT(thread_id) DO UPDATE SET
         last_active_at = excluded.last_active_at,
         status         = excluded.status,
         budget         = excluded.budget`
    ).run({
      threadId: row.threadId,
      createdAt: row.createdAt,
      lastActiveAt: row.lastActiveAt,
      status: row.status,
      budget: JSON.stringify(row.budget),
    });
  }

  remove(threadId: string): void {
    this.db.prepare("DELETE FROM runs WHERE thread_id = ?").run(threadId);
  }
}

/**
 * In-memory cache in front of an optional durable backend. Reads are memory
 * only, writes go to both, so resume latency never depends on disk.
 */
export class RunStore {
  private readonly cache = new Map<string, RunContext>();

  constructor(
    private readonly backend: RunStoreBackend | null,
    private readonly budgetFactory: (restore?: BudgetSnapshot) => RunBudget
  ) {}

  /** Reloads persisted runs on startup. Returns how many were restored. */
  hydrate(): number {
    if (!this.backend) return 0;
    let restored = 0;
    for (const row of this.backend.loadAll()) {
      // A run interrupted mid-flight has no client to stream to and cannot be
      // resumed (the client only resumes 'paused' runs), so drop it.
      if (row.status === "running") {
        this.backend.remove(row.threadId);
        continue;
      }
      this.cache.set(row.threadId, {
        threadId: row.threadId,
        budget: this.budgetFactory(row.budget),
        createdAt: row.createdAt,
        lastActiveAt: row.lastActiveAt,
        status: row.status,
      });
      restored++;
    }
    return restored;
  }

  set(ctx: RunContext): void {
    this.cache.set(ctx.threadId, ctx);
    this.persist(ctx);
  }

  get(threadId: string): RunContext | undefined {
    return this.cache.get(threadId);
  }

  delete(threadId: string): void {
    this.cache.delete(threadId);
    this.backend?.remove(threadId);
  }

  values(): IterableIterator<RunContext> {
    return this.cache.values();
  }

  /** Writes the current status and budget, e.g. after a pause or a spend. */
  persist(ctx: RunContext): void {
    this.backend?.upsert({
      threadId: ctx.threadId,
      createdAt: ctx.createdAt,
      lastActiveAt: ctx.lastActiveAt,
      status: ctx.status,
      budget: ctx.budget.snapshot(),
    });
  }
}

/** Convenience factory used by the server and by tests. */
export function createRunStore(
  backend: RunStoreBackend | null,
  budgetFactory: (restore?: BudgetSnapshot) => RunBudget
): RunStore {
  const store = new RunStore(backend, budgetFactory);
  store.hydrate();
  return store;
}
