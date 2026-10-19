import fs from "fs";
import path from "path";
import Database from "better-sqlite3";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { MemorySaver, type BaseCheckpointSaver } from "@langchain/langgraph";
import { log, type BudgetSnapshot, type RunBudget } from "./observability";
import { SqliteRunBackend, createRunStore, type RunStore } from "./run-store";

// ─── Durable state ──────────────────────────────────────────────────────────
//
// Two things must outlive the process for human-in-the-loop to be safe:
//   1. the graph checkpoint (LangGraph saves at the `interrupt()`),
//   2. the run context (budget spent so far, pause status).
// Both live in one SQLite file by default. Set DATA_DIR to relocate it, or
// CHECKPOINT_DB=memory to opt into ephemeral dev-only storage.

export interface Persistence {
  /** Passed to `buildEditorialGraph`; restores paused runs after a restart. */
  checkpointer: BaseCheckpointSaver;
  runStore: RunStore;
  close(): void;
  /** "sqlite" or "memory" — logged at startup and reported by /api/config. */
  kind: "sqlite" | "memory";
}

export interface PersistenceOptions {
  dataDir?: string;
  /** Explicit database path; overrides `dataDir`. "memory" selects MemorySaver. */
  dbPath?: string;
  budgetFactory: (restore?: BudgetSnapshot) => RunBudget;
}

function openDatabase(dbPath: string): Database.Database {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  return db;
}

export function createPersistence(opts: PersistenceOptions): Persistence {
  const defaultPath = opts.dataDir
    ? path.join(opts.dataDir, "editorial.sqlite")
    : "data/editorial.sqlite";
  const configured = opts.dbPath ?? process.env.CHECKPOINT_DB ?? defaultPath;

  if (configured === "memory") {
    log("warn", "persistence_ephemeral", {
      reason: "CHECKPOINT_DB=memory: paused runs are lost on restart",
    });
    return {
      checkpointer: new MemorySaver(),
      runStore: createRunStore(null, opts.budgetFactory),
      close: () => undefined,
      kind: "memory",
    };
  }

  const dbPath = path.isAbsolute(configured) ? configured : path.resolve(process.cwd(), configured);
  const db = openDatabase(dbPath);
  const runStore = createRunStore(new SqliteRunBackend(db), opts.budgetFactory);
  log("info", "persistence_ready", { kind: "sqlite", path: dbPath });

  return {
    checkpointer: SqliteSaver.fromConnString(dbPath) as unknown as BaseCheckpointSaver,
    runStore,
    close: () => db.close(),
    kind: "sqlite",
  };
}
