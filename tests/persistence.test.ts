import fs from "fs";
import os from "os";
import path from "path";
import Database from "better-sqlite3";
import { describe, expect, it } from "vitest";
import { RunBudget, resolvePricing, DEFAULT_PRICING } from "../src/observability";
import { SqliteRunBackend, createRunStore, type RunContext } from "../src/run-store";
import { createPersistence } from "../src/persistence";

const LIMITS = { maxTotalTokens: 100_000, maxLlmCalls: 20, maxUsd: 1 };

function tempDbPath(name: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "editorial-test-"));
  return path.join(dir, `${name}.sqlite`);
}

describe("budget snapshots", () => {
  it("survives serialisation with spend intact", () => {
    const budget = new RunBudget(LIMITS, resolvePricing(undefined));
    budget.record("writer", "gpt-4.1", { inputTokens: 1000, outputTokens: 500 });
    budget.recordWebSearch("researcher", 0.03);

    const restored = new RunBudget(LIMITS, resolvePricing(undefined), budget.snapshot());
    expect(restored.totalTokens).toBe(1500);
    expect(restored.webSearches).toBe(1);
    expect(restored.usd).toBeCloseTo(budget.usd, 9);
    expect(restored.byNode.writer.calls).toBe(1);
  });

  it("enforces the cost budget once a snapshot has spend", () => {
    const budget = new RunBudget({ ...LIMITS, maxUsd: 0.001 }, resolvePricing(undefined));
    budget.record("writer", "gpt-4.1", { inputTokens: 1000, outputTokens: 500 });
    expect(() => budget.assertCanSpend("reviewer")).toThrow(/cost budget/i);
  });

  it("prices runs with the default table so the USD budget is never inert", () => {
    const budget = new RunBudget(LIMITS, resolvePricing(undefined));
    budget.record("writer", "gpt-4.1", { inputTokens: 1_000_000, outputTokens: 0 });
    expect(budget.usd).toBeCloseTo(DEFAULT_PRICING["gpt-4.1"][0], 6);
  });

  it("lets an operator price override the defaults", () => {
    const pricing = resolvePricing('{"gpt-4.1":[0.5,1]}');
    expect(pricing["gpt-4.1"]).toEqual([0.5, 1]);
    expect(pricing["gpt-4o"]).toEqual(DEFAULT_PRICING["gpt-4o"]);
  });
});

describe("durable run store", () => {
  const makeContext = (threadId: string, status: RunContext["status"]): RunContext => {
    const budget = new RunBudget(LIMITS, resolvePricing(undefined));
    budget.record("researcher", "gpt-4.1-mini", { inputTokens: 100, outputTokens: 50 });
    return { threadId, budget, createdAt: 1, lastActiveAt: 2, status };
  };

  it("restores paused runs after a restart, keeping their budget", () => {
    const dbPath = tempDbPath("runs");
    const db = new Database(dbPath);

    const first = createRunStore(
      new SqliteRunBackend(db),
      restore => new RunBudget(LIMITS, resolvePricing(undefined), restore)
    );
    first.set(makeContext("thread-paused", "paused"));

    // Simulate a restart: a brand-new store over the same file.
    const second = createRunStore(
      new SqliteRunBackend(db),
      restore => new RunBudget(LIMITS, resolvePricing(undefined), restore)
    );

    const restored = second.get("thread-paused");
    expect(restored?.status).toBe("paused");
    expect(restored?.budget.totalTokens).toBe(150);
    db.close();
  });

  it("drops runs that were mid-flight when the process stopped", () => {
    const dbPath = tempDbPath("inflight");
    const db = new Database(dbPath);

    const first = createRunStore(
      new SqliteRunBackend(db),
      restore => new RunBudget(LIMITS, resolvePricing(undefined), restore)
    );
    first.set(makeContext("thread-running", "running"));

    const second = createRunStore(
      new SqliteRunBackend(db),
      restore => new RunBudget(LIMITS, resolvePricing(undefined), restore)
    );
    expect(second.get("thread-running")).toBeUndefined();
    db.close();
  });

  it("removes a run from both cache and disk", () => {
    const dbPath = tempDbPath("delete");
    const db = new Database(dbPath);
    const store = createRunStore(
      new SqliteRunBackend(db),
      restore => new RunBudget(LIMITS, resolvePricing(undefined), restore)
    );

    store.set(makeContext("thread-done", "paused"));
    store.delete("thread-done");
    expect(store.get("thread-done")).toBeUndefined();

    const reopened = createRunStore(
      new SqliteRunBackend(db),
      restore => new RunBudget(LIMITS, resolvePricing(undefined), restore)
    );
    expect(reopened.get("thread-done")).toBeUndefined();
    db.close();
  });
});

describe("persistence wiring", () => {
  it("opens a sqlite checkpointer and run store on disk", () => {
    const dbPath = tempDbPath("wiring");
    const persistence = createPersistence({
      dbPath,
      budgetFactory: restore => new RunBudget(LIMITS, resolvePricing(undefined), restore),
    });
    expect(persistence.kind).toBe("sqlite");
    expect(persistence.runStore).toBeDefined();
    persistence.close();
    expect(fs.existsSync(dbPath)).toBe(true);
  });

  it("supports an explicit ephemeral mode for development", () => {
    const persistence = createPersistence({
      dbPath: "memory",
      budgetFactory: restore => new RunBudget(LIMITS, resolvePricing(undefined), restore),
    });
    expect(persistence.kind).toBe("memory");
    persistence.close();
  });
});
