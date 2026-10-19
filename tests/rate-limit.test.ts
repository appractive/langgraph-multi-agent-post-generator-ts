import { beforeAll, describe, expect, it } from "vitest";
import request from "supertest";
import type { Express } from "express";

// Separate file: vitest gives each test file its own module registry, so this
// environment can set a tiny limit without affecting the other server tests.
process.env.NODE_ENV = "test";
process.env.OPENAI_API_KEY = "sk-test-not-used";
process.env.CHECKPOINT_DB = "memory";
process.env.RATE_LIMIT_MAX = "2";
process.env.RATE_LIMIT_WINDOW_MS = "60000";

let app: Express;

beforeAll(async () => {
  ({ app } = await import("../src/server"));
});

describe("rate limiting", () => {
  it("stops a caller after the configured number of generation requests", async () => {
    const first = await request(app).post("/api/generate").send({});
    const second = await request(app).post("/api/generate").send({});
    const third = await request(app).post("/api/generate").send({});

    expect(first.status).toBe(400); // validation, but counted
    expect(second.status).toBe(400);
    expect(third.status).toBe(429);
    expect(third.body.error).toMatch(/too many/i);
  });
});
