import { beforeAll, describe, expect, it } from "vitest";
import request from "supertest";
import type { Express } from "express";

// The server reads its configuration at module load, so the environment has to
// be in place before it is imported.
process.env.NODE_ENV = "test";
process.env.OPENAI_API_KEY = "sk-test-not-used";
process.env.APP_API_KEY = "test-key";
process.env.CHECKPOINT_DB = "memory";
process.env.MODEL_ALLOWLIST = "gpt-4o,gpt-4.1";

const AUTH = { Authorization: "Bearer test-key" };

let app: Express;

beforeAll(async () => {
  ({ app } = await import("../src/server"));
});

describe("health and configuration endpoints", () => {
  it("reports liveness without authentication", async () => {
    const res = await request(app).get("/healthz");
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: "ok" });
  });

  it("tells the UI that a key is required, without leaking it", async () => {
    const res = await request(app).get("/api/config");
    expect(res.status).toBe(200);
    expect(res.body.requiresApiKey).toBe(true);
    expect(JSON.stringify(res.body)).not.toContain("test-key");
  });
});

describe("authentication boundary", () => {
  it("rejects a request with no credentials", async () => {
    const res = await request(app).post("/api/generate").send({ topic: "x" });
    expect(res.status).toBe(401);
  });

  it("rejects a wrong key", async () => {
    const res = await request(app)
      .post("/api/generate")
      .set("Authorization", "Bearer wrong-key")
      .send({ topic: "x" });
    expect(res.status).toBe(401);
  });

  it("rejects an unauthenticated model listing", async () => {
    const res = await request(app).get("/api/models");
    expect(res.status).toBe(401);
  });

  it("accepts the x-api-key header as well as a bearer token", async () => {
    const res = await request(app).post("/api/generate").set("x-api-key", "test-key").send({});
    expect(res.status).toBe(400); // authenticated, then failed validation
  });
});

describe("request validation", () => {
  it("requires a topic", async () => {
    const res = await request(app).post("/api/generate").set(AUTH).send({});
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/topic/i);
  });

  it("caps the topic length", async () => {
    const res = await request(app).post("/api/generate").set(AUTH).send({ topic: "a".repeat(301) });
    expect(res.status).toBe(400);
  });

  it("rejects an oversized body", async () => {
    const res = await request(app)
      .post("/api/generate")
      .set(AUTH)
      .send({ topic: "x", sources: Array.from({ length: 5000 }, () => "y".repeat(100)) });
    expect([400, 413]).toContain(res.status);
  });

  it("requires a UUID thread id when continuing", async () => {
    const res = await request(app).post("/api/generate/continue").set(AUTH).send({ threadId: "nope" });
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/threadId/i);
  });

  it("validates the outline schema on resume", async () => {
    const res = await request(app)
      .post("/api/generate/continue")
      .set(AUTH)
      .send({ threadId: "3f1a5c62-0f6e-4a0e-9a2e-6a0f2b7c9d11", outline: { seoTitle: "t" } });
    expect(res.status).toBe(400);
  });
});

describe("thread lifecycle", () => {
  it("404s for a paused thread that does not exist", async () => {
    const res = await request(app)
      .post("/api/generate/continue")
      .set(AUTH)
      .send({
        threadId: "3f1a5c62-0f6e-4a0e-9a2e-6a0f2b7c9d11",
        outline: {
          seoTitle: "Title",
          metaDescription: "Desc",
          sections: [{ heading: "H", keyPoints: ["p"] }],
          estimatedReadingTime: 2,
        },
      });
    expect(res.status).toBe(404);
  });
});

describe("model registry", () => {
  it("serves only allowlisted models", async () => {
    const res = await request(app).get("/api/models").set(AUTH);
    expect(res.status).toBe(200);
    expect(res.body.map((m: { id: string }) => m.id).sort()).toEqual(["gpt-4.1", "gpt-4o"]);
  });
});
