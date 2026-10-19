import { describe, expect, it } from "vitest";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import type { ChatOpenAI } from "@langchain/openai";
import { invokeText } from "../src/llm";
import { RunBudget, resolvePricing, UsageCallbackHandler } from "../src/observability";
import type { RunContext } from "../src/run-store";

// Regression test: model calls used to pass only `{ signal }`, which replaces
// the LangGraph run config and silently disables the usage callback, leaving
// the token budget counting almost nothing (only the researcher, which records
// its own usage). The run config must be forwarded.

function makeContext(): { ctx: RunContext; budget: RunBudget } {
  const budget = new RunBudget({ maxTotalTokens: 100_000, maxLlmCalls: 20 }, resolvePricing(undefined));
  return {
    ctx: { threadId: "t", budget, createdAt: 0, lastActiveAt: 0, status: "running" },
    budget,
  };
}

const model = () =>
  new FakeListChatModel({ responses: ["ok"] }) as unknown as ChatOpenAI;

describe("usage accounting plumbing", () => {
  it("records a call when the run config is forwarded", async () => {
    const { ctx, budget } = makeContext();
    const handler = new UsageCallbackHandler(budget, ctx.threadId);

    const text = await invokeText(model(), [{ role: "user", content: "hi" }], {
      ctx,
      node: "writer",
      config: { callbacks: [handler] },
    });

    expect(text).toBe("ok");
    expect(budget.calls).toBe(1);
  });

  it("keeps the abort signal when both are provided", async () => {
    const { ctx, budget } = makeContext();
    const handler = new UsageCallbackHandler(budget, ctx.threadId);
    const controller = new AbortController();

    await invokeText(model(), [{ role: "user", content: "hi" }], {
      ctx,
      node: "writer",
      signal: controller.signal,
      config: { callbacks: [handler] },
    });

    expect(budget.calls).toBe(1);
  });

  it("documents why dropping the config breaks accounting", async () => {
    const { ctx, budget } = makeContext();

    // No config: callbacks never reach the model, so nothing is recorded.
    await invokeText(model(), [{ role: "user", content: "hi" }], { ctx, node: "writer" });

    expect(budget.calls).toBe(0);
  });
});

// The eval harness and the server both attach the usage handler to
// graph.stream(); this pins that a graph-level callback actually reaches the
// model call inside a node, which is what makes per-stage cost visible.
describe("graph-level callback propagation", () => {
  it("records usage reported from inside a node", async () => {
    const { Annotation, StateGraph, START, END } = await import("@langchain/langgraph");
    const budget = new RunBudget({ maxTotalTokens: 1000, maxLlmCalls: 5 }, resolvePricing(undefined));
    const ctx: RunContext = { threadId: "g1", budget, createdAt: 0, lastActiveAt: 0, status: "running" };

    const State = Annotation.Root({ out: Annotation<string>() });
    const graph = new StateGraph(State)
      .addNode("writer", async (_state, config: unknown) => ({
        out: await invokeText(model(), [{ role: "user", content: "hi" }], {
          ctx,
          node: "writer",
          config: config as never,
        }),
      }))
      .addEdge(START, "writer")
      .addEdge("writer", END)
      .compile();

    for await (const _chunk of await graph.stream({ out: "" }, {
      streamMode: "updates",
      configurable: { thread_id: "g1" },
      callbacks: [new UsageCallbackHandler(budget, "g1")],
    })) { /* drain */ }

    expect(budget.calls).toBe(1);
    expect(budget.byNode.writer?.calls).toBe(1);
  });
});
