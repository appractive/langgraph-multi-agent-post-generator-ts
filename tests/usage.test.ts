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
