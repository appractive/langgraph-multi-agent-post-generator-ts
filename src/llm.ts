import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { BaseMessageLike } from "@langchain/core/messages";
import type { Runnable, RunnableConfig } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { isReasoningModel } from "./models";
import { RunBudget } from "./observability";
import { RunStore, type RunContext, type RunStatus } from "./run-store";

// ─── Run context ────────────────────────────────────────────────────────────
//
// Per-thread runtime objects (budget, …) are kept in a registry keyed by the
// LangGraph thread_id instead of in graph state or `configurable`, so they are
// never serialized into checkpoints or exposed to clients.

export type { RunContext, RunStatus };

// The registry is a RunStore: in-memory by default, SQLite-backed in the
// server so paused runs survive a restart. `configureRunStore` swaps it once at
// startup, before any graph run begins.
let activeStore = new RunStore(null, () => new RunBudget({ maxTotalTokens: 1, maxLlmCalls: 1 }));

export function configureRunStore(store: RunStore): void {
  activeStore = store;
}

export function getRunStore(): RunStore {
  return activeStore;
}

export const runRegistry = {
  set: (ctx: RunContext): void => activeStore.set(ctx),
  get: (threadId: string): RunContext | undefined => activeStore.get(threadId),
  delete: (threadId: string): void => activeStore.delete(threadId),
  values: (): IterableIterator<RunContext> => activeStore.values(),
  persist: (ctx: RunContext): void => activeStore.persist(ctx),
};

export function requireRunContext(config: { configurable?: Record<string, unknown> } | undefined): RunContext {
  const threadId = config?.configurable?.thread_id;
  const ctx = typeof threadId === "string" ? activeStore.get(threadId) : undefined;
  if (!ctx) throw new Error("No run context for this thread; the run may have expired.");
  return ctx;
}

// ─── Model factory ──────────────────────────────────────────────────────────

/** Extra output allowance for reasoning models, whose hidden reasoning counts toward the cap. */
const REASONING_TOKEN_HEADROOM = 8_000;

export interface ChatModelOptions {
  temperature: number;
  maxTokens: number;
}

/**
 * Reasoning models (o-series, gpt-5+) go through the Responses API and do not
 * accept a temperature; other chat models keep using Chat Completions.
 * Every call has an explicit output-token cap.
 */
export function createChatModel(model: string, opts: ChatModelOptions): ChatOpenAI {
  const reasoning = isReasoningModel(model);
  return new ChatOpenAI({
    model,
    ...(reasoning ? {} : { temperature: opts.temperature }),
    maxTokens: reasoning ? opts.maxTokens + REASONING_TOKEN_HEADROOM : opts.maxTokens,
    useResponsesApi: reasoning,
    maxRetries: 2,
    timeout: 180_000,
  });
}

// ─── Typed invocation helpers ───────────────────────────────────────────────

export interface InvokeOptions {
  ctx: RunContext;
  node: string;
  signal?: AbortSignal;
  /**
   * The LangGraph run config. Passing it is what propagates the run's
   * callbacks (usage tracking, tracing) down to the model call — invoking with
   * only `{ signal }` replaces that config and silently disables them.
   */
  config?: RunnableConfig;
}

/** Run config for a model call: inherits callbacks/tracing from the graph. */
function runConfig(opts: InvokeOptions): RunnableConfig {
  const signal = opts.signal ?? opts.config?.signal;
  return { ...(opts.config ?? {}), ...(signal ? { signal } : {}) };
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Plain-text completion with a budget check before the call. */
export async function invokeText(
  llm: ChatOpenAI,
  messages: BaseMessageLike[],
  opts: InvokeOptions
): Promise<string> {
  opts.ctx.budget.assertCanSpend(opts.node);
  const response = await llm.invoke(messages, runConfig(opts));
  return response.text;
}

// Invoke a structured-output LLM with retry: on parse/validation failure the
// error message is fed back to the model so it can correct its own output.
export async function invokeStructuredWithRetry<T>(
  llm: Runnable<BaseLanguageModelInput, T>,
  messages: BaseMessageLike[],
  opts: InvokeOptions & { maxAttempts?: number; repair?: (err: unknown) => T | null }
): Promise<T> {
  const maxAttempts = opts.maxAttempts ?? 3;
  const currentMessages: BaseMessageLike[] = [...messages];

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    opts.ctx.budget.assertCanSpend(opts.node);
    try {
      return await llm.invoke(currentMessages, runConfig(opts));
    } catch (err: unknown) {
      if (opts.signal?.aborted) throw err;
      if (attempt === maxAttempts) {
        // Last resort: try a manual repair if provided
        const repaired = opts.repair ? opts.repair(err) : null;
        if (repaired) return repaired;
        throw err;
      }
      // Feed the failure back so the model fixes its formatting
      currentMessages.push({
        role: "user",
        content: `Your previous response failed to parse or validate: ${errorMessage(err)}. Please respond again with strictly valid JSON that matches the required schema. Do not add any extra text, trailing characters, or malformed syntax.`,
      });
    }
  }
  throw new Error("unreachable");
}

