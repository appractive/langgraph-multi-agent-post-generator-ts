import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { BaseMessageLike } from "@langchain/core/messages";
import type { Runnable } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { isReasoningModel } from "./models";
import type { RunBudget } from "./observability";

// ─── Run context ────────────────────────────────────────────────────────────
//
// Per-thread runtime objects (budget, …) are kept in a registry keyed by the
// LangGraph thread_id instead of in graph state or `configurable`, so they are
// never serialized into checkpoints or exposed to clients.

export interface RunContext {
  threadId: string;
  budget: RunBudget;
  createdAt: number;
  lastActiveAt: number;
  status: "running" | "paused" | "done";
}

const runContexts = new Map<string, RunContext>();

export const runRegistry = {
  set(ctx: RunContext): void { runContexts.set(ctx.threadId, ctx); },
  get(threadId: string): RunContext | undefined { return runContexts.get(threadId); },
  delete(threadId: string): void { runContexts.delete(threadId); },
  values(): IterableIterator<RunContext> { return runContexts.values(); },
};

export function requireRunContext(config: { configurable?: Record<string, unknown> } | undefined): RunContext {
  const threadId = config?.configurable?.thread_id;
  const ctx = typeof threadId === "string" ? runContexts.get(threadId) : undefined;
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
  const response = await llm.invoke(messages, { signal: opts.signal });
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
      return await llm.invoke(currentMessages, { signal: opts.signal });
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

