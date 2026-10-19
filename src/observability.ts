import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { Serialized } from "@langchain/core/load/serializable";
import type { BaseMessage } from "@langchain/core/messages";
import type { LLMResult } from "@langchain/core/outputs";

// ─── Structured logging ─────────────────────────────────────────────────────

type LogLevel = "debug" | "info" | "warn" | "error";
const LEVELS: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };
const minLevel = LEVELS[(process.env.LOG_LEVEL as LogLevel) ?? "info"] ?? LEVELS.info;

/** One JSON object per line, so logs can be shipped to any aggregator. */
export function log(level: LogLevel, event: string, fields: Record<string, unknown> = {}): void {
  if (LEVELS[level] < minLevel) return;
  const line = JSON.stringify({ ts: new Date().toISOString(), level, event, ...fields });
  if (level === "error" || level === "warn") console.error(line);
  else console.log(line);
}

// ─── Token / cost budget ────────────────────────────────────────────────────

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
}

/** USD per 1M tokens: [input, output]. */
export type PricingTable = Record<string, [number, number]>;

export interface BudgetLimits {
  maxTotalTokens: number;
  maxLlmCalls: number;
  /** Only enforced for models present in the pricing table. */
  maxUsd?: number;
}

export class BudgetExceededError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BudgetExceededError";
  }
}

interface NodeUsage extends TokenUsage {
  calls: number;
  usd: number;
}

/**
 * Default USD-per-1M-token prices ([input, output]) so the cost budget is
 * enforced out of the box. Prices change: verify against
 * https://openai.com/api/pricing/ and override with MODEL_PRICING_JSON.
 * Longest matching prefix wins, so "gpt-4.1" also covers dated snapshots.
 */
export const DEFAULT_PRICING: PricingTable = {
  "gpt-4o": [2.5, 10],
  "gpt-4o-mini": [0.15, 0.6],
  "gpt-4.1": [2, 8],
  "gpt-4.1-mini": [0.4, 1.6],
  "gpt-4.1-nano": [0.1, 0.4],
  "gpt-4-turbo": [10, 30],
};

/** Serialised budget state, persisted so a resumed run keeps its spend. */
export interface BudgetSnapshot {
  calls: number;
  inputTokens: number;
  outputTokens: number;
  usd: number;
  webSearches: number;
  byNode: Record<string, NodeUsage>;
}

/** Operator pricing merges over the defaults; explicit entries always win. */
export function resolvePricing(json: string | undefined): PricingTable {
  return { ...DEFAULT_PRICING, ...parsePricing(json) };
}

export function parsePricing(json: string | undefined): PricingTable {
  if (!json) return {};
  try {
    const parsed: unknown = JSON.parse(json);
    if (!parsed || typeof parsed !== "object") return {};
    const table: PricingTable = {};
    for (const [model, value] of Object.entries(parsed)) {
      if (Array.isArray(value) && value.length === 2 && value.every(v => typeof v === "number" && v >= 0)) {
        table[model] = [value[0], value[1]];
      }
    }
    return table;
  } catch {
    log("warn", "pricing_parse_failed");
    return {};
  }
}

/** Longest pricing key that prefixes the model id (e.g. "gpt-4.1" for "gpt-4.1"). */
function priceFor(model: string, pricing: PricingTable): [number, number] | undefined {
  if (pricing[model]) return pricing[model];
  const key = Object.keys(pricing)
    .filter(k => model.startsWith(k))
    .sort((a, b) => b.length - a.length)[0];
  return key ? pricing[key] : undefined;
}

/**
 * Tracks every LLM call of one workflow run (across pause/resume) and refuses
 * new calls once a limit is reached.
 */
export class RunBudget {
  calls = 0;
  inputTokens = 0;
  outputTokens = 0;
  usd = 0;
  webSearches = 0;
  readonly byNode: Record<string, NodeUsage> = {};

  constructor(
    private readonly limits: BudgetLimits,
    private readonly pricing: PricingTable = {},
    restore?: BudgetSnapshot
  ) {
    if (!restore) return;
    this.calls = restore.calls;
    this.inputTokens = restore.inputTokens;
    this.outputTokens = restore.outputTokens;
    this.usd = restore.usd;
    this.webSearches = restore.webSearches ?? 0;
    for (const [node, usage] of Object.entries(restore.byNode ?? {})) {
      this.byNode[node] = { ...usage };
    }
  }

  /** Plain-object state, persisted with the run so a resume keeps its spend. */
  snapshot(): BudgetSnapshot {
    return {
      calls: this.calls,
      inputTokens: this.inputTokens,
      outputTokens: this.outputTokens,
      usd: this.usd,
      webSearches: this.webSearches,
      byNode: this.byNode,
    };
  }

  /**
   * Records one billable web-search tool call. Search calls are priced
   * separately from tokens, so they are tracked (and capped) on their own.
   */
  recordWebSearch(node: string, usdPerCall: number): void {
    this.webSearches++;
    this.calls++;
    this.usd += usdPerCall;
    const entry = (this.byNode[node] ??= { calls: 0, inputTokens: 0, outputTokens: 0, usd: 0 });
    entry.calls++;
    entry.usd += usdPerCall;
  }

  get totalTokens(): number {
    return this.inputTokens + this.outputTokens;
  }

  /** Call before every LLM request. Throws when the run has used its budget. */
  assertCanSpend(node: string): void {
    if (this.calls >= this.limits.maxLlmCalls) {
      throw new BudgetExceededError(`Run stopped at ${node}: LLM call limit of ${this.limits.maxLlmCalls} reached.`);
    }
    if (this.totalTokens >= this.limits.maxTotalTokens) {
      throw new BudgetExceededError(`Run stopped at ${node}: token budget of ${this.limits.maxTotalTokens} reached.`);
    }
    if (this.limits.maxUsd !== undefined && this.usd >= this.limits.maxUsd) {
      throw new BudgetExceededError(`Run stopped at ${node}: cost budget of $${this.limits.maxUsd} reached.`);
    }
  }

  record(node: string, model: string, usage: TokenUsage): void {
    const price = priceFor(model, this.pricing);
    const usd = price ? (usage.inputTokens * price[0] + usage.outputTokens * price[1]) / 1_000_000 : 0;
    this.calls++;
    this.inputTokens += usage.inputTokens;
    this.outputTokens += usage.outputTokens;
    this.usd += usd;
    const entry = (this.byNode[node] ??= { calls: 0, inputTokens: 0, outputTokens: 0, usd: 0 });
    entry.calls++;
    entry.inputTokens += usage.inputTokens;
    entry.outputTokens += usage.outputTokens;
    entry.usd += usd;
  }

  summary() {
    return {
      calls: this.calls,
      inputTokens: this.inputTokens,
      outputTokens: this.outputTokens,
      totalTokens: this.totalTokens,
      webSearches: this.webSearches,
      estimatedUsd: Number(this.usd.toFixed(6)),
      byNode: this.byNode,
    };
  }
}

// ─── LangChain usage callback ───────────────────────────────────────────────

function readUsage(output: LLMResult): TokenUsage | null {
  const generation = output.generations?.[0]?.[0] as { message?: { usage_metadata?: { input_tokens?: number; output_tokens?: number } } } | undefined;
  const meta = generation?.message?.usage_metadata;
  if (meta) return { inputTokens: meta.input_tokens ?? 0, outputTokens: meta.output_tokens ?? 0 };
  const legacy = output.llmOutput?.tokenUsage as { promptTokens?: number; completionTokens?: number } | undefined;
  if (legacy) return { inputTokens: legacy.promptTokens ?? 0, outputTokens: legacy.completionTokens ?? 0 };
  return null;
}

/**
 * Attached once per graph run. Attributes each chat-model call to the LangGraph
 * node that made it, records token usage into the run budget, and logs latency.
 */
export class UsageCallbackHandler extends BaseCallbackHandler {
  name = "usage-tracker";
  private readonly pending = new Map<string, { node: string; model: string; startedAt: number }>();

  constructor(private readonly budget: RunBudget, private readonly threadId: string) {
    super();
    // Record usage before invoke() returns, so the next budget check sees it.
    this.awaitHandlers = true;
  }

  override handleChatModelStart(
    _llm: Serialized,
    _messages: BaseMessage[][],
    runId: string,
    _parentRunId?: string,
    extraParams?: Record<string, unknown>,
    _tags?: string[],
    metadata?: Record<string, unknown>
  ): void {
    const invocation = extraParams?.invocation_params as { model?: string } | undefined;
    this.pending.set(runId, {
      node: typeof metadata?.langgraph_node === "string" ? metadata.langgraph_node : "unknown",
      model: String(metadata?.ls_model_name ?? invocation?.model ?? "unknown"),
      startedAt: Date.now(),
    });
  }

  override handleLLMEnd(output: LLMResult, runId: string): void {
    const call = this.pending.get(runId);
    this.pending.delete(runId);
    if (!call) return;
    const usage = readUsage(output) ?? { inputTokens: 0, outputTokens: 0 };
    this.budget.record(call.node, call.model, usage);
    log("info", "llm_call", {
      thread_id: this.threadId,
      node: call.node,
      model: call.model,
      latency_ms: Date.now() - call.startedAt,
      input_tokens: usage.inputTokens,
      output_tokens: usage.outputTokens,
    });
  }

  override handleLLMError(error: unknown, runId: string): void {
    const call = this.pending.get(runId);
    this.pending.delete(runId);
    if (!call) return;
    // Failed calls still count toward the call limit.
    this.budget.record(call.node, call.model, { inputTokens: 0, outputTokens: 0 });
    log("warn", "llm_call_failed", {
      thread_id: this.threadId,
      node: call.node,
      model: call.model,
      latency_ms: Date.now() - call.startedAt,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

