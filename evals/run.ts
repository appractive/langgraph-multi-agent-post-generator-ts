import "dotenv/config";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { Command, isInterrupted, MemorySaver } from "@langchain/langgraph";
import { buildEditorialGraph, type EditorialStage, type EditorialState } from "../src/graph";
import { RunBudget, resolvePricing } from "../src/observability";
import { createRunStore, type RunContext } from "../src/run-store";
import { configureRunStore, runRegistry } from "../src/llm";
import { DEFAULT_MODEL_CONFIG, type Outline } from "../src/types";
import {
  coverageGaps,
  groundingFailures,
  outlineLimitIssues,
  type EvalFailure,
} from "./checks";

// ─── Eval runner ────────────────────────────────────────────────────────────
//
// Runs the real pipeline over a golden set and asserts grounded behaviour:
// every citation resolves, every factual number traces to the research, outline
// points are covered, and injected instructions never reach the article.
//
//   npm run eval              # whole set
//   npm run eval -- --limit 2 # first two cases
//   npm run eval -- --only hitl-edited-outline
//
// Requires OPENAI_API_KEY. Writes evals/report.json and exits 1 on failure.

interface EvalCase {
  id: string;
  kind: "grounded" | "adversarial" | "hitl";
  topic: string;
  sources?: string[];
  enableWebSearch?: boolean;
  editedOutline?: Outline;
}

interface EvalResult {
  id: string;
  kind: EvalCase["kind"];
  ok: boolean;
  failures: EvalFailure[];
  metrics: {
    latencyMs: number;
    sections: number;
    articleWords: number;
    suggestionCount: number;
    scoreBefore: number | null;
    scoreAfter: number | null;
    calls: number;
    totalTokens: number;
    estimatedUsd: number;
    webSearches: number;
  };
  outline: Outline | null;
  judgeFeedback: string[];
}

const LIMITS = {
  maxTotalTokens: Number(process.env.EVAL_MAX_TOKENS) || 400_000,
  maxLlmCalls: Number(process.env.EVAL_MAX_CALLS) || 40,
  maxUsd: Number(process.env.EVAL_MAX_USD) || 2,
};

const modelConfig = {
  researcher: process.env.EVAL_MODEL_RESEARCHER || DEFAULT_MODEL_CONFIG.researcher,
  planner: process.env.EVAL_MODEL_PLANNER || DEFAULT_MODEL_CONFIG.planner,
  writer: process.env.EVAL_MODEL_WRITER || DEFAULT_MODEL_CONFIG.writer,
  reviewer: process.env.EVAL_MODEL_REVIEWER || DEFAULT_MODEL_CONFIG.reviewer,
  judge: process.env.EVAL_MODEL_JUDGE || DEFAULT_MODEL_CONFIG.judge,
};

const MIN_JUDGE_SCORE = Number(process.env.EVAL_MIN_SCORE) || 60;

// In-memory run store: evals never pause across processes.
configureRunStore(
  createRunStore(null, restore => new RunBudget(LIMITS, resolvePricing(process.env.MODEL_PRICING_JSON), restore))
);

function parseArgs(): { limit?: number; only?: string } {
  const args = process.argv.slice(2);
  const out: { limit?: number; only?: string } = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--limit") out.limit = Number(args[++i]);
    else if (args[i] === "--only") out.only = args[++i];
  }
  return out;
}

/** Runs one case end to end, resuming through the HITL interrupt when asked. */
async function runCase(testCase: EvalCase): Promise<EvalResult> {
  const threadId = crypto.randomUUID();
  const budget = new RunBudget(LIMITS, resolvePricing(process.env.MODEL_PRICING_JSON));
  const ctx: RunContext = {
    threadId,
    budget,
    createdAt: Date.now(),
    lastActiveAt: Date.now(),
    status: "running",
  };
  runRegistry.set(ctx);

  const checkpointer = new MemorySaver();
  const graph = buildEditorialGraph(undefined, undefined, checkpointer);
  const config = { configurable: { thread_id: threadId } };

  const initialState = {
    topic: testCase.topic,
    sources: testCase.sources ?? [],
    enableWebSearch: testCase.enableWebSearch ?? false,
    research: null,
    researchSummary: "",
    outline: null,
    article: "",
    originalArticle: "",
    review: null,
    retryCount: 0,
    reviewScores: [],
    modelConfig,
    hitlStages: testCase.editedOutline ? (["planner"] as EditorialStage[]) : [],
    currentStage: null,
  };

  const startedAt = Date.now();
  let interruptSeen = false;

  for await (const chunk of await graph.stream(initialState, { ...config, streamMode: "updates" })) {
    if (isInterrupted(chunk)) interruptSeen = true;
  }

  if (interruptSeen) {
    if (!testCase.editedOutline) throw new Error("unexpected interrupt for a case without an edited outline");
    for await (const _chunk of await graph.stream(new Command({ resume: testCase.editedOutline }), {
      ...config,
      streamMode: "updates",
    })) {
      // drain
    }
  }

  const finalState = (await graph.getState(config)).values as EditorialState;
  const usage = budget.summary();

  const failures: EvalFailure[] = [];
  const outline = finalState.outline;

  if (!outline) {
    failures.push({ check: "outline.present", detail: "the planner produced no outline" });
  } else {
    for (const issue of outlineLimitIssues(outline)) {
      failures.push({ check: "outline.limits", detail: issue });
    }
    const gaps = coverageGaps(outline, finalState.article ?? "");
    if (gaps.length > 0) {
      failures.push({ check: "outline.coverage", detail: `uncovered key points: ${gaps.join(" | ")}` });
    }
  }

  failures.push(...groundingFailures(finalState.article ?? "", finalState.research ?? null, finalState.researchSummary ?? ""));

  const words = (finalState.article ?? "").split(/\s+/).filter(Boolean).length;
  if (words < 250) {
    failures.push({ check: "article.length", detail: `article is only ${words} words` });
  }

  const scoreAfter = finalState.review?.scoreAfter ?? null;
  if (testCase.kind !== "adversarial" && scoreAfter !== null && scoreAfter < MIN_JUDGE_SCORE) {
    failures.push({ check: "judge.score", detail: `judge scored the final article ${scoreAfter} (< ${MIN_JUDGE_SCORE})` });
  }

  // A human-edited outline must survive to the finished article.
  if (testCase.editedOutline) {
    if (outline?.seoTitle !== testCase.editedOutline.seoTitle) {
      failures.push({
        check: "hitl.outlineApplied",
        detail: `final outline title is "${outline?.seoTitle}", expected "${testCase.editedOutline.seoTitle}"`,
      });
    }
    const gaps = coverageGaps(testCase.editedOutline, finalState.article ?? "");
    if (gaps.length > 0) {
      failures.push({ check: "hitl.editsPreserved", detail: `edited points missing: ${gaps.join(" | ")}` });
    }
  }

  return {
    id: testCase.id,
    kind: testCase.kind,
    ok: failures.length === 0,
    failures,
    metrics: {
      latencyMs: Date.now() - startedAt,
      sections: outline?.sections.length ?? 0,
      articleWords: words,
      suggestionCount: finalState.review?.suggestions.length ?? 0,
      scoreBefore: finalState.review?.scoreBefore ?? null,
      scoreAfter,
      calls: usage.calls,
      totalTokens: usage.totalTokens,
      estimatedUsd: usage.estimatedUsd,
      webSearches: usage.webSearches,
    },
    outline: outline ?? null,
    judgeFeedback: finalState.review?.judgeFeedback ?? [],
  };
}

async function main(): Promise<void> {
  if (!process.env.OPENAI_API_KEY) {
    console.error("evals: OPENAI_API_KEY is not set — cannot run the pipeline. Exiting (skipped).");
    process.exit(2);
  }

  const { limit, only } = parseArgs();
  const corpus = JSON.parse(
    fs.readFileSync(path.join(__dirname, "cases.json"), "utf8")
  ) as { cases: EvalCase[] };

  let cases = corpus.cases;
  if (only) cases = cases.filter(c => c.id === only);
  if (limit) cases = cases.slice(0, limit);

  console.log(`Running ${cases.length} eval case(s) against ${modelConfig.writer} …`);

  const results: EvalResult[] = [];
  for (const testCase of cases) {
    process.stdout.write(`  • ${testCase.id} … `);
    try {
      const result = await runCase(testCase);
      results.push(result);
      console.log(result.ok ? "pass" : `FAIL (${result.failures.length})`);
      for (const failure of result.failures) console.log(`      - ${failure.check}: ${failure.detail}`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      results.push({
        id: testCase.id,
        kind: testCase.kind,
        ok: false,
        failures: [{ check: "run", detail: message }],
        metrics: {
          latencyMs: 0, sections: 0, articleWords: 0, suggestionCount: 0,
          scoreBefore: null, scoreAfter: null, calls: 0, totalTokens: 0,
          estimatedUsd: 0, webSearches: 0,
        },
        outline: null,
        judgeFeedback: [],
      });
      console.log(`ERROR (${message})`);
    }
  }

  const passed = results.filter(r => r.ok).length;
  const totalUsd = results.reduce((sum, r) => sum + r.metrics.estimatedUsd, 0);
  const totalTokens = results.reduce((sum, r) => sum + r.metrics.totalTokens, 0);

  console.log(`\n${passed}/${results.length} passed · ${totalTokens} tokens · $${totalUsd.toFixed(4)}`);

  const reportPath = path.join(__dirname, "report.json");
  fs.writeFileSync(
    reportPath,
    JSON.stringify({
      ranAt: new Date().toISOString(),
      models: modelConfig,
      passed,
      total: results.length,
      totalTokens,
      totalUsd: Number(totalUsd.toFixed(6)),
      results,
    }, null, 2)
  );
  console.log(`Report: ${reportPath}`);

  process.exit(passed === results.length ? 0 : 1);
}

void main();
