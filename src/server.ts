import express, { type Request, type Response, type NextFunction } from 'express';
import rateLimit from 'express-rate-limit';
import crypto from 'crypto';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import { z } from 'zod';
import { Command, isInterrupted } from '@langchain/langgraph';
import {
  buildEditorialGraph,
  OutlineSchema,
  sharedCheckpointer,
  type EditorialStage,
  type EditorialState,
  type HitlInterruptPayload,
} from './graph';
import { DEFAULT_MODEL_CONFIG, type ModelConfig } from './types';
import { INPUT_LIMITS } from './prompt-safety';
import { ModelRegistry, parseAllowlist } from './models';
import {
  log,
  parsePricing,
  RunBudget,
  UsageCallbackHandler,
  type BudgetLimits,
} from './observability';
import { runRegistry, type RunContext } from './llm';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;
const openai = new OpenAI();

app.use(express.json({ limit: '64kb' }));
app.use(express.static(path.join(__dirname, '../public')));

// ─── Health Check ───────────────────────────────────────────────────────────

app.get('/healthz', (_req, res) => {
  res.status(200).json({ status: 'ok' });
});

// ─── Guardrails: Authentication, Rate Limiting & Budgets ────────────────────

function requireApiKey(req: Request, res: Response, next: NextFunction): void {
  const configuredKey = process.env.APP_API_KEY?.trim();
  if (!configuredKey) {
    next();
    return;
  }

  const bearer = req.headers.authorization?.startsWith('Bearer ')
    ? req.headers.authorization.slice(7).trim()
    : undefined;
  const headerKey = typeof req.headers['x-api-key'] === 'string'
    ? req.headers['x-api-key'].trim()
    : undefined;
  const candidate = bearer || headerKey || '';

  const expectedBuf = Buffer.from(configuredKey, 'utf8');
  const candidateBuf = Buffer.from(candidate, 'utf8');
  if (
    expectedBuf.length !== candidateBuf.length ||
    !crypto.timingSafeEqual(expectedBuf, candidateBuf)
  ) {
    res.status(401).json({ error: 'Unauthorized: valid API key required.' });
    return;
  }
  next();
}

const generateLimiter = rateLimit({
  windowMs: Number(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
  max: Number(process.env.RATE_LIMIT_MAX) || 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many generation requests, please try again later.' },
});

const modelsLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 60,
  standardHeaders: true,
  legacyHeaders: false,
});

function getBudgetLimits(): BudgetLimits {
  const maxTotalTokens = Number(process.env.MAX_REQUEST_TOKENS) || 120_000;
  const maxLlmCalls = Number(process.env.MAX_LLM_CALLS) || 15;
  const maxUsdRaw = process.env.MAX_REQUEST_USD ? Number(process.env.MAX_REQUEST_USD) : undefined;
  return {
    maxTotalTokens,
    maxLlmCalls,
    ...(maxUsdRaw && Number.isFinite(maxUsdRaw) && maxUsdRaw > 0 ? { maxUsd: maxUsdRaw } : {}),
  };
}

// ─── Model Registry ─────────────────────────────────────────────────────────

export const modelRegistry = new ModelRegistry(
  async () => {
    const list = await openai.models.list();
    const ids: string[] = [];
    for await (const m of list) {
      ids.push(m.id);
    }
    return ids;
  },
  parseAllowlist(process.env.MODEL_ALLOWLIST)
);

app.get('/api/models', modelsLimiter, requireApiKey, async (req, res) => {
  try {
    const force = Boolean(req.query.t);
    const models = await modelRegistry.list(force);
    res.json(models);
  } catch (err: unknown) {
    log('error', 'models_fetch_failed', {
      error: err instanceof Error ? err.message : String(err),
    });
    const fallback = await modelRegistry.list(false);
    res.json(fallback);
  }
});

// ─── Request Validation Schemas ─────────────────────────────────────────────

const ModelConfigSchema = z.object({
  researcher: z.string().min(1).max(80).optional(),
  planner: z.string().min(1).max(80).optional(),
  writer: z.string().min(1).max(80).optional(),
  reviewer: z.string().min(1).max(80).optional(),
  judge: z.string().min(1).max(80).optional(),
}).strict();

export const GenerateRequestSchema = z.object({
  topic: z.string().trim().min(1, 'Topic is required').max(INPUT_LIMITS.topicChars),
  sources: z.array(z.string().trim().max(INPUT_LIMITS.sourceEntryChars)).max(INPUT_LIMITS.sourceEntries).optional().default([]),
  enableWebSearch: z.boolean().optional().default(true),
  hitlEnabled: z.boolean().optional().default(false),
  modelConfig: ModelConfigSchema.optional(),
});

export const ContinueRequestSchema = z.object({
  threadId: z.string().uuid('Valid threadId is required'),
  outline: OutlineSchema,
});

// ─── Thread Lifecycle & Expiry ──────────────────────────────────────────────

const THREAD_TTL_MS = 30 * 60 * 1000;

async function cleanupExpiredThreads(): Promise<void> {
  const now = Date.now();
  for (const ctx of runRegistry.values()) {
    if (now - ctx.lastActiveAt > THREAD_TTL_MS) {
      runRegistry.delete(ctx.threadId);
      await sharedCheckpointer.deleteThread(ctx.threadId).catch(() => undefined);
    }
  }
}

setInterval(() => {
  void cleanupExpiredThreads();
}, 5 * 60 * 1000).unref();

// ─── Graph Stream Runner ────────────────────────────────────────────────────

const EDITORIAL_STAGES = new Set<EditorialStage>(['researcher', 'planner', 'writer', 'reviewer']);

function sendSse(res: Response, event: unknown): void {
  if (!res.destroyed && !res.writableEnded) {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }
}

type GraphStreamInput = Parameters<ReturnType<typeof buildEditorialGraph>['stream']>[0];

async function streamEditorialRun(
  res: Response,
  ctx: RunContext,
  input: GraphStreamInput,
  signal: AbortSignal
): Promise<void> {
  let activeStage: EditorialStage | null = null;
  let interruptedPayload: HitlInterruptPayload | null = null;
  const runStart = Date.now();

  try {
    const graph = buildEditorialGraph(
      stage => {
        activeStage = stage;
        sendSse(res, { node: stage, status: 'running' });
      },
      (stage, update) => {
        sendSse(res, { node: stage, status: 'done', data: update });
      }
    );

    const callbackHandler = new UsageCallbackHandler(ctx.budget, ctx.threadId);
    const stream = await graph.stream(input, {
      streamMode: 'updates',
      signal,
      configurable: { thread_id: ctx.threadId },
      callbacks: [callbackHandler],
    });

    for await (const chunk of stream) {
      if (signal.aborted || res.destroyed || res.writableEnded) return;
      ctx.lastActiveAt = Date.now();

      if (isInterrupted(chunk)) {
        const interruptEntry = chunk.__interrupt__?.[0];
        if (interruptEntry?.value) {
          interruptedPayload = interruptEntry.value as HitlInterruptPayload;
        }
        continue;
      }

      for (const nodeName of Object.keys(chunk)) {
        if (EDITORIAL_STAGES.has(nodeName as EditorialStage)) {
          activeStage = nodeName as EditorialStage;
        }
      }
    }

    if (signal.aborted || res.destroyed || res.writableEnded) return;

    const snapshot = await graph.getState({ configurable: { thread_id: ctx.threadId } });
    const finalValues = snapshot.values as EditorialState;

    if (interruptedPayload) {
      ctx.status = 'paused';
      ctx.lastActiveAt = Date.now();
      log('info', 'run_paused', {
        thread_id: ctx.threadId,
        stage: interruptedPayload.stage,
        usage: ctx.budget.summary(),
      });
      sendSse(res, {
        node: 'hitl_pause',
        data: {
          threadId: ctx.threadId,
          stage: interruptedPayload.stage,
          outline: interruptedPayload.outline ?? finalValues.outline,
        },
      });
      return;
    }

    ctx.status = 'done';
    const usage = ctx.budget.summary();
    log('info', 'run_complete', {
      thread_id: ctx.threadId,
      latency_ms: Date.now() - runStart,
      usage,
    });
    sendSse(res, {
      node: 'complete',
      data: {
        threadId: ctx.threadId,
        article: finalValues.article,
        researchSummary: finalValues.researchSummary,
        usage,
      },
    });
    runRegistry.delete(ctx.threadId);
    await sharedCheckpointer.deleteThread(ctx.threadId).catch(() => undefined);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    log('error', 'run_failed', {
      thread_id: ctx.threadId,
      stage: activeStage ?? 'workflow',
      error: message,
      usage: ctx.budget.summary(),
    });
    if (!signal.aborted && !res.destroyed && !res.writableEnded) {
      sendSse(res, {
        node: activeStage ?? 'workflow',
        status: 'error',
        error: message,
      });
    }
    if (ctx.status !== 'paused') {
      runRegistry.delete(ctx.threadId);
      await sharedCheckpointer.deleteThread(ctx.threadId).catch(() => undefined);
    }
  } finally {
    if (!res.writableEnded) res.end();
  }
}

// ─── Endpoints ──────────────────────────────────────────────────────────────

app.post('/api/generate', generateLimiter, requireApiKey, async (req, res) => {
  const parsed = GenerateRequestSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({
      error: parsed.error.issues.map(i => `${i.path.join('.') || 'body'}: ${i.message}`).join('; '),
    });
    return;
  }

  const { topic, sources, enableWebSearch, hitlEnabled, modelConfig: rawModels } = parsed.data;
  const resolvedModels: ModelConfig = {
    researcher: rawModels?.researcher || DEFAULT_MODEL_CONFIG.researcher,
    planner: rawModels?.planner || DEFAULT_MODEL_CONFIG.planner,
    writer: rawModels?.writer || DEFAULT_MODEL_CONFIG.writer,
    reviewer: rawModels?.reviewer || DEFAULT_MODEL_CONFIG.reviewer,
    judge: rawModels?.judge || DEFAULT_MODEL_CONFIG.judge,
  };

  for (const [role, modelId] of Object.entries(resolvedModels)) {
    if (!(await modelRegistry.isAllowed(modelId))) {
      res.status(400).json({ error: `Disallowed or unsupported model for ${role}: ${modelId}` });
      return;
    }
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  const controller = new AbortController();
  req.on('aborted', () => controller.abort());
  res.on('close', () => {
    if (!res.writableEnded) controller.abort();
  });

  const threadId = crypto.randomUUID();
  const budget = new RunBudget(getBudgetLimits(), parsePricing(process.env.MODEL_PRICING_JSON));
  const ctx: RunContext = {
    threadId,
    budget,
    createdAt: Date.now(),
    lastActiveAt: Date.now(),
    status: 'running',
  };
  runRegistry.set(ctx);

  const initialState: EditorialState = {
    topic,
    sources: sources.filter(Boolean),
    enableWebSearch,
    research: null,
    researchSummary: '',
    outline: null,
    article: '',
    originalArticle: '',
    review: null,
    retryCount: 0,
    reviewScores: [],
    modelConfig: resolvedModels,
    hitlStages: hitlEnabled ? ['planner'] : [],
    currentStage: null,
  };

  await streamEditorialRun(res, ctx, initialState, controller.signal);
});

app.post('/api/generate/continue', generateLimiter, requireApiKey, async (req, res) => {
  const parsed = ContinueRequestSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({
      error: parsed.error.issues.map(i => `${i.path.join('.') || 'body'}: ${i.message}`).join('; '),
    });
    return;
  }

  const { threadId, outline } = parsed.data;
  const ctx = runRegistry.get(threadId);
  if (!ctx || ctx.status !== 'paused') {
    res.status(404).json({ error: 'Paused workflow thread not found or already completed.' });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  const controller = new AbortController();
  req.on('aborted', () => controller.abort());
  res.on('close', () => {
    if (!res.writableEnded) controller.abort();
  });

  ctx.status = 'running';
  ctx.lastActiveAt = Date.now();

  const resumeCommand = new Command<unknown, Partial<EditorialState>, EditorialStage | 'hitl_review'>({
    resume: outline,
  });
  await streamEditorialRun(res, ctx, resumeCommand, controller.signal);
});

export { app };

if (require.main === module) {
  app.listen(PORT, () => {
    log('info', 'server_listening', { port: Number(PORT), url: `http://localhost:${PORT}` });
  });
}
