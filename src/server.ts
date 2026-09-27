import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import { buildEditorialGraph, type EditorialStage, type EditorialState } from './graph';
import { DEFAULT_MODEL_CONFIG } from './types';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;
const openai = new OpenAI();

app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// ─── Model Strength Classification ─────────────────────────────────────────
// Maps model ID patterns to a strength tier + friendly label.

interface ModelInfo {
  id: string;
  name: string;
  strength: 'flagship' | 'standard' | 'fast';
  strengthLabel: string;
}

const STRENGTH_RULES: { pattern: RegExp; strength: ModelInfo['strength']; label: string }[] = [
  // Mini / nano / cheap models
  { pattern: /mini|nano|3\.5/i,             strength: 'fast',      label: 'Fast & Cheap' },
  // Flagship / premier models
  { pattern: /^gpt-4o|^gpt-4\.1$|^gpt-4-turbo|^gpt-5/i, strength: 'flagship',  label: 'Flagship' },
  // Standard models
  { pattern: /^gpt-4/i,                    strength: 'standard',  label: 'Standard' },
];

function classifyModel(id: string): Pick<ModelInfo, 'strength' | 'strengthLabel'> {
  for (const rule of STRENGTH_RULES) {
    if (rule.pattern.test(id)) {
      return { strength: rule.strength, strengthLabel: rule.label };
    }
  }
  return { strength: 'standard', strengthLabel: 'Standard' };
}

// Patterns of models that are NOT standard chat completions models or are unsupported in /v1/chat/completions
const NON_CHAT_PATTERNS = [
  /instruct/i,        // e.g. gpt-3.5-turbo-instruct (only supported in v1/completions)
  /transcribe/i,      // e.g. gpt-4o-transcribe, gpt-4o-transcribe-diarize
  /diarize/i,
  /tts/i,             // e.g. gpt-4o-mini-tts
  /audio/i,           // audio models
  /realtime/i,        // realtime WebSockets/WebRTC models
  /image/i,           // image models
  /search/i,          // e.g. gpt-4o-search-preview, gpt-5-search-api
  /codex/i,
  /whisper/i,
  /embedding/i,
  /moderation/i,
  /dall-e/i,
  /davinci/i,
  /babbage/i,
  /curie/i,
  /ada/i,
  /canary/i,
  /internal/i,
  /live/i,
  /sora/i,
  /computer-use/i,
  /-pro$/i,           // e.g. gpt-5.4-pro, gpt-5.5-pro, o1-pro (only supported in v1/responses or not chat models)
  /chat-latest$/i,    // deprecated models like gpt-5.3-chat-latest
  /\d{4}-\d{2}-\d{2}/, // dated snapshots e.g. 2024-08-06
  /-\d{4}$/,          // dated snapshots e.g. -0613, -1106
  /-\d{6}$/,
  /-16k$/
];

// Explicit set of models that do not work with standard /v1/chat/completions + structured output
const DISALLOWED_CHAT_MODELS = new Set([
  'o1', 'o1-pro', 'o3', 'o3-mini', 'o4-mini',
  'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5.5',
  'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
  'gpt-6-astra', 'gpt-6-luna', 'gpt-6-sol',
  'chat-latest'
]);

function isSupportedChatModel(id: string): boolean {
  if (!id.startsWith('gpt-')) return false;
  if (DISALLOWED_CHAT_MODELS.has(id)) return false;
  return !NON_CHAT_PATTERNS.some(pattern => pattern.test(id));
}

// In-memory cache (refreshed at most every 5 minutes)
let cachedModels: ModelInfo[] | null = null;
let cacheTimestamp = 0;
const CACHE_TTL_MS = 5 * 60 * 1000;

const FALLBACK_MODELS: ModelInfo[] = [
  { id: 'gpt-4o', name: 'gpt-4o', strength: 'flagship', strengthLabel: 'Flagship' },
  { id: 'gpt-4o-mini', name: 'gpt-4o-mini', strength: 'fast', strengthLabel: 'Fast & Cheap' },
  { id: 'gpt-4.1', name: 'gpt-4.1', strength: 'flagship', strengthLabel: 'Flagship' },
  { id: 'gpt-4.1-mini', name: 'gpt-4.1-mini', strength: 'fast', strengthLabel: 'Fast & Cheap' },
  { id: 'gpt-4-turbo', name: 'gpt-4-turbo', strength: 'flagship', strengthLabel: 'Flagship' },
  { id: 'gpt-4', name: 'gpt-4', strength: 'standard', strengthLabel: 'Standard' },
  { id: 'gpt-3.5-turbo', name: 'gpt-3.5-turbo', strength: 'fast', strengthLabel: 'Fast & Cheap' },
];

const PRIORITY_ORDER: Record<string, number> = {
  'gpt-4o': 1,
  'gpt-4.1': 2,
  'gpt-4-turbo': 3,
  'gpt-5.4': 4,
  'gpt-5.2': 5,
  'gpt-5.1': 6,
  'gpt-4': 10,
  'gpt-4o-mini': 20,
  'gpt-4.1-mini': 21,
  'gpt-4.1-nano': 22,
  'gpt-5.4-mini': 23,
  'gpt-5.4-nano': 24,
  'gpt-3.5-turbo': 30,
};

app.get('/api/models', async (_req, res) => {
  try {
    const now = Date.now();
    if (cachedModels && now - cacheTimestamp < CACHE_TTL_MS) {
      return res.json(cachedModels);
    }

    const list = await openai.models.list();
    const models: ModelInfo[] = [];

    for await (const m of list) {
      if (!isSupportedChatModel(m.id)) continue;

      const { strength, strengthLabel } = classifyModel(m.id);
      models.push({ id: m.id, name: m.id, strength, strengthLabel });
    }

    // Sort: flagship first, then standard, then fast, ordered by priority
    const ORDER: Record<string, number> = { flagship: 0, standard: 1, fast: 2 };
    models.sort((a, b) => {
      const tierDiff = (ORDER[a.strength] ?? 1) - (ORDER[b.strength] ?? 1);
      if (tierDiff !== 0) return tierDiff;
      const pA = PRIORITY_ORDER[a.id] ?? 50;
      const pB = PRIORITY_ORDER[b.id] ?? 50;
      if (pA !== pB) return pA - pB;
      return a.id.localeCompare(b.id);
    });

    cachedModels = models.length > 0 ? models : FALLBACK_MODELS;
    cacheTimestamp = now;
    res.json(cachedModels);
  } catch (err: any) {
    console.error('Failed to fetch models:', err.message);
    res.json(FALLBACK_MODELS);
  }
});

const EDITORIAL_STAGES = new Set<EditorialStage>(["researcher", "planner", "writer", "reviewer"]);

function sendSse(res: any, event: unknown) {
  if (!res.destroyed && !res.writableEnded) {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }
}

async function runGraph(res: any, state: EditorialState, signal: AbortSignal) {
  let currentState = state;
  let activeStage: EditorialStage | null = null;
  let paused = false;

  try {
    const graph = buildEditorialGraph(stage => {
      activeStage = stage;
      sendSse(res, { node: stage, status: "running" });
    });
    const updates = await graph.stream(state, { streamMode: "updates", signal });

    for await (const chunk of updates) {
      if (signal.aborted || res.destroyed || res.writableEnded) return;

      for (const [nodeName, update] of Object.entries(chunk)) {
        const nodeUpdate = update as Partial<EditorialState>;
        currentState = { ...currentState, ...nodeUpdate };

        if (nodeName === "pause") {
          paused = true;
          sendSse(res, {
            node: "hitl_pause",
            data: {
              stage: nodeUpdate.pausedStage,
              nextNode: nodeUpdate.nextNode,
              fullState: currentState,
            },
          });
        } else if (EDITORIAL_STAGES.has(nodeName as EditorialStage)) {
          activeStage = nodeName as EditorialStage;
          sendSse(res, { node: nodeName, status: "done", data: nodeUpdate });
        }
      }
    }

    if (!paused && !signal.aborted && !res.destroyed && !res.writableEnded) {
      sendSse(res, { node: "complete", data: { article: currentState.article } });
    }
  } catch (error: any) {
    if (!signal.aborted && !res.destroyed && !res.writableEnded) {
      sendSse(res, {
        node: activeStage ?? "workflow",
        status: "error",
        error: error.message,
      });
    }
  } finally {
    if (!res.writableEnded) res.end();
  }
}

// ─── Endpoints ──────────────────────────────────────────────────────────────

app.post('/api/generate', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  const controller = new AbortController();
  req.on('aborted', () => controller.abort());
  res.on('close', () => {
    if (!res.writableEnded) controller.abort();
  });

  const { hitlEnabled, hitlStages: _hitlStages, ...rawState } = req.body;

  const state: EditorialState = {
    ...rawState,
    modelConfig: { ...DEFAULT_MODEL_CONFIG, ...(rawState.modelConfig || {}) },
    sources: rawState.sources || [],
    enableWebSearch: rawState.enableWebSearch !== false,
    researchSummary: "",
    outline: null,
    article: "",
    originalArticle: "",
    review: null,
    retryCount: 0,
    reviewScores: [],
    hitlStages: hitlEnabled ? ["planner"] : [],
    startNode: null,
    currentStage: null,
    pausedStage: null,
    nextNode: null,
  };

  await runGraph(res, state, controller.signal);
});

app.post('/api/generate/continue', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  const controller = new AbortController();
  req.on('aborted', () => controller.abort());
  res.on('close', () => {
    if (!res.writableEnded) controller.abort();
  });

  const { fullState, nextNode, hitlEnabled, hitlStages: _hitlStages } = req.body;

  if (!nextNode) {
    // Nothing left to run
    res.write(`data: ${JSON.stringify({ node: "complete", data: { article: fullState?.article } })}\n\n`);
    res.end();
    return;
  }

  if (!EDITORIAL_STAGES.has(nextNode as EditorialStage)) {
    sendSse(res, { node: "workflow", status: "error", error: `Unknown node: ${nextNode}` });
    res.end();
    return;
  }

  const state: EditorialState = {
    ...fullState,
    hitlStages: hitlEnabled ? ["planner"] : [],
    startNode: nextNode,
  };

  await runGraph(res, state, controller.signal);
});

app.listen(PORT, () => {
  console.log(`🚀 Server running locally at http://localhost:${PORT}`);
});
