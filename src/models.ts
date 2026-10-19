// ─── Model capability checks ────────────────────────────────────────────────
//
// Models are filtered by capability rather than by a hard-coded list of IDs:
// - only text-generation families (gpt-*, o-series) are offered;
// - modality-specific variants (audio, image, realtime, embeddings, …) are excluded;
// - reasoning models (o-series, gpt-5+) are routed to the Responses API without
//   a temperature, so new releases work without code changes;
// - very expensive tiers are excluded unless the operator allowlists them.

export interface ModelInfo {
  id: string;
  name: string;
  strength: "flagship" | "standard" | "fast";
  strengthLabel: string;
}

/** Variants that are not general text-generation models. */
const NON_TEXT_PATTERNS = [
  /instruct/i,        // legacy completions-only models
  /transcribe|diarize|tts|audio|whisper/i,
  /realtime/i,
  /image|dall-e|sora/i,
  /embedding|moderation/i,
  /search/i,          // search-preview models take different parameters
  /codex/i,
  /computer-use/i,
  /davinci|babbage|curie|\bada\b/i,
  /canary|internal/i,
];

/** Dated snapshots and moving aliases; the stable alias is offered instead. */
const SNAPSHOT_PATTERNS = [
  /\d{4}-\d{2}-\d{2}/,
  /-\d{4}$/,
  /-\d{6}$/,
  /-16k$/,
  /-latest$/i,
];

/** Expensive tiers that are opt-in through MODEL_ALLOWLIST. */
const COST_GUARD_PATTERNS = [
  /-pro\b/i,
  /deep-research/i,
];

const TEXT_MODEL_FAMILY = /^(gpt-\d|o\d)/i;

export function isReasoningModel(id: string): boolean {
  if (/^o\d/i.test(id)) return true;
  const major = id.match(/^gpt-(\d+)/i);
  return !!major && Number(major[1]) >= 5 && !/-chat\b/i.test(id);
}

export function isSupportedTextModel(id: string): boolean {
  if (!TEXT_MODEL_FAMILY.test(id)) return false;
  if (NON_TEXT_PATTERNS.some(p => p.test(id))) return false;
  if (SNAPSHOT_PATTERNS.some(p => p.test(id))) return false;
  return !COST_GUARD_PATTERNS.some(p => p.test(id));
}

export function classifyModel(id: string): Pick<ModelInfo, "strength" | "strengthLabel"> {
  if (/mini|nano|3\.5/i.test(id)) return { strength: "fast", strengthLabel: "Fast & Cheap" };
  if (/^gpt-4o|^gpt-4\.1$|^gpt-4-turbo|^o\d/i.test(id) || isReasoningModel(id)) {
    return { strength: "flagship", strengthLabel: "Flagship" };
  }
  return { strength: "standard", strengthLabel: "Standard" };
}

export function toModelInfo(id: string): ModelInfo {
  return { id, name: id, ...classifyModel(id) };
}

const TIER_ORDER: Record<ModelInfo["strength"], number> = { flagship: 0, standard: 1, fast: 2 };

/** Flagship → standard → fast, then newest-looking version first, then by name. */
export function sortModels(models: ModelInfo[]): ModelInfo[] {
  const version = (id: string) => Number(id.match(/^(?:gpt-|o)(\d+(?:\.\d+)?)/i)?.[1] ?? 0);
  return [...models].sort((a, b) =>
    TIER_ORDER[a.strength] - TIER_ORDER[b.strength]
    || version(b.id) - version(a.id)
    || a.id.localeCompare(b.id));
}

export const FALLBACK_MODEL_IDS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"];

export function parseAllowlist(value: string | undefined): string[] | null {
  const ids = (value ?? "").split(",").map(s => s.trim()).filter(Boolean);
  return ids.length ? [...new Set(ids)] : null;
}

/**
 * Source of truth for which models may be used. With MODEL_ALLOWLIST set, that
 * list is authoritative. Otherwise the live OpenAI model list is filtered by
 * capability (cached), falling back to a small known-good list.
 */
export class ModelRegistry {
  private cache: { models: ModelInfo[]; at: number } | null = null;

  constructor(
    private readonly listLiveIds: () => Promise<string[]>,
    private readonly allowlist: string[] | null,
    private readonly ttlMs = 5 * 60 * 1000
  ) {}

  async list(force = false): Promise<ModelInfo[]> {
    if (this.allowlist) return sortModels(this.allowlist.map(toModelInfo));
    if (!force && this.cache && Date.now() - this.cache.at < this.ttlMs) return this.cache.models;

    let ids: string[];
    try {
      ids = (await this.listLiveIds()).filter(isSupportedTextModel);
    } catch {
      ids = [];
    }
    const models = sortModels((ids.length ? ids : FALLBACK_MODEL_IDS).map(toModelInfo));
    // Only cache successful live lookups so a transient outage is retried.
    if (ids.length) this.cache = { models, at: Date.now() };
    return models;
  }

  async isAllowed(id: string): Promise<boolean> {
    return (await this.list()).some(m => m.id === id);
  }
}

