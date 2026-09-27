import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import type { RunnableConfig } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";
import type { Outline, OutlineSection, Review, ReviewScore, ModelConfig } from "./types";

// ─── Zod Schemas ────────────────────────────────────────────────────────────

const ResearchSummarySchema = z.object({
  summary: z.string().describe("Comprehensive summary of key facts"),
  keyFacts: z.array(z.string()).describe("5-10 key facts"),
  relevantQuotes: z.array(z.string()).describe("Verbatim quotes present in retrieved pages or supplied source text; empty if there are none"),
});

const OutlineSchema = z.object({
  seoTitle: z.string().describe("SEO-optimized title, max 60 chars"),
  metaDescription: z.string().describe("Meta description, max 155 chars"),
  sections: z.array(z.object({
    heading: z.string().describe("H2 heading"),
    keyPoints: z.array(z.string()).describe("2-4 key points"),
  })),
  estimatedReadingTime: z.number(),
});

const SuggestionSchema = z.object({
  description: z.string().describe("Brief description of the improvement that was made"),
  before: z.string().describe("Exact short excerpt (a phrase or sentence) copied verbatim from the original article that was changed or removed; empty string for pure additions"),
  after: z.string().describe("Exact short excerpt copied verbatim from the revised article that replaced the original text, or the newly added text"),
});

const ReviewSchema = z.object({
  scoreBefore: z.number().min(1).max(100).describe("Quality score of the incoming draft before applying edits"),
  suggestions: z.array(SuggestionSchema).describe("Concrete improvements that were actually applied, each with verbatim before/after excerpts; empty when no edits are needed"),
  improvedArticle: z.string().min(1).describe("The complete, polished, ready-to-publish article in full Markdown"),
});

const PostReviewScoreSchema = z.object({
  scoreAfter: z.number().min(1).max(100).describe("Quality score of the revised article after suggestions were applied"),
});

const QUALITY_CRITERIA = `Technical accuracy, clarity and readability, SEO, grammar and spelling, engagement and flow, completeness, and appropriate use of examples.`;
const QUALITY_SCALE = `90-100: excellent and publish-ready; 75-89: good with minor issues; 60-74: acceptable but needs revision; below 60: needs significant revision.`;

// ─── LangGraph State ────────────────────────────────────────────────────────

export type EditorialStage = "researcher" | "planner" | "writer" | "reviewer";

const EditorialStateAnnotation = Annotation.Root({
  topic: Annotation<string>(),
  sources: Annotation<string[]>(),
  enableWebSearch: Annotation<boolean>(),
  researchSummary: Annotation<string>(),
  outline: Annotation<Outline | null>(),
  article: Annotation<string>(),
  originalArticle: Annotation<string>(),
  review: Annotation<Review | null>(),
  retryCount: Annotation<number>(),
  reviewScores: Annotation<ReviewScore[]>(),
  modelConfig: Annotation<ModelConfig>(),
  hitlStages: Annotation<EditorialStage[]>(),
  startNode: Annotation<EditorialStage | null>(),
  currentStage: Annotation<EditorialStage | null>(),
  pausedStage: Annotation<EditorialStage | null>(),
  nextNode: Annotation<EditorialStage | null>(),
});

export type EditorialState = typeof EditorialStateAnnotation.State;
type EditorialStateType = EditorialState;
type EditorialStateUpdate = typeof EditorialStateAnnotation.Update;

// ─── LLM Configuration ──────────────────────────────────────────────────────

function createLLM(modelName: string) {
  return new ChatOpenAI({ modelName, temperature: 0.7 });
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[char]!);
}

function renderProvidedSource(reference: string): string {
  const value = reference.trim();
  const urlMatch = value.match(/https?:\/\/[^\s<>"']+/i);
  const status = urlMatch ? 'User-provided link' : 'Provided as researcher context';
  if (!urlMatch) {
    return `<li><span class="source-reference">${escapeHtml(value)}</span><span class="source-provenance">${status}</span></li>`;
  }

  const candidate = urlMatch[0].replace(/[),.;]+$/, '');
  try {
    const url = new URL(candidate);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return `<li><span class="source-reference">${escapeHtml(value)}</span><span class="source-provenance">Provided as researcher context</span></li>`;
    }
    const link = `<a href="${escapeHtml(url.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(candidate)}</a>`;
    return `<li><span class="source-reference">${escapeHtml(value).replace(escapeHtml(candidate), link)}</span><span class="source-provenance">${status}</span></li>`;
  } catch {
    return `<li><span class="source-reference">${escapeHtml(value)}</span><span class="source-provenance">Provided as researcher context</span></li>`;
  }
}

function collectWebSources(response: any): Array<{ title: string; url: string }> {
  const sources = new Map<string, string>();
  const addSource = (url: string, title = '') => {
    if (!url || !/^https?:\/\//i.test(url)) return;
    if (!sources.has(url) || (!sources.get(url) && title)) sources.set(url, title);
  };

  for (const item of response.output || []) {
    if (item.type === 'web_search_call' && 'sources' in item.action) {
      for (const source of item.action.sources || []) addSource(source.url);
    }
    if (item.type !== 'message') continue;
    for (const content of item.content || []) {
      if (content.type !== 'output_text') continue;
      for (const annotation of content.annotations || []) {
        if (annotation.type === 'url_citation') addSource(annotation.url, annotation.title);
      }
    }
  }

  return Array.from(sources, ([url, title]) => ({ title, url }));
}

function renderInternetSource(source: { title: string; url: string }): string {
  try {
    const url = new URL(source.url);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return '';
    const label = source.title || source.url;
    return `<li><a href="${escapeHtml(url.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a><span class="source-provenance">Retrieved by OpenAI web search</span></li>`;
  } catch {
    return '';
  }
}

// ─── Structured Output Helpers ──────────────────────────────────────────────

// Invoke a structured-output LLM with retry: on parse/validation failure the
// error message is fed back to the model so it can correct its own output.
async function invokeStructuredWithRetry<T>(
  llm: any,
  messages: any[],
  opts: { maxAttempts?: number; repair?: (err: any) => T | null; signal?: AbortSignal } = {}
): Promise<T> {
  const maxAttempts = opts.maxAttempts ?? 3;
  const currentMessages = [...messages];

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await llm.invoke(currentMessages, { signal: opts.signal });
    } catch (err: any) {
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
        content: `Your previous response failed to parse or validate: ${err.message}. Please respond again with strictly valid JSON that matches the required schema. Do not add any extra text, trailing characters, or malformed syntax.`,
      });
    }
  }
  throw new Error("unreachable");
}

// Attempt to salvage an outline from the model's raw (possibly malformed) text
function repairOutlineFromError(err: any): Outline | null {
  const text = typeof err?.message === "string" ? err.message : "";
  // LangChain includes the raw text in "Failed to parse. Text: ..."
  const match = text.match(/Text:\s*"?([\s\S]*?)"?\s*\.?\s*Error:/);
  const raw = match ? match[1] : null;
  if (!raw) return null;

  let parsed: any;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // Try trimming to the last complete section object before corruption
    try {
      const cleaned = raw.replace(/,\s*"estimatedReadingTime".*$/s, "}").replace(/,\s*[}\]]+\s*}?\s*$/, "}");
      parsed = JSON.parse(cleaned);
    } catch {
      return null;
    }
  }
  if (!parsed || !Array.isArray(parsed.sections)) return null;

  // Keep only well-formed sections
  const sections: OutlineSection[] = parsed.sections.filter(
    (s: any) => s && typeof s.heading === "string" && Array.isArray(s.keyPoints)
  );
  if (sections.length === 0) return null;

  return {
    seoTitle: typeof parsed.seoTitle === "string" ? parsed.seoTitle : "Untitled",
    metaDescription: typeof parsed.metaDescription === "string" ? parsed.metaDescription : "",
    sections,
    estimatedReadingTime: typeof parsed.estimatedReadingTime === "number" ? parsed.estimatedReadingTime : Math.max(1, Math.ceil(sections.length * 1.5)),
  };
}

// ─── Suggestion Verification ────────────────────────────────────────────────
//
// Reviewer suggestions are verified against a real word-level diff of the
// incoming and revised articles. Text is tokenized into lowercase words with
// Markdown syntax, punctuation, and quote/dash variants stripped, so edits that
// are invisible in the rendered comparison (curly quotes, bold markers, heading
// levels, whitespace, commas) cannot justify a claimed improvement.

function tokenizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[\u2018\u2019\u201B\u2032]/g, "'")
    .replace(/[\u201C\u201D\u201F\u2033]/g, '"')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1") // links/images -> label
    .match(/[a-z0-9]+(?:'[a-z0-9]+)*/g) ?? [];
}

// Marks which tokens of `a` were deleted and which tokens of `b` were inserted,
// using an LCS diff (common prefix/suffix trimmed first to keep it cheap).
function diffTokens(a: string[], b: string[]): { deleted: boolean[]; inserted: boolean[] } {
  const deleted = new Array<boolean>(a.length).fill(false);
  const inserted = new Array<boolean>(b.length).fill(false);

  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start++;
  let endA = a.length;
  let endB = b.length;
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) { endA--; endB--; }

  const n = endA - start;
  const m = endB - start;
  if (n === 0 || m === 0) {
    for (let i = start; i < endA; i++) deleted[i] = true;
    for (let j = start; j < endB; j++) inserted[j] = true;
    return { deleted, inserted };
  }

  // lcs[i][j] = LCS length of a[start+i..endA) and b[start+j..endB)
  const width = m + 1;
  const lcs = new Uint16Array((n + 1) * width); // article LCS lengths stay < 65536
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i * width + j] = a[start + i] === b[start + j]
        ? lcs[(i + 1) * width + j + 1] + 1
        : Math.max(lcs[(i + 1) * width + j], lcs[i * width + j + 1]);
    }
  }

  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[start + i] === b[start + j]) { i++; j++; }
    else if (lcs[(i + 1) * width + j] >= lcs[i * width + j + 1]) { deleted[start + i] = true; i++; }
    else { inserted[start + j] = true; j++; }
  }
  for (; i < n; i++) deleted[start + i] = true;
  for (; j < m; j++) inserted[start + j] = true;
  return { deleted, inserted };
}

// Returns every start index where `needle` occurs contiguously in `haystack`.
function findTokenRuns(haystack: string[], needle: string[]): number[] {
  const hits: number[] = [];
  if (needle.length === 0 || needle.length > haystack.length) return hits;
  outer: for (let i = 0; i <= haystack.length - needle.length; i++) {
    for (let k = 0; k < needle.length; k++) {
      if (haystack[i + k] !== needle[k]) continue outer;
    }
    hits.push(i);
  }
  return hits;
}

// Largest number of changed tokens covered by any occurrence of the excerpt.
// Returns -1 when the excerpt cannot be located at all.
function changedTokensInExcerpt(tokens: string[], changed: boolean[], excerpt: string[]): number {
  const hits = findTokenRuns(tokens, excerpt);
  if (hits.length === 0) return -1;
  let best = 0;
  for (const start of hits) {
    let count = 0;
    for (let k = 0; k < excerpt.length; k++) if (changed[start + k]) count++;
    best = Math.max(best, count);
  }
  return best;
}

// A real edit must add or remove at least this many words inside the quoted
// excerpts (e.g. a one-word replacement = 1 deleted + 1 inserted).
const MIN_CHANGED_WORDS = 2;
const MIN_EXCERPT_WORDS = 3;

type VerifiedSuggestion = { description: string };

// Keep only suggestions that correspond to a real, visible change:
// - `after` must be locatable in the revised article.
// - `before` (if given) must be locatable in the original article.
// - The quoted excerpts must cover at least MIN_CHANGED_WORDS words that the
//   diff shows were actually inserted (in `after`) or deleted (in `before`).
// Suggestions that quote unchanged text, or text changed only cosmetically,
// are discarded so the UI never lists an edit that did not happen.
export function verifySuggestions(
  raw: Array<{ description: unknown; before: unknown; after: unknown }>,
  originalArticle: string,
  improvedArticle: string
): VerifiedSuggestion[] {
  const originalTokens = tokenizeWords(originalArticle);
  const improvedTokens = tokenizeWords(improvedArticle);
  const { deleted, inserted } = diffTokens(originalTokens, improvedTokens);
  if (!deleted.some(Boolean) && !inserted.some(Boolean)) return [];

  const seen = new Set<string>();
  const verified: VerifiedSuggestion[] = [];

  for (const s of raw) {
    const description = typeof s?.description === "string" ? s.description.trim() : "";
    const before = typeof s?.before === "string" ? s.before : "";
    const after = typeof s?.after === "string" ? s.after : "";
    if (!description || seen.has(description)) continue;

    const afterTokens = tokenizeWords(after);
    const beforeTokens = tokenizeWords(before);
    if (afterTokens.length < MIN_EXCERPT_WORDS && beforeTokens.length < MIN_EXCERPT_WORDS) continue;

    const insertedCount = afterTokens.length
      ? changedTokensInExcerpt(improvedTokens, inserted, afterTokens)
      : 0;
    if (insertedCount < 0) continue; // `after` not found in the revision

    let deletedCount = 0;
    if (beforeTokens.length) {
      deletedCount = changedTokensInExcerpt(originalTokens, deleted, beforeTokens);
      if (deletedCount < 0) continue; // `before` not found in the original
    }

    if (insertedCount + deletedCount < MIN_CHANGED_WORDS) continue;

    seen.add(description);
    verified.push({ description });
  }

  return verified;
}

// ─── Nodes ──────────────────────────────────────────────────────────────────

export async function researcherNode(state: EditorialStateType, config?: { signal?: AbortSignal }): Promise<Partial<EditorialStateType>> {
  const openai = new OpenAI();
  const maxAttempts = 3;
  let response: any;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      response = await openai.responses.parse({
        model: state.modelConfig.researcher,
        instructions: [
          state.enableWebSearch
            ? 'You are an expert research analyst. Search the live web for current, authoritative information about the topic.'
            : 'You are an expert research analyst. Use only supplied source material and your general knowledge; do not imply that you searched the web.',
          'Use the provided references where relevant. Base factual claims on retrieved pages or supplied source text, and distinguish uncertain or conflicting evidence.',
          'Only include verbatim quotes found in retrieved pages or supplied source text. Never invent quotations, URLs, or sources.',
          'Return a useful synthesis, key facts, and quotes that can be traced to the retrieved or supplied sources.',
          'You MUST respond with strictly valid JSON matching the required schema. Do not include any extra text, markdown fences, or trailing characters outside the JSON object.'
        ].join(' '),
        input: `Topic: ${state.topic}\n\nProvided references and data:\n${state.sources.length ? state.sources.join('\n') : 'None'}`,
        tools: state.enableWebSearch ? [{ type: 'web_search', search_context_size: 'medium' }] : [],
        tool_choice: state.enableWebSearch ? 'required' : 'none',
        include: state.enableWebSearch ? ['web_search_call.action.sources'] : [],
        text: { format: zodTextFormat(ResearchSummarySchema, 'research_summary') },
        max_output_tokens: 4096,
      }, { signal: config?.signal });
      break; // success
    } catch (err: any) {
      if (config?.signal?.aborted) throw err;
      const isParseError = err instanceof SyntaxError || /invalid structured output/i.test(err?.message ?? '');
      if (!isParseError || attempt === maxAttempts) {
        if (isParseError) {
          throw new Error(`Researcher failed to return valid JSON after ${maxAttempts} attempts: ${err.message}`);
        }
        throw err;
      }
      console.warn(`Researcher structured output parse failed (attempt ${attempt}/${maxAttempts}), retrying...`);
    }
  }

  const result = response.output_parsed as z.infer<typeof ResearchSummarySchema> | null;
  if (!result) throw new Error('Researcher did not return a structured research summary.');
  const internetSources = collectWebSources(response);

  const sourceList = state.sources.length > 0
    ? `<ul class="source-reference-list">\n${state.sources.map(renderProvidedSource).join('\n')}\n</ul>`
    : '<p>No links, datasets, or source notes were supplied.</p>';
  const providedMaterial = state.sources.length > 0
    ? 'Source entries above were supplied to the researcher as prompt context. Successfully retrieved pages are listed under Internet Sources.'
    : 'No user-provided source material was available.';
  const internetSourceList = internetSources.length > 0
    ? `<ul class="source-reference-list">\n${internetSources.map(renderInternetSource).filter(Boolean).join('\n')}\n</ul>`
    : `<p>${state.enableWebSearch ? 'Web search ran but returned no citeable URLs.' : 'Web search was disabled for this run.'}</p>`;
  const internetProvenance = state.enableWebSearch
    ? `OpenAI web search retrieved ${internetSources.length} citeable source${internetSources.length === 1 ? '' : 's'}.`
    : 'No internet sources were retrieved because web search was disabled.';
  const quotes = result.relevantQuotes.length > 0
    ? `\n\n### Relevant Quotes\n${result.relevantQuotes.map((quote: string) => `> ${quote}`).join("\n\n")}`
    : '';
  const researchSummary = `## Research Summary\n${result.summary}\n\n### Key Facts\n${result.keyFacts.map((f: string) => `- ${f}`).join("\n")}${quotes}\n\n### Information Provenance\n- **Internet:** ${internetProvenance}\n- **LLM knowledge:** Used to synthesize the findings; its underlying training sources are not traceable here.\n- **User-provided material:** ${providedMaterial}\n\n### Internet Sources\n\n${internetSourceList}\n\n### User-Provided Sources and Data\n\n${sourceList}`;
  
  return { researchSummary };
}

export async function plannerNode(state: EditorialStateType, config?: { signal?: AbortSignal }): Promise<Partial<EditorialStateType>> {
  const llm = createLLM(state.modelConfig.planner).withStructuredOutput(OutlineSchema);
  const researchContext = state.researchSummary ? `\n\nResearch:\n${state.researchSummary}` : "";

  const outline = await invokeStructuredWithRetry<Outline>(llm, [
    { role: "system", content: "Create a detailed 4-6 section blog post outline. Respond with strictly valid JSON only — no extra text, no trailing characters." },
    { role: "user", content: `Topic: ${state.topic}${researchContext}` },
  ], { repair: repairOutlineFromError, signal: config?.signal });

  return { outline };
}

// Every outline section heading and key point, numbered so the writer and the
// coverage checker can refer to them. Human (HITL) edits land here, so each
// entry is treated as a mandatory requirement, not a loose hint.
function outlineRequirements(outline: Outline): string[] {
  const items: string[] = [];
  for (const section of outline.sections) {
    const heading = section.heading.trim();
    for (const point of section.keyPoints) {
      const p = point.trim();
      if (p) items.push(heading ? `[${heading}] ${p}` : p);
    }
    if (heading && section.keyPoints.every(p => !p.trim())) items.push(`[${heading}] (section must be written)`);
  }
  return items;
}

function formatOutlineForPrompt(outline: Outline): string {
  const sections = outline.sections
    .map((s: OutlineSection) => `## ${s.heading}\n${s.keyPoints.map(p => `- ${p}`).join("\n")}`)
    .join("\n\n");
  return `Title: ${outline.seoTitle}\nMeta description: ${outline.metaDescription}\n\nOutline:\n${sections}`;
}

const CoverageSchema = z.object({
  missing: z.array(z.number().int()).describe("Numbers of the requirements that the article does not clearly and explicitly address; empty if all are covered"),
});

// Ask a checker which outline requirements the draft failed to cover.
async function findMissingRequirements(
  model: string,
  requirements: string[],
  article: string,
  signal?: AbortSignal
): Promise<string[]> {
  if (requirements.length === 0) return [];
  const llm = new ChatOpenAI({ modelName: model, temperature: 0 }).withStructuredOutput(CoverageSchema);
  const result = await invokeStructuredWithRetry<z.infer<typeof CoverageSchema>>(llm, [
    {
      role: "system",
      content: "You verify that an article covers every requirement from its outline. A requirement is covered only if the article explicitly states or clearly discusses it, including its specific claim (e.g. 'no room for cats' must actually be said, not merely implied). Return the numbers of requirements that are missing or contradicted.",
    },
    {
      role: "user",
      content: `Requirements:\n${requirements.map((r, i) => `${i + 1}. ${r}`).join("\n")}\n\nArticle:\n${article}`,
    },
  ], { signal });
  return Array.from(new Set(result.missing))
    .filter(n => n >= 1 && n <= requirements.length)
    .map(n => requirements[n - 1]);
}

const WRITER_SYSTEM_PROMPT = `Write a complete blog article in Markdown format based on the outline.
- The outline may have been edited by a human. Every heading and every key point is a mandatory requirement: cover each one explicitly in its section, even if it seems unusual, off-topic, or humorous.
- Preserve the specific meaning of each key point; do not drop, soften, or contradict it. Weave it naturally into the prose.
- Use the outline's section headings as the article's H2 headings, in order.`;

export async function writerNode(state: EditorialStateType, config?: { signal?: AbortSignal }): Promise<Partial<EditorialStateType>> {
  const llm = createLLM(state.modelConfig.writer);
  const outline = state.outline!;
  const outlineText = formatOutlineForPrompt(outline);
  const reviewText = state.review ? `\n\nPrevious review suggestions:\n${state.review.suggestions.join("\n")}` : "";

  const response = await llm.invoke([
    { role: "system", content: WRITER_SYSTEM_PROMPT },
    { role: "user", content: `${outlineText}${reviewText}` },
  ], { signal: config?.signal });
  let article = String(response.content);

  // Verify the draft honors every outline point; revise once if any are missing.
  const requirements = outlineRequirements(outline);
  const missing = await findMissingRequirements(state.modelConfig.reviewer, requirements, article, config?.signal);
  if (missing.length > 0) {
    const revision = await llm.invoke([
      { role: "system", content: `${WRITER_SYSTEM_PROMPT}\n- You are revising your draft. Keep everything that is already good; only add or rewrite what is needed to cover the missing requirements. Return the complete article.` },
      { role: "user", content: `${outlineText}\n\nYour draft omitted these required outline points. Cover each one explicitly in the matching section:\n${missing.map(m => `- ${m}`).join("\n")}\n\nDraft:\n${article}` },
    ], { signal: config?.signal });
    const revised = String(revision.content).trim();
    if (revised) article = revised;
  }

  return { article };
}

export async function reviewerNode(state: EditorialStateType, config?: { signal?: AbortSignal }): Promise<Partial<EditorialStateType>> {
  const llm = new ChatOpenAI({ modelName: state.modelConfig.reviewer, temperature: 0.2 }).withStructuredOutput(ReviewSchema);
  const result = await invokeStructuredWithRetry<z.infer<typeof ReviewSchema>>(llm, [
    {
      role: "system",
      content: `You are a senior technical editor specializing in AI content. Score the incoming article before editing it using these criteria: ${QUALITY_CRITERIA}

Score scale: ${QUALITY_SCALE}

CRITICAL RULES:
- scoreBefore must evaluate only the incoming article, before any edits.
- PRESERVE the original article's length and depth. Do NOT shorten, truncate, or summarize.
- Make only targeted, surgical edits — fix errors, improve awkward phrasing, strengthen transitions.
- Apply every listed suggestion to improvedArticle. Suggestions must describe edits you actually made, not recommendations left for the reader.
- For every suggestion, copy the exact "before" excerpt verbatim from the incoming article and the exact "after" excerpt verbatim from your improvedArticle. Do not paraphrase, truncate, or rewrite the excerpts — they are checked verbatim against both texts, and suggestions whose excerpts cannot be found are discarded.
- Every suggestion must describe a real change: the "before" text must no longer appear in the revised article, or the "after" text must not already appear in the incoming article. Never quote an unchanged sentence as both "before" and "after" — such suggestions are discarded.
- If a suggestion is a pure addition with no replaced text, leave "before" as an empty string and quote the newly added text verbatim in "after". The added text must not already exist in the incoming article.
- Suggestions are verified with a word-level diff that ignores Markdown syntax, punctuation, quote styles, and whitespace. Changes of only that kind do not count, and neither does describing content the incoming article already contains (e.g. "added an introduction" when one already exists). Only list edits that add, remove, or replace actual words.
- Each suggestion must correspond to its own distinct edit. If you describe an improvement, you must actually rewrite or add the corresponding words in improvedArticle.
- Do not return an unchanged copy while listing suggestions. If no useful edits are needed, return the original article and an empty suggestions array.
- If an outline is provided, it was approved (possibly edited) by a human. Every key point in it is mandatory: never remove, soften, or contradict content that covers an outline point, even if it seems off-topic or unusual.
- The improvedArticle MUST be the complete, ready-to-publish version in full Markdown format.`
    },
    { role: "user", content: `${state.outline ? `Approved outline (all key points must remain covered):\n${formatOutlineForPrompt(state.outline)}\n\n` : ""}Review this article:\n\n${state.article}` },
  ], {
    signal: config?.signal,
    repair: (err: any): z.infer<typeof ReviewSchema> | null => {
      // Salvage whatever the model produced if it at least parsed
      const text = typeof err?.message === "string" ? err.message : "";
      const match = text.match(/Text:\s*"?\s*(\{[\s\S]*?\})\s*"?\s*\.?\s*Error:/) || text.match(/(\{[\s\S]*\})/);
      if (!match) return null;
      try {
        const raw = JSON.parse(match[1]);
        const scoreBefore = Number(raw.scoreBefore ?? raw.score);
        const article = typeof raw.improvedArticle === "string" && raw.improvedArticle.trim().length > 0
          ? raw.improvedArticle
          : state.article;
        const salvagedSuggestions = Array.isArray(raw.suggestions)
          ? verifySuggestions(raw.suggestions, state.article, article)
              .map(v => {
                const r = raw.suggestions.find((s: any) => s && s.description === v.description) ?? {};
                return { description: v.description, before: String((r as any).before ?? ""), after: String((r as any).after ?? "") };
              })
          : [];
        return {
          scoreBefore: Number.isFinite(scoreBefore) && scoreBefore > 0 ? Math.min(100, scoreBefore) : 70,
          suggestions: article.trim() === state.article.trim() ? [] : salvagedSuggestions,
          improvedArticle: article,
        };
      } catch {
        return null;
      }
    }
  });

  const scoreAfterLLM = new ChatOpenAI({ modelName: state.modelConfig.reviewer, temperature: 0.2 })
    .withStructuredOutput(PostReviewScoreSchema);
  const postReviewScore = await invokeStructuredWithRetry<z.infer<typeof PostReviewScoreSchema>>(scoreAfterLLM, [
    {
      role: "system",
      content: `You are an independent quality assessor. Score only the final article you are given, after editorial suggestions have been applied. Use these criteria: ${QUALITY_CRITERIA} Score scale: ${QUALITY_SCALE} Do not compare it with an earlier version or change the article.`
    },
    { role: "user", content: `Score this revised article after review:\n\n${result.improvedArticle}` },
  ], { signal: config?.signal });

  const changedArticle = result.improvedArticle.trim() !== state.article.trim();
  const verified = changedArticle
    ? verifySuggestions(result.suggestions, state.article, result.improvedArticle)
    : [];
  const review: Review = {
    scoreBefore: result.scoreBefore,
    suggestions: verified.map(s => s.description),
    improvedArticle: result.improvedArticle,
    scoreAfter: postReviewScore.scoreAfter,
  };

  return {
    review,
    originalArticle: state.article,
    article: review.improvedArticle,
    retryCount: state.retryCount + 1,
    reviewScores: [...(state.reviewScores || []), { before: review.scoreBefore, after: review.scoreAfter }],
  };
}

function getNextNode(state: EditorialState): EditorialStage | null {
  switch (state.currentStage) {
    case "researcher": return "planner";
    case "planner": return "writer";
    case "writer": return "reviewer";
    case "reviewer":
      return state.review && state.review.scoreAfter < 75 && state.retryCount < 2
        ? "writer"
        : null;
    default: return null;
  }
}

function withProgress(
  stage: EditorialStage,
  node: (state: EditorialState, config?: { signal?: AbortSignal }) => Promise<EditorialStateUpdate>,
  onStageStart?: (stage: EditorialStage) => void,
  onStageDone?: (stage: EditorialStage, update: EditorialStateUpdate) => void
) {
  return async (state: EditorialState, config?: RunnableConfig): Promise<EditorialStateUpdate> => {
    onStageStart?.(stage);
    const update = await node(state, { signal: config?.signal });
    onStageDone?.(stage, update);
    return { ...update, currentStage: stage };
  };
}

export function buildEditorialGraph(
  onStageStart?: (stage: EditorialStage) => void,
  onStageDone?: (stage: EditorialStage, update: EditorialStateUpdate) => void
) {
  return new StateGraph(EditorialStateAnnotation)
    .addNode("researcher", withProgress("researcher", researcherNode, onStageStart, onStageDone))
    .addNode("planner", withProgress("planner", plannerNode, onStageStart, onStageDone))
    .addNode("writer", withProgress("writer", writerNode, onStageStart, onStageDone))
    .addNode("reviewer", withProgress("reviewer", reviewerNode, onStageStart, onStageDone))
    .addNode("pause", (state: EditorialState): EditorialStateUpdate => ({
      pausedStage: state.currentStage,
      nextNode: getNextNode(state),
    }))
    .addConditionalEdges(START, (state: EditorialState) => {
      if (state.startNode) return state.startNode;
      return "researcher";
    }, {
      researcher: "researcher",
      planner: "planner",
      writer: "writer",
      reviewer: "reviewer",
    })
    .addEdge("researcher", "planner")
    .addConditionalEdges("planner", (state: EditorialState) =>
      state.hitlStages.includes("planner") ? "pause" : "writer", {
        pause: "pause",
        writer: "writer",
      })
    .addEdge("writer", "reviewer")
    .addConditionalEdges("reviewer", (state: EditorialState) => getNextNode(state) ?? "done", {
      writer: "writer",
      done: END,
    })
    .addEdge("pause", END)
    .compile();
}
