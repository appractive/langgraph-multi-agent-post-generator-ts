import {
  Annotation,
  StateGraph,
  START,
  END,
  interrupt,
  MemorySaver,
  type BaseCheckpointSaver,
  type LangGraphRunnableConfig,
} from "@langchain/langgraph";
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";
import { traceable } from "langsmith/traceable";
import type {
  Outline,
  OutlineSection,
  Review,
  ReviewScore,
  ModelConfig,
  Research,
  ResearchFinding,
  RubricCriterionResult,
} from "./types";
import {
  INPUT_LIMITS,
  UNTRUSTED_DATA_POLICY,
  wrapUntrusted,
} from "./prompt-safety";
import {
  buildSourceIndex,
  resolveSourceRefs,
  formatResearchForPrompt,
  finalizeCitations,
} from "./citations";
import {
  RUBRIC_ANCHORS,
  RubricJudgementSchema,
  formatRubricForPrompt,
  scoreRubric,
  rubricFeedback,
} from "./rubric";
import {
  tokenizeWords,
  diffTokens,
  findTokenRuns,
  verifySuggestions,
} from "./suggestions";
import {
  createChatModel,
  invokeText,
  invokeStructuredWithRetry,
  requireRunContext,
  type RunContext,
} from "./llm";
import { log } from "./observability";

export { tokenizeWords, diffTokens, findTokenRuns, verifySuggestions };

// ─── Zod Schemas ────────────────────────────────────────────────────────────

const FindingOutputSchema = z.object({
  text: z.string().describe("Fact or verbatim quote"),
  sourceRefs: z.array(z.string()).describe("Full URLs of retrieved web pages or user reference labels such as 'U1', 'U2' that directly support this item; empty array if none"),
});

const ResearchSummarySchema = z.object({
  summary: z.string().describe("Comprehensive summary of key facts"),
  keyFacts: z.array(FindingOutputSchema).describe("5-10 key facts, each citing supporting sourceRefs"),
  relevantQuotes: z.array(FindingOutputSchema).describe("Verbatim quotes present in retrieved pages or supplied source text with their sourceRefs; empty if there are none"),
});

export const OutlineSchema = z.object({
  seoTitle: z.string().min(1).max(INPUT_LIMITS.outlineFieldChars).describe("SEO-optimized title, max 60 chars"),
  metaDescription: z.string().min(1).max(INPUT_LIMITS.outlineFieldChars).describe("Meta description, max 155 chars"),
  sections: z.array(z.object({
    heading: z.string().min(1).max(INPUT_LIMITS.outlineFieldChars).describe("H2 heading"),
    keyPoints: z.array(z.string().min(1).max(INPUT_LIMITS.outlineFieldChars)).min(1).max(INPUT_LIMITS.outlineKeyPoints).describe("2-4 key points"),
  })).min(1).max(INPUT_LIMITS.outlineSections),
  estimatedReadingTime: z.number().min(1).max(120),
});

const SuggestionSchema = z.object({
  description: z.string().describe("Brief description of the improvement that was made"),
  before: z.string().describe("Exact short excerpt (a phrase or sentence) copied verbatim from the original article that was changed or removed; empty string for pure additions"),
  after: z.string().describe("Exact short excerpt copied verbatim from the revised article that replaced the original text, or the newly added text"),
});

const ReviewEditSchema = z.object({
  suggestions: z.array(SuggestionSchema).describe("Concrete improvements that were actually applied, each with verbatim before/after excerpts; empty when no edits are needed"),
  improvedArticle: z.string().min(1).describe("The complete, polished, ready-to-publish article in full Markdown"),
});

// ─── LangGraph State ────────────────────────────────────────────────────────

export type EditorialStage = "researcher" | "planner" | "writer" | "reviewer";

const EditorialStateAnnotation = Annotation.Root({
  topic: Annotation<string>(),
  sources: Annotation<string[]>(),
  enableWebSearch: Annotation<boolean>(),
  research: Annotation<Research | null>(),
  researchSummary: Annotation<string>(),
  outline: Annotation<Outline | null>(),
  article: Annotation<string>(),
  originalArticle: Annotation<string>(),
  review: Annotation<Review | null>(),
  retryCount: Annotation<number>(),
  reviewScores: Annotation<ReviewScore[]>(),
  modelConfig: Annotation<ModelConfig>(),
  hitlStages: Annotation<EditorialStage[]>(),
  currentStage: Annotation<EditorialStage | null>(),
});

export type EditorialState = typeof EditorialStateAnnotation.State;
type EditorialStateType = EditorialState;
type EditorialStateUpdate = typeof EditorialStateAnnotation.Update;

export interface HitlInterruptPayload {
  stage: EditorialStage;
  outline: Outline | null;
}

// ─── HTML & Source Helpers ──────────────────────────────────────────────────

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[char]!);
}

function renderProvidedSource(reference: string, index: number): string {
  const value = reference.trim();
  const urlMatch = value.match(/https?:\/\/[^\s<>"']+/i);
  const status = urlMatch ? 'User-provided link' : 'Provided as researcher context';
  const badge = `<strong>[${index + 1}]</strong> `;
  if (!urlMatch) {
    return `<li><span class="source-reference">${badge}${escapeHtml(value)}</span><span class="source-provenance">${status}</span></li>`;
  }

  const candidate = urlMatch[0].replace(/[),.;]+$/, '');
  try {
    const url = new URL(candidate);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return `<li><span class="source-reference">${badge}${escapeHtml(value)}</span><span class="source-provenance">Provided as researcher context</span></li>`;
    }
    const link = `<a href="${escapeHtml(url.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(candidate)}</a>`;
    return `<li><span class="source-reference">${badge}${escapeHtml(value).replace(escapeHtml(candidate), link)}</span><span class="source-provenance">${status}</span></li>`;
  } catch {
    return `<li><span class="source-reference">${badge}${escapeHtml(value)}</span><span class="source-provenance">Provided as researcher context</span></li>`;
  }
}

type ParsedOpenAIResponse = Awaited<ReturnType<OpenAI["responses"]["parse"]>>;

function collectWebSources(response: ParsedOpenAIResponse): Array<{ title: string; url: string }> {
  const sources = new Map<string, string>();
  const addSource = (url?: string, title = '') => {
    if (!url || !/^https?:\/\//i.test(url)) return;
    if (!sources.has(url) || (!sources.get(url) && title)) sources.set(url, title);
  };

  for (const item of response.output || []) {
    if (item.type === 'web_search_call' && 'action' in item && item.action && typeof item.action === 'object' && 'sources' in item.action) {
      const actionSources = (item.action as { sources?: Array<{ url?: string }> }).sources;
      for (const source of actionSources || []) addSource(source.url);
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

/** Number of billable web-search tool calls in a Responses API result. */
function countWebSearchCalls(response: ParsedOpenAIResponse): number {
  return (response.output || []).filter(item => item.type === 'web_search_call').length;
}

function renderInternetSource(source: { id: number; title: string; url?: string }): string {
  if (!source.url) return '';
  try {
    const url = new URL(source.url);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return '';
    const label = source.title || source.url;
    return `<li><strong>[${source.id}]</strong> <a href="${escapeHtml(url.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a><span class="source-provenance">Retrieved by OpenAI web search</span></li>`;
  } catch {
    return '';
  }
}

// Attempt to salvage an outline from the model's raw (possibly malformed) text
export function repairOutlineFromError(err: unknown): Outline | null {
  const text = err instanceof Error ? err.message : "";
  // LangChain includes the raw text in "Failed to parse. Text: ..."
  const match = text.match(/Text:\s*"?([\s\S]*?)"?\s*\.?\s*Error:/);
  const raw = match ? match[1] : null;
  if (!raw) return null;

  let parsed: Record<string, unknown> | null = null;
  try {
    parsed = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    // Try trimming to the last complete section object before corruption
    try {
      const cleaned = raw.replace(/,\s*"estimatedReadingTime".*$/s, "}").replace(/,\s*[}\]]+\s*}?\s*$/, "}");
      parsed = JSON.parse(cleaned) as Record<string, unknown>;
    } catch {
      return null;
    }
  }
  if (!parsed || !Array.isArray(parsed.sections)) return null;

  // Keep only well-formed sections
  const sections: OutlineSection[] = parsed.sections.filter(
    (s): s is OutlineSection =>
      Boolean(s && typeof s === "object" && typeof (s as OutlineSection).heading === "string" && Array.isArray((s as OutlineSection).keyPoints))
  );
  if (sections.length === 0) return null;

  return {
    seoTitle: typeof parsed.seoTitle === "string" ? parsed.seoTitle : "Untitled",
    metaDescription: typeof parsed.metaDescription === "string" ? parsed.metaDescription : "",
    sections,
    estimatedReadingTime: typeof parsed.estimatedReadingTime === "number" ? parsed.estimatedReadingTime : Math.max(1, Math.ceil(sections.length * 1.5)),
  };
}

// ─── Nodes ──────────────────────────────────────────────────────────────────

const callResearcherResponses = traceable(
  async (
    openai: OpenAI,
    params: Parameters<OpenAI["responses"]["parse"]>[0],
    signal?: AbortSignal
  ) => openai.responses.parse(params, { signal }),
  { name: "researcher_responses_parse", run_type: "llm" }
);

export async function researcherNode(
  state: EditorialStateType,
  config?: LangGraphRunnableConfig
): Promise<Partial<EditorialStateType>> {
  const ctx = requireRunContext(config);
  const openai = new OpenAI();
  const maxAttempts = 3;
  let response: ParsedOpenAIResponse | undefined;

  const numberedRefs = state.sources.length
    ? state.sources.map((s, idx) => `[U${idx + 1}] ${s}`).join("\n")
    : "None";

  const userInput = [
    wrapUntrusted("topic", state.topic, INPUT_LIMITS.topicChars),
    wrapUntrusted("sources", numberedRefs, INPUT_LIMITS.sourceEntries * INPUT_LIMITS.sourceEntryChars),
  ].join("\n\n");

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    ctx.budget.assertCanSpend("researcher");
    const callStart = Date.now();
    try {
      const parsedResponse = await callResearcherResponses(openai, {
        model: state.modelConfig.researcher,
        instructions: [
          UNTRUSTED_DATA_POLICY,
          state.enableWebSearch
            ? 'You are an expert research analyst. Search the live web for current, authoritative information about the topic inside <topic>.'
            : 'You are an expert research analyst. Use only supplied source material inside <sources> and your general knowledge; do not imply that you searched the web.',
          'Use the provided references where relevant. Base factual claims on retrieved pages or supplied source text, and distinguish uncertain or conflicting evidence.',
          'For each keyFact and relevantQuote, populate sourceRefs with the exact full URLs of retrieved web pages or the user reference labels (e.g. "U1", "U2") that directly support it. Use an empty array if a fact comes only from general background knowledge.',
          'Only include verbatim quotes found in retrieved pages or supplied source text. Never invent quotations, URLs, or sources.',
          'You MUST respond with strictly valid JSON matching the required schema. Do not include any extra text, markdown fences, or trailing characters outside the JSON object.'
        ].join(' '),
        input: userInput,
        tools: state.enableWebSearch ? [{ type: 'web_search', search_context_size: 'medium' }] : [],
        tool_choice: state.enableWebSearch ? 'required' : 'none',
        include: state.enableWebSearch ? ['web_search_call.action.sources'] : [],
        text: { format: zodTextFormat(ResearchSummarySchema, 'research_summary') },
        max_output_tokens: 4096,
      }, config?.signal);
      response = parsedResponse;

      const usage = {
        inputTokens: parsedResponse.usage?.input_tokens ?? 0,
        outputTokens: parsedResponse.usage?.output_tokens ?? 0,
      };
      ctx.budget.record("researcher", state.modelConfig.researcher, usage);

      // Web-search tool calls are billed separately from tokens, so they count
      // against the run budget on their own (WEB_SEARCH_USD_PER_CALL).
      const searchCalls = countWebSearchCalls(parsedResponse);
      const searchUsd = Number(process.env.WEB_SEARCH_USD_PER_CALL) || 0;
      for (let i = 0; i < searchCalls; i++) ctx.budget.recordWebSearch("researcher", searchUsd);
      log("info", "llm_call", {
        thread_id: ctx.threadId,
        node: "researcher",
        model: state.modelConfig.researcher,
        latency_ms: Date.now() - callStart,
        input_tokens: usage.inputTokens,
        output_tokens: usage.outputTokens,
      });
      break; // success
    } catch (err: unknown) {
      if (config?.signal?.aborted) throw err;
      ctx.budget.record("researcher", state.modelConfig.researcher, { inputTokens: 0, outputTokens: 0 });
      const msg = err instanceof Error ? err.message : String(err);
      log("warn", "researcher_attempt_failed", {
        thread_id: ctx.threadId,
        attempt,
        max_attempts: maxAttempts,
        latency_ms: Date.now() - callStart,
        error: msg,
      });
      const isParseError = err instanceof SyntaxError || /invalid structured output/i.test(msg);
      if (!isParseError || attempt === maxAttempts) {
        if (isParseError) {
          throw new Error(`Researcher failed to return valid JSON after ${maxAttempts} attempts: ${msg}`);
        }
        throw err;
      }
    }
  }

  const result = (response?.output_parsed ?? null) as z.infer<typeof ResearchSummarySchema> | null;
  if (!result || !response) throw new Error('Researcher did not return a structured research summary.');
  const internetSources = collectWebSources(response);
  const sourceIndex = buildSourceIndex(state.sources, internetSources);

  const keyFacts: ResearchFinding[] = result.keyFacts.map(f => ({
    text: f.text,
    sourceIds: resolveSourceRefs(f.sourceRefs, sourceIndex),
  }));
  const quotes: ResearchFinding[] = result.relevantQuotes.map(q => ({
    text: q.text,
    sourceIds: resolveSourceRefs(q.sourceRefs, sourceIndex),
  }));

  const research: Research = {
    summary: result.summary,
    keyFacts,
    quotes,
    sources: sourceIndex,
  };

  const webSourceEntries = sourceIndex.filter(s => s.origin === "web");
  const sourceList = state.sources.length > 0
    ? `<ul class="source-reference-list">\n${state.sources.map(renderProvidedSource).join('\n')}\n</ul>`
    : '<p>No links, datasets, or source notes were supplied.</p>';
  const providedMaterial = state.sources.length > 0
    ? 'Source entries above were supplied to the researcher as prompt context. Successfully retrieved pages are listed under Internet Sources.'
    : 'No user-provided source material was available.';
  const internetSourceList = webSourceEntries.length > 0
    ? `<ul class="source-reference-list">\n${webSourceEntries.map(renderInternetSource).filter(Boolean).join('\n')}\n</ul>`
    : `<p>${state.enableWebSearch ? 'Web search ran but returned no citeable URLs.' : 'Web search was disabled for this run.'}</p>`;
  const internetProvenance = state.enableWebSearch
    ? `OpenAI web search retrieved ${webSourceEntries.length} citeable source${webSourceEntries.length === 1 ? '' : 's'}.`
    : 'No internet sources were retrieved because web search was disabled.';

  const formatCiteBadge = (ids: number[]) => ids.length ? ` [${ids.join(", ")}]` : "";
  const quotesBlock = quotes.length > 0
    ? `\n\n### Relevant Quotes\n${quotes.map(q => `> ${q.text}${formatCiteBadge(q.sourceIds)}`).join("\n\n")}`
    : '';
  const researchSummary = `## Research Summary\n${result.summary}\n\n### Key Facts\n${keyFacts.map(f => `- ${f.text}${formatCiteBadge(f.sourceIds)}`).join("\n")}${quotesBlock}\n\n### Information Provenance\n- **Internet:** ${internetProvenance}\n- **LLM knowledge:** Used to synthesize the findings; its underlying training sources are not traceable here.\n- **User-provided material:** ${providedMaterial}\n\n### Internet Sources\n\n${internetSourceList}\n\n### User-Provided Sources and Data\n\n${sourceList}`;

  return { research, researchSummary };
}

export async function plannerNode(
  state: EditorialStateType,
  config?: LangGraphRunnableConfig
): Promise<Partial<EditorialStateType>> {
  const ctx = requireRunContext(config);
  const llm = createChatModel(state.modelConfig.planner, { temperature: 0.7, maxTokens: 2048 })
    .withStructuredOutput(OutlineSchema);

  const promptParts = [wrapUntrusted("topic", state.topic, INPUT_LIMITS.topicChars)];
  if (state.research) {
    promptParts.push(wrapUntrusted("research", formatResearchForPrompt(state.research), INPUT_LIMITS.researchPromptChars));
  } else if (state.researchSummary) {
    promptParts.push(wrapUntrusted("research", state.researchSummary, INPUT_LIMITS.researchPromptChars));
  }

  const outline = await invokeStructuredWithRetry<Outline>(llm, [
    {
      role: "system",
      content: `${UNTRUSTED_DATA_POLICY}\nCreate a detailed 4-6 section blog post outline grounded in the supplied research. Respond with strictly valid JSON only — no extra text, no trailing characters.`,
    },
    { role: "user", content: promptParts.join("\n\n") },
  ], { ctx, node: "planner", repair: repairOutlineFromError, config });

  return { outline };
}

/**
 * Dedicated HITL node using LangGraph's built-in `interrupt()`.
 * If HITL is enabled for the planner, pauses execution and awaits a verified
 * Outline via `Command({ resume: editedOutline })` on the same `thread_id`.
 */
export async function hitlReviewNode(
  state: EditorialStateType
): Promise<Partial<EditorialStateType>> {
  if (!state.hitlStages.includes("planner")) {
    return {};
  }
  const payload: HitlInterruptPayload = {
    stage: "planner",
    outline: state.outline,
  };
  const resumed: unknown = interrupt(payload);
  const parsed = OutlineSchema.safeParse(resumed);
  if (parsed.success) {
    return { outline: parsed.data };
  }
  log("warn", "hitl_resume_payload_rejected", {
    reason: "resumed outline failed validation; keeping the planner's outline",
    issues: parsed.error.issues.map(i => `${i.path.join(".") || "outline"}: ${i.message}`),
  });
  return {};
}

// Every outline section heading and key point, numbered so the writer and the
// coverage checker can refer to them. Human (HITL) edits land here, so each
// entry is treated as a mandatory requirement, not a loose hint.
export function outlineRequirements(outline: Outline): string[] {
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
export async function findMissingRequirements(
  model: string,
  requirements: string[],
  article: string,
  ctx: RunContext,
  config?: LangGraphRunnableConfig
): Promise<string[]> {
  if (requirements.length === 0) return [];
  const llm = createChatModel(model, { temperature: 0, maxTokens: 1024 }).withStructuredOutput(CoverageSchema);
  const reqText = requirements.map((r, i) => `${i + 1}. ${r}`).join("\n");
  const result = await invokeStructuredWithRetry<z.infer<typeof CoverageSchema>>(llm, [
    {
      role: "system",
      content: `${UNTRUSTED_DATA_POLICY}\nYou verify that an article covers every requirement from its outline. A requirement is covered only if the article explicitly states or clearly discusses it, including its specific claim (e.g. 'no room for cats' must actually be said, not merely implied). Return the numbers of requirements that are missing or contradicted.`,
    },
    {
      role: "user",
      content: `${wrapUntrusted("requirements", reqText, INPUT_LIMITS.researchPromptChars)}\n\n${wrapUntrusted("article", article, INPUT_LIMITS.articlePromptChars)}`,
    },
  ], { ctx, node: "writer", config });
  return Array.from(new Set(result.missing))
    .filter(n => n >= 1 && n <= requirements.length)
    .map(n => requirements[n - 1]);
}

const WRITER_SYSTEM_PROMPT = `${UNTRUSTED_DATA_POLICY}
Write a complete blog article in Markdown format based on the outline in <outline> and grounded in the research in <research>.
- The outline may have been edited by a human. Every heading and every key point is a mandatory requirement: cover each one explicitly in its section, even if it seems unusual, off-topic, or humorous.
- Preserve the specific meaning of each key point; do not drop, soften, or contradict it. Weave it naturally into the prose.
- Use the outline's section headings as the article's H2 headings, in order.
- Ground factual claims, numbers, dates, and quotes in <research>. Do not invent facts, statistics, quotes, or sources.
- When numbered sources are provided in <research>, cite them inline using bracketed numbers like [1] or [1, 2] immediately after the supported claim or quote. Only cite source numbers that appear in <research>.
- Do not add a separate Sources or References section at the end; the system appends the verified Sources list automatically.`;

export async function writerNode(
  state: EditorialStateType,
  config?: LangGraphRunnableConfig
): Promise<Partial<EditorialStateType>> {
  const ctx = requireRunContext(config);
  const llm = createChatModel(state.modelConfig.writer, { temperature: 0.7, maxTokens: 6144 });
  const outline = state.outline!;
  const outlineBlock = wrapUntrusted("outline", formatOutlineForPrompt(outline), INPUT_LIMITS.researchPromptChars);
  const researchBlock = wrapUntrusted("research", formatResearchForPrompt(state.research), INPUT_LIMITS.researchPromptChars);

  const feedbackItems = [
    ...(state.review?.suggestions ?? []),
    ...(state.review?.judgeFeedback ?? []),
  ];
  const feedbackBlock = feedbackItems.length > 0
    ? `\n\n${wrapUntrusted("feedback", feedbackItems.join("\n"), INPUT_LIMITS.feedbackChars)}`
    : "";

  let article = await invokeText(llm, [
    { role: "system", content: WRITER_SYSTEM_PROMPT },
    { role: "user", content: `${outlineBlock}\n\n${researchBlock}${feedbackBlock}` },
  ], { ctx, node: "writer", config });

  // Verify the draft honors every outline point; revise once if any are missing.
  const requirements = outlineRequirements(outline);
  const missing = await findMissingRequirements(state.modelConfig.reviewer, requirements, article, ctx, config);
  if (missing.length > 0) {
    const missingBlock = wrapUntrusted("missing", missing.map(m => `- ${m}`).join("\n"), INPUT_LIMITS.feedbackChars);
    const draftBlock = wrapUntrusted("draft", article, INPUT_LIMITS.articlePromptChars);
    const revised = (await invokeText(llm, [
      {
        role: "system",
        content: `${WRITER_SYSTEM_PROMPT}\n- You are revising your draft. Keep everything that is already good, including existing inline [n] citations; only add or rewrite what is needed to cover the missing requirements in <missing>. Return the complete article.`,
      },
      {
        role: "user",
        content: `${outlineBlock}\n\n${researchBlock}\n\nYour draft omitted these required outline points. Cover each one explicitly in the matching section:\n${missingBlock}\n\n${draftBlock}`,
      },
    ], { ctx, node: "writer", config })).trim();
    if (revised) article = revised;
  }

  article = finalizeCitations(article, state.research?.sources ?? []);
  return { article };
}

/**
 * Scores an article using the independent judge model and the fixed weighted
 * rubric. The editor model (`reviewer`) never grades its own work.
 */
async function evaluateWithJudge(
  judgeModel: string,
  article: string,
  outline: Outline | null,
  research: Research | null,
  ctx: RunContext,
  config?: LangGraphRunnableConfig
): Promise<{ score: number; criteria: RubricCriterionResult[]; feedback: string[] }> {
  const llm = createChatModel(judgeModel, { temperature: 0, maxTokens: 2048 })
    .withStructuredOutput(RubricJudgementSchema);

  const userParts: string[] = [];
  if (outline) {
    userParts.push(wrapUntrusted("outline", formatOutlineForPrompt(outline), INPUT_LIMITS.researchPromptChars));
  }
  userParts.push(wrapUntrusted("research", formatResearchForPrompt(research), INPUT_LIMITS.researchPromptChars));
  userParts.push(wrapUntrusted("article", article, INPUT_LIMITS.articlePromptChars));

  const judgement = await invokeStructuredWithRetry(llm, [
    {
      role: "system",
      content: `${UNTRUSTED_DATA_POLICY}
You are an independent editorial judge. Evaluate the article inside <article> strictly against the fixed rubric below, verifying claims and [n] citations against <research> and requirement coverage against <outline>.

Rubric criteria:
${formatRubricForPrompt()}

Scoring anchors for every criterion (integer 0–4):
${RUBRIC_ANCHORS}

Return one entry for every criterion id with a concrete 1–2 sentence rationale and an integer score from 0 to 4. Be strict and objective.`,
    },
    { role: "user", content: userParts.join("\n\n") },
  ], { ctx, node: "reviewer", config });

  const { score, criteria } = scoreRubric(judgement);
  return { score, criteria, feedback: rubricFeedback(criteria) };
}

export async function reviewerNode(
  state: EditorialStateType,
  config?: LangGraphRunnableConfig
): Promise<Partial<EditorialStateType>> {
  const ctx = requireRunContext(config);

  // Pick a distinct judge model when reviewer and judge were configured identically
  // so the model that edits the draft is not the same model that grades the result.
  const configuredJudge = state.modelConfig.judge || state.modelConfig.writer;
  const judgeModel = configuredJudge === state.modelConfig.reviewer
    ? (state.modelConfig.reviewer === "gpt-4.1" ? "gpt-4o" : "gpt-4.1")
    : configuredJudge;

  // 1. Independent judge scores the incoming draft before any edits
  const beforeEval = await evaluateWithJudge(
    judgeModel,
    state.article,
    state.outline,
    state.research,
    ctx,
    config
  );

  // 2. Editor model performs targeted edits (does NOT assign scores)
  const editorLlm = createChatModel(state.modelConfig.reviewer, { temperature: 0.2, maxTokens: 6144 })
    .withStructuredOutput(ReviewEditSchema);

  const editorInputs: string[] = [];
  if (state.outline) {
    editorInputs.push(wrapUntrusted("outline", formatOutlineForPrompt(state.outline), INPUT_LIMITS.researchPromptChars));
  }
  editorInputs.push(wrapUntrusted("research", formatResearchForPrompt(state.research), INPUT_LIMITS.researchPromptChars));
  if (beforeEval.feedback.length > 0) {
    editorInputs.push(wrapUntrusted("feedback", beforeEval.feedback.join("\n"), INPUT_LIMITS.feedbackChars));
  }
  editorInputs.push(wrapUntrusted("article", state.article, INPUT_LIMITS.articlePromptChars));

  const editResult = await invokeStructuredWithRetry<z.infer<typeof ReviewEditSchema>>(editorLlm, [
    {
      role: "system",
      content: `${UNTRUSTED_DATA_POLICY}
You are a senior technical editor specializing in AI content. Edit the article in <article> to address any judge findings in <feedback>, verify facts and [n] citations against <research>, and ensure every point in <outline> remains covered.

CRITICAL RULES:
- PRESERVE the original article's length and depth. Do NOT shorten, truncate, or summarize.
- Make only targeted, surgical edits — fix errors, improve awkward phrasing, strengthen transitions, and ensure factual claims carry accurate [n] citations from <research>.
- Never invent facts, quotes, or source numbers not present in <research>.
- Do not add a separate Sources or References section; the system appends the verified Sources section automatically.
- Apply every listed suggestion to improvedArticle. Suggestions must describe edits you actually made, not recommendations left for the reader.
- For every suggestion, copy the exact "before" excerpt verbatim from the incoming article and the exact "after" excerpt verbatim from your improvedArticle. Do not paraphrase, truncate, or rewrite the excerpts — they are checked verbatim against both texts, and suggestions whose excerpts cannot be found are discarded.
- Every suggestion must describe a real change: the "before" text must no longer appear in the revised article, or the "after" text must not already appear in the incoming article. Never quote an unchanged sentence as both "before" and "after" — such suggestions are discarded.
- If a suggestion is a pure addition with no replaced text, leave "before" as an empty string and quote the newly added text verbatim in "after". The added text must not already exist in the incoming article.
- Suggestions are verified with a word-level diff that ignores Markdown syntax, punctuation, quote styles, and whitespace. Changes of only that kind do not count, and neither does describing content the incoming article already contains. Only list edits that add, remove, or replace actual words.
- Each suggestion must correspond to its own distinct edit. If you describe an improvement, you must actually rewrite or add the corresponding words in improvedArticle.
- Do not return an unchanged copy while listing suggestions. If no useful edits are needed, return the original article and an empty suggestions array.
- If an outline is provided, it was approved (possibly edited) by a human. Every key point in it is mandatory: never remove, soften, or contradict content that covers an outline point, even if it seems off-topic or unusual.
- The improvedArticle MUST be the complete, ready-to-publish version in full Markdown format.`
    },
    { role: "user", content: editorInputs.join("\n\n") },
  ], {
    ctx,
    node: "reviewer",
    config,
    repair: (err: unknown): z.infer<typeof ReviewEditSchema> | null => {
      const text = err instanceof Error ? err.message : "";
      const match = text.match(/Text:\s*"?\s*(\{[\s\S]*?\})\s*"?\s*\.?\s*Error:/) || text.match(/(\{[\s\S]*\})/);
      if (!match) return null;
      try {
        const raw = JSON.parse(match[1]) as Record<string, unknown>;
        const article = typeof raw.improvedArticle === "string" && raw.improvedArticle.trim().length > 0
          ? raw.improvedArticle
          : state.article;
        const rawSuggestions = Array.isArray(raw.suggestions)
          ? (raw.suggestions as Array<{ description: unknown; before: unknown; after: unknown }>)
          : [];
        const salvagedSuggestions = verifySuggestions(rawSuggestions, state.article, article)
          .map(v => {
            const r = rawSuggestions.find(s => s && s.description === v.description);
            return {
              description: v.description,
              before: String(r?.before ?? ""),
              after: String(r?.after ?? ""),
            };
          });
        return {
          suggestions: article.trim() === state.article.trim() ? [] : salvagedSuggestions,
          improvedArticle: article,
        };
      } catch {
        return null;
      }
    }
  });

  const finalizedArticle = finalizeCitations(
    editResult.improvedArticle,
    state.research?.sources ?? []
  );

  // 3. Independent judge scores the revised article with the same fixed rubric
  const afterEval = finalizedArticle.trim() === state.article.trim()
    ? beforeEval
    : await evaluateWithJudge(
        judgeModel,
        finalizedArticle,
        state.outline,
        state.research,
        ctx,
        config
      );

  const changedArticle = finalizedArticle.trim() !== state.article.trim();
  const verified = changedArticle
    ? verifySuggestions(editResult.suggestions, state.article, finalizedArticle)
    : [];

  const review: Review = {
    scoreBefore: beforeEval.score,
    suggestions: verified.map(s => s.description),
    improvedArticle: finalizedArticle,
    scoreAfter: afterEval.score,
    rubric: afterEval.criteria,
    judgeFeedback: afterEval.feedback,
    judgeModel,
  };

  return {
    review,
    originalArticle: state.article,
    article: review.improvedArticle,
    retryCount: state.retryCount + 1,
    reviewScores: [...(state.reviewScores || []), { before: review.scoreBefore, after: review.scoreAfter }],
  };
}

function getNextAfterReview(state: EditorialState): "writer" | "done" {
  return state.review && state.review.scoreAfter < 75 && state.retryCount < 2
    ? "writer"
    : "done";
}

function withProgress(
  stage: EditorialStage,
  node: (state: EditorialState, config?: LangGraphRunnableConfig) => Promise<EditorialStateUpdate>,
  onStageStart?: (stage: EditorialStage) => void,
  onStageDone?: (stage: EditorialStage, update: EditorialStateUpdate) => void
) {
  const traced = traceable(
    async (state: EditorialState, config?: LangGraphRunnableConfig) => node(state, config),
    { name: `editorial_${stage}`, run_type: "chain" }
  );

  return async (state: EditorialState, config?: LangGraphRunnableConfig): Promise<EditorialStateUpdate> => {
    const threadId = typeof config?.configurable?.thread_id === "string" ? config.configurable.thread_id : "unknown";
    const startedAt = Date.now();
    onStageStart?.(stage);
    log("info", "node_start", { thread_id: threadId, node: stage });
    try {
      const update = await traced(state, config);
      log("info", "node_end", { thread_id: threadId, node: stage, latency_ms: Date.now() - startedAt });
      onStageDone?.(stage, update);
      return { ...update, currentStage: stage };
    } catch (err: unknown) {
      log("error", "node_error", {
        thread_id: threadId,
        node: stage,
        latency_ms: Date.now() - startedAt,
        error: err instanceof Error ? err.message : String(err),
      });
      throw err;
    }
  };
}

export const sharedCheckpointer = new MemorySaver();

export function buildEditorialGraph(
  onStageStart?: (stage: EditorialStage) => void,
  onStageDone?: (stage: EditorialStage, update: EditorialStateUpdate) => void,
  checkpointer: BaseCheckpointSaver = sharedCheckpointer
) {
  return new StateGraph(EditorialStateAnnotation)
    .addNode("researcher", withProgress("researcher", researcherNode, onStageStart, onStageDone))
    .addNode("planner", withProgress("planner", plannerNode, onStageStart, onStageDone))
    .addNode("hitl_review", hitlReviewNode)
    .addNode("writer", withProgress("writer", writerNode, onStageStart, onStageDone))
    .addNode("reviewer", withProgress("reviewer", reviewerNode, onStageStart, onStageDone))
    .addConditionalEdges(START, (state: EditorialState) => {
      if (!state.enableWebSearch && (!state.sources || state.sources.length === 0)) {
        return "planner";
      }
      return "researcher";
    }, {
      researcher: "researcher",
      planner: "planner",
    })
    .addEdge("researcher", "planner")
    .addEdge("planner", "hitl_review")
    .addEdge("hitl_review", "writer")
    .addEdge("writer", "reviewer")
    .addConditionalEdges("reviewer", getNextAfterReview, {
      writer: "writer",
      done: END,
    })
    .compile({ checkpointer });
}
