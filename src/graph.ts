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

const ReviewSchema = z.object({
  scoreBefore: z.number().min(1).max(100).describe("Quality score of the incoming draft before applying edits"),
  suggestions: z.array(z.string()).describe("Concrete improvements that are actually applied; empty when no edits are needed"),
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

// ─── Nodes ──────────────────────────────────────────────────────────────────

export async function researcherNode(state: EditorialStateType, config?: { signal?: AbortSignal }): Promise<Partial<EditorialStateType>> {
  const openai = new OpenAI();
  const response = await openai.responses.parse({
    model: state.modelConfig.researcher,
    instructions: [
      state.enableWebSearch
        ? 'You are an expert research analyst. Search the live web for current, authoritative information about the topic.'
        : 'You are an expert research analyst. Use only supplied source material and your general knowledge; do not imply that you searched the web.',
      'Use the provided references where relevant. Base factual claims on retrieved pages or supplied source text, and distinguish uncertain or conflicting evidence.',
      'Only include verbatim quotes found in retrieved pages or supplied source text. Never invent quotations, URLs, or sources.',
      'Return a useful synthesis, key facts, and quotes that can be traced to the retrieved or supplied sources.'
    ].join(' '),
    input: `Topic: ${state.topic}\n\nProvided references and data:\n${state.sources.length ? state.sources.join('\n') : 'None'}`,
    tools: state.enableWebSearch ? [{ type: 'web_search', search_context_size: 'medium' }] : [],
    tool_choice: state.enableWebSearch ? 'required' : 'none',
    include: state.enableWebSearch ? ['web_search_call.action.sources'] : [],
    text: { format: zodTextFormat(ResearchSummarySchema, 'research_summary') },
    max_output_tokens: 3000,
  }, { signal: config?.signal });
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

export async function writerNode(state: EditorialStateType, config?: { signal?: AbortSignal }): Promise<Partial<EditorialStateType>> {
  const llm = createLLM(state.modelConfig.writer);
  const outlineText = state.outline!.sections.map((s: OutlineSection) => `## ${s.heading}\n${s.keyPoints.map(p => `- ${p}`).join("\n")}`).join("\n\n");
  const reviewText = state.review ? `\nPrevious review suggestions:\n${state.review.suggestions.join("\n")}` : "";

  const response = await llm.invoke([
    { role: "system", content: "Write a complete blog article in Markdown format based on the outline." },
    { role: "user", content: `Title: ${state.outline!.seoTitle}\n\nOutline:\n${outlineText}${reviewText}` },
  ], { signal: config?.signal });

  return { article: String(response.content) };
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
- Do not return an unchanged copy while listing suggestions. If no useful edits are needed, return the original article and an empty suggestions array.
- The improvedArticle MUST be the complete, ready-to-publish version in full Markdown format.`
    },
    { role: "user", content: `Review this article:\n\n${state.article}` },
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
        return {
          scoreBefore: Number.isFinite(scoreBefore) && scoreBefore > 0 ? Math.min(100, scoreBefore) : 70,
          suggestions: article.trim() === state.article.trim()
            ? []
            : (Array.isArray(raw.suggestions) ? raw.suggestions : []),
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

  const review: Review = { ...result, scoreAfter: postReviewScore.scoreAfter };

  return {
    review,
    originalArticle: state.originalArticle || state.article,
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
  onStageStart?: (stage: EditorialStage) => void
) {
  return async (state: EditorialState, config?: RunnableConfig): Promise<EditorialStateUpdate> => {
    onStageStart?.(stage);
    const update = await node(state, { signal: config?.signal });
    return { ...update, currentStage: stage };
  };
}

export function buildEditorialGraph(onStageStart?: (stage: EditorialStage) => void) {
  const pauseIfSelected = (stage: EditorialStage, nextStage: EditorialStage) => (state: EditorialState) =>
    state.hitlStages.includes(stage) ? "pause" : nextStage;

  return new StateGraph(EditorialStateAnnotation)
    .addNode("researcher", withProgress("researcher", researcherNode, onStageStart))
    .addNode("planner", withProgress("planner", plannerNode, onStageStart))
    .addNode("writer", withProgress("writer", writerNode, onStageStart))
    .addNode("reviewer", withProgress("reviewer", reviewerNode, onStageStart))
    .addNode("pause", (state: EditorialState): EditorialStateUpdate => ({
      pausedStage: state.currentStage,
      nextNode: getNextNode(state),
    }))
    .addConditionalEdges(START, (state: EditorialState) => {
      if (state.startNode) return state.startNode;
      return state.enableWebSearch !== false || state.sources.length > 0 ? "researcher" : "planner";
    }, {
      researcher: "researcher",
      planner: "planner",
      writer: "writer",
      reviewer: "reviewer",
    })
    .addConditionalEdges("researcher", pauseIfSelected("researcher", "planner"), {
      pause: "pause",
      planner: "planner",
    })
    .addConditionalEdges("planner", pauseIfSelected("planner", "writer"), {
      pause: "pause",
      writer: "writer",
    })
    .addConditionalEdges("writer", pauseIfSelected("writer", "reviewer"), {
      pause: "pause",
      reviewer: "reviewer",
    })
    .addConditionalEdges("reviewer", (state: EditorialState) => {
      if (state.hitlStages.includes("reviewer")) return "pause";
      return getNextNode(state) ?? "done";
    }, {
      pause: "pause",
      writer: "writer",
      done: END,
    })
    .addEdge("pause", END)
    .compile();
}
