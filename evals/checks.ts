import type { Outline, Research } from "../src/types";

// ─── Deterministic eval checks ──────────────────────────────────────────────
//
// Every assertion here is computable from the pipeline's own output, so a
// regression in grounding shows up without paying for a second judge model.

export interface EvalFailure {
  check: string;
  detail: string;
}

/** Words too common to prove a key point was covered. */
const STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
  "with", "as", "at", "by", "from", "that", "this", "these", "those", "is",
  "are", "was", "were", "be", "been", "being", "it", "its", "their", "they",
  "you", "your", "we", "our", "how", "why", "what", "when", "which", "who",
  "can", "will", "should", "must", "may", "not", "no", "all", "any", "each",
]);

export function contentWords(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g) ?? []).filter(
    word => word.length > 2 && !STOPWORDS.has(word)
  );
}

/** Bracketed citation numbers used in the article, deduplicated. */
export function citedSourceIds(article: string): number[] {
  const ids = new Set<number>();
  for (const match of article.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
    for (const part of match[1].split(",")) {
      const id = Number(part.trim());
      if (Number.isInteger(id) && id > 0) ids.add(id);
    }
  }
  return [...ids].sort((a, b) => a - b);
}

/**
 * Numbers that carry a factual claim — percentages, decimals, thousands and
 * years — which must appear in the research, otherwise the writer invented them.
 */
export function factualNumbers(text: string): string[] {
  const found = new Set<string>();
  for (const match of text.matchAll(/(?<![\w.])(\d{1,3}(?:\.\d+)?%|\d+\.\d+|\d{1,3}(?:,\d{3})+|\b(?:19|20)\d{2}\b)/g)) {
    found.add(match[1]);
  }
  return [...found];
}

export function unsupportedNumbers(article: string, researchText: string): string[] {
  const haystack = researchText.replace(/\s+/g, " ");
  return factualNumbers(article).filter(number => !haystack.includes(number));
}

/** Required phrases (from a case definition) that the article must contain. */
export function missingRequired(article: string, required: string[] | undefined): string[] {
  if (!required || required.length === 0) return [];
  const haystack = normalize(article);
  return required.filter(term => !haystack.includes(normalize(term)));
}

/** Strings that must never survive from untrusted input into the article. */
export const INJECTION_CANARIES = ["<script", "javascript:", "onerror=", "CANARY-INJECTION"];

export function injectionLeaks(article: string): string[] {
  const lower = article.toLowerCase();
  return INJECTION_CANARIES.filter(canary => lower.includes(canary.toLowerCase()));
}

function normalize(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

/**
 * Sections whose heading never appears in the article. This is the structural
 * assertion: the writer is told to use the outline headings as its H2s.
 */
export function headingGaps(outline: Outline, article: string): string[] {
  const haystack = normalize(article);
  return outline.sections
    .filter(section => {
      const heading = normalize(section.heading);
      return heading.length > 0 && !haystack.includes(heading);
    })
    .map(section => section.heading);
}

/**
 * Words that describe structure rather than content ("side-by-side
 * comparison", "summary of key points"). An outline point made only of these
 * cannot be checked by vocabulary, so it is reported, never failed.
 */
const META_WORDS = new Set([
  "comparison", "compare", "comparing", "comparative", "analysis", "analyse",
  "analyze", "overview", "summary", "conclusion", "introduction", "section",
  "sections", "point", "points", "discussed", "discuss", "various", "metrics",
  "side", "overall", "final", "key", "main", "brief", "explain", "explains",
  "describe", "describes", "provide", "provides", "future", "outlook",
  "based", "using", "according", "regarding", "example", "examples",
  "detail", "details", "specific", "specifically", "include", "includes",
]);

/** A token counts as meta when it, or every hyphen-separated part, is meta. */
function isMetaToken(word: string): boolean {
  if (META_WORDS.has(word)) return true;
  // Hyphenated filler such as "side-by-side" is meta when every part is.
  return word.split("-").every(part => META_WORDS.has(part) || STOPWORDS.has(part));
}

/** True when a key point has enough distinctive vocabulary to be checkable. */
export function isCheckable(point: string): boolean {
  return contentWords(point).some(word => !isMetaToken(word));
}

/**
 * Warn-only: key points whose distinctive vocabulary is absent. Paraphrase is
 * legitimate writing, so this surfaces in the report without failing a run.
 */
export function keyPointGaps(outline: Outline, article: string): string[] {
  const articleWords = new Set(contentWords(article));
  const gaps: string[] = [];
  for (const section of outline.sections) {
    for (const point of section.keyPoints) {
      if (!isCheckable(point)) continue;
      // Compare only distinctive vocabulary: meta words such as "comparison"
      // or "metrics" describe structure and would drown out the real content.
      const words = contentWords(point).filter(word => !isMetaToken(word));
      if (words.length === 0) continue;
      const hits = words.filter(word => articleWords.has(word)).length;
      // A point is covered when most of its distinctive vocabulary appears.
      if (hits / words.length < 0.6) gaps.push(`${section.heading}: ${point}`);
    }
  }
  return gaps;
}

/** SEO length limits the planner prompt states. */
export function outlineLimitIssues(outline: Outline): string[] {
  const issues: string[] = [];
  if (outline.seoTitle.length > 60) {
    issues.push(`seoTitle is ${outline.seoTitle.length} chars (max 60)`);
  }
  if (outline.metaDescription.length > 155) {
    issues.push(`metaDescription is ${outline.metaDescription.length} chars (max 155)`);
  }
  if (outline.sections.length < 3) {
    issues.push(`only ${outline.sections.length} sections (expected at least 3)`);
  }
  return issues;
}

export function researchText(research: Research | null, fallback: string): string {
  if (research) {
    const parts = [research.summary];
    for (const fact of research.keyFacts) parts.push(fact.text);
    for (const quote of research.quotes) parts.push(quote.text);
    for (const source of research.sources) parts.push(`${source.title ?? ""} ${source.url ?? ""}`);
    return parts.join("\n");
  }
  return fallback;
}

/** Citation and grounding assertions for one finished article. */
export function groundingFailures(
  article: string,
  research: Research | null,
  researchSummary: string
): EvalFailure[] {
  const failures: EvalFailure[] = [];

  const sourceCount = research?.sources.length ?? 0;
  const cited = citedSourceIds(article);
  const outOfRange = cited.filter(id => id > sourceCount);
  if (outOfRange.length > 0) {
    failures.push({
      check: "citations.inRange",
      detail: `cited sources that do not exist: ${outOfRange.join(", ")} (only ${sourceCount} available)`,
    });
  }
  if (sourceCount > 0 && cited.length === 0) {
    failures.push({ check: "citations.present", detail: "article cites no sources although research found some" });
  }

  const unsupported = unsupportedNumbers(article, researchText(research, researchSummary));
  if (unsupported.length > 0) {
    failures.push({
      check: "faithfulness.numbers",
      detail: `numbers absent from the research: ${unsupported.join(", ")}`,
    });
  }

  const leaks = injectionLeaks(article);
  if (leaks.length > 0) {
    failures.push({ check: "injection.leaks", detail: `article contains ${leaks.join(", ")}` });
  }

  return failures;
}
