import type { Research, ResearchFinding, ResearchSource } from "./types";

// ─── Source index ───────────────────────────────────────────────────────────
//
// Every source the writer may cite gets a stable number. User-supplied
// references come first (the researcher sees them as U1, U2, …, which map to
// ids 1, 2, …), followed by pages retrieved by web search. Facts and quotes are
// linked to these ids; URLs the model mentions that are not in the index are
// treated as fabricated and dropped.

const MAX_TITLE_CHARS = 200;

export function normalizeUrl(raw: string): string | null {
  try {
    const url = new URL(raw.trim());
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    url.hash = "";
    for (const key of [...url.searchParams.keys()]) {
      if (/^utm_/i.test(key)) url.searchParams.delete(key);
    }
    return url.href.replace(/\/$/, "").toLowerCase();
  } catch {
    return null;
  }
}

/** First http(s) URL in a free-text reference, with trailing punctuation removed. */
export function extractUrl(text: string): string | undefined {
  const match = text.match(/https?:\/\/[^\s<>"'`]+/i);
  if (!match) return undefined;
  const candidate = match[0].replace(/[),.;:!?\]]+$/, "");
  return normalizeUrl(candidate) ? candidate : undefined;
}

function shorten(text: string, max = MAX_TITLE_CHARS): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
}

export function buildSourceIndex(
  userEntries: string[],
  webSources: Array<{ title: string; url: string }>
): ResearchSource[] {
  const sources: ResearchSource[] = [];
  const seenUrls = new Set<string>();

  for (const entry of userEntries) {
    const url = extractUrl(entry);
    const key = url ? normalizeUrl(url) : null;
    if (key) seenUrls.add(key);
    sources.push({ id: sources.length + 1, title: shorten(entry), url, origin: "user" });
  }

  for (const web of webSources) {
    const key = normalizeUrl(web.url);
    if (!key || seenUrls.has(key)) continue;
    seenUrls.add(key);
    sources.push({ id: sources.length + 1, title: shorten(web.title || web.url), url: web.url, origin: "web" });
  }

  return sources;
}

/**
 * Maps the references a model attached to a finding (URLs or user labels such
 * as "U2") to source ids. Unknown references are dropped.
 */
export function resolveSourceRefs(refs: string[], sources: ResearchSource[]): number[] {
  const byUrl = new Map<string, number>();
  for (const source of sources) {
    const key = source.url ? normalizeUrl(source.url) : null;
    if (key) byUrl.set(key, source.id);
  }
  const userCount = sources.filter(s => s.origin === "user").length;

  const ids = new Set<number>();
  for (const ref of refs) {
    const label = ref.trim().match(/^\[?U(\d+)\]?$/i);
    if (label) {
      const n = Number(label[1]);
      if (n >= 1 && n <= userCount) ids.add(n);
      continue;
    }
    const key = normalizeUrl(ref);
    const id = key ? byUrl.get(key) : undefined;
    if (id !== undefined) ids.add(id);
  }
  return [...ids].sort((a, b) => a - b);
}

// ─── Prompt formatting ──────────────────────────────────────────────────────

function formatFinding(finding: ResearchFinding): string {
  const cite = finding.sourceIds.length ? ` [${finding.sourceIds.join(", ")}]` : " [general knowledge, no source]";
  return `${finding.text}${cite}`;
}

/** Plain-text research block (facts, quotes, numbered sources) for prompts. */
export function formatResearchForPrompt(research: Research | null): string {
  if (!research) return "No research is available.";
  const parts = [`Summary:\n${research.summary}`];
  if (research.keyFacts.length) {
    parts.push(`Key facts:\n${research.keyFacts.map(f => `- ${formatFinding(f)}`).join("\n")}`);
  }
  if (research.quotes.length) {
    parts.push(`Verbatim quotes:\n${research.quotes.map(q => `- "${q.text}"${q.sourceIds.length ? ` [${q.sourceIds.join(", ")}]` : ""}`).join("\n")}`);
  }
  parts.push(research.sources.length
    ? `Numbered sources:\n${research.sources.map(s => `[${s.id}] ${s.title}${s.url && !s.title.includes(s.url) ? ` — ${s.url}` : ""}`).join("\n")}`
    : "Numbered sources: none");
  return parts.join("\n\n");
}

// ─── Inline citations ───────────────────────────────────────────────────────

// [1] or [1, 3] — but not Markdown links "[1](…)" or footnotes "[^1]".
const CITATION_PATTERN = /\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\](?!\()/g;

/** Source ids cited in the article, in order of first appearance. */
export function citedSourceIds(article: string): number[] {
  const ids: number[] = [];
  for (const match of article.matchAll(CITATION_PATTERN)) {
    for (const part of match[1].split(",")) {
      const id = Number(part.trim());
      if (!ids.includes(id)) ids.push(id);
    }
  }
  return ids;
}

/** Removes citation numbers that do not refer to a known source. */
export function normalizeCitations(article: string, sourceCount: number): string {
  return article.replace(/ ?\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\](?!\()/g, (whole, group: string) => {
    const valid = [...new Set(group.split(",").map(p => Number(p.trim())))]
      .filter(id => Number.isInteger(id) && id >= 1 && id <= sourceCount);
    if (valid.length === 0) return "";
    const leading = whole.startsWith(" ") ? " " : "";
    return `${leading}[${valid.join(", ")}]`;
  });
}

const SOURCES_HEADING = /^(#{1,3})\s*(sources|references|bibliography|works cited|citations)\s*:?\s*$/i;

/** Removes a model-written Sources/References section; the canonical one is generated. */
export function stripSourcesSection(article: string): string {
  const lines = article.split("\n");
  const out: string[] = [];
  let skipLevel = 0;
  for (const line of lines) {
    const heading = line.match(/^(#{1,6})\s/);
    if (skipLevel) {
      if (heading && heading[1].length <= skipLevel) skipLevel = 0;
      else continue;
    }
    const sources = line.trim().match(SOURCES_HEADING);
    if (sources) {
      skipLevel = sources[1].length;
      continue;
    }
    out.push(line);
  }
  return out.join("\n").replace(/\s+$/, "");
}

function escapeMarkdownText(text: string): string {
  return text.replace(/([\\[\]*_`<>])/g, "\\$1");
}

function markdownUrl(raw: string): string | null {
  if (!normalizeUrl(raw)) return null;
  return new URL(raw).href.replace(/\(/g, "%28").replace(/\)/g, "%29");
}

export function renderSourcesSection(sources: ResearchSource[], citedIds: number[]): string {
  const cited = sources.filter(s => citedIds.includes(s.id));
  if (cited.length === 0) return "";
  const lines = cited.map(source => {
    const href = source.url ? markdownUrl(source.url) : null;
    const title = escapeMarkdownText(source.title);
    const body = href ? `[${title}](${href})` : `${title} *(user-provided reference)*`;
    return `- **[${source.id}]** ${body}`;
  });
  return `## Sources\n\n${lines.join("\n")}`;
}

/**
 * Deterministic citation post-processing: drop any model-written source list,
 * remove citations to unknown sources, and append the canonical Sources section
 * listing exactly the sources the article cites.
 */
export function finalizeCitations(article: string, sources: ResearchSource[]): string {
  const body = normalizeCitations(stripSourcesSection(article), sources.length);
  const section = renderSourcesSection(sources, citedSourceIds(body));
  return section ? `${body}\n\n${section}\n` : `${body}\n`;
}

