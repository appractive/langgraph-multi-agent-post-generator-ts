// ─── Suggestion Verification ────────────────────────────────────────────────
//
// Reviewer suggestions are verified against a real word-level diff of the
// incoming and revised articles. Text is tokenized into lowercase words with
// Markdown syntax, punctuation, and quote/dash variants stripped, so edits that
// are invisible in the rendered comparison (curly quotes, bold markers, heading
// levels, whitespace, commas) cannot justify a claimed improvement.

export function tokenizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[\u2018\u2019\u201B\u2032]/g, "'")
    .replace(/[\u201C\u201D\u201F\u2033]/g, '"')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1") // links/images -> label
    .match(/[a-z0-9]+(?:'[a-z0-9]+)*/g) ?? [];
}

// Marks which tokens of `a` were deleted and which tokens of `b` were inserted,
// using an LCS diff (common prefix/suffix trimmed first to keep it cheap).
export function diffTokens(a: string[], b: string[]): { deleted: boolean[]; inserted: boolean[] } {
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
export function findTokenRuns(haystack: string[], needle: string[]): number[] {
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

export type VerifiedSuggestion = { description: string };

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

