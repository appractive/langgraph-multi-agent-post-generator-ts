import { z } from "zod";
import type { RubricCriterionResult } from "./types";

// ─── Fixed judging rubric ───────────────────────────────────────────────────
//
// Drafts are scored by a separate judge model against fixed questions. The
// judge returns an integer 0–4 per criterion; the overall 1–100 score is
// computed here, deterministically, so the model never picks the final number
// and the reviewer never grades its own edits.

export const RUBRIC_CRITERIA = [
  {
    id: "grounding",
    label: "Accuracy & grounding",
    weight: 2,
    question: "Are factual claims supported by the research, with [n] citations that point to the right numbered source, and no fabricated facts, statistics, quotes, or sources?",
  },
  {
    id: "coverage",
    label: "Outline coverage",
    weight: 1.5,
    question: "Does the article explicitly cover every heading and key point of the approved outline without contradicting any of them?",
  },
  {
    id: "clarity",
    label: "Clarity & readability",
    weight: 1,
    question: "Is the prose clear, concise, and easy to follow for the intended audience, with jargon explained?",
  },
  {
    id: "structure",
    label: "Structure & flow",
    weight: 1,
    question: "Does the article have a logical order, a strong introduction and conclusion, and smooth transitions between sections?",
  },
  {
    id: "language",
    label: "Grammar & spelling",
    weight: 1,
    question: "Is the article free of grammar, spelling, and punctuation errors?",
  },
  {
    id: "engagement",
    label: "Engagement & examples",
    weight: 1,
    question: "Does the article hold attention and use concrete, relevant examples instead of generic filler?",
  },
  {
    id: "seo",
    label: "SEO",
    weight: 0.5,
    question: "Do the title, headings, and opening paragraph use the topic's key terms naturally and stay consistent with the meta description?",
  },
] as const;

export type RubricCriterionId = (typeof RUBRIC_CRITERIA)[number]["id"];

export const RUBRIC_MAX_POINTS = 4;

export const RUBRIC_ANCHORS = "0 = fails the criterion; 1 = major problems; 2 = several noticeable issues; 3 = minor issues only; 4 = fully meets the criterion with no issues.";

const criterionIds = RUBRIC_CRITERIA.map(c => c.id) as [RubricCriterionId, ...RubricCriterionId[]];

export const RubricJudgementSchema = z.object({
  criteria: z.array(z.object({
    id: z.enum(criterionIds).describe("Rubric criterion id"),
    rationale: z.string().describe("One or two sentences citing specific evidence from the article"),
    score: z.number().int().min(0).max(RUBRIC_MAX_POINTS).describe("Integer score using the anchors"),
  })).describe("Exactly one entry per rubric criterion"),
});

export type RubricJudgement = z.infer<typeof RubricJudgementSchema>;

export function formatRubricForPrompt(): string {
  return RUBRIC_CRITERIA
    .map(c => `- ${c.id} (${c.label}): ${c.question}`)
    .join("\n");
}

/**
 * Converts a judgement into per-criterion results and a weighted 1–100 score.
 * Missing criteria score 0 so an incomplete judgement cannot inflate the total.
 */
export function scoreRubric(judgement: RubricJudgement): { score: number; criteria: RubricCriterionResult[] } {
  const byId = new Map<string, { score: number; rationale: string }>();
  for (const entry of judgement.criteria) {
    if (!byId.has(entry.id)) byId.set(entry.id, entry);
  }

  let earned = 0;
  let possible = 0;
  const criteria = RUBRIC_CRITERIA.map(criterion => {
    const entry = byId.get(criterion.id);
    const raw = entry ? Math.round(entry.score) : 0;
    const score = Math.min(RUBRIC_MAX_POINTS, Math.max(0, Number.isFinite(raw) ? raw : 0));
    earned += score * criterion.weight;
    possible += RUBRIC_MAX_POINTS * criterion.weight;
    return {
      id: criterion.id,
      label: criterion.label,
      score,
      max: RUBRIC_MAX_POINTS,
      rationale: entry?.rationale?.trim() || "Not assessed by the judge.",
    };
  });

  return { score: Math.max(1, Math.round((earned / possible) * 100)), criteria };
}

/** Rationales for criteria below `threshold`, used as rewrite guidance. */
export function rubricFeedback(criteria: RubricCriterionResult[], threshold = 3): string[] {
  return criteria
    .filter(c => c.score < threshold)
    .map(c => `${c.label} (${c.score}/${c.max}): ${c.rationale}`);
}

