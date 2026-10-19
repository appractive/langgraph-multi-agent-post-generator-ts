export interface OutlineSection {
  heading: string;
  keyPoints: string[];
}

export interface Outline {
  seoTitle: string;
  metaDescription: string;
  sections: OutlineSection[];
  estimatedReadingTime: number;
}

export interface RubricCriterionResult {
  id: string;
  label: string;
  score: number;
  max: number;
  rationale: string;
}

export interface Review {
  scoreBefore: number;
  scoreAfter: number;
  suggestions: string[];
  improvedArticle: string;
  /** Rubric breakdown of the revised article, produced by the independent judge. */
  rubric: RubricCriterionResult[];
  /** Judge feedback on weak criteria, passed to the writer if another pass runs. */
  judgeFeedback: string[];
  judgeModel: string;
}

export interface ReviewScore {
  before: number;
  after: number;
}

/** A numbered source the writer can cite as [id]. */
export interface ResearchSource {
  id: number;
  title: string;
  url?: string;
  origin: "web" | "user";
}

/** A fact or quote with the ids of the sources that support it. */
export interface ResearchFinding {
  text: string;
  sourceIds: number[];
}

export interface Research {
  summary: string;
  keyFacts: ResearchFinding[];
  quotes: ResearchFinding[];
  sources: ResearchSource[];
}

export interface ModelConfig {
  researcher: string;
  planner: string;
  writer: string;
  reviewer: string;
  /** Scores drafts. Kept separate from the reviewer so the editor never grades its own work. */
  judge: string;
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  researcher: "gpt-4.1-mini",
  planner: "gpt-4o-mini",
  writer: "gpt-4.1",
  reviewer: "gpt-4o",
  judge: "gpt-4.1",
};
