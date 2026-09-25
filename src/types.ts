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

export interface Review {
  scoreBefore: number;
  scoreAfter: number;
  suggestions: string[];
  improvedArticle: string;
}

export interface ReviewScore {
  before: number;
  after: number;
}

export interface ModelConfig {
  researcher: string;
  planner: string;
  writer: string;
  reviewer: string;
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  researcher: "gpt-4.1-mini",
  planner: "gpt-4o-mini",
  writer: "gpt-4.1",
  reviewer: "gpt-4o",
};
