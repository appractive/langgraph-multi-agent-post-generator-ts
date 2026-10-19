import { describe, expect, it } from "vitest";
import {
  citedSourceIds,
  coverageGaps,
  groundingFailures,
  injectionLeaks,
  outlineLimitIssues,
  unsupportedNumbers,
} from "../evals/checks";
import type { Outline, Research } from "../src/types";

const outline: Outline = {
  seoTitle: "A short title",
  metaDescription: "A meta description",
  estimatedReadingTime: 4,
  sections: [
    { heading: "Coordination", keyPoints: ["Handoffs cost more than work"] },
    { heading: "Cats", keyPoints: ["The team has no room for cats"] },
  ],
};

const research: Research = {
  summary: "Coordination overhead grows faster than headcount. Adoption reached 42.5% in 2019.",
  keyFacts: [{ text: "Adoption reached 42.5% in 2019", sourceIds: [1] }],
  quotes: [],
  sources: [
    { id: 1, origin: "user", title: "Report" },
    { id: 2, origin: "web", title: "Example", url: "https://example.com" },
  ],
};

describe("citation checks", () => {
  it("reads single and grouped citation markers", () => {
    expect(citedSourceIds("Text [1] more [2, 1] and [3].")).toEqual([1, 2, 3]);
  });

  it("flags citations that point past the available sources", () => {
    const failures = groundingFailures("Claim [7] here.", research, "");
    expect(failures.map(f => f.check)).toContain("citations.inRange");
  });

  it("flags an article that cites nothing although sources exist", () => {
    const failures = groundingFailures("An uncited article.", research, "");
    expect(failures.map(f => f.check)).toContain("citations.present");
  });

  it("accepts a grounded article", () => {
    expect(groundingFailures("Adoption reached 42.5% in 2019 [1].", research, "")).toEqual([]);
  });
});

describe("faithfulness checks", () => {
  it("extracts factual numbers but not small counts", () => {
    expect(unsupportedNumbers("We saw 42.5% growth in 2019 across 3 teams.", "42.5% 2019"))
      .toEqual([]);
  });

  it("flags a statistic the research never states", () => {
    expect(unsupportedNumbers("Usage grew 87.3% last year.", "usage grew")).toEqual(["87.3%"]);
  });
});

describe("injection checks", () => {
  it("detects scripts, handlers, and canaries", () => {
    expect(injectionLeaks('<script>alert(1)</script>')[0]).toBe("<script");
    expect(injectionLeaks("see javascript:void(0)")).toContain("javascript:");
    expect(injectionLeaks("img onerror=alert")).toContain("onerror=");
    expect(injectionLeaks("CANARY-INJECTION")).toContain("CANARY-INJECTION");
  });

  it("passes clean prose", () => {
    expect(injectionLeaks("A normal article about coordination.")).toEqual([]);
  });
});

describe("outline checks", () => {
  it("flags over-long SEO fields and thin outlines", () => {
    const issues = outlineLimitIssues({
      ...outline,
      seoTitle: "x".repeat(61),
      metaDescription: "y".repeat(156),
      sections: [outline.sections[0]],
    });
    expect(issues).toHaveLength(3);
  });

  it("finds key points the article does not discuss", () => {
    const article = "Coordination: handoffs cost more than work in most teams.";
    expect(coverageGaps(outline, article)).toEqual(["Cats: The team has no room for cats"]);
  });

  it("accepts an article that covers every point", () => {
    const article = "Handoffs cost more than work. The team has no room for cats at all.";
    expect(coverageGaps(outline, article)).toEqual([]);
  });
});
