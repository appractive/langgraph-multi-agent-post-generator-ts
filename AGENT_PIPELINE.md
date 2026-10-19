# Agent Pipeline

`buildEditorialGraph()` in `src/graph.ts` compiles a LangGraph `StateGraph`: it owns node execution, state updates, transitions, the revision loop, and the human-review pause. The Express server drives the graph and streams node updates to the browser over SSE.

```mermaid
flowchart TD
    start((Start)) --> researchCheck{Web search enabled<br/>or sources supplied?}
    researchCheck -->|Yes| researcher[Researcher<br/>Searches, summarises, indexes sources]
    researchCheck -->|No| planner[Planner<br/>Creates the outline]
    researcher --> planner
    planner --> hitlReview{hitl_review<br/>interrupt() when enabled}
    hitlReview -->|Human approves or edits the outline| writer[Writer<br/>Drafts, checks coverage, revises]
    writer --> reviewer[Reviewer<br/>Judge scores, editor edits, judge re-scores]
    reviewer --> retryCheck{scoreAfter < 75<br/>and retryCount < 2?}
    retryCheck -->|Yes| writer
    retryCheck -->|No| finish((Complete))

    researcher -.-> state[(State channels)]
    planner -.-> state
    writer -.-> state
    reviewer -.-> state
    state -.-> checkpoint[("SQLite checkpoint<br/>keyed by thread_id")]
```

## Nodes

| Node | Reads | Writes | Why it exists |
| --- | --- | --- | --- |
| `researcher` | `topic`, `sources`, `enableWebSearch` | `research`, `researchSummary` | Grounds the article. Uses the Responses API with the `web_search` tool; every fact and quote carries `sourceRefs` that are resolved into a numbered source index. |
| `planner` | `topic`, `research` | `outline` | Turns evidence into a 4–6 section outline with SEO fields. Structured output (Zod) with a repair path. |
| `hitl_review` | `outline`, `hitlStages` | `outline` (only if resumed) | Pauses with `interrupt()` rather than ending the run. See below. |
| `writer` | `outline`, `research`, `review` | `article` | Drafts, then asks a checker which outline points are missing and revises once if any are. Finalises `[n]` citations. |
| `reviewer` | `article`, `outline`, `research` | `review`, `article`, `originalArticle`, `retryCount`, `reviewScores` | Three steps: an independent **judge** scores the draft, an **editor** model applies targeted edits, the judge scores the result again. |

### Why the judge is separate from the editor

A model that edits an article and then grades its own edits inflates the score, which makes the `scoreAfter < 75` gate meaningless. The default config therefore uses `gpt-4o` to edit and `gpt-4.1` to judge (`DEFAULT_MODEL_CONFIG` in `src/types.ts`), and `reviewerNode` explicitly picks a different judge when the two are configured identically.

### Why suggestions are verified

The editor returns "suggestions" describing what it changed. Models routinely claim edits they did not make. `verifySuggestions()` (`src/suggestions.ts`) diffs the incoming and revised article at word level and discards any suggestion whose quoted `before`/`after` excerpts do not cover real word changes. The UI never lists an edit that did not happen.

## State channels

`EditorialStateAnnotation` holds `topic`, `sources`, `enableWebSearch`, `research`, `researchSummary`, `outline`, `article`, `originalArticle`, `review`, `retryCount`, `reviewScores`, `modelConfig`, `hitlStages`, and `currentStage`. Nodes return partial updates; LangGraph merges them.

## Human in the loop

`hitlReviewNode` calls LangGraph's `interrupt(payload)` when `hitlStages` contains `planner`. The graph stops **with its checkpoint saved**, not by ending the invocation:

1. The server streams `{ node: "hitl_pause", data: { threadId, stage, outline } }`.
2. The client posts `{ threadId, outline }` to `/api/generate/continue` — validated against `OutlineSchema`, so a malformed edit is rejected with 400 rather than silently ignored.
3. The server loads the checkpoint for that `threadId`, resumes with `new Command({ resume: outline })`, and the run continues at `writer`.

The client never holds graph state, so it cannot forge an article, skip a stage, or change the model mid-run. Runs paused longer than 30 minutes are swept, and resuming a thread whose checkpoint is gone returns **409** instead of failing obscurely.

## Streaming protocol

`POST /api/generate` and `POST /api/generate/continue` both return `text/event-stream`. Each `data:` frame is one of:

| Event | Payload |
| --- | --- |
| stage start | `{ node, status: "running" }` |
| stage end | `{ node, status: "done", data }` |
| pause | `{ node: "hitl_pause", data: { threadId, stage, outline } }` |
| completion | `{ node: "complete", data: { threadId, article, researchSummary, usage } }` |
| failure | `{ node, status: "error", error }` |

`usage` is the run budget snapshot: calls, input/output tokens, estimated USD, web searches, and a per-node breakdown.

## Failure and cost behaviour

- **Cancellation.** A client disconnect aborts the `AbortSignal`, which is threaded from the HTTP request through every model call — a closed tab stops spending immediately.
- **Retries.** Structured output is retried up to 3 times, feeding the validation error back to the model; outline and review also have salvage paths for partially malformed JSON.
- **Budget.** Every model call is preceded by `budget.assertCanSpend(node)`. Exceeding calls, tokens, or USD raises `BudgetExceededError`, which the server reports as a normal stage error.
- **Revision loop.** At most one extra writer→reviewer pass (`retryCount < 2`), so the pipeline cannot oscillate.
