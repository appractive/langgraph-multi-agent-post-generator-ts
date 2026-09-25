# Agent Architecture

The article generator uses the compiled LangGraph `StateGraph` from `buildEditorialGraph()` in `src/graph.ts` to orchestrate its in-process agent nodes. LangGraph owns the stage transitions, shared state, retry branch, and pause node; LangChain's `ChatOpenAI` provides chat-model integration, while the researcher uses the OpenAI Responses API directly. `runGraph()` in the Express server streams LangGraph node updates to the browser over SSE. Research runs when web search is enabled or source material is supplied; otherwise the graph starts with planning. After review, a score below 75 triggers one additional writing and review pass.

LangGraph state is scoped to a graph invocation. For a selected human-review stage, a LangGraph pause node ends the current invocation and returns the stage, next node, and full state to the browser over SSE. After approval and any edits, the browser posts that state to `/api/generate/continue`; the server starts a new graph invocation at the saved next node. No persistent LangGraph checkpointer is configured.

```mermaid
flowchart TD
    user[User] --> browser[Browser web app]
    browser -->|GET /api/models| server[Node.js / Express server<br/>Internal app API and workflow runner]
    server -->|OpenAI Models API: GET /v1/models| models[OpenAI Models API]
    browser -->|POST /api/generate<br/>SSE progress stream| server
    server -->|Invoke compiled StateGraph| researchCheck

    subgraph workflow[LangGraph StateGraph, in-process nodes]
        researchCheck{Web search enabled<br/>or sources supplied?}
        researchCheck -->|Yes| researcher[Researcher agent<br/>Searches and summarizes evidence]
        researchCheck -->|No| planner[Planner agent<br/>Creates article outline]
        researcher --> planner

        planner --> writer[Writer agent<br/>Drafts article in Markdown]
        writer --> reviewer[Reviewer agent<br/>Edits article and scores it]
        reviewer --> qualityCheck{Post-review score below 75<br/>and retry available?}
        qualityCheck -->|Yes: one retry| writer
        qualityCheck -->|No| result[Publish-ready article]

        state[(LangGraph shared state)]
        researcher -. research summary .-> state
        planner -. outline .-> state
        writer -. draft .-> state
        reviewer -. edits, scores, retry count .-> state
    end

    researcher -->|OpenAI Responses API<br/>POST /v1/responses<br/>web_search tool when enabled| responses[OpenAI Responses API]
    planner -->|OpenAI Chat Completions API<br/>POST /v1/chat/completions| chat[OpenAI Chat Completions API]
    writer -->|OpenAI Chat Completions API<br/>POST /v1/chat/completions| chat
    reviewer -->|OpenAI Chat Completions API<br/>POST /v1/chat/completions| chat
    result -->|SSE result| browser

    server -->|SSE pause node: stage, next node, full state| browser
    browser -->|POST /api/generate/continue<br/>edited state and next node| server
    browser -->|Abort active request| server
```

`/api/models`, `/api/generate`, and `/api/generate/continue` are routes served by this app; they are not OpenAI endpoints. Human-in-the-loop pauses can be enabled for selected stages. The score-based rewrite path is limited to one additional writer/reviewer pass.
