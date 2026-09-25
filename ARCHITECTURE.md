# Agent Architecture

The article generator runs a staged workflow over shared editorial state. The agents are in-process steps in the Node.js server, not separate HTTP services. The browser calls the app's internal routes; the server calls OpenAI APIs for model work. Research is included when web search is enabled or source material is supplied; otherwise the workflow starts with planning. After review, an article scoring below 75 gets one rewrite and re-review pass.

```mermaid
flowchart TD
    user[User] --> browser[Browser web app]
    browser -->|GET /api/models| server[Node.js / Express server<br/>Internal app API and workflow runner]
    server -->|OpenAI Models API: GET /v1/models| models[OpenAI Models API]
    browser -->|POST /api/generate<br/>SSE progress stream| server
    server --> researchCheck

    subgraph workflow[In-process agent workflow, not separate HTTP APIs]
        researchCheck{Web search enabled<br/>or sources supplied?}
        researchCheck -->|Yes| researcher[Researcher agent<br/>Searches and summarizes evidence]
        researchCheck -->|No| planner[Planner agent<br/>Creates article outline]
        researcher --> planner

        planner --> writer[Writer agent<br/>Drafts article in Markdown]
        writer --> reviewer[Reviewer agent<br/>Edits article and scores it]
        reviewer --> qualityCheck{Post-review score below 75<br/>and retry available?}
        qualityCheck -->|Yes: one retry| writer
        qualityCheck -->|No| result[Publish-ready article]

        state[(Shared editorial state)]
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

    server -. optional checkpoint after selected stages .-> browser
    browser -->|POST /api/generate/continue<br/>resume after approval| server
    browser -->|Stop: abort generation request| stopped[Generation stopped]
```

`/api/models`, `/api/generate`, and `/api/generate/continue` are routes served by this app; they are not OpenAI endpoints. Human-in-the-loop checkpoints can be enabled for selected stages. The server pauses after a selected stage and resumes from its next node when the browser posts to `/api/generate/continue`. Stopping aborts the active generation request. The score-based rewrite path is limited to one additional writer/reviewer pass.
