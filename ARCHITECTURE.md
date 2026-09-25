# Agent Architecture

The article generator runs a staged workflow over shared editorial state. Research is included when web search is enabled or source material is supplied; otherwise the workflow starts with planning. After review, an article scoring below 75 gets one rewrite and re-review pass.

```mermaid
flowchart TD
    user[User: topic and optional sources] --> app[Web app]
    app --> api[Generation API]
    api --> researchCheck{Web search enabled<br/>or sources supplied?}

    researchCheck -->|Yes| researcher[Researcher agent<br/>Searches web when enabled<br/>and summarizes evidence]
    researchCheck -->|No| planner[Planner agent<br/>Creates article outline]
    researcher --> planner

    planner --> writer[Writer agent<br/>Drafts article in Markdown]
    writer --> reviewer[Reviewer agent<br/>Edits article and scores it]
    reviewer --> qualityCheck{Post-review score below 75<br/>and retry available?}
    qualityCheck -->|Yes: one retry| writer
    qualityCheck -->|No| result[Publish-ready article]

    api -. optional checkpoint after selected stages .-> human{Human approval}
    human -->|Approve and continue| api
    human -->|Stop pipeline| stopped[Generation stopped]

    state[(Shared editorial state)]
    researcher -. research summary .-> state
    planner -. outline .-> state
    writer -. draft .-> state
    reviewer -. edits, scores, retry count .-> state
```

Human-in-the-loop checkpoints can be enabled for selected stages. The server pauses after a selected stage and resumes from its next node when approved. The score-based rewrite path is limited to one additional writer/reviewer pass.
