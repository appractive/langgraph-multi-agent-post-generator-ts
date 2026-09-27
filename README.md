# Agentic Article Generator

Agentic Article Generator turns a topic into a publish-ready article through a LangGraph `StateGraph` workflow. The researcher gathers context, the planner creates an outline, the writer drafts the article, and the reviewer edits and scores it. LangChain provides chat-model integration. You can monitor each stage, choose models, and pause the workflow for human review.

## Requirements

- Node.js 20 or later and npm
- An OpenAI API key with access to the models you want to use
- Docker, if you plan to run the containerized version

## Run Locally

1. Install dependencies from the lockfile:

   ```bash
   npm ci
   ```

2. Copy `.env.example` to `.env` in the project root, then replace the API key placeholder:

   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   ```

   Do not commit `.env` or share your API key. To use a different port, set `PORT` in this file; the default is `3001`.

3. Start the development server:

   ```bash
   npm run dev
   ```

4. Open [http://localhost:3001](http://localhost:3001) (or the port you configured).

To run the compiled server locally instead, use `npm run build` followed by `npm start`.

## Use the App

1. Enter a topic in the topic field. A topic is required.
2. Optionally add source URLs, reports, books, or other reference notes in **Sources & references**, one entry per line.
3. Open **Configuration** to adjust the workflow.

   **Researcher Agent:** Searches the live web when enabled. Turn it off to skip live web search; if you provide sources, the researcher still uses those references. With web search off and no supplied sources, research is skipped and the workflow starts at planning.

   **Human in the Loop:** When enabled, pauses after the Planner creates the outline so you can review or edit it before the Writer drafts the article. Use **Approve & Continue** to proceed or **Stop Pipeline** to end the run.

   **Models:** Select one model for all stages or expand **Customize each stage** to choose separate researcher, planner, writer, and reviewer models. Use **Refresh models** to reload models available to your API key. Model availability depends on your OpenAI account.
4. Select **Generate**. Follow the Research, Outline, Draft, and Review tabs as the pipeline runs. If the post-review quality score is below 75, the workflow performs one additional writing and review pass.
5. Read the finished article in **Final**. Use **Copy Article** to copy it to the clipboard. Use **Stop generating** to cancel an in-progress generation.

Live web research uses OpenAI's web search tool and may incur API charges. Review generated content and sources before publishing.

## Run with Docker

Build the image from the project root:

```bash
docker build -t agentic-article-generator .
```

Run it with the project `.env` file:

```bash
docker run --rm -p 3001:3001 --env-file .env agentic-article-generator
```

Open [http://localhost:3001](http://localhost:3001).

When done.
 Stop the container with `Ctrl+C`. Because the run command uses `--rm`, Docker removes the container automatically when it stops. To remove the image when you're finished, run:

```bash
docker image rm agentic-article-generator
```

If you omitted `--rm`, 
list containers with `docker ps -a`
Stop the container (if it is currently running):
`docker stop <container_id_or_name>`
 remove the stopped container with:
 `docker rm <container_id>`
 Delete the image:
 `docker rmi agentic-article-generator`

## Available npm Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Run the TypeScript server for local development |
| `npm run build` | Compile TypeScript into `dist/` |
| `npm start` | Run the compiled server; build first |
