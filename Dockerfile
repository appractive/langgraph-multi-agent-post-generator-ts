# Build Stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
# Full install: native modules (better-sqlite3) compile once, here.
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

# Run Stage
FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3001
# The container must accept connections from the published port.
ENV HOST=0.0.0.0
# Checkpoints and run budgets live here; mount a volume to keep paused runs.
ENV DATA_DIR=/app/data

COPY package.json package-lock.json ./
COPY --from=builder /app/node_modules ./node_modules
RUN npm prune --omit=dev && npm cache clean --force \
 && mkdir -p /app/data && chown -R node:node /app/data

COPY --from=builder --chown=node:node /app/dist ./dist
COPY --chown=node:node public ./public

USER node
VOLUME ["/app/data"]
EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:' + (process.env.PORT || 3001) + '/healthz').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["node", "dist/server.js"]
