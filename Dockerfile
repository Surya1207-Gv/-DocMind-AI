# =============================================================================
# DocMind AI — single-image deployment
# =============================================================================
# Stage 1 builds the React SPA; stage 2 runs FastAPI and serves that build from
# the same origin. One image, one service, one public URL — no CORS, no second
# deployment to keep in sync.
# =============================================================================

# --- Stage 1: build the React SPA -------------------------------------------
FROM node:20-alpine AS frontend-build

WORKDIR /build

# Install dependencies first so this layer caches across source-only changes.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install

COPY frontend/ ./
# No VITE_API_BASE_URL: the SPA calls same-origin /api in production.
RUN npm run build


# --- Stage 2: FastAPI runtime -----------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DATA_DIR=/data \
    FRONTEND_DIST_DIR=/app/frontend/dist \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (runtime only — no test packages in the image).
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Application source.
COPY backend /app/backend

# Compiled SPA from stage 1.
COPY --from=frontend-build /build/dist /app/frontend/dist

# Bundled demo PDF (used when DEMO_SEED=true).
COPY assets /app/assets

# Writable state lives in DATA_DIR so a mounted volume can persist it.
RUN mkdir -p /data/uploads /data/faiss_indices

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# Hosts such as Render inject $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
