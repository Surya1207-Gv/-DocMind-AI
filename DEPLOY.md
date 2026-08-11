# DocMind AI — Production Deployment Guide

This document outlines deployment configurations and operational requirements for DocMind AI.

---

## 1. Architecture & Persistence Requirements

DocMind AI uses two local storage systems for state and information retrieval:
1. **SQLite Database (`backend/docmind.db`):** Multi-tenant accounts, conversation history, document metadata, and background analytics.
2. **FAISS Vector Store (`backend/faiss_indices/`):** Persistent index files (`index.faiss`, `index.pkl`) per document.

> ⚠️ **CRITICAL FOR CLOUD PLATFORMS:** When deploying to serverless or ephemeral container platforms (such as Render, Railway, Fly.io), you **must attach a persistent volume disk** mounted to `/app/backend/faiss_indices`, `/app/backend/uploads`, and `/app/backend/docmind.db`. Otherwise, vector indexes and chat histories will be lost on container restarts.

---

## 2. Docker Compose Deployment (Recommended)

### Prerequisites
- Docker Engine 24+ & Docker Compose v2+
- OpenRouter API Key

### Launch Stack
```bash
# 1. Clone repository
git clone https://github.com/Surya1207-Gv/-DocMind-AI.git
cd -DocMind-AI

# 2. Configure environment variables
cp backend/.env.example backend/.env
# Add your OPENROUTER_API_KEY and generate a secure JWT_SECRET_KEY

# 3. Start containers
docker-compose up --build -d
```

- **Frontend UI:** `http://localhost:3000`
- **Backend API:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Health & Readiness Check:** `http://localhost:8000/api/health`

---

## 3. Cloud Deployment: Railway

1. **Create New Project on Railway:** Select "Deploy from GitHub repo".
2. **Deploy Backend Service:**
   - Root directory: `/`
   - Dockerfile path: `backend/Dockerfile`
   - Set Environment Variables:
     - `OPENROUTER_API_KEY`: your API key
     - `JWT_SECRET_KEY`: generated 64-char hex string
     - `ALLOWED_ORIGINS`: your frontend Railway URL
   - **Attach Volume:** Add a Persistent Volume mounted to `/app/backend`.
3. **Deploy Frontend Service:**
   - Root directory: `frontend`
   - Dockerfile path: `frontend/Dockerfile`
   - Add environment variable `VITE_API_BASE_URL` pointing to the Railway Backend public URL.

---

## 4. Cloud Deployment: Render

1. **Backend Web Service:**
   - Environment: `Docker`
   - Docker Context: `.`
   - Dockerfile Path: `backend/Dockerfile`
   - Add Environment Variables (`OPENROUTER_API_KEY`, `JWT_SECRET_KEY`, `ALLOWED_ORIGINS`).
   - Add **Persistent Disk:** Mount `/app/backend/faiss_indices` (Size: 1GB+).
2. **Frontend Static Site / Web Service:**
   - Environment: `Docker`
   - Docker Context: `frontend`
   - Dockerfile Path: `frontend/Dockerfile`

---

## 5. Production Health Probes & Monitoring

The backend exposes health checks and real-time operational telemetry:
- `GET /api/health`: Validates database connectivity, FAISS directory write access, and LLM configuration.
- `GET /api/metrics` (or `GET /api/telemetry`): Returns query counts, latency breakdowns, and zero-hit retrieval rates.
