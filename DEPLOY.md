# Deployment Guide

DocMind AI ships as **one Docker image**. A Node stage compiles the React SPA;
a Python stage runs FastAPI and serves that build from the same origin.

That means: one service, one URL, no CORS configuration, and one thing to keep
running. The deployed artifact is the same image you can run locally, so there
is no "works on my machine" gap between local and production.

---

## 1 · Render (recommended, free)

### Blueprint deploy

1. Push the repository to GitHub.
2. In the [Render dashboard](https://dashboard.render.com): **New → Blueprint**,
   select the repository. Render reads [`render.yaml`](render.yaml).
3. Provide the secrets Render prompts for:

   | Variable | Value |
   |---|---|
   | `OPENROUTER_API_KEY` | your key from [openrouter.ai/keys](https://openrouter.ai/keys) |
   | `DEMO_PASSWORD` | any password for the public demo account |
   | `GEMINI_API_KEY` | optional — leave blank |

   `JWT_SECRET_KEY` is generated automatically by Render (`generateValue: true`).

4. Deploy. First build takes ~5–10 minutes (installing `faiss-cpu` and building
   the SPA). The app is then live at `https://<service-name>.onrender.com`.

### Manual deploy (without the blueprint)

**New → Web Service**, then:

| Setting | Value |
|---|---|
| Runtime | Docker |
| Dockerfile path | `./Dockerfile` |
| Docker context | `.` |
| Health check path | `/api/health` |
| Plan | Free |

Add the environment variables listed in [`.env.example`](.env.example). At
minimum: `OPENROUTER_API_KEY`, `JWT_SECRET_KEY`, `DATA_DIR=/data`,
`DEMO_SEED=true`.

### Free-tier characteristics

- **Cold starts.** The instance sleeps after ~15 minutes idle. The next request
  takes 30–60 seconds. Open the URL a minute before demoing it.
- **Ephemeral disk.** Free instances have no persistent volume: uploads, indices
  and chat history reset on restart. `DEMO_SEED=true` re-indexes the bundled
  sample PDF at every boot so the app is never empty.
- **Persistence** requires a paid instance. Switch `plan: free` to
  `plan: starter` in `render.yaml` and uncomment the `disk:` block, which mounts
  a volume at `/data` — the value of `DATA_DIR`, under which uploads, FAISS
  indices, the SQLite database and logs all live.

---

## 2 · Local Docker

```bash
cp .env.example .env      # set OPENROUTER_API_KEY and JWT_SECRET_KEY
docker compose up --build
```

Everything is served at <http://localhost:8000>: the UI at `/`, the API under
`/api`, Swagger docs at `/docs`. A named volume (`docmind-data`) persists state
across restarts.

---

## 3 · Hugging Face Spaces (alternative)

The same image works on Spaces:

1. Create a Space with SDK **Docker**.
2. Push this repository to it.
3. Add `OPENROUTER_API_KEY` and `JWT_SECRET_KEY` as **Repository secrets**.
4. Add `app_port: 8000` to the Space's `README.md` front matter.

Spaces do not sleep as aggressively as Render's free tier, which makes cold
starts less likely, but free Spaces have no persistent storage either — keep
`DEMO_SEED=true`.

---

## 4 · Any Docker host

```bash
docker build -t docmind .
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=... \
  -e JWT_SECRET_KEY=... \
  -e DATA_DIR=/data \
  -v docmind-data:/data \
  docmind
```

The container honours `$PORT` if the platform injects one (Render, Railway,
Fly.io, Cloud Run all do), defaulting to 8000.

---

## Post-deploy verification

```bash
BASE=https://<your-service>.onrender.com

curl -s $BASE/api/health     # {"status":"healthy", ...}
curl -s $BASE/api/info       # active model names
curl -sI $BASE/ | head -1    # 200 — the SPA
```

`"status":"degraded"` means one of the probes failed. The response body names
which: `database`, `faiss_indices_writable`, or `llm_provider`
(`missing_api_key` means no LLM key reached the process).

---

## Operational notes

- **Secrets** are only ever read from the environment. `backend/.env` is
  gitignored and excluded from the Docker build context; never bake a key into
  an image or commit one.
- **Retrieval tuning** (`TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`,
  `RELEVANCE_THRESHOLD`, `VECTOR_WEIGHT`) is environment-driven and needs no
  redeploy of code — only a restart. Note that changing chunking parameters only
  affects **newly uploaded** documents; existing indices keep their original
  chunking until re-uploaded.
- **Monitoring**: `GET /api/metrics` returns query counts, mean latency and the
  zero-hit rate. A rising zero-hit rate is the earliest signal that retrieval has
  degraded.
- **Model swaps** need no code change: set `LLM_MODEL` or `EMBEDDING_MODEL`.
  Changing the embedding model invalidates existing indices — documents must be
  re-uploaded, since vectors from different models are not comparable.
