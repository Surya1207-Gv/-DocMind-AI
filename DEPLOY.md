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

## Persistence

### What is stored where

Every piece of durable state lives on the local filesystem under `DATA_DIR`
(`/data` in the container). There is no external database.

| State | Location | Survives restart? |
|---|---|---|
| User accounts, password hashes | `$DATA_DIR/docmind.db` (SQLite, `users`) | Only with a mounted disk |
| Chat history | `$DATA_DIR/docmind.db` (`chat_messages`) | Only with a mounted disk |
| Analytics, quizzes | `$DATA_DIR/docmind.db` (`analytics`, `quizzes`) | Only with a mounted disk |
| Document inventory | `$DATA_DIR/docmind.db` (`documents`) | Only with a mounted disk |
| Uploaded originals | `$DATA_DIR/uploads/<doc_id>.<ext>` | Only with a mounted disk |
| FAISS vector indices | `$DATA_DIR/faiss_indices/<doc_id>/` | Only with a mounted disk |
| Logs | `$DATA_DIR/docmind.log` | Only with a mounted disk |

**On Render's free plan there is no mounted disk, so none of the above
survives.** The container is replaced whenever the instance sleeps or
redeploys, and `docmind.db` is recreated empty.

This is worth stating precisely because of how it presents: a user registers,
uses the app, comes back the next day, and cannot log in. That looks exactly
like an authentication bug. It is not — the account row no longer exists. The
authentication code itself is covered by
`backend/tests/test_regression_startup_persistence.py`, which proves the boot
sequence never modifies or deletes an existing user.

To tell the two apart on a running instance, check `GET /api/health`:

```json
"storage": {
  "backend": "sqlite",
  "data_dir": "/data",
  "database_existed_at_boot": false,      // ← a fresh database: prior data is gone
  "database_url_present_but_unused": false
}
```

`database_existed_at_boot: false` on every deploy means `DATA_DIR` is
ephemeral. The same warning is logged once at startup.

### Fix 1 — persistent disk (recommended, no code changes)

Attach a volume at `DATA_DIR` and everything above persists as-is:

```yaml
plan: starter          # a disk requires a paid instance
disk:
  name: docmind-data
  mountPath: /data     # must equal DATA_DIR
  sizeGB: 1
```

This is the whole fix. SQLite in WAL mode is entirely adequate for this
application's write volume, and a single-service deployment is the
architecture the project is built around.

### Fix 2 — PostgreSQL (not implemented; migration path only)

**This build has no PostgreSQL support.** There is no `DATABASE_URL` handling,
no SQLAlchemy layer, and no driver in `requirements.txt`. If you attach a
Render Postgres instance, `DATABASE_URL` will be set and the application will
ignore it — `/api/health` reports this as
`database_url_present_but_unused: true` so it cannot pass unnoticed.

The migration is contained but real. What it would require:

1. **One file holds all SQL.** `backend/database.py` is the only module that
   issues queries; `backend/main.py` touches a connection exactly once (the
   health probe). Nothing else in the codebase knows the database exists. This
   is the reason the migration is tractable at all.
2. **No abstraction exists yet.** `database.py` uses `sqlite3` directly, so a
   migration means introducing one (SQLAlchemy Core, or a thin dialect shim).
3. **SQLite-specific constructs that must change** — all inside `database.py`:
   - `?` placeholders → `%s` (psycopg) or named parameters
   - `PRAGMA journal_mode` / `synchronous` / `foreign_keys` → removed; Postgres
     enforces foreign keys unconditionally
   - `sqlite3.IntegrityError` / `OperationalError` → `psycopg.errors.*`
   - `INSERT OR REPLACE` (analytics, quizzes) → `INSERT ... ON CONFLICT ... DO UPDATE`
   - `COLLATE NOCASE` (the case-insensitive username and email lookups) →
     `citext`, or `LOWER(col) = LOWER(%s)` with a matching functional index
   - `ORDER BY rowid` (chat history ordering) → an explicit ordering column;
     Postgres has no rowid
   - `sqlite3.Row` → `psycopg.rows.dict_row`
4. **Connection pooling.** `get_db_connection()` opens a fresh connection per
   call, which is free for SQLite and expensive for Postgres. A pool becomes
   necessary.
5. **A real migration tool.** The current schema evolves via
   `CREATE TABLE IF NOT EXISTS` plus guarded `ALTER TABLE` in `init_db()`. That
   is acceptable for a single-file database and not for a shared one.
6. **Uploads and FAISS indices still need a disk.** Postgres would persist the
   relational state only. The vector indices and original files are on the
   filesystem, so a disk (or object storage, a larger change) is *still*
   required. **Postgres alone does not solve the persistence problem.**

Given point 6, Fix 1 is the smaller and more complete change for this
architecture. Postgres becomes worth the migration when the deployment needs
more than one instance sharing state — which is a different requirement from
"data should survive a restart".

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
