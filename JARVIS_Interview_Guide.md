# JARVIS — Complete Interview Preparation Guide

All answers are written in first-person, interview-ready language. Each answer explains the concept simply first, then goes technically deep.

---

## 1. Project Overview

### Question 1:
Can you give me a high-level overview of the JARVIS project?

### Answer:
JARVIS is a full-stack Cognitive Operating System I built for solo founders. Think of it as a smart productivity journal that combines AI-powered decision logging, real-time Socratic coaching during work sessions, and semantic memory search — all in one cohesive interface.

Technically, it is a Node.js + Express + TypeScript backend serving a Vanilla JavaScript + HTML/CSS frontend, with a SQLite database (WAL-mode optimized) storing all user data. The AI layer is powered by Google Gemini API, and real-time updates across browser tabs are handled through WebSockets.

The core idea: every time you make a decision, capture an insight, or log a mistake — JARVIS stores it, embeds it as a vector, and makes it searchable by meaning rather than just keywords. During a focused work session, the AI acts as a Socratic coach, automatically reacting when you log something and periodically checking in to keep you focused.

**Interviewer follow-ups to prepare for:**
- "Why build this for solo founders specifically?" Solo founders make hundreds of micro-decisions daily with no one to validate their thinking. A Notion doc is passive — JARVIS is reactive.
- "What problem does this solve that a Notion doc cannot?" JARVIS has active AI coaching, semantic search, and structured memory types with confidence scoring.

---

### Question 2:
What are the core features of JARVIS?

### Answer:
JARVIS has five main feature areas:

1. Cognitive Memory Logging — Users log Decisions (with reasoning + confidence score 1-10), Insights (observations, thoughts), and Mistakes (with root cause + preventive rule). All stored in SQLite.

2. AI-Powered Semantic Search — Every entry is converted to a vector embedding using Gemini's text-embedding-004 model. Searching "scaling issue" finds entries about "AWS RDS Aurora" even if the word "scaling" never appears — it matches by meaning, not keywords.

3. Focused Work Sessions with Socratic AI — You declare a task, start a session, and JARVIS opens with a focused clarifying question. Every 90 seconds it checks in. When you log something mid-session, it reacts immediately with a pointed question or observation.

4. BYOK (Bring Your Own Key) — Users can store their personal Gemini API key in the database. The backend checks for a user-level key first, falls back to the server key. This lets power users bypass shared API quota limits.

5. Confidence Analytics — A Chart.js visualization showing the confidence distribution of all decisions, helping users audit their decision-making quality over time.

---

### Question 3:
Why did you call it a "Cognitive Operating System"?

### Answer:
Because it is infrastructure for thinking, not just a chat tool or note-taking app. Just like an OS manages computer resources, JARVIS manages cognitive resources — where you are spending mental energy, what decisions you are making confidently vs. tentatively, what patterns of mistakes keep recurring.

The three memory types map to a thinking framework: Decisions track choices and their confidence, Insights track observations and learning, Mistakes close the feedback loop by documenting failure plus a rule to prevent recurrence. Together they build a searchable cognitive history over time.

---

## 2. Architecture

### Question 1:
Walk me through the architecture of JARVIS.

### Answer:
JARVIS follows a classic client-server architecture with four distinct layers:

**Frontend:** Pure Vanilla HTML5, CSS3, and JavaScript — no framework. index.html is the main dashboard shell; landing.html is onboarding; signup.html and login.html handle auth. app.js (~1,460 lines) is the single frontend controller managing all state, DOM manipulation, and API calls. speech.js is a separate IIFE module handling the Web Speech API for voice dictation.

**Backend:** Node.js + Express with TypeScript source files (server.ts, routes in routes/ai.ts, auth.ts, memory.ts). Compiled to JS and run via node server/server.js. Organized into route modules: Auth, Memory CRUD, and AI Proxy. WebSocket server runs on the same HTTP server instance.

**Database:** Single SQLite file at server/jarvis.db. Tables: users, decisions, insights, mistakes, sessions, embeddings. Uses better-sqlite3 library (synchronous, single-threaded, very fast). WAL mode enabled for improved concurrent read performance.

**AI Layer:** All AI calls go through the backend — the browser never touches the Gemini API key directly. Three AI functions: chat, embed, search. Supports Gemini (default), OpenAI, and Anthropic via a unified callLLM() dispatcher. Embeddings stored as JSON-serialized float arrays in the embeddings table.

**Real-Time Layer:** ws npm package powers a WebSocket server. On any memory write, the server calls broadcastUpdate() — pushes a memory:update event to all connected clients. Frontend handles this in handleRealtimeUpdate() — instantly updating the memory feed in all open tabs.

---

### Question 2:
Why did you choose this architecture over React + REST API or GraphQL?

### Answer:
Simple answer: right tool for the right job.

**Why Vanilla JS over React?** JARVIS is a single-page application with one primary view. React's value shines at scale with dozens of composable components. For a single-page tool with a predictable layout, vanilla JS has zero build step, zero bundle size, and zero framework upgrades to maintain. The tradeoff is that app.js is larger (~1,460 lines), but it is well-organized into discrete init* functions.

**Why SQLite over PostgreSQL?** JARVIS is designed for personal use. SQLite is serverless, requires zero configuration, stores everything in a single file, and better-sqlite3 gives synchronous, extremely fast queries. For a single-user cognitive tool, you do not need a connection pool or separate database process. If scaling to thousands of users, I would migrate to PostgreSQL.

**Why Express over Fastify/Koa?** Express has the largest ecosystem, the most familiar middleware patterns, and excellent TypeScript support. Right choice for readability and development speed.

**Why not GraphQL?** The data relationships in JARVIS are simple: users own entries, entries have a type. REST maps perfectly to this. GraphQL flexibility is valuable for arbitrary query composition — not needed here.

---

### Question 3:
How does the frontend and backend communicate?

### Answer:
Through two channels:

1. HTTP REST API — The frontend app.js has a const api = {...} object with async methods for every backend interaction. It uses native fetch() to call endpoints like POST /api/decisions, GET /api/memory, POST /api/ai/chat. Every request includes a Bearer JWT token in the Authorization header, read from localStorage.

2. WebSockets — On page load, initWebSocket() opens a persistent WebSocket connection to the server. The server maintains a Set of connected clients. When a write operation happens, the server calls broadcastUpdate(), which iterates the set and sends a JSON message to every OPEN client. The client-side ws.onmessage handler dispatches to handleRealtimeUpdate() which updates in-memory state and re-renders the feed.

This hybrid approach means: HTTP for actions, WebSocket for reactions. The frontend initiates changes via REST, and all connected clients get notified in real-time via WebSocket.

---

## 3. Backend — Node.js / Express / TypeScript

### Question 1:
Explain how you structured your Express server.

### Answer:
The entry point is server/server.js. Here is the startup sequence:

1. Load .env from the server/ directory using dotenv
2. Create an Express app with cors and express.json() middleware
3. Create an HTTP server wrapping the Express app (http.createServer(app)) — needed because the WebSocket server must attach to a raw HTTP server, not just Express
4. Initialize the WebSocket server on that HTTP server
5. Register route modules: /api/auth (no auth guard), /api/memory (JWT guarded), /api/ai (JWT guarded)
6. Serve static files from the project root, with / pointing to landing.html
7. Listen on PORT (default 3001)

The broadcastUpdate function is attached to app.locals so route modules can call req.app.locals.broadcastUpdate(data) without circular imports. Routes are split into three files — auth.ts, memory.ts, ai.ts — each exporting an Express Router.

---

### Question 2:
How does authentication work in JARVIS?

### Answer:
JARVIS uses stateless JWT (JSON Web Token) authentication.

**Signup flow:**
1. User submits email, username, password, passwordConfirm
2. Backend validates: all fields present, passwords match, min 6 chars
3. Checks uniqueness: getUserByEmail() and getUserByUsername() queries
4. Hashes password with bcryptjs.hash(password, 10) — 10 salt rounds
5. Generates a random 7-char user ID (Math.random().toString(36).slice(2, 9))
6. Inserts user into SQLite users table
7. Signs a JWT with { userId, email } payload, JWT_SECRET from env, 7-day expiry
8. Returns { user, token } to frontend

**Login:** Accepts emailOrUsername (flexible — either works), tries email lookup then username lookup, compares password against bcrypt hash using bcrypt.compare(), returns a new JWT on success.

**verifyAuth middleware** runs before every protected route:
1. Reads Authorization: Bearer token header
2. Calls jwt.verify(token, JWT_SECRET) — throws if expired or invalid
3. Attaches req.userId and req.userEmail to the request object via TypeScript declaration merging in types.ts
4. Calls next() to proceed

**Frontend:** JWT stored in localStorage as jarvis_token. getAuthHeaders() prepends it to every fetch call. On 401 response, token is cleared and user is redirected to login.html.

**Security trade-offs I acknowledge:**
- localStorage is vulnerable to XSS; httpOnly cookies are safer in production
- No refresh token currently; a production system would use short-lived access tokens + refresh tokens
- JWT is stateless so there is no revocation mechanism without a blocklist

---

### Question 3:
How did you implement the TypeScript transition?

### Answer:
The backend was originally JavaScript. I migrated it to TypeScript by:

1. Adding TypeScript dependencies: typescript, ts-node, ts-jest, and @types/* packages for all libraries
2. Creating tsconfig.json with strict mode, targeting ES2020, CommonJS module output
3. Defining shared types in types.ts: interfaces for User, Decision, Insight, Mistake, Session, Embedding, MemoryData, JWTPayload, SignupBody, LoginBody, InsightCard
4. Declaration merging for Express Request: added userId?: string and userEmail?: string to the Express Request interface so TypeScript knows req.userId is valid after verifyAuth runs
5. Converting route files to .ts, adding type annotations to every function parameter and return type
6. Updating database.ts to use typed prepared statement results

The practical benefit: TypeScript caught several potential undefined dereference bugs at compile time that would have been silent runtime errors in JS, especially around nullable database query results (getUserById() returns User | undefined).

The routes directory has both .js and .ts files — the .js files are the compiled output that actually runs. The TypeScript source is what I author and maintain.

---

### Question 4:
How does the database layer work? What is WAL mode and why did you enable it?

### Answer:
The database layer is in server/database.ts, using better-sqlite3 — a synchronous SQLite binding for Node.js.

**Schema:** Six tables — users, decisions, insights, mistakes, sessions, embeddings. All primary keys are TEXT strings. user_id is a TEXT column on all content tables, defaulting to 'default-user' for backward compatibility.

**Prepared Statements:** Every query is pre-compiled as a db.prepare() statement object stored in a stmts object. Prepared statements are parsed once by SQLite and stored as an execution plan — reused across all requests. This is significantly faster than re-parsing SQL strings on every request, and completely prevents SQL injection because parameters are always bound separately from the query structure.

**WAL Mode (Write-Ahead Logging):** db.pragma('journal_mode = WAL') is called at startup. By default SQLite uses a rollback journal — writes lock the entire database file. WAL mode allows concurrent readers while a write is happening, because writers append to a separate WAL file instead of modifying the database in-place. For JARVIS, even while a decision is being written, the memory feed read query can proceed without blocking.

**Migrations:** Runtime schema migrations using PRAGMA table_info() to inspect existing column structure. If user_id is missing (older database), the migration backs up tables, recreates them with the new schema, restores data, and drops the backups — all within a single db.exec() call which runs atomically. The whole block runs inside try/catch so it is safe on first run.

**Public API:** The memory export object wraps all statement calls in typed methods (insertDecision(), deleteDecision(), buildContext(), etc.) — a clean repository layer that routes import. No raw SQL outside database.ts.

---

## 4. AI Integration (Gemini + LLM Proxy)

### Question 1:
How does the AI integration work? Walk me through a chat message flow.

### Answer:
All AI calls are proxied through the backend. Here is a complete message flow:

1. User types a message and hits Enter
2. Frontend calls api.chat(text, state.session.conversationHistory) — a fetch() POST to /api/ai/chat with { prompt, history } in the body and a Bearer JWT in the header
3. verifyAuth middleware validates the JWT, attaches req.userId to the request
4. POST /api/ai/chat handler in ai.ts:
   - Calls getGeminiKey(req) — checks if the user has a personal API key in the users table; falls back to process.env.GEMINI_API_KEY
   - Calls buildSystemPrompt(userId) which calls memory.buildContext(userId) — queries SQLite for the user's last 10 decisions, insights, and mistakes and formats them as a text block injected into the system prompt
   - Calls callLLM() which dispatches to callGemini() based on LLM_PROVIDER env var
5. callGemini() constructs the request with system_instruction, contents (history + current message), and generationConfig. Calls Gemini's REST endpoint.
6. Response parsed: extracts candidates[0].content.parts[0].text
7. Backend returns { reply } to frontend
8. Frontend adds the response as a new .assistant-msg div and appends it to state.session.conversationHistory

Key design: The system prompt is built fresh on every request from live SQLite data. If you log a new decision, the very next AI message automatically knows about it.

---

### Question 2:
Explain how semantic search works in JARVIS. What is cosine similarity?

### Answer:
Semantic search lets users search their memory by meaning, not just exact keyword matches.

**Storing embeddings:**
1. When a user saves a decision/insight/mistake, the frontend immediately calls api.embedEntry(id, type, text) — fire-and-forget (non-blocking)
2. This hits POST /api/ai/embed on the backend
3. The handler calls generateEmbedding(text, apiKey) — sends the text to Gemini's text-embedding-004 model
4. Gemini returns a 768-dimensional float vector representing the semantic meaning of that text
5. This vector is JSON-serialized and stored in the embeddings table

**Searching:**
1. User types a search query
2. Frontend calls api.semanticSearch(query, topK=5)
3. Backend handler POST /api/ai/search:
   - Embeds the query text using the same Gemini model
   - Loads all stored embeddings for the user from SQLite
   - Computes cosine similarity between query vector and each stored vector
   - Sorts by score descending, returns top K results with similarity scores

**What is cosine similarity?**
Think of each embedding as a direction in 768-dimensional space. Cosine similarity measures the angle between two vectors — if two texts mean roughly the same thing, their vectors point in similar directions, giving a score close to 1.0. If they mean different things, vectors diverge, giving a score near 0.

Formula: cos(theta) = (A dot B) / (|A| * |B|) — the dot product divided by the product of magnitudes.

My implementation in ai.ts:
```
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const magB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return magA && magB ? dot / (magA * magB) : 0;
}
```

**Why not a vector database like Pinecone?**
For a personal tool with hundreds of entries, brute-force cosine similarity is fast enough — O(n) where n is tiny. A vector database uses ANN (Approximate Nearest Neighbor) algorithms like HNSW to search millions of vectors in milliseconds, but introduces external infrastructure and cost. Not warranted at this scale.

---

### Question 3:
What is BYOK and how did you implement it?

### Answer:
BYOK (Bring Your Own Key) lets users supply their own Gemini API key instead of sharing the server key. This matters because shared keys hit shared rate limits, power users want control over their own API spend, and it improves privacy since queries go through their own Google account.

Implementation:
1. The users table has an api_key TEXT column (nullable)
2. A BYOK modal on the dashboard lets users paste their Gemini key
3. Frontend calls api.saveApiKey(key) — POST /api/ai/apikey
4. Backend: memory.updateUserApiKey(req.userId, api_key || null) stores or clears the key in the user's row
5. Every AI request uses getGeminiKey(req):

```
function getGeminiKey(req) {
  const user = memory.getUserById(req.userId);
  if (user?.api_key) return user.api_key;  // User key takes priority
  const serverKey = process.env.GEMINI_API_KEY;
  if (!serverKey) throw new Error('GEMINI_API_KEY not set');
  return serverKey;
}
```

Security consideration: The API key is stored as plaintext in SQLite. In production I would encrypt it at rest using AES-256, with the encryption key derived from a server-side secret.

---

### Question 4:
How does the multi-provider LLM support work?

### Answer:
The LLM_PROVIDER environment variable controls which LLM is used. Three separate caller functions handle each provider's unique API format:

- callGemini() — Gemini format: system_instruction, contents array with parts
- callOpenAI() — OpenAI format: flat messages array with system/user/assistant roles
- callAnthropic() — Anthropic format: separate system parameter, custom anthropic-version: 2023-06-01 header

callLLM() is the single entry point using a switch statement to dispatch:
```
switch (provider) {
  case 'gemini':    return callGemini(...);
  case 'openai':   return callOpenAI(...);
  case 'anthropic': return callAnthropic(...);
}
```

convertHistory() handles the message format difference — Gemini uses { role: 'model', parts: [{text}] } while OpenAI/Anthropic use { role: 'assistant', content: '...' }.

Design pattern: This is a classic Strategy Pattern — the algorithm for calling an LLM is encapsulated in interchangeable functions, selected at runtime based on configuration.

---

## 5. Real-Time WebSockets

### Question 1:
How does the real-time update system work?

### Answer:
JARVIS uses WebSockets to push live updates to all connected clients whenever memory data changes.

Server side setup:
```
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
const clients = new Set();

wss.on('connection', (ws) => {
  clients.add(ws);
  ws.on('close', () => clients.delete(ws));
});

function broadcastUpdate(data) {
  const message = JSON.stringify(data);
  clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) client.send(message);
  });
}
app.locals.broadcastUpdate = broadcastUpdate;
```

After any write (e.g., saving a decision), the route calls:
```
req.app.locals.broadcastUpdate({
  type: 'memory:update',
  action: 'decision_added',
  userId,
  data: { id, description, reasoning, confidence, time }
});
```

Client side:
```
state.ws = new WebSocket(`ws://${window.location.host}`);
state.ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'memory:update') handleRealtimeUpdate(data);
};
state.ws.onclose = () => setTimeout(initWebSocket, 3000); // auto-reconnect
```

handleRealtimeUpdate() checks if the item already exists in local state (deduplication guard), adds it if not, then calls updateMemoryIfActive() to re-render the visible tab.

Why WebSockets over polling? Polling hits the server every N seconds regardless of changes — wasteful and laggy. WebSockets maintain a persistent bidirectional connection so updates are instantaneous with no unnecessary traffic.

---

## 6. Frontend — Vanilla JS, CSS, HTML

### Question 1:
Why did you use Vanilla JavaScript instead of React, Vue, or Angular?

### Answer:
This was a deliberate architectural decision, not a gap in knowledge.

Right sizing: JARVIS is a single-view dashboard with roughly 15 interactive sections. React shines at scale with dozens of composable components. For a single-page tool with predictable layout, vanilla JS has zero build step, zero bundle size, zero framework upgrades to maintain.

Performance: No virtual DOM overhead. DOM updates in JARVIS are targeted — when a decision is added, I call updateMemoryIfActive('decisions') which re-renders just the memory feed, not the entire page.

Learning depth: Writing everything from scratch forced me to understand every byte — event delegation, state management, DOM lifecycle, fetch API, WebSocket API. I can explain exactly why every line exists.

Trade-off I acknowledge: As the app grows with multiple views and shared components, a framework becomes worthwhile. If adding a mobile app or team dashboard, I would migrate the frontend to React.

---

### Question 2:
How do you manage state on the frontend?

### Answer:
State is managed through a single const state = {...} object declared at the top of app.js. It is a plain JavaScript object acting as the single source of truth:

```
const state = {
  session: {
    active: false, id: null, startTime: null,
    timerInterval: null, task: '',
    checkinInterval: null, conversationHistory: [],
  },
  memory: { decisions: [], insights: [], mistakes: [], sessions: [] },
  activeMemoryTab: 'decisions',
  isAIThinking: false,
  ws: null,
  searchQuery: '',
  filterConfidence: 0,
  semanticResults: null,
  memorySortOrder: 'newest',
  confidenceChart: null,
};
```

State mutations happen three ways:
1. User actions (save button click) — optimistic update, then API call
2. API responses — update state, then re-render
3. WebSocket events — update state, then re-render

There is no reactive binding — when state changes, the responsible function explicitly calls render functions like renderMemoryFeed(), updateMemoryCount(). This is intentional: explicit and debuggable.

Optimistic UI: When a user saves a decision, it is added to state.memory.decisions immediately before the API call completes. This makes the UI feel instant. If the API fails, a toast shows the error.

---

### Question 3:
How does the CSS design system work? What is glassmorphism?

### Answer:
The entire visual design is built with CSS custom properties defined in the :root pseudo-class in index.css. This creates a design token layer:

```
:root {
  --bg-base: #0d0f10;
  --bg-surface: #141618;
  --text-primary: #e8eaed;
  --accent: #4A9EFF;
  --accent-dim: rgba(74, 158, 255, 0.1);
  --font-sans: 'Inter', system-ui, sans-serif;
  --spacing-md: 24px;
  --radius-xl: 16px;
  --transition: 0.15s ease;
}
```

Dark and light themes are implemented via a [data-theme='light'] attribute selector that overrides these variables. Switching themes is one line of JS: document.documentElement.setAttribute('data-theme', 'light').

Glassmorphism is a modern UI trend where panels appear as frosted glass — semi-transparent background with a backdrop blur filter:
```
.panel {
  background: rgba(20, 22, 24, 0.85);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.06);
}
```

Why no Tailwind? Tailwind adds a build step and creates utility-class markup that is hard to read for complex custom designs. Raw CSS with variables gives pixel-perfect control and a readable stylesheet structure.

---

### Question 4:
How does the Voice Dictation feature work?

### Answer:
Voice dictation is in speech.js using the browser's native Web Speech API (window.SpeechRecognition or window.webkitSpeechRecognition for Chrome).

The flow:
1. A .mic-btn[data-target] button on any input triggers recognition
2. SpeechRecognition is instantiated with lang: 'en-US', interimResults: false, maxAlternatives: 1
3. When the user speaks, recognition.onresult fires with a transcript string
4. The transcript is immediately placed in the target input field
5. Optionally the transcript is sent to POST /api/ai/summarize on the backend to condense it
6. A confirm() dialog shows the summary and asks the user to approve before replacing the raw transcript

The module is wrapped in an IIFE (Immediately Invoked Function Expression) to avoid polluting the global scope. It initializes on DOMContentLoaded by scanning for all .mic-btn[data-target] buttons and binding them.

Browser compatibility: webkitSpeechRecognition is Chrome-only. The code gracefully degrades — if the API is unavailable, clicking the mic button shows an alert instead of crashing.

---

## 7. Database Design

### Question 1:
Explain the database schema and your design decisions.

### Answer:
JARVIS uses six tables in SQLite:

users — id, email (UNIQUE), username (UNIQUE), password (bcrypt hash), created_at, updated_at, google_id (for future OAuth), api_key (BYOK)

decisions — id (TEXT PK), user_id, description, reasoning, confidence (INTEGER 1-10), time (ISO 8601 TEXT)

insights — id, user_id, text, time

mistakes — id, user_id, description, cause, rule, time

sessions — id, user_id, task, start_time, end_time (nullable), duration (formatted string like "01:23:45")

embeddings — id ("emb_" + entry_id), entry_id, entry_type ('decision'/'insight'/'mistake'), user_id, embedding (JSON TEXT of 768 floats), time

Key design decisions:

TEXT primary keys: Random short strings generated on the client (Math.random().toString(36).slice(2, 9)) before the API call — this enables optimistic UI updates because the ID exists before the server confirms the insert.

TEXT for timestamps: ISO 8601 strings (new Date().toISOString()). SQLite has no native datetime type, but ISO strings sort lexicographically correctly — ORDER BY time DESC works perfectly.

Embeddings as JSON TEXT: SQLite has no VECTOR type. JSON-serialized float arrays are human-readable during debugging and trivial to parse with JSON.parse(). At this scale, the overhead is negligible.

No strict foreign key constraints: Foreign keys are disabled during inserts (PRAGMA foreign_keys = OFF) to handle the migration period. This was a pragmatic decision during schema evolution.

---

### Question 2:
How did you handle database migrations?

### Answer:
I wrote a runtime migration system that runs every time the server starts. It uses PRAGMA table_info(table_name) to inspect the current schema and check if expected columns exist.

For adding the user_id column to existing tables:
```
const checkColumn = db.prepare('PRAGMA table_info(decisions)').all();
const hasUserIdColumn = checkColumn.some(col => col.name === 'user_id');
if (!hasUserIdColumn) {
  db.exec(`
    CREATE TABLE decisions_backup AS SELECT * FROM decisions;
    DROP TABLE decisions;
    CREATE TABLE decisions (id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'default-user', ...);
    INSERT INTO decisions SELECT id, 'default-user', description, ... FROM decisions_backup;
    DROP TABLE decisions_backup;
  `);
}
```

The whole block runs inside try/catch — if tables do not exist yet (first run), the migration is skipped gracefully.

Why this approach? Production apps use migration frameworks like Flyway or Prisma Migrate which version migrations with numbered files. For a personal project, runtime PRAGMA checks are simple and self-contained. The tradeoff: migrations run on every startup (fast, but slightly redundant after first run), and there is no rollback mechanism.

---

## 8. API Design

### Question 1:
Describe the REST API design of JARVIS.

### Answer:
The API follows RESTful conventions organized under /api/:

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | /api/auth/signup | None | Create account |
| POST | /api/auth/login | None | Login, get JWT |
| GET/POST | /api/auth/verify | Bearer | Validate token |
| GET | /api/memory | Bearer | Get all memory entries |
| GET | /api/memory/sessions | Bearer | Get all sessions |
| DELETE | /api/memory/:type/:id | Bearer | Delete an entry |
| POST | /api/decisions | Bearer | Create decision |
| POST | /api/insights | Bearer | Create insight |
| POST | /api/mistakes | Bearer | Create mistake |
| POST | /api/sessions | Bearer | Create/update session |
| POST | /api/ai/chat | Bearer | Send AI message |
| POST | /api/ai/embed | Bearer | Generate + store embedding |
| POST | /api/ai/search | Bearer | Semantic search |
| GET | /api/ai/apikey | Bearer | Check BYOK key status |
| POST | /api/ai/apikey | Bearer | Save/clear BYOK key |
| POST | /api/ai/insights | Bearer | Generate AI insight cards |
| GET | /api/health | None | Health check + provider status |

Response format: All endpoints return JSON. Errors include { error: "message" }. Success responses include { ok: true } and relevant data. HTTP status codes are used correctly: 400 bad request, 401 unauthorized, 409 conflict, 500 server error.

---

### Question 2:
How do you handle errors in the API?

### Answer:
Error handling happens at two levels:

Route level: Every route handler is wrapped in try/catch. On error it returns res.status(500).json({ error: e.message }). For expected errors, specific status codes are returned before the catch block:
```
if (!email || !username) return res.status(400).json({ error: 'All fields are required.' });
if (memory.getUserByEmail(email)) return res.status(409).json({ error: 'Email already in use.' });
```

Frontend level: The api.* methods use try/catch around fetch calls. On 401, the token is cleared and the user is redirected to login.html. For other errors, a toast notification is shown.

AI failures are fail-silent: When an AI call fails during a session (check-in, reaction to logged entry), the error is caught, logged to console, and the UI continues without interruption. A failed AI check-in should never block a user from logging their work.

---

## 9. Security

### Question 1:
What security measures did you implement in JARVIS?

### Answer:
Several layers of security are in place:

1. Password hashing: bcryptjs with 10 salt rounds. Never storing plaintext passwords. bcrypt.compare() for verification — timing-safe comparison prevents timing attacks.

2. JWT authentication: Stateless tokens, server-signed with process.env.JWT_SECRET. Never hardcoded secrets in source code (.env + .gitignore).

3. API key proxy pattern: The Gemini/OpenAI/Anthropic API keys live only on the server in environment variables. The browser never sees them. Every AI request goes through the backend which injects the key before forwarding.

4. Route protection: All data endpoints require a valid JWT. Only signup, login, health check, and static files are public.

5. Input validation: Required fields are checked before any database operation. The type parameter in DELETE /api/memory/:type/:id is validated against an allowlist (decisions, insights, mistakes).

6. SQL injection prevention: All queries use better-sqlite3 prepared statements with parameterized bindings. No string concatenation into SQL.

7. XSS prevention: The escapeHtml() function in app.js escapes HTML special characters before inserting user-generated content into the DOM.

8. CORS: cors() middleware enabled for development. In production I would configure it with an origin allowlist.

Known limitation: The BYOK API key is stored as plaintext in SQLite. Production should encrypt it at rest.

---

## 10. Performance Optimization

### Question 1:
What performance optimizations did you implement?

### Answer:
Several intentional optimizations:

1. SQLite WAL mode: Allows concurrent reads during writes, reducing lock contention. Critical for the real-time update flow where a write triggers a broadcast and other clients immediately issue reads.

2. Prepared statements: All SQL is pre-compiled. Queries run fast without re-parsing overhead.

3. Optimistic UI updates: Decisions/insights/mistakes appear in the feed immediately on click, before the API confirms. The user perceives zero latency.

4. Fire-and-forget embedding generation: api.embedEntry() is called with no await — it runs in the background and any failure is silently caught. The user's save action completes instantly without waiting for the Gemini embedding API.

5. History trimming: The AI conversation history array is capped at 20 messages (conversationHistory.splice(0, 2) when length > 20). This prevents the API request payload from growing unboundedly.

6. disableThinking for chat: When calling Gemini 2.5 models, thinkingConfig: { thinkingBudget: 0 } is set to disable extended thinking, reducing latency significantly for quick chat replies.

7. Zero build step frontend: No webpack, no bundler, no transpilation. The browser loads exactly what is written, with no overhead.

8. Database connection singleton: The db instance in database.ts is created once at module load and reused across all requests.

---

## 11. Testing

### Question 1:
How did you test JARVIS?

### Answer:
JARVIS has a multi-layered testing approach:

Integration Tests (Jest + Supertest):
- tests/ai.test.js — Tests AI routes: mocks Gemini API response, verifies /api/ai/chat returns expected structure, tests error handling when API key is missing
- tests/api.test.js — Tests CRUD routes: signs up a test user, creates decisions/insights/mistakes, verifies they are returned, deletes them, verifies deletion
- tests/session.test.js — Tests session lifecycle: start, check status, update with end time and duration
- tests/theme.test.js — Uses jest-environment-jsdom to test theme-toggle UI behavior (DOM manipulation, localStorage persistence)

E2E Tests (Puppeteer):
- tests/manual_user_journey.js — Automates a real browser: loads the app at localhost:3001, fills in forms, clicks buttons, verifies DOM state. Tests the complete signup to dashboard to log decision to verify memory feed flow.

Test setup:
- jest.config.js configures separate test environments (node for API tests, jsdom for DOM tests)
- Supertest imports the Express app from server.js and makes HTTP requests in-process — no need to start a real server
- Test users are created fresh in each test file to avoid state pollution

What I would add in production: API contract tests, database seeding/teardown fixtures for test isolation, performance benchmarks for cosine similarity search.

---

## 12. Deployment

### Question 1:
How would you deploy JARVIS?

### Answer:
JARVIS has a Dockerfile and docker-compose.yml for containerized deployment.

Dockerfile (simplified):
```
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3001
CMD ["node", "server/server.js"]
```

Run: npm run docker:build then npm run docker:run — maps port 3001.

Cloud deployment options:
- Railway / Render / Fly.io: Push to GitHub, connect repo, set env vars (GEMINI_API_KEY, JWT_SECRET, PORT), auto-deploy
- AWS EC2: Deploy the Docker container, use Nginx as a reverse proxy for HTTPS termination
- Persistent SQLite: Mount a volume for server/jarvis.db — otherwise the database resets on every container restart

Main SQLite scaling challenge: In containerized environments, the SQLite file must be on a persistent volume. With multiple container instances, SQLite becomes a bottleneck (single writer). At that point, migrating to PostgreSQL is the path forward.

---

## 13. Design Patterns

### Question 1:
What design patterns are used in JARVIS?

### Answer:
Several patterns are present, both explicit and emergent:

1. Strategy Pattern — callLLM() dispatches to callGemini(), callOpenAI(), or callAnthropic() based on LLM_PROVIDER config. The strategy is selected at runtime without changing the calling code.

2. Module Pattern — speech.js is wrapped in an IIFE (function() { ... })(), creating a private scope. All internal variables are hidden; only DOM event behavior is exposed.

3. Repository Pattern — The memory export in database.ts abstracts all data access behind named methods (insertDecision(), getAll(), buildContext()). Routes never write raw SQL — they call the repository.

4. Facade Pattern — The api = { ... } object in app.js is a facade over fetch(). Route handling, headers management, error parsing, and URL construction are encapsulated behind simple method calls like api.saveDecision(entry).

5. Observer Pattern — The WebSocket broadcast system. The server (subject) maintains a Set of clients (observers). When state changes, all observers are notified via broadcastUpdate().

6. Singleton — The SQLite database connection (const db = new Database(DB_PATH)) is created once at module import time and shared across all prepared statements.

---

## 14. Challenges and Debugging

### Question 1:
What was the hardest technical challenge you faced building JARVIS?

### Answer:
The most complex challenge was the database migration system. When I added multi-user support, I needed to add a user_id column to all content tables — but existing users already had data without that column. SQLite does not support ALTER TABLE ... ADD COLUMN ... NOT NULL without a default value, and my foreign key constraints made it trickier.

The solution was a backup-recreate-restore migration pattern: create a backup table, drop the original, create it with the new schema, insert from backup with the default value ('default-user'), drop the backup. I also had to handle foreign key constraints on old tables preventing the migration — so I added a second migration pass to remove those constraints.

The challenge was making this idempotent — it had to run safely on every server startup without breaking if it had already run. Using PRAGMA table_info() to check column existence before migrating solved this.

Second major challenge: The WebSocket server in test environments. ws behaves differently when imported by Jest — the test environment has no real HTTP server. I wrapped the instantiation in try/catch and provided a mock { on: () => {} } object when WebSocket.Server throws, so tests can import the server module without crashing.

---

### Question 2:
How did you handle the API key security concern?

### Answer:
This was a conscious design decision made early: the API key never leaves the server.

Initially I was tempted to have the frontend call Gemini directly — it would have been simpler. But that would expose the API key in the browser's network tab. Even obfuscated in a config file, any user with DevTools open could extract it.

The backend proxy pattern solves this completely: the frontend calls /api/ai/chat, the server reads the key from process.env.GEMINI_API_KEY (never sent to the client), makes the Gemini API call server-side, and returns only the AI response. The key is invisible to the browser.

The .gitignore file excludes .env from version control. The .env.example file is committed with placeholder values so other developers know what variables to set.

---

## 15. Scalability

### Question 1:
How would you scale JARVIS to support 10,000 users?

### Answer:
JARVIS in its current form is a personal tool. But here is exactly what I would change to scale it:

Database to PostgreSQL: SQLite has a single writer limitation. With concurrent users, PostgreSQL with connection pooling handles thousands of simultaneous connections. The migration would be largely schema-compatible since I am already using standard SQL.

Vector Storage to pgvector or Pinecone: Currently semantic search loads all embeddings for a user from SQLite and does brute-force cosine similarity in Node.js. At thousands of entries per user at scale, this breaks down. Migration to pgvector (PostgreSQL extension) enables ANN search that scales to billions of vectors.

JWT to Refresh Token Pattern: Current 7-day tokens mean a compromised token stays valid for 7 days. Add short-lived access tokens (15 minutes) plus a refresh token (longer-lived, stored in httpOnly cookie + database).

WebSocket to Redis Pub/Sub: With multiple Node.js instances, the in-memory clients Set only contains connections to that instance. A user on instance A will not get updates triggered on instance B. Solution: Redis Pub/Sub as a message bus — all instances subscribe, broadcasts come through Redis so all instances forward to their local clients.

API Rate Limiting: Add express-rate-limit middleware to prevent abuse of AI endpoints.

---

## 16. Behavioral Questions

### Question 1:
What did you learn from building JARVIS?

### Answer:
The deepest lesson was about architectural clarity before code. I started with a simple static HTML page and kept adding features — first localStorage, then a backend, then auth, then WebSockets, then semantic search. Each addition required refactoring earlier decisions.

If I were starting over, I would spend a week defining the full data model, the API contract, and the authentication flow before writing a single line of feature code. The migration pain I experienced (adding user_id columns to existing tables) was entirely preventable with upfront schema design.

Technically, I learned how vector embeddings actually work — not just conceptually, but by implementing the cosine similarity function from scratch and debugging why searches were returning unexpected results (it was a normalization issue with the dot product calculation).

I also learned how to work with TypeScript in an incremental migration — taking existing JS files and adding types layer by layer without breaking the running application.

---

### Question 2:
How do you prioritize features in a solo project?

### Answer:
I use a prioritization framework with three questions: Does this unblock other features? Does this directly add user value? Is this technically interesting enough to teach me something?

For JARVIS, I built in this order:
1. Core memory logging — fundamental; everything else depends on it
2. Backend + SQLite persistence — nothing is valuable if it is not saved
3. Authentication — multi-user support enables sharing and deployment
4. AI chat during sessions — high user value; the core differentiator
5. Semantic search — technically deep and high value
6. BYOK — enables real-world deployment without shared quota limits
7. Sessions tab, export, confidence chart — polish

I explicitly deprioritized things that were technically interesting but not user-impactful — like building a custom vector database or implementing streaming AI responses.

---

### Question 3:
How did you approach debugging a hard bug?

### Answer:
The hardest bug was the WebSocket crash in the test environment. When running npm test, Jest would import server.js which would immediately try to create a WebSocket.Server — and crash because there was no HTTP server in the test context.

My debugging process:
1. Isolated the error: Read the stack trace fully — it pointed to the WebSocket.Server constructor in server.js
2. Understood the root cause: The ws library's Server constructor requires a running HTTP server when passed a { server } option — which does not exist in the Jest test context
3. Explored solutions: Lazy-initialize only if not in test environment (fragile), try/catch with a no-op mock fallback (simple), or avoid exporting the server module from tests
4. Chose the simplest fix: try/catch with a mock { on: () => {} } object when WebSocket.Server throws
5. Verified: npm test passed; npm start still worked with real WebSockets

The lesson: always isolate first, understand the root cause before reaching for a fix, and pick the simplest solution that does not compromise the production path.

---

## 17. Tricky / Deep-Understanding Questions

### Question 1:
Why does the backend use INSERT OR REPLACE instead of INSERT?

### Answer:
INSERT OR REPLACE handles the case where an entry with the same PRIMARY KEY already exists — instead of throwing a UNIQUE constraint violation, it deletes the old row and inserts the new one. This is SQLite's form of upsert (update-or-insert).

In JARVIS, this matters because if a WebSocket broadcast fires and the frontend tries to save the same entry twice (network retry, optimistic update conflict), the second insert will not crash — it silently overwrites with identical data. It is a defensive measure against duplicate-save bugs.

The trade-off: INSERT OR REPLACE deletes and re-inserts, which resets any columns not specified in the INSERT. For JARVIS's schema, all columns are specified explicitly so this is safe.

---

### Question 2:
What happens if the Gemini API is down or rate-limited during a session?

### Answer:
JARVIS is designed to degrade gracefully at every AI call point:

- Session start greeting fails: startSession() has a try/catch around the opening AI message. If it fails, a static fallback message appears. The session timer and logging still work perfectly.

- AI check-in fails: triggerAICheckin() catches all errors, logs to console with console.warn, and does nothing else. The check-in simply does not appear.

- Reaction to logged entry fails: reactToCapture() is similarly fail-silent. The entry is already saved; AI commentary is optional enrichment, not a blocker.

- User reply fails: Shows a categorized error in the chat — different messages for API key issues, rate limits, and network failures.

This "AI is enhancement, not requirement" philosophy means the core logging and session timer functionality never depends on a third-party API being available.

---

### Question 3:
Why are embeddings stored as TEXT (JSON string) instead of a BLOB?

### Answer:
I chose JSON TEXT for three reasons:

1. Human readability during debugging: I can SELECT embedding FROM embeddings LIMIT 1 in SQLite's CLI and actually read the numbers, which helped enormously when I was debugging the cosine similarity implementation.

2. JSON.parse() is fast enough: Parsing a 768-float JSON string takes microseconds. At the scale JARVIS operates — hundreds of entries — this overhead is negligible.

3. Simpler code: better-sqlite3 requires explicit Buffer handling for BLOBs. JSON strings are stored and retrieved as plain strings with full TypeScript type safety.

In a production vector search system, I would store embeddings in a purpose-built vector database which uses optimized binary formats internally.

---

### Question 4:
The conversation history is capped at 20 messages. Why, and what are the risks?

### Answer:
The cap removes the oldest two messages when the array exceeds 20 entries.

Why: LLM APIs charge by token count. A long conversation history makes each API request increasingly expensive and slow. Most LLMs also have context window limits — keeping history bounded prevents hitting those limits.

The risk: Removing the oldest messages means the AI forgets early conversation context. In a long session, JARVIS might not remember a commitment the user made at the start when answering a question at the end.

Better approach in production: Instead of simple truncation, use a sliding window with summarization — when history exceeds N messages, send the oldest N/2 messages to the AI with a prompt: "Summarize the key points of this conversation in 3 sentences." Replace those messages with the summary. This preserves the gist without using the full token budget.

---

### Question 5:
How is the buildContext() function used to give the AI memory?

### Answer:
buildContext(userId) in database.ts is called inside buildSystemPrompt(userId) in ai.ts before every AI request. It queries SQLite for up to 10 decisions, 10 insights, and 10 mistakes for the user, and formats them as a plain text block that is injected directly into the Gemini system prompt.

This means JARVIS has contextual memory of your cognitive history in every conversation without needing a vector database — the SQLite query provides recent context for free. For older or more specific entries, the semantic search feature handles retrieval.

---

## Quick Reference — Technology Justification Table

| Technology | Why Chosen | Known Trade-off |
|-----------|-----------|----------------|
| Node.js | Non-blocking I/O, same language as frontend | Single-threaded; CPU-bound tasks block event loop |
| Express | Mature ecosystem, TypeScript support, fast dev | Slower than Fastify; minimal structure enforcement |
| TypeScript | Compile-time safety, catches bugs early | Build step required; learning curve |
| SQLite (better-sqlite3) | Zero-config, serverless, fast sync API | Single writer; not for horizontal scaling |
| WAL mode | Concurrent reads during writes | Slightly more disk I/O; WAL file to manage |
| JWT | Stateless auth, no server-side session store | No easy revocation; 7-day expiry risk |
| bcryptjs | Industry-standard password hashing | Slower than argon2id (acceptable for this use case) |
| Gemini text-embedding-004 | 768-dim embeddings, free tier available | External API dependency; adds embedding latency |
| WebSocket (ws) | Low overhead, native browser support | No auto-reconnect (implemented manually) |
| Vanilla JS | Zero build step, zero dependencies, full control | More code to write; no component reusability |
| Chart.js | Easy API, good-looking charts, lightweight | Less customizable than D3.js |
| Puppeteer | Real browser E2E tests | Slow, requires Chrome, flaky on CI |
| Jest + Supertest | Standard Node testing, in-process HTTP testing | Integration tests rather than pure unit tests |

---

Generated for JARVIS v2.0.0 — Personal Cognitive Operating System
Tech Stack: Node.js, Express, TypeScript, SQLite, Gemini API, WebSockets, Vanilla JS
