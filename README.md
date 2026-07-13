# SkyBridge Support Agent — Phase 8: Deployment

FastAPI backend + static HTML/JS chat UI, serving the LangGraph agent built across
Phases 4–7 (RAG retrieval, booking/policy tools, per-thread memory, feedback-driven
adaptation), now packaged as a single deployable service.

## Project layout

Everything below lives at the **root** of this zip/repo — there is no extra wrapper
folder to `cd` into first.

```
agent/
  config.py     # env loading, OpenAI/Pinecone clients, LangSmith setup
  kb.py          # PDF extraction, chunking, embedding, retrieval (Phase 4)
  tools.py       # booking_lookup, policy_calculator (Phase 5)
  memory.py      # SQLite-backed sessions + feedback (Phase 6/7, persistent)
  tracing.py     # LangSmith @traceable wrapper with PII redaction
  graph.py       # AgentState, all nodes, routing, graph builder
api/
  main.py        # FastAPI app: /chat, /feedback, /health, /conversations
static/
  index.html     # chat UI (sidebar, trace panel, escalation cards, feedback, theme)
requirements.txt
.env.example
Procfile
SkyBridge_Policy_Handbook.pdf
```

`agent/` has no FastAPI or Colab-specific code in it — it's the single shared
implementation referenced in the Problem Framing Document's deployment architecture
note. A notebook can add this folder to `sys.path` and `from agent import graph, kb,
tools` to run the exact same logic that's deployed here.

## Local setup

1. Unzip, then from the unzipped folder: `pip install -r requirements.txt`
2. `cp .env.example .env` and fill in your real `OPENAI_API_KEY` and `PINECONE_API_KEY`
   (LangSmith key is optional — see below).
3. `SkyBridge_Policy_Handbook.pdf` is already included at the root — no need to add it.
4. `uvicorn api.main:app --reload`
5. Open `http://localhost:8000` — the UI is served at the root path.

## LangSmith tracing (optional but recommended)

Set `LANGSMITH_API_KEY` in `.env` to enable tracing. Every graph node
(`resolve_reference`, `classify`, `retrieve`, `booking_lookup`, `policy_calculator`,
`adaptive_escalate`, `compose`) is wrapped with `@traceable`, so each one appears as a
separate step in your LangSmith project, with PII redacted (PNR-like tokens, email
addresses) from the traced inputs/outputs before they're sent — see `agent/tracing.py`
for the exact redaction logic.

**Not independently verified end-to-end**: this project's sandbox has no PyPI/network
access, so the `langsmith` package itself (and its `process_inputs`/`process_outputs`
hooks specifically) could not be installed and tested against a live LangSmith account
here. Confirm on your first real run that traces actually appear in your LangSmith
project and that redaction is applied as expected; if `process_inputs`/`process_outputs`
aren't supported in your installed version, run `pip install -U langsmith`.

If `LANGSMITH_API_KEY` is not set, `@traceable` becomes a transparent no-op (verified —
see the test suite) and the agent behaves identically, just without traces.

## Deploying to Render

1. Push this repository to GitHub (the contents of this zip, unzipped, as the repo root
   — no `app/` subfolder involved).
2. Create a new Render Web Service, point it at the repo. Leave the root directory as
   the repo root (default) — do not set a subdirectory.
3. Build command: `pip install -r requirements.txt`
4. Start command: uses `Procfile` automatically (`uvicorn api.main:app --host 0.0.0.0
   --port $PORT`).
5. Set environment variables in Render's dashboard (never commit `.env`):
   `OPENAI_API_KEY`, `PINECONE_API_KEY`, optionally `LANGSMITH_API_KEY` and
   `LANGSMITH_PROJECT`.
6. `SkyBridge_Policy_Handbook.pdf` is already committed at the repo root alongside the
   app code — it's a fictional public-facing policy document, not a secret, so no special
   handling is needed for it.

**Note on Python version:** this repo includes a `.python-version` file pinning Render to
Python 3.11.9. Without it, Render may default to a much newer Python version that
doesn't yet have pre-built wheels for `pydantic-core` (a `fastapi`/`pydantic`
dependency), causing pip to try compiling it from source via Rust/maturin — which fails
on Render's build environment (read-only filesystem where the Rust cargo cache tries to
write). If you see a build error mentioning `maturin`, `cargo`, or `pydantic-core`:
1. Confirm `.python-version` is present at the repo root and was actually committed/pushed.
2. As a second, independent safeguard, also set the `PYTHON_VERSION` environment variable
   to `3.11.9` in the Render dashboard (Environment tab) — Render supports both
   mechanisms, and setting both removes any ambiguity about which one Render is reading.
3. After changing either, trigger a fresh deploy (not just a restart) and check the top of
   the build log for the Python version Render reports using, to confirm the pin took
   effect before debugging further.

**Note on `openai`/`httpx` version compatibility:** `requirements.txt` pins
`httpx==0.27.2` alongside `openai==1.51.0`. This is required, not optional — newer
`httpx` versions removed a `proxies` parameter that this version of the `openai` SDK's
internal HTTP client still passes, causing a `TypeError: Client.__init__() got an
unexpected keyword argument 'proxies'` at import time (the app fails to even start). If
you ever upgrade `openai` in the future, check whether it still needs this `httpx` pin or
has been updated to work with newer `httpx` releases.

## Logging, latency, and error handling

- Every `/chat` call is timed; latency in milliseconds is logged and also returned in
  the API response (and shown in the UI under each message).
- All logs are structured dicts printed to stdout, which Render (and most PaaS
  platforms) capture directly as the service's log stream — no extra logging
  infrastructure was added for this project's scale.
- **Graceful failure handling**: the graph invocation in `/chat` is wrapped in
  try/except. An unhandled exception (verified in testing by simulating an OpenAI API
  outage) is logged with its full traceback server-side, but the client only ever
  receives a clean `503` with a safe, generic message — never a raw stack trace, and the
  server process does not crash. The app also recovers cleanly on the next request once
  the underlying issue clears (also verified in testing).
- A KB build failure at startup (e.g. bad Pinecone key, missing PDF) does not prevent
  the process from starting — `/health` still responds, and `/chat` fails cleanly with a
  503 rather than the whole service refusing to boot.

## Known limitations and deployment assumptions

- **SQLite persistence is single-instance, not shared.** `agent/memory.py` replaced
  Phase 6/7's in-memory Python dicts with a SQLite file, so session/feedback state now
  survives a process restart within the same running instance — a real improvement, and
  directly tested (see test suite: state survives a simulated restart). However, it is
  **not** a shared/multi-instance store: if this service is ever scaled to more than one
  Render instance, each instance would have its own separate SQLite file and its own view
  of sessions and feedback. Fixing that would require a real shared database (Postgres,
  etc.), which is out of scope here.
- **Render free-tier cold starts.** Per the Problem Framing Document's known limitation:
  Render's free tier spins down on inactivity, introducing a cold-start delay of up to
  ~60 seconds on the first request after an idle period. The `/health` endpoint and the
  UI's status indicator will reflect this (showing "Starting up..." until the KB index
  finishes loading).
- **No authentication.** Any visitor to the deployed URL can chat as any customer and
  submit feedback for any thread_id they know or guess. This is acceptable for a capstone
  demo but would need real auth (and per-user thread ownership checks) before handling
  real customers.
- **LangSmith integration is unverified end-to-end** (see above) due to no PyPI access
  in the build environment — confirm traces appear correctly on first real deployment.
- **The UI's conversation switcher replays past messages as plain bubbles, without full
  trace metadata.** A `GET /history/{thread_id}` endpoint returns the stored
  role/content turns for a thread, and the UI calls it both when switching conversations
  in the sidebar and on page load (so a browser refresh doesn't blank the current
  conversation). Replayed messages don't have a Trace panel or feedback buttons, because
  only the response text is stored historically, not per-turn intent/tools/retrieval
  metadata — that's a display limitation, not a memory gap: the full conversation
  context is still used server-side for classification and retrieval on the next turn
  regardless of what's visually replayed.
