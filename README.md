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

Set `LANGSMITH_API_KEY` in `.env` to enable tracing.

**PII redaction — two attempts, here's where it actually landed.**

*Attempt 1:* redact PII inside `agent/tracing.py`'s `@traceable` wrapper (`process_inputs`/
`process_outputs`), applied per node function. Live testing against a real LangSmith
project showed this was incomplete: LangGraph has its own built-in, automatic LangSmith
tracing, separate from that wrapper. Once `LANGCHAIN_TRACING_V2` is enabled, LangGraph
traces the entire graph invocation and every routing function
(`route_after_classify`, etc.) on its own, capturing the full raw state — real traces
confirmed the top-level "LangGraph" run and routing-function runs showed unredacted
PNRs, even though individually-decorated functions redacted correctly for their own
input/output.

*Attempt 2 (a deliberate overcorrection, flagged as such at the time):* set
`LANGSMITH_HIDE_INPUTS`/`LANGSMITH_HIDE_OUTPUTS` globally, which guarantees no PII leaks
anywhere — but hides *all* content, not just PII, making LangSmith traces show only
names/timing/hierarchy with zero field-level visibility. This closed the leak but lost
too much debugging value to be the right long-term answer.

**Current approach:** a `Client` configured with a recursive `anonymizer`
(`agent/config.py`'s `TRACED_CLIENT`, built with LangSmith's `create_anonymizer` against
the same PNR/email regex patterns used elsewhere in this project), applied to LangGraph's
automatic tracing via a `tracing_context(client=TRACED_CLIENT)` wrapper around the actual
`graph.invoke()` call (see `agent/tracing.py`'s `graph_tracing_context()`, used in
`api/main.py`'s `/chat` handler). This is LangSmith's own documented pattern for this
exact scenario — their PII-removal reference repo has a dedicated LangGraph example for
it — and should mask only PII-matching strings while preserving all other field-level
visibility (`intent_category`, `tools_used`, etc.) in the LangSmith UI.

**Still not independently verified end-to-end**: this is the second attempt at this
specific problem, and like the first, could not be tested against a live LangSmith
account in this build environment (no PyPI/network access in the sandbox). On your next
real run, check traces at **all three levels** — the top-level "LangGraph" run, a
routing-function run (e.g. `route_after_classify`), and an individual node run — and
confirm PII is masked at each, not just the innermost one (that's precisely the gap the
first attempt had). If PII still leaks through the top-level/routing-function runs
specifically, the guaranteed fallback is the blunt `LANGSMITH_HIDE_INPUTS`/
`LANGSMITH_HIDE_OUTPUTS` approach from Attempt 2 above — it trades away field-level
visibility but is documented to work regardless of trace source.

If `LANGSMITH_API_KEY` is not set, `@traced_node` becomes a transparent no-op (verified —
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

- **SQLite persistence only survives within a single running container — and on Render's
  free tier, that's a much shorter window than "until you redeploy."** Per Render's own
  documentation, a free web service's filesystem is wiped not only on redeploys but on
  every **restart and every idle spin-down** — and free services spin down automatically
  after just 15 minutes with no incoming traffic. In practice this means: leave the app
  idle for 15+ minutes, and the next request gets a fresh container with an empty
  `agent_state.db` — all conversations and feedback history are gone, not just the
  browser's active thread pointer. This is a materially bigger limitation than "state
  resets across intentional redeploys" (which is what an earlier version of this note
  said) — on the free tier, it effectively resets on any normal gap in usage. A paid
  Render instance type with an attached persistent disk (or a managed database like
  Render Postgres) would be needed to actually fix this; SQLite-on-local-disk was chosen
  here as a deliberate, documented "better than Phase 6/7's in-memory dicts, but still not
  production-durable" middle ground, not a claim of true persistence.
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
