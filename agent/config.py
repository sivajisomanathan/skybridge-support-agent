"""
Shared configuration for the SkyBridge support agent.
Imported by both the FastAPI app (api/main.py) and any notebook/script that
needs the same clients and constants -- this is the single source of truth
referenced in the Problem Framing Document's deployment architecture note.
"""
import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_KEYS = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(missing)}. "
        f"Set them in a .env file (see .env.example)."
    )

# --- LangSmith tracing ---
# Tracing is opt-in via LANGSMITH_API_KEY. If it's not set, the app still runs
# fine -- @traceable becomes a no-op wrapper (see agent/tracing.py) rather than
# a hard failure. This keeps local development possible without a LangSmith
# account, while still satisfying the observability requirement when deployed.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "skybridge-support-agent"))
    os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
    # CRITICAL: LangGraph has its own built-in, automatic LangSmith tracing --
    # completely separate from agent/tracing.py's @traceable wrapper. Once
    # LANGCHAIN_TRACING_V2 is on, LangGraph traces the entire graph invocation
    # and every routing function on its own, capturing the full raw state
    # (including PNRs) -- this bypasses per-function redaction entirely,
    # confirmed by inspecting real traces in a live LangSmith project (the
    # top-level "LangGraph" run and routing-function runs showed unredacted
    # PNRs even though individually-decorated node functions redacted
    # correctly). Per LangSmith's own documentation, the only mechanism that
    # reliably applies regardless of which code path generates a trace is
    # hiding inputs/outputs globally at the client/transport level:
    os.environ.setdefault("LANGSMITH_HIDE_INPUTS", "true")
    os.environ.setdefault("LANGSMITH_HIDE_OUTPUTS", "true")
    # Trade-off, stated plainly: this hides ALL input/output content in
    # LangSmith (not just PII) -- run names, node execution order/hierarchy,
    # per-node latency, and error status remain visible, but you lose
    # content-level inspection (intent_category, tools_used, etc.) directly
    # in the LangSmith UI. Full content-level detail is still available in
    # this app's own structured stdout logs (see api/main.py's _log calls).
    TRACING_ENABLED = True
else:
    TRACING_ENABLED = False

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
INDEX_NAME = "skybridge-policy-kb"
EMBED_DIM = 1536
SIMILARITY_THRESHOLD = 0.30

MAX_HISTORY_TURNS = 3

# SQLite file for session/feedback persistence. NOTE (documented in README):
# on Render's free tier, the filesystem is ephemeral across redeploys and does
# NOT survive a new deploy -- only a paid persistent disk add-on would. This
# still improves on Phase 6/7's pure in-memory dicts, which lost state on
# every process restart, including ones that happen without a redeploy
# (e.g. a crash, or a free-tier idle spin-down).
DB_PATH = os.getenv("AGENT_DB_PATH", "agent_state.db")
