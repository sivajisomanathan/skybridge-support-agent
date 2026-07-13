"""
Shared configuration for the SkyBridge support agent.
Imported by both the FastAPI app (api/main.py) and any notebook/script that
needs the same clients and constants -- this is the single source of truth
referenced in the Problem Framing Document's deployment architecture note.
"""
import os
import re
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
    TRACING_ENABLED = True
else:
    TRACING_ENABLED = False

# --- PII-aware LangSmith client ---
# See README's LangSmith section for the full history. Short version:
# LangGraph has its own built-in, automatic LangSmith tracing (separate from
# agent/tracing.py's per-function @traceable wrapper), so per-function
# redaction alone misses the top-level graph run and routing-function runs.
# The fix is a Client with a recursive `anonymizer`, applied via
# tracing_context() around graph.invoke() (see agent/tracing.py's
# graph_tracing_context()).
#
# A first attempt at this failed with "'str' object has no attribute 'sub'"
# in Render's logs, on every single trace -- root cause: create_anonymizer's
# rules must use COMPILED regex objects (re.compile(...)) for "pattern", not
# raw strings. Passing raw strings meant LangSmith tried to call .sub() (a
# compiled-pattern method) on a plain str internally, which fails, silently,
# in a background thread -- explaining why traces stopped appearing
# entirely rather than just failing to redact.
TRACED_CLIENT = None
if TRACING_ENABLED:
    from langsmith import Client
    from langsmith.anonymizer import create_anonymizer

    _ANONYMIZER_RULES = [
        {"pattern": re.compile(r"\b[A-Z0-9]{5,8}\b"), "replace": "[REDACTED]"},
        {"pattern": re.compile(r"\S+@\S+"), "replace": "[REDACTED_EMAIL]"},
    ]
    TRACED_CLIENT = Client(anonymizer=create_anonymizer(_ANONYMIZER_RULES))

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
