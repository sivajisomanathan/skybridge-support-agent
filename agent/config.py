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
    # langsmith's SDK reads LANGCHAIN_API_KEY; accept LANGSMITH_API_KEY as the
    # user-facing name and mirror it, since both names appear in LangSmith's
    # own docs across versions.
    os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
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
