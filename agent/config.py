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
    TRACING_ENABLED = True
else:
    TRACING_ENABLED = False

# --- PII-aware LangSmith client (see agent/tracing.py) ---
# LangGraph has its own built-in, automatic LangSmith tracing, separate from
# any @traceable-decorated function -- confirmed via real traces in a live
# LangSmith project showing unredacted PNRs in the top-level "LangGraph" run
# and routing-function runs, even though individually-decorated node
# functions redacted correctly for their own specific input/output.
#
# The correct fix, per LangSmith's own documentation and a dedicated example
# in their PII-removal reference repo for this exact scenario
# (langchain-ai/langsmith-pii-removal, langgraph-example/agent.py): build a
# Client with a recursive `anonymizer` function, and make LangGraph's
# automatic tracing route through THAT client for the duration of each graph
# invocation via `tracing_context`. This masks PII specifically (any string
# matching the PNR/email patterns) while preserving all other field-level
# visibility (intent_category, tools_used, etc.) in the LangSmith UI --
# unlike blanket LANGSMITH_HIDE_INPUTS/OUTPUTS, which hides everything.
#
# CAVEAT: this specific mechanism (a custom anonymizer client covering
# LangGraph's automatic traces via tracing_context, not just explicitly
# decorated functions) could not be verified against a live installation in
# this build environment (no PyPI access). This replaces an earlier attempt
# (per-function process_inputs/process_outputs) that real testing showed was
# incomplete. Confirm on your next real run that PII is masked at ALL trace
# levels (the top-level "LangGraph" run, routing-function runs, and
# individual node runs) -- not just the innermost ones. If PII still leaks
# through the top-level/routing-function runs specifically, the guaranteed
# (but blunt) fallback is setting LANGSMITH_HIDE_INPUTS=true and
# LANGSMITH_HIDE_OUTPUTS=true, which trades away all field-level visibility
# in LangSmith but is documented to work regardless of trace source.
TRACED_CLIENT = None
if TRACING_ENABLED:
    from langsmith import Client
    from langsmith.anonymizer import create_anonymizer

    _ANONYMIZER_RULES = [
        {"pattern": r"\b[A-Z0-9]{5,8}\b", "replace": "[REDACTED]"},
        {"pattern": r"\S+@\S+", "replace": "[REDACTED_EMAIL]"},
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
