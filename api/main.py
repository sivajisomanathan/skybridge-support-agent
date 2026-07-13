"""
FastAPI backend for the SkyBridge support agent.

Endpoints:
  POST /chat       - run one turn of the agent
  POST /feedback   - submit thumbs up/down (+ optional reason) for a turn
  GET  /health     - basic liveness check
  GET  /conversations - list threads for the UI sidebar
  /                - serves the static chat UI

Deployment concerns addressed here (Phase 8 requirements):
  - Latency capture: every /chat call is timed and logged.
  - Error capture + graceful failure: the graph invocation is wrapped in
    try/except. An unhandled exception (e.g. the OpenAI API being down)
    returns a clean JSON error response with a safe, generic customer-facing
    message -- it does not crash the server process, and does not leak a
    stack trace to the client.
  - Structured logging to stdout (Render captures stdout as its log stream)
    in addition to LangSmith tracing at the node level.
"""
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import kb, memory
from agent.graph import build_graph, AgentState
from agent.tracing import graph_tracing_context

app = FastAPI(title="SkyBridge Support Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # same-origin in production (UI is served by this app); "*" only matters for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None
PDF_PATH = "SkyBridge_Policy_Handbook.pdf"


def _redact(text: str, max_len: int = 80) -> str:
    redacted = re.sub(r"\b[A-Z0-9]{5,8}\b", "[REDACTED]", text)
    redacted = re.sub(r"\S+@\S+", "[REDACTED_EMAIL]", redacted)
    return redacted[:max_len]


def _log(event: dict) -> None:
    """Structured log line to stdout. Render (and most PaaS platforms) capture
    stdout directly as the application's log stream -- no extra logging
    infrastructure needed for this project's scale."""
    print({"timestamp": datetime.now(timezone.utc).isoformat(), **event})


@app.on_event("startup")
def startup() -> None:
    global _graph
    memory.init_db()
    try:
        chunks = kb.build_index(PDF_PATH)
        _log({"event": "startup", "status": "ok", "kb_chunks": len(chunks)})
    except Exception as e:
        # Deliberately do NOT re-raise here: a KB build failure (e.g. missing
        # PDF, bad API key) should not prevent the process from starting at
        # all -- /health should still respond, and /chat should fail with a
        # clear 503 rather than the whole app refusing to boot. This is the
        # graceful-failure behavior for a startup-time problem specifically.
        _log({"event": "startup", "status": "error", "error": str(e)})
    _graph = build_graph()


class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str


class FeedbackRequest(BaseModel):
    thread_id: str
    intent_category: str
    rating: str  # "up" | "down"
    reason: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "kb_ready": kb._index is not None}


@app.get("/conversations")
def conversations():
    return memory.list_conversations()


@app.get("/history/{thread_id}")
def history(thread_id: str):
    """Returns this thread's stored turns, for the UI to replay into the chat
    window on load/switch. Only role+content are stored per turn (not the
    full per-turn metadata like intent/tools/trace) -- replayed messages
    render as plain bubbles without a Trace panel, which is a display
    limitation, not a data one: the full context is still used server-side
    for classification/retrieval on the NEXT turn regardless."""
    return memory.get_history(thread_id)


@app.post("/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    start = time.monotonic()

    try:
        if kb._index is None:
            # Graceful failure: KB never came up at startup (e.g. bad
            # Pinecone key). Fail this request cleanly instead of crashing on
            # a null index inside retrieve_node.
            raise RuntimeError("Knowledge base is not available right now.")

        history = memory.get_history(thread_id)
        initial_state: AgentState = {
            "thread_id": thread_id, "user_input": req.message, "resolved_input": None,
            "conversation_history": history, "pnr": None, "intent_category": None,
            "outcome_type": None, "retrieved_context": None, "retrieved_sections": [],
            "booking_record": None, "calc_result": None, "tools_used": [],
            "feedback_adapted": False, "final_response": None,
        }
        final_state = None
        with graph_tracing_context():
            final_state = _graph.invoke(initial_state, config={"recursion_limit": 10})

        memory.touch_conversation(thread_id, title_hint=req.message)
        memory.append_turn(thread_id, req.message, final_state["final_response"])

        latency_ms = round((time.monotonic() - start) * 1000)
        _log({
            "event": "chat", "thread_id": thread_id, "input_preview": _redact(req.message),
            "intent_category": final_state.get("intent_category"),
            "tools_used": final_state.get("tools_used"),
            "outcome_type": final_state.get("outcome_type"),
            "feedback_adapted": final_state.get("feedback_adapted", False),
            "latency_ms": latency_ms, "status": "ok",
        })

        return {
            "thread_id": thread_id,
            "response": final_state["final_response"],
            "intent_category": final_state.get("intent_category"),
            "outcome_type": final_state.get("outcome_type"),
            "tools_used": final_state.get("tools_used"),
            "retrieved_sections": final_state.get("retrieved_sections"),
            "resolved_input": final_state.get("resolved_input"),
            "feedback_adapted": final_state.get("feedback_adapted", False),
            "latency_ms": latency_ms,
        }

    except Exception as e:
        latency_ms = round((time.monotonic() - start) * 1000)
        _log({
            "event": "chat", "thread_id": thread_id, "input_preview": _redact(req.message),
            "status": "error", "error": str(e), "traceback": traceback.format_exc(),
            "latency_ms": latency_ms,
        })
        # Graceful failure: the customer gets a clean, generic message and a
        # 503, never a raw exception or a crashed connection.
        raise HTTPException(
            status_code=503,
            detail="I'm having trouble processing that right now. Please try again in a moment, "
                   "or contact support directly if this continues.",
        )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    new_count = memory.submit_feedback(req.thread_id, req.intent_category, req.rating)
    _log({
        "event": "feedback", "thread_id": req.thread_id, "intent_category": req.intent_category,
        "rating": req.rating, "reason_given": bool(req.reason), "negative_count_after": new_count,
    })
    return {"negative_count_after": new_count}


# --- Static UI ---
STATIC_DIR = Path(__file__).parent.parent / "static"

@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
