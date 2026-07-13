"""
LangSmith tracing. Two things happen when LANGSMITH_API_KEY is set:
  1. agent/config.py sets LANGSMITH_HIDE_INPUTS/LANGSMITH_HIDE_OUTPUTS=true
     globally -- this is the actual, working PII-redaction mechanism. See
     the README's LangSmith section for the two prior attempts that didn't
     hold up under live testing before landing here.
  2. traced_node still wraps each node function in @traceable, purely so
     each node shows up as a clearly-named, separate step in the trace
     hierarchy (LangGraph's own automatic tracing already does this too,
     so this is mostly redundant/defense-in-depth at this point, kept
     because it's inert and matches the original per-node observability
     goal even with content hidden).

graph_tracing_context() is now a deliberate no-op. An earlier version wrapped
graph.invoke() in tracing_context(client=...) to redirect LangGraph's
automatic tracing through a custom anonymizer -- live testing showed this
stopped ALL tracing from appearing, for reasons that couldn't be diagnosed
without a live install to test against. Given two consecutive live-only
failures on the fine-grained approach, this was reverted in favor of the
simple, documented-safe global hide flags, which don't require wrapping the
invocation in anything and shouldn't share that failure mode.
"""
import re
from contextlib import contextmanager, nullcontext
from agent.config import TRACING_ENABLED


def redact_text(text) -> str:
    if not isinstance(text, str):
        return text
    redacted = re.sub(r"\b[A-Z0-9]{5,8}\b", "[REDACTED]", text)
    redacted = re.sub(r"\S+@\S+", "[REDACTED_EMAIL]", redacted)
    return redacted


def _redact_recursive(obj):
    """Kept for defense-in-depth on individually-decorated functions, though
    LANGSMITH_HIDE_INPUTS/OUTPUTS (agent/config.py) is the mechanism actually
    relied on now. Harmless either way: if hide_inputs/outputs is active,
    this never even gets a chance to matter, since there's no input/output
    payload for it to act on in the first place."""
    if isinstance(obj, dict):
        return {k: _redact_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_recursive(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


def _redact_dict(data: dict) -> dict:
    return _redact_recursive(data)


if TRACING_ENABLED:
    from langsmith import traceable

    def traced_node(name: str):
        def decorator(fn):
            return traceable(
                name=name,
                run_type="chain",
                process_inputs=_redact_dict,
                process_outputs=_redact_dict,
            )(fn)
        return decorator
else:
    def traced_node(name: str):
        """No-op decorator when tracing is disabled -- the node runs exactly
        as written, just without a LangSmith run being created."""
        def decorator(fn):
            return fn
        return decorator


def graph_tracing_context():
    """Deliberate no-op -- see module docstring for why the earlier
    tracing_context()-based version was reverted."""
    return nullcontext()
