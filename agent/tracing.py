"""
LangSmith tracing: a Client-level anonymizer (agent/config.py's TRACED_CLIENT)
covers LangGraph's own automatic tracing via graph_tracing_context() below,
and a per-function @traceable wrapper (traced_node) provides defense-in-depth
redaction for individually-decorated node functions. See agent/config.py for
the full explanation of why both layers exist and what real testing found.
"""
import re
from contextlib import contextmanager, nullcontext
from agent.config import TRACING_ENABLED, TRACED_CLIENT


def redact_text(text) -> str:
    if not isinstance(text, str):
        return text
    redacted = re.sub(r"\b[A-Z0-9]{5,8}\b", "[REDACTED]", text)
    redacted = re.sub(r"\S+@\S+", "[REDACTED_EMAIL]", redacted)
    return redacted


def _redact_recursive(obj):
    """Walks any nested dict/list structure and applies redact_text() to every
    string value found, regardless of key name or nesting depth. Kept as the
    process_inputs/process_outputs implementation for individually-decorated
    functions (defense-in-depth) -- the primary mechanism covering LangGraph's
    own automatic tracing is TRACED_CLIENT's anonymizer, applied via
    graph_tracing_context() around the graph.invoke() call itself.
    """
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
    from langsmith.run_helpers import tracing_context

    def traced_node(name: str):
        def decorator(fn):
            return traceable(
                name=name,
                run_type="chain",
                process_inputs=_redact_dict,
                process_outputs=_redact_dict,
            )(fn)
        return decorator

    @contextmanager
    def graph_tracing_context():
        """Wrap graph.invoke() in this so LangGraph's own automatic tracing
        (the top-level graph run + every routing function) routes through
        TRACED_CLIENT's anonymizer, not just individually-decorated
        functions. See agent/config.py for why this two-layer approach
        exists and what's unverified about it."""
        with tracing_context(client=TRACED_CLIENT):
            yield
else:
    def traced_node(name: str):
        """No-op decorator when tracing is disabled -- the node runs exactly
        as written, just without a LangSmith run being created."""
        def decorator(fn):
            return fn
        return decorator

    def graph_tracing_context():
        return nullcontext()
