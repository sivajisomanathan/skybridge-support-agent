"""
Per-function LangSmith tracing for individually-decorated node functions, with
recursive redaction applied to their own inputs/outputs.

IMPORTANT: this is defense-in-depth, not the primary redaction mechanism. The
primary, guaranteed mechanism is LANGSMITH_HIDE_INPUTS/LANGSMITH_HIDE_OUTPUTS,
set globally in agent/config.py -- required because LangGraph has its own
built-in, automatic LangSmith tracing (separate from this module's @traceable
wrapper) that traces the full graph invocation and every routing function on
its own, bypassing whatever redaction is applied only to individually-
decorated functions here. This was discovered via real traces in a live
LangSmith project showing unredacted PNRs in the top-level "LangGraph" run
and routing-function runs (route_after_classify, etc.), even though functions
wrapped with @traced_node redacted correctly for their own specific
input/output. See agent/config.py for the full explanation of the fix.
"""
import re
from agent.config import TRACING_ENABLED


def redact_text(text) -> str:
    if not isinstance(text, str):
        return text
    redacted = re.sub(r"\b[A-Z0-9]{5,8}\b", "[REDACTED]", text)
    redacted = re.sub(r"\S+@\S+", "[REDACTED_EMAIL]", redacted)
    return redacted


def _redact_recursive(obj):
    """Walks any nested dict/list structure and applies redact_text() to every
    string value found, regardless of key name or nesting depth.

    This replaces an earlier, narrower version that only redacted a fixed
    allowlist of top-level keys (user_input, resolved_input,
    conversation_history). That approach missed PII appearing under other
    keys -- e.g. the `pnr` field, or a PNR nested inside `booking_record`
    -- which real testing against a live LangSmith project surfaced (see
    README). Blanket recursive string redaction is more robust: it doesn't
    depend on correctly guessing every key a PNR might end up under, current
    or future, including whatever nesting shape LangSmith's process_inputs/
    process_outputs hooks actually pass (which could not be verified before
    a live LangSmith account was available to test against).
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
