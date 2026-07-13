"""
LangSmith tracing for each graph node, with PII redaction applied BEFORE
anything is sent to LangSmith -- per the Problem Framing Document's explicit
requirement ("Observability: LangSmith (with PII redaction applied before
tracing)").

Design note: this project's LLM calls use the raw OpenAI SDK directly
(client.chat.completions.create), not langchain_openai's ChatOpenAI wrapper,
so automatic LangSmith auto-instrumentation (which hooks LangChain's chat
model classes) does not apply here. Instead, each graph node function is
wrapped with LangSmith's `traceable` decorator, which works with any Python
function regardless of what it calls internally.

Redaction strategy: the same regex-based redaction used in every prior
phase's JSONL logging (PNR-like tokens, email addresses) is applied to
free-text fields in both the traced inputs and outputs, via `traceable`'s
process_inputs/process_outputs hooks. If tracing is disabled (no
LANGSMITH_API_KEY set), @traced_node becomes a plain no-op passthrough --
the agent works identically either way, just without traces.

CAVEAT: `process_inputs`/`process_outputs` are LangSmith SDK features that
could not be verified against a live installation in this environment (no
PyPI access in the build sandbox). Confirm on first real run that traces
appear correctly in your LangSmith project; if process_inputs/process_outputs
aren't supported in your installed langsmith version, `pip install -U
langsmith` and retry, or fall back to manually redacting fields inside each
node function before returning them.
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
