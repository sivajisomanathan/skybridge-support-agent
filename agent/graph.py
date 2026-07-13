"""
LangGraph agent: state schema, nodes, routing, and graph construction.
Logic carried over from Phase 5-7, with two changes for deployment:
  1. Every node is wrapped with @traced_node (LangSmith tracing + PII redaction).
  2. Memory/feedback reads and writes go through agent.memory's SQLite-backed
     functions instead of Phase 6/7's in-memory dicts.
"""
import json
import re
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END

from agent.config import client, CHAT_MODEL
from agent.kb import retrieve_grounded
from agent.tools import booking_lookup, policy_calculator
from agent.memory import has_prior_negative_feedback
from agent.tracing import traced_node


class AgentState(TypedDict):
    thread_id: str
    user_input: str
    resolved_input: Optional[str]
    conversation_history: List[dict]
    pnr: Optional[str]
    intent_category: Optional[str]
    outcome_type: Optional[str]
    retrieved_context: Optional[str]
    retrieved_sections: List[str]
    booking_record: Optional[dict]
    calc_result: Optional[dict]
    tools_used: List[str]
    feedback_adapted: bool
    final_response: Optional[str]


RESOLVE_PROMPT = """Given a conversation history and a new customer message, rewrite the
new message as a fully standalone question IF it references something from earlier in the
conversation (e.g. "that", "it", "this one"). If the new message is already standalone,
return it completely unchanged. Respond with ONLY the resulting message text -- no
explanation, no quotes, no extra formatting.
"""


@traced_node("resolve_reference")
def resolve_reference_node(state: AgentState) -> dict:
    history = state.get("conversation_history") or []
    if not history:
        return {"resolved_input": state["user_input"]}
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": RESOLVE_PROMPT},
            {"role": "user", "content": f"Conversation history:\n{history_text}\n\nNew message: {state['user_input']}"},
        ],
        temperature=0,
    )
    return {"resolved_input": resp.choices[0].message.content.strip()}


PNR_PATTERN = re.compile(r"\b[A-Z]{2}\d{4}\b", re.IGNORECASE)
CLASSIFY_PROMPT = """Classify the customer message into exactly one intent category.
Respond ONLY with a JSON object: {"intent_category": "delay_compensation | seat_change | cancellation_refund | baggage | billing | other"}
"""


@traced_node("classify")
def classify_node(state: AgentState) -> dict:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": CLASSIFY_PROMPT},
                  {"role": "user", "content": state["resolved_input"]}],
        temperature=0,
    )
    try:
        parsed = json.loads(resp.choices[0].message.content)
        intent = parsed.get("intent_category", "other")
    except json.JSONDecodeError:
        intent = "other"
    pnr_match = PNR_PATTERN.search(state["resolved_input"]) or PNR_PATTERN.search(state["user_input"])
    pnr = pnr_match.group(0).upper() if pnr_match else None
    return {"intent_category": intent, "pnr": pnr}


@traced_node("retrieve")
def retrieve_node(state: AgentState) -> dict:
    retrieval = retrieve_grounded(state["resolved_input"])
    if retrieval["grounded"]:
        context = "\n\n".join(f"[{m['section_id']}] {m['title']}\n{m['text']}" for m in retrieval["matches"])
        sections = [m["section_id"] for m in retrieval["matches"]]
    else:
        context = "NO_RELEVANT_POLICY_FOUND"
        sections = []
    return {"retrieved_context": context, "retrieved_sections": sections}


@traced_node("booking_lookup")
def booking_lookup_node(state: AgentState) -> dict:
    if not state.get("pnr"):
        return {"booking_record": {"booking_found": False, "reason": "no_pnr_provided"}, "tools_used": state["tools_used"]}
    record = booking_lookup(state["pnr"])
    return {"booking_record": record, "tools_used": state["tools_used"] + ["booking_lookup"]}


@traced_node("policy_calculator")
def policy_calculator_node(state: AgentState) -> dict:
    booking = state.get("booking_record") or {}
    intent = state["intent_category"]
    if intent == "delay_compensation":
        result = policy_calculator("delay_compensation", delay_hours=booking.get("delay_hours"))
    elif intent == "cancellation_refund":
        result = policy_calculator("refund_fee", fare_type=booking.get("fare_type"))
    else:
        result = None
    tools_used = state["tools_used"] + (["policy_calculator"] if result else [])
    return {"calc_result": result, "tools_used": tools_used}


@traced_node("adaptive_escalate")
def adaptive_escalate_node(state: AgentState) -> dict:
    response = (
        "I want to make sure you get this resolved properly -- since my last answer on "
        "this topic didn't fully help, I'm escalating this directly to a human agent rather "
        "than trying the same automated response again."
    )
    return {
        "outcome_type": "escalation",
        "final_response": response,
        "tools_used": [],
        "retrieved_context": "SKIPPED_DUE_TO_ADAPTATION",
        "retrieved_sections": [],
        "feedback_adapted": True,
    }


COMPOSE_PROMPT = """You are a customer support assistant for SkyBridge Airlines (a fictional airline).

Hard safety rules (never break these):
1. You cannot execute any action (no cancellations, refunds, rebookings, or account changes).
   If asked to perform an action, politely refuse and explain you can only provide information,
   then offer to escalate to a human agent.
2. You must ALWAYS escalate to a human agent for: damaged/lost baggage claims, billing or
   payment disputes (e.g. double charges), or anything ambiguous / outside airline policy topics.
3. You will be given retrieved policy context, and possibly a booking record, a calculated
   result, and prior conversation turns. Use ONLY the retrieved/tool information for policy
   facts -- do not invent figures. Conversation history is for tone and continuity only.
4. If a booking lookup failed or no PNR was provided, do NOT guess the customer's fare type
   or delay length. Ask for the correct booking reference, or offer to escalate.
5. If the retrieved context is NO_RELEVANT_POLICY_FOUND, say clearly you don't have verified
   information on this topic and offer to escalate -- do NOT guess.
6. Keep responses concise (2-4 sentences) and professional.

Respond ONLY with a JSON object (no markdown fences, no extra text):
{
  "intent_category": "delay_compensation | seat_change | cancellation_refund | baggage | billing | other",
  "outcome_type": "answer | refusal | escalation | insufficient_information",
  "response": "the customer-facing message, citing policy sections and/or booking specifics when used"
}
"""


@traced_node("compose")
def compose_node(state: AgentState) -> dict:
    context_parts = [f"Retrieved policy context:\n{state['retrieved_context']}"]
    if state.get("booking_record"):
        context_parts.append(f"Booking record: {json.dumps(state['booking_record'])}")
    if state.get("calc_result"):
        context_parts.append(f"Calculated result: {json.dumps(state['calc_result'])}")
    history = state.get("conversation_history") or []
    if history:
        history_text = "\n".join(f"{t['role']}: {t['content']}" for t in history)
        context_parts.append(f"Prior conversation this session:\n{history_text}")
    context_parts.append(f"Customer message: {state['user_input']}")
    if state["resolved_input"] != state["user_input"]:
        context_parts.append(f"(Resolved standalone question: {state['resolved_input']})")
    user_message = "\n\n".join(context_parts)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": COMPOSE_PROMPT},
                  {"role": "user", "content": user_message}],
        temperature=0.2,
    )
    raw_text = resp.choices[0].message.content
    try:
        parsed = json.loads(raw_text)
        outcome_type = parsed.get("outcome_type")
        response_text = parsed.get("response", raw_text)
    except json.JSONDecodeError:
        outcome_type = None
        response_text = raw_text
    return {"outcome_type": outcome_type, "final_response": response_text}


def route_after_classify(state: AgentState) -> str:
    if has_prior_negative_feedback(state["thread_id"], state["intent_category"]):
        return "adaptive_escalate"
    return "retrieve"


def route_after_retrieve(state: AgentState) -> str:
    intent = state["intent_category"]
    if intent in ("seat_change", "cancellation_refund", "delay_compensation"):
        return "booking_lookup"
    return "compose"


def route_after_booking(state: AgentState) -> str:
    booking = state.get("booking_record") or {}
    intent = state["intent_category"]
    if not booking.get("booking_found"):
        return "compose"
    if intent in ("cancellation_refund", "delay_compensation"):
        return "policy_calculator"
    return "compose"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("resolve_reference", resolve_reference_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("booking_lookup", booking_lookup_node)
    workflow.add_node("policy_calculator", policy_calculator_node)
    workflow.add_node("adaptive_escalate", adaptive_escalate_node)
    workflow.add_node("compose", compose_node)

    workflow.set_entry_point("resolve_reference")
    workflow.add_edge("resolve_reference", "classify")
    workflow.add_conditional_edges("classify", route_after_classify, {
        "retrieve": "retrieve",
        "adaptive_escalate": "adaptive_escalate",
    })
    workflow.add_conditional_edges("retrieve", route_after_retrieve, {
        "booking_lookup": "booking_lookup",
        "compose": "compose",
    })
    workflow.add_conditional_edges("booking_lookup", route_after_booking, {
        "policy_calculator": "policy_calculator",
        "compose": "compose",
    })
    workflow.add_edge("policy_calculator", "compose")
    workflow.add_edge("adaptive_escalate", END)
    workflow.add_edge("compose", END)

    return workflow.compile()
