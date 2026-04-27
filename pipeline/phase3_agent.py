"""
Phase 3 — Agentic Decision Layer.

Uses LangChain agent with ChatOllama and a custom order lookup tool.
Escalation decisions are deterministic (never LLM-decided).
Generates grounded draft replies using KB context from Phase 2.
"""

import json
from pathlib import Path

import requests
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.logger import get_logger

logger = get_logger(__name__)

# Try importing agent framework — fall back gracefully if unavailable
try:
    from langchain.agents import AgentExecutor, create_react_agent
    AGENT_AVAILABLE = True
except ImportError:
    try:
        from langchain_classic.agents import AgentExecutor, create_react_agent
        AGENT_AVAILABLE = True
    except ImportError:
        AGENT_AVAILABLE = False
        logger.warning("AgentExecutor not available — using direct LLM fallback")


@tool
def order_lookup(order_id: str) -> str:
    """Look up the current status and details of a customer order by order ID."""
    try:
        resp = requests.get(
            f"http://localhost:5050/api/orders/{order_id}", timeout=5
        )
        if resp.status_code == 404:
            return "Order not found in system."
        return str(resp.json())
    except Exception as e:
        return f"Lookup failed: {str(e)}"


def should_escalate(triage: dict, order_data: str) -> tuple[bool, str]:
    """
    Deterministic escalation check. The LLM never decides escalation.

    Args:
        triage: The Phase 1 triage result dict.
        order_data: Raw string response from order_lookup tool.

    Returns:
        Tuple of (should_escalate: bool, reason: str).
        Returns (False, "") if no escalation conditions are met.
    """
    # Condition 1: Order not found
    if "not found" in order_data.lower():
        return True, "Order ID not found in system"

    # Condition 2: Order delayed past 14-day SLA
    try:
        if "days_since_order" in order_data:
            # Parse days_since_order from the string representation
            for part in order_data.split(","):
                if "days_since_order" in part:
                    days = int("".join(c for c in part.split(":")[-1] if c.isdigit()))
                    if days > 14:
                        return True, f"Order delayed {days} days, past 14-day SLA"
                    break
    except (ValueError, IndexError):
        pass

    # Condition 3: Lost in transit
    if "lost_in_transit" in order_data.lower():
        return True, "Order marked as lost in transit"

    # Condition 4: High urgency + angry sentiment
    if triage.get("urgency") == "high" and triage.get("sentiment") == "angry":
        return True, "High urgency + angry sentiment, human required"

    # Condition 5: Complaint with no order reference
    entities = triage.get("entities", {})
    order_id = entities.get("order_id") if entities else None
    if triage.get("intent") == "complaint" and not order_id:
        return True, "Complaint with no order reference"

    return False, ""


def _generate_reply(llm, triage: dict, rag_result: dict, order_data: str,
                    escalate: bool, escalation_reason: str) -> str:
    """
    Use the LLM to generate a draft customer reply grounded in KB context.

    Args:
        llm: The ChatOllama instance.
        triage: Phase 1 triage result.
        rag_result: Phase 2 RAG result.
        order_data: Order lookup result string.
        escalate: Whether the ticket is being escalated.
        escalation_reason: Reason for escalation, if any.

    Returns:
        A draft reply string addressed to the customer.
    """
    kb_answer = rag_result.get("grounded_answer", "No KB context available.")

    if escalate:
        prompt_text = f"""You are a customer support agent for Stryde, a premium sneaker brand.
A customer has written in with an issue. This ticket is being ESCALATED to a senior agent.
Write a brief, empathetic reply acknowledging the issue and informing the customer that
their case has been escalated to a senior support specialist who will contact them within 2 hours.

Customer message: {triage['raw_text']}

Knowledge base context: {kb_answer}

Order data: {order_data}

Escalation reason: {escalation_reason}

Write a warm, professional reply in 3-4 sentences. Address the customer directly."""
    else:
        prompt_text = f"""You are a customer support agent for Stryde, a premium sneaker brand.
A customer has written in with an issue. Based on the knowledge base and order data,
write a helpful, complete reply that resolves their concern.

Customer message: {triage['raw_text']}

Knowledge base context: {kb_answer}

Order data: {order_data}

Write a warm, professional reply in 3-5 sentences. Address the customer directly.
Ground your response in the knowledge base information provided."""

    try:
        response = llm.invoke(prompt_text)
        return response.content
    except Exception as e:
        logger.error(f"Reply generation failed: {e}")
        if escalate:
            return ("Thank you for reaching out. We understand your concern and have "
                    "escalated your case to our senior support team. A specialist will "
                    "contact you within 2 hours during business hours.")
        return ("Thank you for contacting Stryde support. We are looking into your "
                "request and will get back to you shortly.")


def run_agent(triage_result: dict, rag_result: dict) -> dict:
    """
    Run the Phase 3 agent: look up order, check escalation, draft reply.

    Args:
        triage_result: Output dict from Phase 1.
        rag_result: Output dict from Phase 2.

    Returns:
        Dict matching the Phase 3 output schema with decision,
        escalation_reason, order_data, draft_reply, etc.
    """
    ticket_id = triage_result["ticket_id"]
    logger.info(f"Running agent for ticket {ticket_id}")

    llm = ChatOllama(model="llama3.1", temperature=0.2)

    # Step 1: Look up order if order_id exists
    entities = triage_result.get("entities", {})
    order_id = entities.get("order_id") if entities else None
    order_data_str = "No order ID provided in ticket."

    if order_id:
        logger.info(f"Looking up order {order_id}")
        order_data_str = order_lookup.invoke(order_id)

    # Step 2: Deterministic escalation check
    escalate, reason = should_escalate(triage_result, order_data_str)

    if escalate:
        logger.info(f"Ticket {ticket_id} ESCALATED: {reason}")
    else:
        logger.info(f"Ticket {ticket_id} RESOLVED")

    # Step 3: Generate draft reply
    draft_reply = _generate_reply(
        llm, triage_result, rag_result, order_data_str, escalate, reason
    )

    # Step 4: Determine KB source used
    chunks = rag_result.get("retrieved_chunks", [])
    kb_source = chunks[0]["source"] if chunks else "none"

    # Step 5: Parse order data for the output packet
    order_data_parsed = None
    if order_id and "not found" not in order_data_str.lower() and "failed" not in order_data_str.lower():
        try:
            order_data_parsed = eval(order_data_str)  # from str repr of dict
        except Exception:
            order_data_parsed = {"raw": order_data_str}

    # Build final resolution packet
    result = {
        "ticket_id": ticket_id,
        "decision": "escalate" if escalate else "resolve",
        "escalation_reason": reason if escalate else None,
        "order_data": order_data_parsed,
        "draft_reply": draft_reply,
        "kb_context_used": kb_source,
        "resolved": not escalate,
    }

    logger.debug(f"Agent result for {ticket_id}: {result['decision']}")
    return result


if __name__ == "__main__":
    from pipeline.phase1_triage import triage_ticket
    from pipeline.phase2_rag import load_rag_chain, retrieve_and_answer

    project_root = Path(__file__).resolve().parent.parent
    tickets = json.loads(
        (project_root / "data" / "tickets.json").read_text(encoding="utf-8")
    )

    chain = load_rag_chain()

    # Test with 3 tickets: one resolve, one escalate (delay), one escalate (not found)
    test_tickets = [tickets[1], tickets[0], tickets[7]]  # T002, T001, T008

    for ticket in test_tickets:
        print(f"\n{'='*60}")
        print(f"Ticket {ticket['ticket_id']}: {ticket['text'][:80]}...")

        triage = triage_ticket(ticket)
        rag = retrieve_and_answer(triage, chain)
        outcome = run_agent(triage, rag)

        print(f"Decision:   {outcome['decision'].upper()}")
        if outcome["escalation_reason"]:
            print(f"Reason:     {outcome['escalation_reason']}")
        print(f"KB Source:  {outcome['kb_context_used']}")
        print(f"Reply:      {outcome['draft_reply'][:200]}...")