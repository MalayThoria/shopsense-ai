"""
Phase 3 — Agentic Decision Layer.

Uses LangChain AgentExecutor with create_react_agent and ChatOllama.
Custom order_lookup tool is called through the agent's reasoning loop.
Escalation decisions are deterministic (never LLM-decided).
Generates grounded draft replies using KB context from Phase 2.
"""

import json
from pathlib import Path

import requests
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

from utils.logger import get_logger

logger = get_logger(__name__)

# Import agent framework — handle different langchain versions
try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent


# ── Tool ─────────────────────────────────────────────────────────────

@tool
def order_lookup(order_id: str) -> str:
    """Look up the current status and details of a customer order by order ID.
    Use this when a customer mentions an order number like ORD-XXXX."""
    try:
        resp = requests.get(
            f"http://localhost:5050/api/orders/{order_id}", timeout=5
        )
        if resp.status_code == 404:
            return "Order not found in system."
        return json.dumps(resp.json())
    except Exception as e:
        return f"Lookup failed: {str(e)}"


# ── Escalation rules (deterministic, never LLM-decided) ─────────────

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
    # Rule 1: API failure (S10) or order not found
    if "failed" in order_data.lower():
        return True, "Order lookup API unavailable, human review required"
    if "not found" in order_data.lower():
        return True, "Order ID not found in system"

    # Parse order data JSON
    order_dict = {}
    try:
        order_dict = json.loads(order_data)
    except (json.JSONDecodeError, TypeError):
        pass

    # Rule 2: Lost in transit (checked BEFORE SLA for correct reason)
    status = order_dict.get("status", "")
    if status == "lost_in_transit":
        return True, "Order marked as lost in transit"

    # Rule 3: Order delayed past 14-day SLA
    days = order_dict.get("days_since_order", 0)
    if isinstance(days, int) and days > 14:
        return True, f"Order delayed {days} days, past 14-day SLA"

    # Rule 4: High urgency + angry sentiment
    if triage.get("urgency") == "high" and triage.get("sentiment") == "angry":
        return True, "High urgency + angry sentiment, human required"

    # Rule 5: High-stakes intent with no order reference
    entities = triage.get("entities", {})
    order_id = entities.get("order_id") if entities else None
    high_stakes_intents = {
        "complaint",
        "refund_inquiry",
        "warranty_claim",
        "billing_dispute",
    }
    if (
        triage.get("intent") in high_stakes_intents
        and triage.get("urgency") == "high"
        and not order_id
    ):
        return True, f"High-urgency {triage.get('intent')} with no order reference"

    return False, ""


# ── ReAct prompt template for the agent ──────────────────────────────

REACT_PROMPT = PromptTemplate.from_template(
"""You are a customer support agent for Stryde, a premium Indian D2C sneaker and apparel brand.
You have access to the following tools:

{tools}

Use the following format:

Question: the customer's question or issue you must address
Thought: think about what you need to do to help this customer
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat if needed)
Thought: I now know enough to write a helpful reply
Final Answer: a warm, professional customer support reply in 3-5 sentences

Important context from our knowledge base:
{kb_context}

Order data (if available):
{order_data}

Customer ID: {customer_id}
This ticket has been marked as: {decision}
{escalation_info}

Address the customer politely. You may reference their customer ID ({customer_id}) for personalization.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)


# ── Agent runner ─────────────────────────────────────────────────────

def run_agent(triage_result: dict, rag_result: dict) -> dict:
    """
    Run the Phase 3 agent: look up order, check escalation, draft reply.

    The order is looked up directly (outside the agent) first for the
    deterministic escalation check. The AgentExecutor then generates
    the draft reply with access to the order_lookup tool.

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
    tools = [order_lookup]

    # Step 1: Look up order directly (outside agent) for escalation check
    entities = triage_result.get("entities", {})
    order_id = entities.get("order_id") if entities else None
    customer_id = triage_result.get("customer_id", "")
    order_data_str = "No order ID provided in ticket."
    ownership_mismatch = False

    if order_id:
        logger.info(f"Looking up order {order_id}")
        order_data_str = order_lookup.invoke(order_id)

        # Verify the order actually belongs to this customer
        try:
            order_dict_check = json.loads(order_data_str)
            order_customer = order_dict_check.get("customer_id", "")
            if order_customer and customer_id and order_customer != customer_id:
                ownership_mismatch = True
                logger.warning(
                    f"Ownership mismatch: ticket from {customer_id} but "
                    f"order {order_id} belongs to {order_customer}"
                )
        except (json.JSONDecodeError, TypeError):
            pass

    # Step 2: Deterministic escalation check
    if ownership_mismatch:
        escalate, reason = True, "Order ID does not belong to ticket customer (possible fraud or typo)"
    else:
        escalate, reason = should_escalate(triage_result, order_data_str)

    # Step 3: Build agent and generate draft reply
    kb_answer = rag_result.get("grounded_answer", "No KB context available.")

    # Build clearer decision context for the agent
    if escalate:
        decision_str = "ESCALATED — inform the customer their case is being escalated to a senior agent who will contact them within 2 hours"
    else:
        decision_str = "RESOLVED — provide a helpful, complete reply"

    # Tell the agent whether order lookup succeeded
    if order_id and order_data_str.startswith("{"):
        order_status_note = "ORDER LOOKUP SUCCEEDED — use the order data above. Do NOT say the order cannot be found."
    elif order_id:
        order_status_note = "ORDER LOOKUP FAILED — acknowledge the order ID but explain that details are unavailable."
    else:
        order_status_note = "NO ORDER ID PROVIDED — ask politely for the order number if needed."

    escalation_info = f"Escalation reason: {reason}" if escalate else ""
    customer_id = triage_result.get("customer_id", "valued customer")

    try:
        # Create the ReAct agent with the prompt
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=REACT_PROMPT,
        )

        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        # Run the agent
        agent_response = agent_executor.invoke({
            "input": triage_result["raw_text"],
            "kb_context": kb_answer,
            "order_data": order_data_str,
            "customer_id": customer_id,
            "decision": f"{decision_str}\n\n{order_status_note}",
            "escalation_info": escalation_info,
        })

        draft_reply = agent_response.get("output", "")

        # If agent hit iteration limit or returned empty, fall back
        if not draft_reply or "agent stopped" in draft_reply.lower() or "iteration limit" in draft_reply.lower():
            logger.warning(f"Agent hit iteration limit for {ticket_id}, using fallback")
            draft_reply = _fallback_reply(llm, triage_result, rag_result,
                                           order_data_str, escalate, reason)

    except Exception as e:
        logger.error(f"Agent failed for {ticket_id}: {e}, using fallback LLM call")
        draft_reply = _fallback_reply(llm, triage_result, rag_result,
                                       order_data_str, escalate, reason)

    # Step 4: Determine KB source used
    chunks = rag_result.get("retrieved_chunks", [])
    kb_source = chunks[0]["source"] if chunks else "none"

    # Step 5: Parse order data for the output packet
    order_data_parsed = None
    if order_id and "not found" not in order_data_str.lower() and "failed" not in order_data_str.lower():
        try:
            order_data_parsed = json.loads(order_data_str)
        except (json.JSONDecodeError, TypeError):
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


def _fallback_reply(llm, triage: dict, rag_result: dict, order_data: str,
                    escalate: bool, escalation_reason: str) -> str:
    """
    Fallback reply generation if AgentExecutor fails or hits iteration limit.

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
    customer_id = triage.get("customer_id", "valued customer")

    if escalate:
        prompt_text = f"""You are a customer support agent for Stryde, a premium sneaker brand.
A customer has written in with an issue. This ticket is being ESCALATED to a senior agent.
Write a brief, empathetic reply acknowledging the issue and informing the customer that
their case has been escalated to a senior support specialist who will contact them within 2 hours.

Customer ID: {customer_id}
Customer message: {triage['raw_text']}
Knowledge base context: {kb_answer}
Order data: {order_data}
Escalation reason: {escalation_reason}

Write a warm, professional reply in 3-4 sentences. Address the customer directly using their customer ID."""
    else:
        prompt_text = f"""You are a customer support agent for Stryde, a premium sneaker brand.
A customer has written in with an issue. Based on the knowledge base and order data,
write a helpful, complete reply that resolves their concern.

Customer ID: {customer_id}
Customer message: {triage['raw_text']}
Knowledge base context: {kb_answer}
Order data: {order_data}

Write a warm, professional reply in 3-5 sentences. Address the customer directly using their customer ID.
Ground your response in the knowledge base information provided."""

    try:
        response = llm.invoke(prompt_text)
        return response.content
    except Exception as e:
        logger.error(f"Fallback reply generation failed: {e}")
        if escalate:
            return (f"Dear {customer_id}, thank you for reaching out. We understand your "
                    "concern and have escalated your case to our senior support team. "
                    "A specialist will contact you within 2 hours during business hours.")
        return (f"Dear {customer_id}, thank you for contacting Stryde support. "
                "We are looking into your request and will get back to you shortly.")


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