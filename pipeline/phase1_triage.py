"""
Phase 1 — Ticket Triage & Classification.

Uses the ollama Python library directly (no LangChain) to classify
customer support tickets by intent, urgency, sentiment, and entities.
"""

import json
from pathlib import Path

import ollama

from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a customer support ticket classifier for Stryde, a D2C sneaker and apparel brand.

Analyze the customer message and return ONLY a valid JSON object with no extra text, no markdown fences, no explanation — just the raw JSON object.

The JSON must have exactly these fields:
{
  "intent": one of ["order_status", "return_request", "refund_inquiry", "product_question", "complaint", "shipping_delay", "cancellation", "warranty_claim", "billing_dispute", "feedback"],
  "urgency": one of ["low", "medium", "high"],
  "sentiment": one of ["positive", "neutral", "frustrated", "angry"],
  "entities": {
    "order_id": extracted order ID like "ORD-XXXX" or null if not present,
    "product_name": product name mentioned or null if not present,
    "days_mentioned": number of days mentioned or null if not present
  },
  "confidence": a float between 0 and 1 indicating your confidence
}

Rules:
- If the customer mentions waiting many days, is using caps, exclamation marks, or threatening language, set urgency to "high".
- If the customer is calm and just asking a question, set urgency to "low".
- Extract order IDs only if they match the pattern ORD-XXXX.
- Be precise with intent classification. A customer asking "where is my order" is "order_status", not "shipping_delay". A customer complaining about repeated late deliveries is "complaint" or "shipping_delay" depending on context.
- "warranty_claim" is for product defects like sole separation, stitching issues, or anything covered by manufacturing warranty.
- "billing_dispute" is for double charges, wrong amounts charged, or any payment-related disputes.
- "feedback" is for positive comments, praise, compliments, or general non-issue messages where the customer is not asking for help.
- "product_question" is ONLY for genuine product inquiries (sizing, features, comparisons) — not praise.
- Return ONLY the JSON object. Nothing else."""

STRICT_PROMPT = """You are a JSON-only classifier. Return ONLY a valid JSON object. 
Absolutely no text before or after the JSON. No markdown. No explanation.
Just a single JSON object with keys: intent, urgency, sentiment, entities, confidence.
Valid intents: order_status, return_request, refund_inquiry, product_question, complaint, shipping_delay, cancellation, warranty_claim, billing_dispute, feedback.
Valid urgency: low, medium, high.
Valid sentiment: positive, neutral, frustrated, angry.
entities must have: order_id (string or null), product_name (string or null), days_mentioned (integer or null).
confidence must be a float between 0 and 1."""


def triage_ticket(ticket: dict) -> dict:
    """
    Classify a single customer support ticket using llama3.1 via Ollama.

    Args:
        ticket: Dict with keys ticket_id, customer_id, text, channel.

    Returns:
        Dict matching the Phase 1 output schema with intent, urgency,
        sentiment, entities, confidence, plus ticket_id, customer_id,
        and raw_text injected.
    """
    logger.info(f"Triaging ticket {ticket['ticket_id']}")

    result = _call_ollama(ticket["text"], SYSTEM_PROMPT)

    # Retry with stricter prompt if first attempt failed
    if result is None:
        logger.warning(f"Retry with strict prompt for {ticket['ticket_id']}")
        result = _call_ollama(ticket["text"], STRICT_PROMPT)

    # Fallback if both attempts fail
    if result is None:
        logger.error(f"Both attempts failed for {ticket['ticket_id']}, using fallback")
        result = {
            "intent": "unknown",
            "urgency": "medium",
            "sentiment": "neutral",
            "entities": {
                "order_id": None,
                "product_name": None,
                "days_mentioned": None,
            },
            "confidence": 0.0,
        }

    # Inject ticket metadata
    result["ticket_id"] = ticket["ticket_id"]
    result["customer_id"] = ticket["customer_id"]
    result["raw_text"] = ticket["text"]

    # Ensure entities dict has all required keys
    if "entities" not in result or not isinstance(result["entities"], dict):
        result["entities"] = {
            "order_id": None,
            "product_name": None,
            "days_mentioned": None,
        }
    for key in ("order_id", "product_name", "days_mentioned"):
        result["entities"].setdefault(key, None)

    logger.debug(f"Triage result for {ticket['ticket_id']}: {result['intent']} / {result['urgency']}")
    return result


def _call_ollama(text: str, system_prompt: str) -> dict | None:
    """
    Make a single ollama.chat() call and parse the JSON response.

    Args:
        text: The raw customer message.
        system_prompt: The system prompt to use.

    Returns:
        Parsed dict on success, None on failure.
    """
    try:
        response = ollama.chat(
            model="llama3.1",
            format="json",
            options={"temperature": 0},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return None


def triage_all_tickets(tickets_path: str) -> list[dict]:
    """
    Batch-triage all tickets from a JSON file.

    Args:
        tickets_path: Path to the tickets.json file.

    Returns:
        List of triage result dicts for every ticket.
    """
    tickets_file = Path(tickets_path)
    tickets = json.loads(tickets_file.read_text(encoding="utf-8"))

    results = []
    total = len(tickets)

    for i, ticket in enumerate(tickets, 1):
        print(f"Triaging ticket {i} of {total}...")
        result = triage_ticket(ticket)
        results.append(result)

    # Save results
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "triage_results.json"
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {len(results)} triage results to {output_file}")
    logger.info(f"Batch triage complete: {len(results)} tickets processed")
    return results


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "tickets.json"
    results = triage_all_tickets(str(data_path))

    # Print summary
    from collections import Counter
    intents = Counter(r["intent"] for r in results)
    urgencies = Counter(r["urgency"] for r in results)
    print("\n--- Intent Distribution ---")
    for intent, count in intents.most_common():
        print(f"  {intent}: {count}")
    print("\n--- Urgency Distribution ---")
    for urgency, count in urgencies.most_common():
        print(f"  {urgency}: {count}")