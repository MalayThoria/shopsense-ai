"""
ShopSense AI Pipeline Orchestrator.

Connects Phase 1 (Triage), Phase 2 (RAG), and Phase 3 (Agent) into a
single end-to-end pipeline. The RAG chain is initialised once at module
level for performance.
"""

import json
from pathlib import Path

from pipeline.phase1_triage import triage_ticket
from pipeline.phase2_rag import load_rag_chain, retrieve_and_answer
from pipeline.phase3_agent import run_agent

from utils.logger import get_logger

logger = get_logger(__name__)

# Initialise RAG chain once at module level
logger.info("Initialising RAG chain for orchestrator...")
rag_chain = load_rag_chain()
logger.info("Orchestrator ready")


def process_ticket(ticket: dict) -> dict:
    """
    Full end-to-end ShopSense AI pipeline.

    Runs a single customer ticket through all three phases:
      Phase 1 — Triage & classification
      Phase 2 — RAG knowledge retrieval
      Phase 3 — Agentic decision & draft reply

    Args:
        ticket: Dict with keys ticket_id, customer_id, text, channel.

    Returns:
        Dict with keys triage, rag, outcome containing the full
        output from each phase.
    """
    logger.info(f"Processing ticket {ticket['ticket_id']} through full pipeline")

    # Phase 1 — Triage
    triage = triage_ticket(ticket)
    logger.info(
        f"  Phase 1 done: intent={triage['intent']}, "
        f"urgency={triage['urgency']}, sentiment={triage['sentiment']}"
    )

    # Phase 2 — RAG
    rag = retrieve_and_answer(triage, rag_chain)
    logger.info(f"  Phase 2 done: {len(rag.get('retrieved_chunks', []))} chunks retrieved")

    # Phase 3 — Agent
    outcome = run_agent(triage, rag)
    logger.info(f"  Phase 3 done: decision={outcome['decision']}")

    return {
        "triage": triage,
        "rag": rag,
        "outcome": outcome,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    tickets_path = project_root / "data" / "tickets.json"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))

    # Process first 5 tickets
    results = []
    for i, ticket in enumerate(tickets[:5], 1):
        print(f"\n{'='*70}")
        print(f"TICKET {i}/5 — {ticket['ticket_id']}")
        print(f"{'='*70}")
        print(f"Text: {ticket['text'][:100]}...")

        result = process_ticket(ticket)
        results.append(result)

        # Pretty-print summary
        t = result["triage"]
        o = result["outcome"]
        print(f"\n  Intent:     {t['intent']}")
        print(f"  Urgency:    {t['urgency']}")
        print(f"  Sentiment:  {t['sentiment']}")
        print(f"  Order ID:   {t['entities'].get('order_id', 'None')}")
        print(f"  Decision:   {o['decision'].upper()}")
        if o.get("escalation_reason"):
            print(f"  Esc Reason: {o['escalation_reason']}")
        print(f"  KB Source:  {o['kb_context_used']}")
        print(f"  Reply:      {o['draft_reply'][:150]}...")

    # Save results
    output_file = output_dir / "pipeline_results.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{'='*70}")
    print(f"Saved {len(results)} results to {output_file}")

    # Print summary
    resolved = sum(1 for r in results if r["outcome"]["decision"] == "resolve")
    escalated = sum(1 for r in results if r["outcome"]["decision"] == "escalate")
    print(f"\nSUMMARY: {resolved} resolved, {escalated} escalated")

    if escalated > 0:
        print("\nEscalation reasons:")
        for r in results:
            if r["outcome"]["decision"] == "escalate":
                print(f"  {r['outcome']['ticket_id']}: {r['outcome']['escalation_reason']}")