# ShopSense AI — Architecture

## System diagram

```
# ShopSense AI — Architecture

## System diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SHOPSENSE AI — HOW IT WORKS                          ║
║              From customer ticket to resolved/escalated reply                ║
╚══════════════════════════════════════════════════════════════════════════════╝


  STEP 0:  A customer sends a message
  ─────────────────────────────────────

      "Hi, my order ORD-4892 hasn't arrived in 17 days!"
                            │
                            ▼
                ┌───────────────────────┐
                │   Streamlit UI   OR   │
                │   Batch script        │
                └───────────┬───────────┘
                            │ ticket = {id, text, channel}
                            ▼
                ┌───────────────────────┐
                │     ORCHESTRATOR      │     ← The "boss" that runs all 3 phases
                │   process_ticket()    │       in order, one after another
                └───────────┬───────────┘
                            │
                            ▼

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │   PHASE 1 — UNDERSTAND THE TICKET                                      │
  │   ────────────────────────────────                                     │
  │                                                                        │
  │   "What is the customer asking? How urgent? How angry?"                │
  │                                                                        │
  │     ┌────────────────────────────────────────────────────┐             │
  │     │  llama3.1 (running locally via Ollama)             │             │
  │     │  format="json", temperature=0 (deterministic)      │             │
  │     │  Reads the message and returns structured JSON:    │             │
  │     │                                                    │             │
  │     │     intent:     shipping_delay                     │             │
  │     │     urgency:    high                               │             │
  │     │     sentiment:  frustrated                         │             │
  │     │     order_id:   ORD-4892                           │             │
  │     │     days:       17                                 │             │
  │     │     confidence: 0.95                               │             │
  │     └────────────────────────────────────────────────────┘             │
  │                                                                        │
  │   10 intent classes: order_status, return_request, refund_inquiry,      │
  │   product_question, complaint, shipping_delay, cancellation,           │
  │   warranty_claim, billing_dispute, feedback                            │
  │                                                                        │
  │   Safety net: If JSON parsing fails → retry with stricter prompt       │
  │               If still fails → safe fallback (never crashes)           │
  │                                                                        │
  └────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │   PHASE 2 — FIND THE RIGHT POLICY                                      │
  │   ────────────────────────────────                                     │
  │                                                                        │
  │   "What does Stryde's policy say about this?"                          │
  │                                                                        │
  │                                                                        │
  │   ┌──────────────────────┐         ┌────────────────────────────┐      │
  │   │  KNOWLEDGE BASE      │         │   1. Build search query    │      │
  │   │  (ChromaDB)          │         │      from intent + text    │      │
  │   │                      │         │                            │      │
  │   │  8 policy documents: │         │   2. Search ChromaDB       │      │
  │   │  • Returns           │ ◄───────┤      → top 3 best chunks   │      │
  │   │  • Refunds           │         │      (MMR for diversity)   │      │
  │   │  • Shipping          │         │      (MiniLM-L6-v2 embeds) │      │
  │   │  • Warranty          │         │                            │      │
  │   │  • Payments          │         │   3. Send chunks + query   │      │
  │   │  • Loyalty           │         │      to llama3.1           │      │
  │   │  • Sizes             │         │      via RetrievalQA chain │      │
  │   │  • Escalations       │         │                            │      │
  │   └──────────────────────┘         │   4. Get grounded answer   │      │
  │                                    │      based ONLY on policy  │      │
  │                                    └────────────┬───────────────┘      │
  │                                                 │                      │
  │   Output:  grounded_answer + which policy was used                     │
  │                                                                        │
  └────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │   PHASE 3 — DECIDE & REPLY (Real LangChain Agent)                      │
  │   ──────────────────────────────────────────────                       │
  │                                                                        │
  │   "Look up the order, decide resolve vs escalate, write reply"         │
  │                                                                        │
  │                                                                        │
  │   STEP 3A — DETERMINISTIC ORDER LOOKUP & ESCALATION CHECK              │
  │   ────────────────────────────────────────────────────────             │
  │                                                                        │
  │     order_id: ORD-4892                                                 │
  │           │                                                            │
  │           ▼                                                            │
  │     ┌─────────────────────────┐                                        │
  │     │  Flask API (port 5050)  │                                        │
  │     │  Reads orders.json      │                                        │
  │     │                         │                                        │
  │     │  Returns JSON:          │                                        │
  │     │   status: in_transit    │                                        │
  │     │   days: 17              │                                        │
  │     └────────────┬────────────┘                                        │
  │                  │                                                     │
  │                  ▼                                                     │
  │     ┌─────────────────────────────────────────────────┐                │
  │     │  Ownership check: does order belong to this     │                │
  │     │  customer? If not → escalate immediately        │                │
  │     └────────────────────────┬────────────────────────┘                │
  │                              │                                         │
  │                              ▼                                         │
  │     ┌─────────────────────────────────────────────────┐                │
  │     │  should_escalate()  — RULES, not LLM            │                │
  │     │                                                 │                │
  │     │   1. API failure or order not found              │                │
  │     │   2. Status is "lost_in_transit"                │                │
  │     │   3. Days since order > 14                      │ ← MATCHES!     │
  │     │   4. High urgency AND angry sentiment           │                │
  │     │   5. High-stakes intent + high urgency          │                │
  │     │      + no order ID                              │                │
  │     │                                                 │                │
  │     │   → Decision: ESCALATE / RESOLVE                │                │
  │     └────────────────────────┬────────────────────────┘                │
  │                              │                                         │
  │                              ▼                                         │
  │                                                                        │
  │   STEP 3B — REACT AGENT DRAFTS THE REPLY                               │
  │   ───────────────────────────────────────                              │
  │                                                                        │
  │     ┌──────────────────────────────────────────────────┐               │
  │     │  LangChain AgentExecutor                         │               │
  │     │  + create_react_agent()                          │               │
  │     │  + ChatOllama (llama3.1, temp=0.2)               │               │
  │     │                                                  │               │
  │     │  Tools available:                                │               │
  │     │    • @tool order_lookup(order_id)                │               │
  │     │                                                  │               │
  │     │  ReAct reasoning loop:                           │               │
  │     │    Thought → Action → Observation → ...          │               │
  │     │    (max 5 iterations)                            │               │
  │     │                                                  │               │
  │     │  Context injected into prompt:                   │               │
  │     │    • KB grounded answer (from Phase 2)           │               │
  │     │    • Order data                                  │               │
  │     │    • Customer ID                                 │               │
  │     │    • Decision (resolve/escalate)                 │               │
  │     │    • Escalation reason (if any)                  │               │
  │     │                                                  │               │
  │     │  Fallback: direct LLM call if agent fails        │               │
  │     └────────────────────┬─────────────────────────────┘               │
  │                          │                                             │
  │                          ▼                                             │
  │     "Dear C101, sorry for the delay on ORD-4892.                       │
  │      Your case has been escalated to our senior team                   │
  │      and someone will contact you within 2 hours..."                   │
  │                                                                        │
  └────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼

  FINAL OUTPUT — Resolution Packet
  ─────────────────────────────────

       ┌──────────────────────────────────────┐
       │  decision:           ESCALATE        │
       │  escalation_reason:  17-day delay    │
       │  order_data:         {…}             │
       │  draft_reply:        "Dear C101..."  │
       │  kb_source:          shipping.md     │
       │  resolved:           false           │
       └──────────────────────────────────────┘
                       │
                       ▼
       Sent to:
        • Streamlit UI (visualized for support agent)
        • outputs/pipeline_results.json (for audit)
        • logs/pipeline.log (for debugging)


╔══════════════════════════════════════════════════════════════════════════════╗
║  THE BIG IDEA                                                                ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║                                                                              ║
║  Phase 1 = "What did they say?"     →  CLASSIFY the ticket                  ║
║  Phase 2 = "What's our policy?"     →  RETRIEVE grounded knowledge          ║
║  Phase 3 = "What should we do?"     →  DECIDE (rules) + DRAFT (agent)       ║
║                                                                              ║
║  The LLM does the "soft" work: understanding, retrieving, writing.           ║
║  The RULES do the "hard" work: deciding when a human must intervene.         ║
║  A LangChain ReAct agent handles the reply drafting with tool access.        ║
║  This makes the system intelligent, agentic, AND auditable.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```
```

## Data flow per ticket

1. A ticket dict (`ticket_id`, `customer_id`, `text`, `channel`) enters the orchestrator from the UI or a batch run.
2. **Phase 1** calls Ollama with a strict JSON schema. On parse failure, the orchestrator retries with a stricter prompt; on a second failure, it falls back to a neutral default packet and marks `confidence = 0.0`.
3. **Phase 2** constructs a query from `intent + raw_text`, retrieves the top 3 chunks from Chroma, and runs a `RetrievalQA` "stuff" chain to produce a grounded answer with source attribution.
4. **Phase 3** (a) calls the mock Order API via a LangChain `@tool` if an `order_id` was extracted, (b) runs **deterministic** escalation rules over the triage + order data — never asks the LLM whether to escalate, (c) drafts a customer reply with the LLM, conditioning on the KB context, the order data, and the resolve/escalate decision.
5. The orchestrator returns a single packet combining all three phase outputs.

## Component contracts

| Phase | Input | Output |
|---|---|---|
| 1 — Triage | `{ticket_id, customer_id, text, channel}` | `{intent, urgency, sentiment, entities, confidence, raw_text, ticket_id, customer_id}` |
| 2 — RAG | triage dict | `{ticket_id, query_used, retrieved_chunks[], grounded_answer}` |
| 3 — Agent | triage dict + rag dict | `{ticket_id, decision, escalation_reason, order_data, draft_reply, kb_context_used, resolved}` |

## Escalation rules (deterministic)

The agent **never** delegates the resolve/escalate decision to the LLM. The five rules in [`pipeline/phase3_agent.py`](../pipeline/phase3_agent.py) `should_escalate()`:

1. Order ID not found in the system.
2. Order delayed beyond the 14-day shipping SLA.
3. Order flagged `lost_in_transit`.
4. `urgency == "high"` **and** `sentiment == "angry"`.
5. `intent == "complaint"` with no order reference.

This guarantees auditable, testable behavior on the highest-stakes part of the workflow.

## Why three phases, not one big agent

We considered a single autonomous agent that decided everything (classify, retrieve, look up, escalate, reply). We rejected it because the escalation decision needs deterministic guarantees a customer-support team can audit, and because chaining specialized stages is dramatically easier to evaluate, debug, and improve in production. See [`tradeoffs.md`](tradeoffs.md) for the full rationale.
