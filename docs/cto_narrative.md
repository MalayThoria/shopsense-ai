# ShopSense AI — CTO Narrative

*Day 30 of the Stryde engagement. Written for the Stryde CTO.*

Thirty days into the Stryde engagement, here is where we stand. Your support team was drowning in tickets that fell into three buckets: things the FAQ already answered, things that needed a human, and things nobody could tell apart from each other until an angry customer escalated on Twitter. ShopSense AI is the first cut at fixing that — a single end-to-end pipeline that classifies every incoming ticket, retrieves the right policy, looks up the order, and either drafts a grounded reply or hands the case to a human with a reason attached.

The system runs in three connected phases. **Triage** uses a local `llama3.1` model in JSON mode to extract intent, urgency, sentiment, and entities like order IDs. **RAG retrieval** pulls the most relevant chunks from your eight policy documents in ChromaDB and produces a grounded answer with source attribution — so every reply is traceable back to a policy line, not a hallucination. **The agentic layer** calls a mock Order Lookup API, then applies five deterministic escalation rules (SLA breaches, lost-in-transit, angry-and-urgent, complaints without an order ID, missing orders) before drafting the customer-facing reply. Critically, the LLM never decides whether to escalate. That decision is rule-based, auditable, and testable in isolation.

A deliberate choice we made early was to keep the pipeline local-first. Ollama, Chroma, and sentence-transformers all run on a laptop, which let us iterate without burning API budget and gives Stryde a clean migration path: swap the LLM client for a hosted model when you want the latency win, keep everything else.

What's working today: the orchestrator processes tickets end-to-end, the Streamlit UI lets your support leads demo the flow, and the sample outputs in `outputs/` show the system handling shipping delays, double charges, returns, product questions, and cancellations correctly.

What we'd build in the next 30 days: an offline eval harness over a labeled ticket set so we can track triage accuracy and grounding faithfulness as a regression metric; a feedback loop that captures whether the human agent edited the draft reply (the cheapest source of fine-tuning data we have); multi-tenant support so your wholesale and DTC channels can have separate KBs; and production observability — request traces, token costs, and per-rule escalation counts in a dashboard your ops team actually opens.

The foundation is in. Now we make it earn its keep.

---

