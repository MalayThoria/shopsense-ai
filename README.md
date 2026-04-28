# ShopSense AI

A customer support intelligence platform for **Stryde**, a fictional D2C sneaker and apparel brand. ShopSense ingests a customer ticket and runs it through a single end-to-end pipeline: classify it, retrieve grounded knowledge, look up the order, decide whether to resolve or escalate, and draft a reply.

Built for the Forward Deployed Engineer take-home assignment.

---

## Architecture at a glance

```
                          ┌────────────────────────────────────────┐
   Customer ticket ──►    │            Orchestrator                │
   (UI / JSON file)       │       pipeline/orchestrator.py         │
                          └────────────────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
 ┌─────────────┐                 ┌─────────────────┐               ┌────────────────┐
 │  Phase 1    │                 │     Phase 2     │               │    Phase 3     │
 │  Triage     │  ─triage dict─► │   RAG Retrieval │ ─rag dict──►  │  Agentic       │
 │  (Ollama)   │                 │ (Chroma + LLM)  │               │  Decision      │
 └─────────────┘                 └─────────────────┘               └────────┬───────┘
   intent /                       grounded_answer +                         │
   urgency /                      retrieved_chunks                          ▼
   sentiment /                                                ┌─────────────────────────┐
   entities                                                   │  AgentExecutor (ReAct)   │
                                                              │  + should_escalate()     │
                                                              │  + order ownership check │
                                                              └─────────────────────────┘
                                                                          │
                                                                          ▼
                                                                Resolution packet
                                                                (decision, reply,
                                                                 KB source, order data)
```

A detailed diagram lives in [`docs/architecture.md`](docs/architecture.md).

---

## What it does

1. **Phase 1 — Triage & Classification.** [`pipeline/phase1_triage.py`](pipeline/phase1_triage.py). LLM call to `llama3.1` (via Ollama, JSON mode, `temperature=0`) returning `intent`, `urgency`, `sentiment`, `entities` (`order_id`, `product_name`, `days_mentioned`), and a `confidence` score. 10 intent classes: `order_status`, `return_request`, `refund_inquiry`, `product_question`, `complaint`, `shipping_delay`, `cancellation`, `warranty_claim`, `billing_dispute`, `feedback`. Strict-prompt retry + safe fallback.
2. **Phase 2 — RAG Knowledge Retrieval.** [`pipeline/phase2_rag.py`](pipeline/phase2_rag.py). LangChain `RetrievalQA` over a ChromaDB index of 8 Stryde policy markdown files. MMR retrieval (`k=3, fetch_k=6`) for diverse chunks across different sources. Embedding model: `sentence-transformers/all-MiniLM-L6-v2`. Returns a grounded answer plus the retrieved source chunks.
3. **Phase 3 — Agentic Decision & Action.** [`pipeline/phase3_agent.py`](pipeline/phase3_agent.py). A LangChain `AgentExecutor` with `create_react_agent` handles reply drafting with access to the `order_lookup` tool. **Escalation is deterministic, never LLM-decided** — five rules trigger escalation: API failure or order not found, marked lost-in-transit, delayed past the 14-day SLA, high-urgency + angry sentiment, or a high-stakes intent with no order reference. Includes **order ownership verification** to prevent cross-customer data leaks. Falls back to a direct LLM call if the ReAct agent hits its iteration limit.
4. **Orchestrator.** [`pipeline/orchestrator.py`](pipeline/orchestrator.py) wires the three phases into a single `process_ticket()` call. Processes all 25 tickets end-to-end. The RAG chain is initialised once at module-load to avoid re-loading embeddings on every ticket.

---

## Project layout

```
shopsense-ai/
├── api/
│   └── mock_order_api.py        # Flask order-lookup API on :5050
├── data/
│   ├── tickets.json             # 25 sample customer tickets
│   ├── orders.json              # Mock order database (20 orders)
│   └── knowledge_base/          # 8 Stryde policy markdown files
├── ingest/
│   └── build_vectorstore.py     # One-time KB → ChromaDB ingestion
├── pipeline/
│   ├── __init__.py              # Package init
│   ├── phase1_triage.py         # Phase 1 — Triage & classification
│   ├── phase2_rag.py            # Phase 2 — RAG knowledge retrieval
│   ├── phase3_agent.py          # Phase 3 — Agentic decision layer
│   └── orchestrator.py          # End-to-end pipeline coordinator
├── ui/
│   └── app.py                   # Streamlit demo UI
├── outputs/
│   ├── triage_results.json      # Phase 1 outputs (25 tickets)
│   ├── rag_examples.json        # 3 sample Phase 2 outputs
│   └── pipeline_results.json    # Full end-to-end runs (25 tickets)
├── docs/
│   ├── architecture.md          # System diagram + data flow
│   ├── cto_narrative.md         # Day 30 CTO-facing narrative
│   └── tradeoffs.md             # Engineering trade-off log
├── utils/
│   ├── __init__.py              # Package init
│   └── logger.py                # Shared logging utility
├── logs/                        # Auto-generated logs (gitignored)
├── chroma_db/                   # Persisted vector store (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **Ollama** running locally with the `llama3.1` model pulled
  ```bash
  # Install Ollama from https://ollama.com
  ollama pull llama3.1
  ollama serve            # runs on localhost:11434
  ```
- ~2 GB free disk for the embedding model and ChromaDB index

No API keys required — the project runs entirely locally.

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/MalayThoria/shopsense-ai.git
cd shopsense-ai

# 2. Create a virtual environment and install dependencies
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt

# 3. Make sure Ollama is running and llama3.1 is pulled (see Prerequisites)

# 4. Build the ChromaDB knowledge base (one time)
python -m ingest.build_vectorstore
```

---

## Running the system

You'll need **two terminals** (three if you want the UI). Activate the venv in each.

**Terminal 1 — Mock Order API**
```bash
python -m api.mock_order_api
# Serves on http://localhost:5050
```

**Terminal 2 — Run the full pipeline (batch over all 25 tickets)**
```bash
python -m pipeline.orchestrator
# Processes all 25 tickets and writes outputs/pipeline_results.json
```

**Terminal 3 (optional) — Streamlit UI**
```bash
streamlit run ui/app.py
# Opens http://localhost:8501 with sample-ticket dropdown and live phase results
```

---

## Reproducing the sample outputs

| File | How to regenerate |
|---|---|
| `outputs/triage_results.json` | `python -m pipeline.phase1_triage` |
| `outputs/rag_examples.json` | `python -m pipeline.phase2_rag` |
| `outputs/pipeline_results.json` | `python -m pipeline.orchestrator` |

---

## Sample output

A single ticket processed end-to-end produces this structure:

```json
{
  "triage": {
    "ticket_id": "T001",
    "customer_id": "C101",
    "intent": "order_status",
    "urgency": "high",
    "sentiment": "frustrated",
    "entities": {
      "order_id": "ORD-4892",
      "product_name": null,
      "days_mentioned": 17
    },
    "confidence": 0.95
  },
  "rag": {
    "query_used": "order_status Hi, I placed an order ORD-4892...",
    "retrieved_chunks": [
      {"source": "shipping_info.md", "text": "...", "relevance_score": 0.85}
    ],
    "grounded_answer": "..."
  },
  "outcome": {
    "ticket_id": "T001",
    "decision": "escalate",
    "escalation_reason": "Order delayed 17 days, past 14-day SLA",
    "order_data": {"status": "in_transit", "days_since_order": 17, "...": "..."},
    "draft_reply": "Dear C101, ...",
    "kb_context_used": "shipping_info.md",
    "resolved": false
  }
}
```

**Pipeline summary (25 tickets):** 17 resolved, 8 escalated (32% escalation rate).

Escalation reasons include: SLA breach, lost in transit, order not found, ownership mismatch, high-urgency refund/warranty with no order reference, and angry + high-urgency sentiment.

---

## Tech Stack

| Component | Tool | Reason |
|---|---|---|
| LLM | Ollama + llama3.1 | Local, free, JSON mode for structured output |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Fast, CPU-friendly, no API cost |
| Vector store | ChromaDB | Persistent, metadata support, LangChain integration |
| Retrieval | MMR (k=3, fetch_k=6) | Diverse chunks from multiple KB sources |
| Agent | LangChain AgentExecutor + create_react_agent | ReAct reasoning loop with tool access |
| Orchestration | LangChain (RetrievalQA, @tool) | Retriever + tool abstractions out of box |
| Mock API | Flask | Mirrors real production architecture |
| UI | Streamlit | Rapid demo UI in a single file |

---

## Design Decisions

See [`docs/tradeoffs.md`](docs/tradeoffs.md) for the full trade-off log. Key decisions: deterministic escalation rules over LLM-decided, local-first stack over hosted APIs, ReAct agent with fallback for reply drafting, MMR retrieval for chunk diversity, order ownership verification for security, and a real Flask service for order lookup to mirror production architecture.

For the Day 30 client narrative, see [`docs/cto_narrative.md`](docs/cto_narrative.md).