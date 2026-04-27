# ShopSense AI

A customer support intelligence platform for **Stryde**, a fictional D2C sneaker and apparel brand. ShopSense ingests a customer ticket and runs it through a single end-to-end pipeline: classify it, retrieve grounded knowledge, look up the order, decide whether to resolve or escalate, and draft a reply.

Built for the Forward Deployed Engineer take-home assignment.

---

## Architecture at a glance

```
                          ┌────────────────────────────────────┐
   Customer ticket ──►    │            Orchestrator            │
   (UI / JSON file)       │       pipeline/orchestrator.py     │
                          └────────────────────────────────────┘
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
   entities                                                   │  Mock Order API (Flask) │
                                                              │  localhost:5050         │
                                                              └─────────────────────────┘
                                                                          │
                                                                          ▼
                                                          ┌──────────────────────────────┐
                                                          │ Deterministic escalation     │
                                                          │ rules + LLM reply drafter    │
                                                          └──────────────────────────────┘
                                                                          │
                                                                          ▼
                                                                Resolution packet
                                                                (decision, reply,
                                                                 KB source, order data)
```

A more detailed diagram lives in [`docs/architecture.md`](docs/architecture.md).

---

## What it does

1. **Phase 1 — Triage & Classification.** [`pipeline/phase1_triage.py`](pipeline/phase1_triage.py). LLM call to `llama3.1` (via Ollama, JSON mode) returning `intent`, `urgency`, `sentiment`, `entities` (`order_id`, `product_name`, `days_mentioned`), and a `confidence` score. Strict-prompt retry + safe fallback.
2. **Phase 2 — RAG Knowledge Retrieval.** [`pipeline/phase2_rag.py`](pipeline/phase2_rag.py). LangChain `RetrievalQA` over a ChromaDB index of 8 Stryde policy markdown files (`returns_policy`, `refund_process`, `shipping_info`, `product_warranty`, `payment_methods`, `loyalty_program`, `size_guide`, `escalation_contacts`). Embedding model: `sentence-transformers/all-MiniLM-L6-v2`. Returns a grounded answer plus the retrieved source chunks.
3. **Phase 3 — Agentic Decision & Action.** [`pipeline/phase3_agent.py`](pipeline/phase3_agent.py). LangChain `@tool` (`order_lookup`) hits the mock Order API. **Escalation is deterministic, never LLM-decided** — five rules trigger escalation: order not found, delayed past the 14-day SLA, marked lost-in-transit, high-urgency + angry sentiment, or a complaint with no order reference. Reply is drafted by the LLM, grounded in the Phase 2 KB context.
4. **Orchestrator.** [`pipeline/orchestrator.py`](pipeline/orchestrator.py) wires the three phases into a single `process_ticket()` call. The RAG chain is initialised once at module-load to avoid re-loading embeddings on every ticket.

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
│   ├── triage_results.json      # Sample Phase 1 outputs (25 tickets)
│   ├── rag_examples.json        # 3 sample Phase 2 outputs
│   └── pipeline_results.json    # 5 full end-to-end runs
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
git clone <your-repo-url> shopsense-ai
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
python ingest/build_vectorstore.py
```

---

## Running the system

You'll need **two terminals** (three if you want the UI). Activate the venv in each.

**Terminal 1 — Mock Order API**
```bash
python api/mock_order_api.py
# Serves on http://localhost:5050
```

**Terminal 2 — Run the full pipeline (batch over sample tickets)**
```bash
python pipeline/orchestrator.py
# Processes the first 5 tickets and writes outputs/pipeline_results.json
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
| [`outputs/triage_results.json`](outputs/triage_results.json) | `python pipeline/phase1_triage.py` |
| [`outputs/rag_examples.json`](outputs/rag_examples.json) | `python pipeline/phase2_rag.py` |
| [`outputs/pipeline_results.json`](outputs/pipeline_results.json) | `python pipeline/orchestrator.py` |

---

## Demo flow (for a 5-minute walkthrough)

1. Start `mock_order_api.py` and `streamlit run ui/app.py`.
2. Pick **"🚚 Shipping Delay (ORD-4892)"** from the dropdown — demonstrates the SLA-breach escalation path (17-day delay → escalated).
3. Pick **"✅ Order Status Check (ORD-5021)"** — a calm, low-urgency ticket that resolves with a grounded reply.
4. Pick **"😡 Double Charge (ORD-9999)"** — angry sentiment + missing order → escalated for human review.
5. Open `outputs/pipeline_results.json` to show the full structured packet for each ticket.

---

## Stack

- **LLM**: Ollama + `llama3.1` (local, JSON mode)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store**: ChromaDB (persisted)
- **Orchestration**: LangChain (`RetrievalQA`, `@tool`)
- **Mock API**: Flask
- **UI**: Streamlit

For the design rationale and what we'd build on Day 60, see [`docs/cto_narrative.md`](docs/cto_narrative.md). For the engineering trade-offs taken along the way, see [`docs/tradeoffs.md`](docs/tradeoffs.md).
