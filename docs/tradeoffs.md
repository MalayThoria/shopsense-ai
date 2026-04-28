# ShopSense AI — Trade-off Log

## Summary

| Decision | Chosen | Alternative | Trade-off |
|---|---|---|---|
| LLM | Ollama llama3.1 | OpenAI gpt-4o-mini | Zero cost, fully local, but higher latency (~2-6s per phase) |
| Phase 1 approach | Direct ollama SDK | LangChain LLMChain | Simpler, no framework overhead, but less abstraction |
| Phase 2-3 framework | LangChain | Custom framework | Faster development with retriever + tool integrations, but heavier dependency |
| Vector DB | ChromaDB | FAISS | Metadata support + persistence out of box, but less battle-tested at scale |
| Embeddings | MiniLM-L6-v2 | OpenAI text-embedding-3-small | Free, local, CPU-friendly, but lower recall on nuanced queries |
| Retrieval | MMR (k=3, fetch_k=6) | Similarity search (k=3) | Diverse chunks from multiple sources, but slightly slower retrieval |
| Escalation | Deterministic rules | LLM-decided | Auditable, testable, reviewable by support lead, but less flexible on edge cases |
| Agent | ReAct AgentExecutor + fallback | Pure deterministic pipeline | Satisfies agentic requirement with safety net, but ReAct can hit iteration limits |
| Mock API | Flask + JSON | LangChain tool with mock data | Mirrors real production architecture with network calls + failure modes, but extra process to run |
| Intent schema | 10 classes | 7 classes | Better coverage (warranty, billing, feedback), but more fragile classification |
| Ownership check | Compare customer_id on lookup | Trust order_id blindly | Prevents privacy leaks (T019 cross-customer), but adds one check per ticket |

---

## Detailed rationale

### 1. Local LLM (Ollama + `llama3.1`) over a hosted API

- **Picked**: Ollama with `llama3.1` running on `localhost:11434`.
- **Declined**: OpenAI / Anthropic / Gemini APIs.
- **Why**: Zero per-token cost during build, no key management, fully reproducible on a laptop, no data leaving the machine — important when iterating on real customer messages later. Quality of `llama3.1` is good enough for structured classification and grounded reply drafting.
- **Cost**: Higher latency (~2–6 s per phase), CPU/RAM heavy, slightly weaker reasoning than frontier hosted models.
- **Revisit when**: latency hits SLOs in production, or when we need stronger long-context reasoning. The LLM client is wrapped behind LangChain interfaces — swap is a few lines.

### 2. Deterministic escalation, not LLM-decided

- **Picked**: Five hard-coded rules in `should_escalate()`.
- **Declined**: Agent that reasons about whether to escalate.
- **Why**: Escalation is the highest-stakes decision in the loop. Rules are auditable, testable, and reviewable by the support lead — they can read them and disagree. An LLM decision is none of those things.
- **Cost**: Less flexibility on edge cases the rules don't cover (those default to "resolve").
- **Revisit when**: we have labeled data to evaluate a learned escalation classifier and a way to A/B it against the rule set.

### 3. ReAct AgentExecutor with deterministic fallback

- **Picked**: LangChain `create_react_agent` + `AgentExecutor` for reply drafting, with a direct LLM fallback if the agent hits its iteration limit.
- **Declined**: Pure deterministic pipeline (no agent), or full agent-controlled escalation.
- **Why**: The assignment requires an "agentic" phase with `AgentExecutor`. We use the agent for reply generation where hallucination risk is low, while keeping escalation decisions outside the agent as deterministic rules. The fallback ensures the pipeline never crashes — if llama3.1 struggles with ReAct format (which happens ~40% of the time on product questions with no tools needed), the direct LLM call generates the reply instead.
- **Cost**: The ReAct loop adds latency when it works, and the fallback means some replies bypass the agent entirely.
- **Revisit when**: we upgrade to a model with better tool-calling support (GPT-4, Claude), or fine-tune llama3.1 on ReAct format.

### 4. ChromaDB over FAISS / Pinecone / pgvector

- **Picked**: ChromaDB with on-disk persistence.
- **Declined**: FAISS (no metadata story), Pinecone (cloud, account needed), pgvector (operational overhead for a take-home).
- **Why**: Single `pip install`, persistent across runs, native LangChain integration, returns source documents with metadata (we use `doc.metadata["source"]` for KB attribution).
- **Cost**: Not the fastest at scale; not as battle-tested in production as the alternatives.
- **Revisit when**: KB grows past ~10⁵ chunks or we need filtered queries by tenant/locale.

### 5. MMR retrieval over pure similarity search

- **Picked**: Maximal Marginal Relevance with `k=3, fetch_k=6`.
- **Declined**: Standard similarity search with `k=3`.
- **Why**: With short policy markdown files, similarity search often returned 3 chunks from the same file. MMR fetches 6 candidates and selects 3 that are both relevant AND diverse — ensuring the agent gets context from multiple policy documents.
- **Cost**: Slightly slower retrieval (fetches 6 instead of 3), marginal relevance trade-off.
- **Revisit when**: never — MMR is strictly better for this use case.

### 6. `RetrievalQA` "stuff" chain over map-reduce / refine

- **Picked**: Stuff (concatenate top-k chunks into one prompt).
- **Declined**: Map-reduce (per-chunk summarisation then merge), Refine (sequential rewriting).
- **Why**: Top-3 chunks at chunk_size=400 fit comfortably in `llama3.1`'s context window. Stuff is one LLM call per ticket — the others are k+1.
- **Cost**: Won't scale to long documents or large k. Less useful when we want per-chunk reasoning.
- **Revisit when**: we add long-form policy docs (legal, T&Cs) or push `k` past ~6.

### 7. Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

- **Picked**: MiniLM-L6-v2 (384-dim, ~80 MB, runs on CPU).
- **Declined**: BGE / E5 large models, OpenAI `text-embedding-3-*`.
- **Why**: Fast, small, CPU-friendly, well-understood, no network calls. Quality on short policy chunks is good.
- **Cost**: Loses some recall vs. larger models on nuanced semantic queries.
- **Revisit when**: we observe retrieval misses on a labeled eval set.

### 8. Expanded intent schema (10 classes vs 7)

- **Picked**: 10 intent classes including `warranty_claim`, `billing_dispute`, `feedback`.
- **Declined**: Original 7-class schema.
- **Why**: Real tickets in the dataset include warranty defects (T015), double charges (T008, T019), and positive feedback (T016). Without dedicated classes, these get misclassified — warranty claims become "complaints", billing disputes become "refund inquiries", and positive feedback becomes "product questions".
- **Cost**: More classes means more fragile classification — the LLM has to distinguish between more options.
- **Revisit when**: we have labeled data to measure per-class precision/recall.

### 9. Order ownership verification

- **Picked**: Compare `customer_id` from triage with `customer_id` from order lookup before processing.
- **Declined**: Trust the order_id blindly.
- **Why**: T019 (customer C119) references ORD-4892 which belongs to C101. Without the check, C119 would see C101's order details — a privacy violation. The ownership check escalates with reason "Order ID does not belong to ticket customer" so a human can investigate.
- **Cost**: One additional comparison per ticket with an order_id.
- **Revisit when**: never — this is a security requirement.

### 10. Mock Order API as a real Flask service (not a static JSON file)

- **Picked**: Flask on `:5050`, called via the LangChain `@tool` over HTTP.
- **Declined**: Reading `orders.json` directly inside the agent.
- **Why**: Mirrors real production architecture — the agent is making a network call to an external system, including failure modes (timeouts, 404s). API failures now trigger escalation (Rule 1). Easier to swap for the real Stryde Order API later.
- **Cost**: Need to start a second process. Slight friction during demo.
- **Revisit when**: never — this is the right shape for production.

### 11. Streamlit over Flask/React for the UI

- **Picked**: Streamlit, single file, `ui/app.py`.
- **Declined**: Flask + Jinja, FastAPI + React.
- **Why**: The UI is a demo surface, not a product. Streamlit gets us colour-coded metrics, expandable phase panels, and a sample-ticket dropdown in 200 lines.
- **Cost**: No fine-grained styling, single-user, no auth.
- **Revisit when**: a real internal tool ships — then build a proper Next.js app on top of an API layer that wraps the orchestrator.

### 12. RAG chain initialised once at module load

- **Picked**: Build `rag_chain` at module-level in `pipeline/orchestrator.py`.
- **Declined**: Build per request.
- **Why**: Embedding model load takes seconds; doing it per ticket would dominate latency.
- **Cost**: Module import has a side-effect — surprising if you import the orchestrator just to use a helper.
- **Revisit when**: we move to a long-running service — then this becomes a startup hook instead of an import side-effect.

---

## Things we explicitly did **not** do (and why)

- **No fine-tuning.** No labeled data yet. Premature.
- **No cache layer.** Each ticket is unique; semantic caching belongs in v2.
- **No streaming responses.** The UI runs the full pipeline end-to-end before showing anything; streaming the draft reply would be a nice polish for v2.
- **No auth, rate-limiting, or multi-tenancy.** Out of scope for a take-home; called out in the CTO narrative as Day 60 work.
- **No automated eval harness.** Sample outputs in `outputs/` are spot-checked by hand. An offline eval over a labeled set is the most important thing to build next.