"""
ShopSense AI — Streamlit UI.

A clean, functional interface for running the 3-phase customer support
intelligence pipeline. Provides sample tickets, colour-coded results,
and expandable phase details.
"""

import sys
from pathlib import Path

# Add project root to path so pipeline imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import json
import streamlit as st
from pipeline.orchestrator import process_ticket

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🧠",
    layout="wide",
)

# ── Load real tickets from data/tickets.json ─────────────────────
tickets_path = project_root / "data" / "tickets.json"
all_tickets = json.loads(tickets_path.read_text(encoding="utf-8"))

# Build sample ticket options from real data (pick diverse ones)
SAMPLE_TICKET_IDS = {
    "Select a sample ticket...": None,
    "🚚 Shipping Delay (T001 — ORD-4892)": "T001",
    "✅ Order Status Check (T002 — ORD-5021)": "T002",
    "❌ Cancellation (T003 — ORD-6610)": "T003",
    "📦 Lost in Transit (T004 — ORD-7741)": "T004",
    "😡 Double Charge (T008 — ORD-9999)": "T008",
    "👟 Product Question (T007)": "T007",
    "💳 UPI Double Charge (T019 — ORD-4892)": "T019",
    "🔧 Warranty Claim (T015)": "T015",
    "💬 Positive Feedback (T016)": "T016",
    "😤 Angry Refund (T024)": "T024",
}

# Index tickets by ID for quick lookup
tickets_by_id = {t["ticket_id"]: t for t in all_tickets}

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown(
        """
        **ShopSense AI** processes customer tickets through a 3-phase pipeline:

        **Phase 1 — Triage**
        Classifies intent, urgency, sentiment, and extracts entities
        using llama3.1 via Ollama. 10 intent classes.

        **Phase 2 — RAG Retrieval**
        Searches the Stryde knowledge base (ChromaDB) with MMR retrieval
        and generates a policy-grounded answer using LangChain.

        **Phase 3 — Agent Decision**
        LangChain ReAct AgentExecutor looks up order data, applies
        deterministic escalation rules, verifies order ownership,
        and drafts a customer reply.

        ---
        *Built with Ollama, LangChain, ChromaDB, and Streamlit.*
        """
    )

    st.markdown("---")
    st.markdown("**🔗 Prerequisites**")
    st.markdown(
        """
        - Ollama running at `localhost:11434`
        - Mock API running at `localhost:5050`
        - ChromaDB ingested via `build_vectorstore.py`
        """
    )

# ── Main panel ───────────────────────────────────────────────────
st.title("🧠 ShopSense AI — Customer Support Intelligence")
st.markdown("Select a sample ticket or paste your own, then run it through the full pipeline.")

# Sample ticket dropdown
selected_sample = st.selectbox("Load a sample ticket", options=list(SAMPLE_TICKET_IDS.keys()))

# Get the selected ticket data (if any)
selected_ticket_id = SAMPLE_TICKET_IDS.get(selected_sample)
selected_ticket_data = tickets_by_id.get(selected_ticket_id) if selected_ticket_id else None

# Text area — pre-fill if sample selected
default_text = selected_ticket_data["text"] if selected_ticket_data else ""
ticket_text = st.text_area(
    "Paste customer ticket",
    value=default_text,
    height=150,
    placeholder="Type or paste a customer support message here...",
)

# Channel selector
channel = st.selectbox("Channel", ["email", "chat", "whatsapp"])

# Run button
run_clicked = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# ── Process and display ──────────────────────────────────────────
if run_clicked:
    if not ticket_text.strip():
        st.error("Please enter a ticket message before running the pipeline.")
    else:
        # Use real customer_id from sample data, or a default for custom text
        if selected_ticket_data and ticket_text.strip() == selected_ticket_data["text"].strip():
            ticket = {
                "ticket_id": selected_ticket_data["ticket_id"],
                "customer_id": selected_ticket_data["customer_id"],
                "channel": selected_ticket_data.get("channel", channel),
                "text": ticket_text.strip(),
            }
        else:
            # Custom text pasted by user — no real customer_id
            ticket = {
                "ticket_id": "UI-001",
                "customer_id": "UI-CUSTOM",
                "channel": channel,
                "text": ticket_text.strip(),
            }

        with st.spinner("Running all 3 pipeline phases..."):
            try:
                result = process_ticket(ticket)
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                st.stop()

        triage = result["triage"]
        rag = result["rag"]
        outcome = result["outcome"]

        st.success("Pipeline complete!")
        st.markdown("---")

        # ── Phase 1 — Triage ─────────────────────────────────────
        with st.expander("📋 Phase 1 — Triage Result"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Intent", triage.get("intent", "unknown"))

            with col2:
                urgency = triage.get("urgency", "medium")
                urgency_colors = {"high": "🔴", "medium": "🟠", "low": "🟢"}
                st.metric("Urgency", f"{urgency_colors.get(urgency, '')} {urgency}")

            with col3:
                sentiment = triage.get("sentiment", "neutral")
                sentiment_icons = {
                    "angry": "😡", "frustrated": "😤",
                    "neutral": "😐", "positive": "😊",
                }
                st.metric("Sentiment", f"{sentiment_icons.get(sentiment, '')} {sentiment}")

            st.markdown("**Entities:**")
            entities = triage.get("entities", {})
            ent_col1, ent_col2, ent_col3 = st.columns(3)
            with ent_col1:
                st.write(f"Order ID: `{entities.get('order_id', 'None')}`")
            with ent_col2:
                st.write(f"Product: `{entities.get('product_name', 'None')}`")
            with ent_col3:
                st.write(f"Days mentioned: `{entities.get('days_mentioned', 'None')}`")

            st.write(f"Customer ID: `{triage.get('customer_id', 'N/A')}`")
            st.write(f"Confidence: `{triage.get('confidence', 0)}`")

        # ── Phase 2 — RAG ────────────────────────────────────────
        with st.expander("📚 Phase 2 — Knowledge Base Answer"):
            st.markdown("**Grounded Answer:**")
            st.info(rag.get("grounded_answer", "No answer generated."))

            with st.expander("Retrieved Chunks"):
                chunks = rag.get("retrieved_chunks", [])
                if chunks:
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {i}** — `{chunk.get('source', 'unknown')}`")
                        st.caption(chunk.get("text", ""))
                        st.markdown("---")
                else:
                    st.write("No chunks retrieved.")

        # ── Phase 3 — Agent Decision ─────────────────────────────
        with st.expander("⚡ Phase 3 — Agent Decision", expanded=True):
            decision = outcome.get("decision", "unknown")

            if decision == "escalate":
                st.error(f"🔴 **ESCALATED** — {outcome.get('escalation_reason', '')}")
            else:
                st.success("🟢 **RESOLVED**")

            # Order data
            order_data = outcome.get("order_data")
            if order_data:
                with st.expander("📦 Order Data"):
                    st.json(order_data)

            # Draft reply
            st.markdown("**Draft Reply:**")
            draft_reply = outcome.get("draft_reply", "No reply generated.")
            st.text_area(
                "Reply",
                value=draft_reply,
                height=200,
                label_visibility="collapsed",
            )

            st.markdown(f"*KB Source: `{outcome.get('kb_context_used', 'none')}`*")