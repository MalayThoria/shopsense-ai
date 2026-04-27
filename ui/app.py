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

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🧠",
    layout="wide",
)

# ── Sample tickets for quick demo ────────────────────────────────────
SAMPLE_TICKETS = {
    "Select a sample ticket...": "",
    "🚚 Shipping Delay (ORD-4892)": (
        "Hi, I placed an order ORD-4892 almost 17 days ago and it still "
        "hasn't arrived. The tracking hasn't updated in a week. This is "
        "really frustrating — I needed these sneakers for an event this "
        "weekend. Can someone please look into this urgently?"
    ),
    "✅ Order Status Check (ORD-5021)": (
        "Hey, just checking on my order ORD-5021. It says delivered but "
        "I want to confirm everything is good. Thanks!"
    ),
    "❌ Cancellation (ORD-6610)": (
        "I want to cancel my order ORD-6610. I placed it yesterday and "
        "I changed my mind about the colour. Please cancel it before it ships."
    ),
    "😡 Double Charge (ORD-9999)": (
        "I WAS CHARGED TWICE FOR MY ORDER! I can see two transactions of "
        "₹3299 on my credit card statement. This is absolutely ridiculous. "
        "Fix this NOW or I'm filing a chargeback. Order number is ORD-9999."
    ),
    "👟 Product Question": (
        "I'm looking at the Stryde Runner Pro on your website. Does it "
        "come in wide fit? I have slightly wider feet and I'm between "
        "UK 10 and UK 11. What would you recommend?"
    ),
}

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown(
        """
        **ShopSense AI** processes customer tickets through a 3-phase pipeline:

        **Phase 1 — Triage**
        Classifies intent, urgency, sentiment, and extracts entities
        using llama3.1 via Ollama.

        **Phase 2 — RAG Retrieval**
        Searches the Stryde knowledge base (ChromaDB) and generates
        a policy-grounded answer using LangChain.

        **Phase 3 — Agent Decision**
        Looks up order data, applies deterministic escalation rules,
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

# ── Main panel ───────────────────────────────────────────────────────
st.title("🧠 ShopSense AI — Customer Support Intelligence")
st.markdown("Paste a customer ticket below and run it through the full pipeline.")

# Sample ticket dropdown
selected_sample = st.selectbox("Load a sample ticket", options=list(SAMPLE_TICKETS.keys()))

# Text area — pre-fill if sample selected
default_text = SAMPLE_TICKETS.get(selected_sample, "")
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

# ── Process and display ──────────────────────────────────────────────
if run_clicked:
    if not ticket_text.strip():
        st.error("Please enter a ticket message before running the pipeline.")
    else:
        ticket = {
            "ticket_id": "UI-001",
            "customer_id": "UI-USER",
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

        # ── Phase 1 — Triage ─────────────────────────────────────────
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

            st.write(f"Confidence: `{triage.get('confidence', 0)}`")

        # ── Phase 2 — RAG ────────────────────────────────────────────
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

        # ── Phase 3 — Agent Decision ─────────────────────────────────
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