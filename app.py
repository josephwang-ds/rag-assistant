"""
RAG Knowledge Assistant — Demo ④
Upload documents (PDF / TXT / MD) → ask questions → get answers with source citations.

Business impact: Cuts document lookup from ~15 minutes to under 30 seconds.
"""

import os
import io
import math
import json
import re
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Optional PDF support
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Sample knowledge base ──────────────────────────────────────────────────────
SAMPLE_DOCS = {
    "Return Policy.txt": """
RETURN & REFUND POLICY — Effective January 1, 2024

1. ELIGIBILITY
Items may be returned within 30 days of delivery for a full refund.
Items must be unused, in original packaging, with all tags attached.
Digital products and custom orders are non-refundable.
Sale items marked "Final Sale" cannot be returned.

2. PROCESS
To initiate a return, contact support@store.com with your order number.
We will send a prepaid return shipping label within 1 business day.
Refunds are processed within 5–7 business days after we receive the item.
Original shipping fees are non-refundable unless the item was defective.

3. DAMAGED OR DEFECTIVE ITEMS
Report damaged or defective items within 7 days of delivery.
Attach photos of the damage to your support email.
We will ship a replacement at no cost or issue a full refund including shipping.

4. EXCHANGES
Exchanges are available for size or color changes on eligible items.
Exchange requests must be submitted within 14 days of delivery.
Items must pass the same eligibility criteria as returns.
""",
    "Shipping Guide.txt": """
SHIPPING GUIDE — Updated March 2024

DOMESTIC SHIPPING (US)
Standard Shipping: 5–7 business days — Free on orders over $50, otherwise $5.99
Expedited Shipping: 2–3 business days — $12.99
Overnight Shipping: Next business day — $24.99 (order before 2pm EST)

INTERNATIONAL SHIPPING
Canada & Mexico: 7–14 business days — $14.99
Europe: 10–18 business days — $19.99
Asia-Pacific: 12–21 business days — $24.99
Rest of World: 14–28 business days — $29.99

TRACKING
All orders receive a tracking number via email within 24 hours of shipment.
Track your order at track.store.com or via the carrier's website.

DELAYS
During peak seasons (Nov–Dec), please allow 2–3 extra business days.
We are not responsible for carrier delays due to weather or customs.

ORDER CUTOFF
Orders placed before 12pm EST Monday–Friday ship same day.
Weekend orders ship the following Monday.
""",
    "Product FAQ.txt": """
PRODUCT FAQ

Q: What materials are used in your electronics?
A: All our electronics use RoHS-compliant components. Casings are made from recycled ABS plastic (minimum 30% recycled content). Cables use oxygen-free copper for optimal signal quality.

Q: Do you offer warranty on electronics?
A: Yes. All electronics come with a 1-year manufacturer warranty covering defects in materials and workmanship. Extended 2-year warranties are available for purchase at checkout.

Q: Are your accessories compatible with all devices?
A: Most accessories are designed for universal compatibility. Product pages list specific compatibility. If unsure, contact support before purchasing.

Q: How do I register my product for warranty?
A: Visit warranty.store.com within 30 days of purchase. You'll need your order number and serial number (found on the product box).

Q: Can I use accessories internationally?
A: Gadget X and Gadget Y support 100–240V input. Always check the product label before using abroad. US plugs may require an adapter but no voltage converter.

Q: What is the battery life on Widget A?
A: Widget A provides up to 12 hours of continuous use on a single charge. Charging time is approximately 2 hours with the included USB-C cable.
""",
}

# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    chunk_idx = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text_str = " ".join(chunk_words)
        chunks.append({
            "source": source,
            "chunk_id": chunk_idx,
            "text": chunk_text_str,
        })
        i += chunk_size - overlap
        chunk_idx += 1
    return chunks


def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        if PDF_SUPPORT:
            with pdfplumber.open(uploaded_file) as pdf:
                return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        else:
            st.warning("PDF support requires pdfplumber. Install with: pip install pdfplumber")
            return ""
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


# ── Embeddings & retrieval ─────────────────────────────────────────────────────

def get_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ API key not configured.")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts locally using sentence-transformers (no API cost)."""
    embedder = get_embedder()
    return embedder.encode(texts, convert_to_numpy=True).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def retrieve(query_embedding: list[float], chunks: list[dict], top_k: int = 4) -> list[dict]:
    scored = [
        {**c, "score": cosine_similarity(query_embedding, c["embedding"])}
        for c in chunks
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def answer_question(client: OpenAI, question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_id']}]\n{c['text']}"
        for c in context_chunks
    )
    system = (
        "You are a helpful knowledge base assistant. Answer the user's question using ONLY "
        "the provided context. Be specific and cite the source document name in your answer. "
        "If the answer is not in the context, say so clearly — do not make up information."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


# ── Build index ────────────────────────────────────────────────────────────────

def build_index(docs: dict[str, str]) -> list[dict]:
    all_chunks = []
    for source, text in docs.items():
        all_chunks.extend(chunk_text(text, source))

    with st.spinner(f"Embedding {len(all_chunks)} chunks (local model)…"):
        texts = [c["text"] for c in all_chunks]
        embeddings = embed_texts(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb

    return all_chunks


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("📚 RAG Knowledge Assistant")
st.caption("Upload documents → ask questions → get answers with source citations. No more manual searching.")

with st.sidebar:
    st.header("⚙️ Settings")

    st.divider()
    st.markdown("**Supported formats**")
    st.markdown("- `.txt` / `.md`\n- `.pdf` (requires pdfplumber)\n- Multiple files at once")
    st.divider()
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 8, 4)
    st.divider()
    st.markdown("**Business impact**")
    st.markdown("Cuts document lookup from ~15 min to under 30 seconds. Scales to thousands of pages.")
    st.divider()
    st.markdown("Built by [Joseph Wang](https://josephjwang.com)")

# ── Step 1: Load documents ─────────────────────────────────────────────────────
st.subheader("1 · Load your knowledge base")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_files = st.file_uploader(
        "Upload documents (TXT, MD, PDF)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
with col2:
    use_sample = st.button("▶ Use sample knowledge base", use_container_width=True)

docs = {}
if uploaded_files:
    for f in uploaded_files:
        text = extract_text_from_file(f)
        if text.strip():
            docs[f.name] = text
    st.success(f"Loaded {len(docs)} document(s)")
elif use_sample or "rag_sample" in st.session_state:
    st.session_state["rag_sample"] = True
    docs = SAMPLE_DOCS
    st.info("Using sample knowledge base: Return Policy, Shipping Guide, Product FAQ")

if docs:
    with st.expander("📄 View loaded documents"):
        for name, text in docs.items():
            st.markdown(f"**{name}** ({len(text.split())} words)")
            st.text(text[:300] + "…" if len(text) > 300 else text)
            st.divider()

    # ── Step 2: Build index ────────────────────────────────────────────────────
    index_key = f"index_{hash(tuple(docs.keys()))}"
    if index_key not in st.session_state:
        if st.button("🔧 Build index", type="primary"):
            chunks = build_index(docs)
            st.session_state[index_key] = chunks
            st.success(f"Index built — {len(chunks)} chunks across {len(docs)} documents")
    else:
        chunks = st.session_state[index_key]
        st.success(f"✅ Index ready — {len(chunks)} chunks across {len(docs)} documents")

    # ── Step 3: Ask questions ──────────────────────────────────────────────────
    if index_key in st.session_state:
        chunks = st.session_state[index_key]

        st.subheader("2 · Ask a question")

        sample_qs = [
            "What is the return window?",
            "How long does international shipping take?",
            "What warranty comes with electronics?",
            "How do I report a damaged item?",
            "What is the battery life of Widget A?",
        ]

        st.markdown("**Quick picks:**")
        qcols = st.columns(len(sample_qs))
        chosen_q = None
        for i, q in enumerate(sample_qs):
            with qcols[i]:
                if st.button(q, key=f"rq_{i}", use_container_width=True):
                    chosen_q = q

        question = st.text_input(
            "Or type your own question",
            value=chosen_q or "",
            placeholder="What is the refund processing time?",
        )

        if question and st.button("🔍 Ask", type="primary"):
            client = get_client()
            with st.spinner("Searching knowledge base…"):
                q_emb = embed_texts([question])[0]
                top_chunks = retrieve(q_emb, chunks, top_k=top_k)

            with st.spinner("Generating answer…"):
                answer = answer_question(client, question, top_chunks)
                st.session_state["rag_answer"] = answer
                st.session_state["rag_chunks"] = top_chunks

        if "rag_answer" in st.session_state:
            st.divider()
            st.markdown("**Answer**")
            st.success(st.session_state["rag_answer"])

            with st.expander(f"📎 Retrieved sources (top {top_k} chunks)"):
                for c in st.session_state["rag_chunks"]:
                    st.markdown(f"**{c['source']}** · chunk {c['chunk_id']} · score: `{c['score']:.3f}`")
                    st.text(c["text"][:300] + "…")
                    st.divider()

else:
    st.info("⬆ Upload documents or click **Use sample knowledge base** to get started.")
