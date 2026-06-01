"""
RAG Knowledge Assistant — Demo ④
Upload documents (PDF / TXT / MD) → ask questions → get answers with source citations.

Business impact: Cuts document lookup from ~15 minutes to under 30 seconds.
"""

import os, io, re
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#0f1117; }
  [data-testid="stSidebar"] { background:#1a1f2e; }
  h1,h2,h3,p,label,div { color:#e2e8f0 !important; }
  .section-tag {
    display:inline-block;background:#1e293b;color:#94a3b8 !important;
    font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
    padding:0.3rem 0.8rem;border-radius:4px;margin-bottom:1rem;
  }
  .stButton>button {
    background:#1e293b;border:1px solid #334155;color:#e2e8f0 !important;border-radius:8px;font-size:0.82rem;
  }
  .stButton>button:hover { border-color:#6366f1;background:#2d3748; }
  .stButton>button[kind="primary"] { background:#6366f1 !important;border-color:#6366f1 !important; }
  [data-testid="stFileUploader"] {
    border:2px dashed #334155 !important;border-radius:10px !important;
    padding:1.5rem !important;background:#1a1f2e !important;
  }
  [data-testid="stDownloadButton"]>button {
    background:#1e293b !important;border:1px solid #334155 !important;
    color:#e2e8f0 !important;border-radius:8px !important;
  }
  [data-testid="stDownloadButton"]>button:hover { border-color:#6366f1 !important; }
  .source-card {
    background:#1a1f2e;border:1px solid #334155;border-radius:8px;
    padding:0.8rem 1rem;margin-bottom:0.6rem;
  }
  .privacy-box {
    background:#0c1a2e;border:1px solid #1e3a5f;border-radius:8px;
    padding:0.7rem 1rem;color:#7dd3fc !important;font-size:0.83rem;line-height:1.7;margin-bottom:1rem;
  }
</style>
""", unsafe_allow_html=True)

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

SAMPLE_QUESTIONS = [
    "What is the return window?",
    "How long does international shipping take?",
    "What warranty comes with electronics?",
    "How do I report a damaged item?",
    "What is the battery life of Widget A?",
    "When does my weekend order ship?",
]

# ── Core RAG ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ API key not configured.")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def chunk_text(text: str, source: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    words = text.split()
    chunks, i, idx = [], 0, 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append({"source": source, "chunk_id": idx, "text": " ".join(chunk_words)})
        i += chunk_size - overlap
        idx += 1
    return chunks

def extract_text(f) -> str:
    name = f.name.lower()
    if name.endswith(".pdf"):
        if PDF_SUPPORT:
            with pdfplumber.open(f) as pdf:
                return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        st.warning("PDF support requires pdfplumber — install with `pip install pdfplumber`")
        return ""
    return f.read().decode("utf-8", errors="ignore")

def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, convert_to_numpy=True).tolist()

def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

def retrieve(q_emb, chunks, top_k=4):
    scored = [{**c, "score": cosine_sim(q_emb, c["embedding"])} for c in chunks]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

def build_index(docs: dict) -> list[dict]:
    all_chunks = []
    for source, text in docs.items():
        all_chunks.extend(chunk_text(text, source))
    with st.spinner(f"Building index — embedding {len(all_chunks)} chunks…"):
        texts = [c["text"] for c in all_chunks]
        embeddings = embed_texts(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb
    return all_chunks

def answer_question(client, question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    system = (
        "You are a helpful knowledge base assistant. Answer using ONLY the provided context. "
        "Be specific and cite the source document name. "
        "If the answer is not in the context, say so clearly."
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1, max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 RAG Assistant")
    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload documents or use the sample\n"
        "2. Docs are chunked + embedded locally\n"
        "3. Your question is matched to the most relevant chunks\n"
        "4. AI answers strictly from those chunks — no hallucination"
    )
    st.divider()
    st.markdown("**Supported formats**")
    st.markdown("`.txt` · `.md` · `.pdf`  — multiple files at once")
    st.divider()
    st.markdown("**Business impact**")
    st.markdown("Document lookup: ~15 min → under 30 sec. Scales to thousands of pages.")
    st.divider()
    st.markdown("Built by [Joseph Wang](https://josephjwang.com)")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='background:linear-gradient(90deg,#6366f1,#06b6d4);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-size:2.2rem;font-weight:700;margin-bottom:0.2rem'>📚 RAG Knowledge Assistant</h1>
<p style='color:#94a3b8;font-size:1rem;margin-bottom:1.5rem'>
Upload documents → ask questions → get precise answers with source citations</p>
""", unsafe_allow_html=True)

# ── Step 1: Load documents ─────────────────────────────────────────────────────
st.markdown('<span class="section-tag">Step 1 — Load your knowledge base</span>', unsafe_allow_html=True)

mode = st.radio("Source", ["Use sample knowledge base", "Upload my own documents"],
                horizontal=True, label_visibility="collapsed")

docs = {}

if mode == "Use sample knowledge base":
    if "rag_sample" not in st.session_state:
        st.session_state["rag_sample"] = True
    docs = SAMPLE_DOCS
    st.info("📂 Sample KB loaded: **Return Policy**, **Shipping Guide**, **Product FAQ**")

else:
    st.markdown("""<div class="privacy-box">
    🔒 <b>Your documents stay private.</b> Files are processed entirely in-memory for this session only —
    nothing is stored or sent anywhere except to the AI model to generate answers. When you close the tab, everything is cleared.
    </div>""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        for f in uploaded_files:
            text = extract_text(f)
            if text.strip():
                docs[f.name] = text
        if docs:
            st.success(f"✅ Loaded {len(docs)} document(s)")
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem;color:#475569">
            <div style="font-size:2.5rem">📄</div>
            <p style="margin-top:0.5rem">Upload TXT, MD, or PDF files to get started</p>
        </div>""", unsafe_allow_html=True)

# ── Auto-build index ───────────────────────────────────────────────────────────
if docs:
    with st.expander("📄 View loaded documents"):
        for name, text in docs.items():
            st.markdown(f"**{name}** — {len(text.split())} words")
            st.text(text[:300] + "…" if len(text) > 300 else text)
            st.divider()

    index_key = f"rag_index_{hash(tuple(sorted(docs.keys())))}"
    if index_key not in st.session_state:
        chunks = build_index(docs)
        st.session_state[index_key] = chunks
        st.success(f"✅ Index ready — {len(chunks)} chunks across {len(docs)} document(s)")
    else:
        chunks = st.session_state[index_key]
        st.success(f"✅ Index ready — {len(chunks)} chunks across {len(docs)} document(s)")

    # ── Step 2: Ask ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="section-tag">Step 2 — Ask a question</span>', unsafe_allow_html=True)

    if mode == "Use sample knowledge base":
        st.markdown("**Quick picks:**")
        row1, row2 = SAMPLE_QUESTIONS[:3], SAMPLE_QUESTIONS[3:]
        for row in [row1, row2]:
            rcols = st.columns(3)
            for i, q in enumerate(row):
                with rcols[i]:
                    if st.button(q, key=f"rq_{SAMPLE_QUESTIONS.index(q)}", use_container_width=True):
                        st.session_state["_q_inject"] = q
                        st.rerun()

    if "_q_inject" in st.session_state:
        st.session_state["rag_question"] = st.session_state.pop("_q_inject")

    question = st.text_input(
        "Ask anything about your documents",
        placeholder="e.g. What is the return window?",
        key="rag_question",
        label_visibility="collapsed",
    )

    col_ask, col_clear, _ = st.columns([1, 1, 6])
    with col_ask:
        ask = st.button("🔍 Ask", type="primary", disabled=not question, use_container_width=True)
    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.pop("rag_answer", None)
            st.session_state.pop("rag_chunks", None)
            st.session_state["_q_inject"] = ""
            st.rerun()

    if ask and question:
        client = get_client()
        with st.spinner("Searching knowledge base…"):
            q_emb = embed_texts([question])[0]
            top_chunks = retrieve(q_emb, chunks, top_k=4)
        with st.spinner("Generating answer…"):
            answer = answer_question(client, question, top_chunks)
            st.session_state["rag_answer"] = answer
            st.session_state["rag_chunks"] = top_chunks
        st.session_state["_q_inject"] = ""
        st.rerun()

    # ── Answer ─────────────────────────────────────────────────────────────────
    if st.session_state.get("rag_answer"):
        st.divider()
        st.markdown('<span class="section-tag">Answer</span>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#1e293b;border-left:3px solid #6366f1;border-radius:0 8px 8px 0;"
            f"padding:1rem 1.2rem;color:#e2e8f0;line-height:1.8'>"
            f"{st.session_state['rag_answer']}</div>",
            unsafe_allow_html=True,
        )

        # Source citations
        with st.expander("📎 Source citations"):
            for c in st.session_state["rag_chunks"]:
                st.markdown(
                    f"<div class='source-card'>"
                    f"<span style='color:#6366f1;font-weight:600'>{c['source']}</span>"
                    f"<span style='color:#475569;font-size:0.8rem;margin-left:0.5rem'>chunk {c['chunk_id']} · relevance {c['score']:.2f}</span>"
                    f"<div style='color:#94a3b8;font-size:0.83rem;margin-top:0.4rem;line-height:1.6'>{c['text'][:250]}…</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
