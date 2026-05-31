# RAG Knowledge Assistant — Demo ④

> **Business impact:** Cuts document lookup from ~15 minutes to under 30 seconds. Scales to thousands of pages with zero retraining.

Upload any documents → ask questions in plain English → get precise answers with source citations.

## Live demo

[josephjwang.com/analyst](https://josephjwang.com/analyst)

## What it does

1. Upload `.txt`, `.md`, or `.pdf` files (or use the built-in sample)
2. Documents are chunked and embedded using `text-embedding-3-small`
3. Your question is embedded and matched to the most relevant chunks (cosine similarity)
4. GPT-4o-mini answers using only the retrieved context — no hallucination outside the docs
5. Source document + chunk shown for every answer

## Sample knowledge base

Includes 3 pre-loaded documents:
- **Return Policy** — eligibility, process, damaged items, exchanges
- **Shipping Guide** — domestic/international rates, tracking, cutoff times
- **Product FAQ** — materials, warranty, compatibility, battery life

## Tech stack

- **Frontend:** Streamlit
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Retrieval:** Cosine similarity (numpy, no external vector DB)
- **Generation:** OpenAI GPT-4o-mini with strict context grounding
- **PDF parsing:** pdfplumber (optional)

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

## Architecture

```
Documents → Chunking (400 words, 80 overlap) → Embeddings → In-memory index
Query → Embed → Top-k cosine retrieval → GPT-4o-mini → Answer + citations
```

No external vector database required — index lives in Streamlit session state. For production, swap numpy store with Chroma, Pinecone, or pgvector.

## Related demos

- [AI Data Analyst](../ai-data-analyst) — CSV → LLM analysis + charts
- [ChatBI](../chatbi) — natural language → SQL → results

---

Built by [Joseph Wang](https://josephjwang.com) · Northwestern MSc Data Science · 6 years enterprise analytics
