# RAG Knowledge Assistant

Upload documents and ask questions — get precise answers with source citations.

## Overview

A retrieval-augmented generation (RAG) pipeline built on top of OpenAI embeddings. Documents are chunked, embedded, and stored in an in-memory index. At query time, the most relevant chunks are retrieved by cosine similarity and passed to GPT-4o-mini, which answers strictly from the provided context and cites the source document.

## Features

- **Multi-document upload** — `.txt`, `.md`, and `.pdf` (via pdfplumber) supported; multiple files at once
- **Configurable retrieval** — top-k adjustable in the sidebar
- **Grounded answers** — model is instructed to answer only from retrieved context and to say so when the answer isn't there
- **Source transparency** — every answer shows the source document, chunk index, and similarity score
- **Built-in sample** — 3 pre-loaded documents (return policy, shipping guide, product FAQ)

## Architecture

```
Documents
  → chunk (400 words, 80-word overlap)
  → embed (text-embedding-3-small)
  → in-memory index (numpy)

Query
  → embed
  → top-k cosine retrieval
  → GPT-4o-mini (context-grounded)
  → answer + citations
```

No external vector database required. For production scale, the numpy store is a drop-in swap for Chroma, Pinecone, or pgvector.

## Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Embeddings | OpenAI `text-embedding-3-small` |
| Retrieval | NumPy (cosine similarity) |
| Generation | OpenAI GPT-4o-mini |
| PDF parsing | pdfplumber |

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

## Sample questions

- What is the return window?
- How long does international shipping take?
- What warranty comes with electronics?
- How do I report a damaged item?

## Related

- [AI Data Analyst](https://github.com/josephwang-ds/ai-data-analyst) — CSV → natural language analysis + charts
- [ChatBI](https://github.com/josephwang-ds/chatbi) — natural language → SQL → results

---

[josephjwang.com](https://josephjwang.com) · [github.com/josephwang-ds](https://github.com/josephwang-ds)
