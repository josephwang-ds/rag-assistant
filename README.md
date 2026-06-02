# RAG Knowledge Assistant

Upload documents and ask questions — get precise answers with source citations.

## Overview

A retrieval-augmented generation (RAG) pipeline using local SentenceTransformer embeddings and DeepSeek generation through an OpenAI-compatible client. Documents are chunked, embedded, and stored in an in-memory index. At query time, the most relevant chunks are retrieved by cosine similarity and passed to the LLM, which answers strictly from the provided context and cites the source document.

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
  → embed (all-MiniLM-L6-v2, local)
  → in-memory index (numpy)

Query
  → embed
  → top-k cosine retrieval
  → DeepSeek chat (context-grounded)
  → answer + citations
```

No external vector database required. For production scale, the numpy store is a drop-in swap for Chroma, Pinecone, or pgvector.

## Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Retrieval | NumPy (cosine similarity) |
| Generation | DeepSeek `deepseek-chat` via OpenAI-compatible API |
| PDF parsing | pdfplumber |

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_deepseek_api_key
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
