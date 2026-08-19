# Journal AI

Private journal app with a chat feature that answers questions about your past entries using RAG.

## What it does

You write journal entries, and then you can ask the AI things like "How many gms of protien I've consumed today" or "How many calories have I burnt this week" and it pulls relevant entries and answers based on your own writing.

## Setup

You need:
- Python 3.10+
- [Ollama](https://ollama.ai) (or an OpenRouter API key)
- A [Qdrant Cloud](https://cloud.qdrant.io) account (free tier is fine)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Qdrant and LLM credentials.

If using Ollama locally:
```bash
ollama pull mistral
ollama serve
```

Then:
```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000.

## How the search works

It uses hybrid retrieval — vector embeddings for semantic search plus BM25 for keyword matching, combined with reciprocal rank fusion. This way it catches both "that thing about my cat" and exact terms like "Kafka consumer offset reset" without one approach drowning out the other.

Embeddings are done locally with sentence-transformers. Everything is stored in Qdrant.

## Multi-tenancy

Each user's data is isolated. Every DB query and vector search filters by user ID from the JWT. Users can't see each other's entries or chat context.
