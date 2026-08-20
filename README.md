# Journal AI

Private journal app with a chat feature that answers questions about your past entries using RAG.

Write entries about your day, your workouts, your meals — whatever. Then ask things like "How much protein did I eat this week?" or "What was bugging me last Tuesday?" and it pulls relevant entries and answers from your own writing.

[Video walkthrough](https://drive.google.com/file/d/1gbrsHdX170qX6bh8OUsI92D5QklXjHkP/view)

## Setup

You need:
- Python 3.10+
- [Ollama](https://ollama.ai) (or an OpenRouter API key)
- A [Qdrant Cloud](https://cloud.qdrant.io) account (free tier works fine)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials. For Qdrant, create a free cluster at https://cloud.qdrant.io and copy the cluster URL and API key into your `.env` file.

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

### Docker

```bash
docker compose up --build
```

The compose file mounts `journal.db` into the container so data persists across restarts.

## LLM Providers

Both Ollama and OpenRouter implement the OpenAI `/v1/chat/completions` spec, so the app talks to them the same way. Swapping between them means changing three env vars — no code changes needed:

| Variable | Ollama (default) | OpenRouter |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | `ollama` | your OpenRouter key |
| `LLM_MODEL` | `mistral` | `meta-llama/llama-3.1-8b-instruct:free` |

Both configurations are in `.env.example` — just uncomment the block you want. The `LLMClient` in `app/services/llm.py` handles the rest.

## How the Search Works

Hybrid retrieval — vector embeddings for semantic search plus BM25 for keyword matching, combined with reciprocal rank fusion. This way it catches both "that thing about my cat" and exact terms like "Kafka consumer offset reset" without one approach drowning out the other.

Embeddings are done locally with sentence-transformers (all-MiniLM-L6-v2). Everything is stored in Qdrant.

## Chunking Strategy

Each journal entry is stored as a single vector embedding. The full entry text goes into Qdrant as one point alongside the embedding. This works well for typical journal-length entries (a paragraph to a page), but very long entries may lose some retrieval granularity since the embedding has to represent the entire text as one vector. For a production system you'd probably want to chunk long entries into overlapping segments, but for personal journal entries this keeps things simple without a meaningful hit to retrieval quality.

## Multi-tenancy

Each user's data is isolated. Every DB query and vector search filters by user ID from the JWT. Users can't see each other's entries or chat context.
