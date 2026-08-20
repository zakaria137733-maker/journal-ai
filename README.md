# Journal AI

Private journal app with a chat feature that answers questions about your past entries using RAG.

Write entries about your day, your workouts, your meals — whatever. Then ask things like "How much protein did I eat this week?" or "What was bugging me last Tuesday?" and it pulls relevant entries and answers from your own writing.

[Watch the demo on Loom](https://www.loom.com/share/cf885ffeee774cacb106a51323f34e34)

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

## LLM Abstraction Layer

Both Ollama and OpenRouter implement the OpenAI REST API spec, which means
the same `openai.OpenAI` client works for both — only the `base_url`,
`api_key`, and `model` change. These three values are read from environment
variables, so switching providers requires changing three lines in `.env`
with zero code changes.

To use OpenRouter instead of Ollama, update your `.env`:
```
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your-openrouter-api-key
LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

## How the Search Works

Hybrid retrieval — vector embeddings for semantic search plus BM25 for keyword matching, combined with reciprocal rank fusion. This way it catches both "that thing about my cat" and exact terms like "Kafka consumer offset reset" without one approach drowning out the other.

Embeddings are done locally with sentence-transformers (all-MiniLM-L6-v2). Everything is stored in Qdrant.

## Chunking Strategy

Each journal entry is stored as a single embedding vector. This works well
for short entries but has a known limitation: long entries covering multiple
topics produce averaged vectors that may retrieve poorly for specific
sub-topics. A production improvement would be sentence-level chunking with
parent document retrieval — storing multiple vectors per entry and
re-ranking retrieved chunks before passing to the LLM.

## Multi-tenancy

Each user's data is isolated. Every DB query and vector search filters by user ID from the JWT. Users can't see each other's entries or chat context.
