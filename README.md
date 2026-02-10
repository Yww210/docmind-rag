# DocMind-RAG — Modular RAG System (FAISS + HyDE + FastAPI + Docker)

A production-style, end-to-end Retrieval-Augmented Generation (RAG) system:
- Document ingestion (folder ingest)
- Chunking + embeddings
- FAISS vector search
- Optional HyDE retrieval boost
- LLM answer generation
- Offline evaluation (latency + text metrics)
- FastAPI service + Docker + CI

## Features
- **Modular architecture**: swap embeddings/LLMs easily.
- **FAISS indexing**: fast local vector search.
- **HyDE (optional)**: generate hypothetical answer/doc to improve recall.
- **Evaluation**: runs on JSONL Q/A pairs.
- **Deployment**: FastAPI + Docker + docker-compose.
- **CI**: lint + basic unit tests.

## Quickstart (Local)
### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) Ingest sample docs
```bash
python scripts/ingest_folder.py --path data/samples
```

### 3) Run API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Query
```bash
curl -X POST "http://localhost:8000/query"   -H "Content-Type: application/json"   -d '{"question":"Summarize the key points in demo_doc_1.", "use_hyde": true}'
```

## Docker
```bash
cp .env.example .env
docker compose up --build
```

## API Endpoints
- `GET /health`
- `POST /ingest`  (ingest raw text)
- `POST /query`   (ask question)

## Evaluation
```bash
python scripts/run_eval.py --data data/eval/qa.jsonl --use_hyde true
```

Outputs:
- avg latency
- simple overlap score (lightweight, dependency-free)

## Notes on Keys
This repo supports either:
- OpenAI-compatible API (set `OPENAI_API_KEY`), OR
- Local fallback "mock LLM" mode for demos.

See `.env.example`.

## License
MIT
