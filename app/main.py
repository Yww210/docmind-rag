from fastapi import FastAPI

from app.rag.pipeline import RAGPipeline
from app.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse

app = FastAPI(title="DocMind-RAG", version="1.0.0")
pipeline = RAGPipeline()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    n = pipeline.ingest_text(req.text, source=req.source or "api")
    return IngestResponse(chunks_indexed=n, source=req.source or "api")

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return pipeline.answer(req.question, use_hyde=req.use_hyde)
