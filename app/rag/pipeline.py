from typing import List

from app.config import settings
from app.rag.embeddings import Embedder
from app.rag.index import FaissIndex
from app.rag.ingest import ingest_text
from app.rag.llm import LLM
from app.rag.retriever import Retriever
from app.schemas import QueryResponse, RetrievedChunk


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        dim = self.embedder.embed(["dim_probe"]).shape[1]
        self.index = FaissIndex(dim=dim, index_dir=settings.index_dir)
        self.retriever = Retriever(self.index, self.embedder)
        self.llm = LLM()

    def ingest_text(self, text: str, source: str = "api") -> int:
        return ingest_text(self.index, self.embedder, text, source)

    def _hyde(self, question: str) -> str:
        prompt = (
            "Generate a short passage that would likely appear in relevant documents. "
            "Do NOT mention that it is hypothetical.\n\nQuestion:\n"
            f"{question}\n\nPassage:"
        )
        return self.llm.generate(prompt)

    def answer(self, question: str, use_hyde: bool = False) -> QueryResponse:
        query_for_retrieval = self._hyde(question) if use_hyde else question
        retrieved = self.retriever.retrieve(query_for_retrieval, top_k=settings.top_k)

        context_blocks: List[str] = []
        retrieved_items: List[RetrievedChunk] = []
        for rec, score in retrieved:
            context_blocks.append(f"[{rec.source} | {rec.chunk_id}]\n{rec.text}")
            retrieved_items.append(
                RetrievedChunk(source=rec.source, chunk_id=rec.chunk_id, score=score, text=rec.text)
            )

        context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

        prompt = (
            "Answer the question using ONLY the provided context. "
            "If the context is insufficient, say what is missing.\n\n"
            f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
        )
        ans = self.llm.generate(prompt)
        return QueryResponse(answer=ans, retrieved=retrieved_items)
