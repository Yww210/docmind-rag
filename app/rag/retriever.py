from typing import List, Tuple

from app.config import settings
from app.rag.embeddings import Embedder
from app.rag.index import ChunkRecord, FaissIndex


class Retriever:
    def __init__(self, index: FaissIndex, embedder: Embedder):
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[ChunkRecord, float]]:
        k = top_k or settings.top_k
        qvec = self.embedder.embed([query])[0]
        return self.index.search(qvec, k)
