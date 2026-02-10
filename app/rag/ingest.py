from typing import List

from app.config import settings
from app.rag.chunking import chunk_text
from app.rag.embeddings import Embedder
from app.rag.index import ChunkRecord, FaissIndex


def ingest_text(index: FaissIndex, embedder: Embedder, text: str, source: str) -> int:
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return 0

    vecs = embedder.embed(chunks)
    records: List[ChunkRecord] = []
    for i, ch in enumerate(chunks):
        records.append(ChunkRecord(chunk_id=f"{source}:{i}", source=source, text=ch))
    index.add(vecs, records)
    return len(records)
