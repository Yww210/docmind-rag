from dataclasses import dataclass
from typing import List, Tuple
import os, json
import numpy as np
import faiss

@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    text: str

class FaissIndex:
    def __init__(self, dim: int, index_dir: str):
        self.dim = dim
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        self.index_path = os.path.join(index_dir, "vectors.faiss")
        self.meta_path = os.path.join(index_dir, "meta.jsonl")

        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[ChunkRecord] = []

        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            self.meta = []
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self.meta.append(ChunkRecord(**obj))

    def persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for rec in self.meta:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    def add(self, vectors: np.ndarray, records: List[ChunkRecord]):
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)
        self.meta.extend(records)
        self.persist()

    def search(self, qvec: np.ndarray, top_k: int) -> List[Tuple[ChunkRecord, float]]:
        if qvec.ndim == 1:
            qvec = qvec.reshape(1, -1)
        scores, ids = self.index.search(qvec.astype(np.float32), top_k)
        results = []
        for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
            if idx == -1:
                continue
            if idx < len(self.meta):
                results.append((self.meta[idx], float(score)))
        return results
