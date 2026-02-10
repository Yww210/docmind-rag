from typing import List
import numpy as np
from app.config import settings

class Embedder:
    def __init__(self):
        self.backend = settings.embedding_backend
        if self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.embedding_model)
        else:
            raise ValueError(f"Unsupported EMBEDDING_BACKEND={self.backend}")

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
