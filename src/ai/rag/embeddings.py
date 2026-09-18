import os

import src.config  # noqa: F401  (loads .env before EMBEDDING_MODEL is read)
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))


def embed(texts: list[str]) -> list[list[float]]:
    return _model.encode(texts, normalize_embeddings=True).tolist()