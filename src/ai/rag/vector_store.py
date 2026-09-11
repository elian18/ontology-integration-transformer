import os
import chromadb

from src.config import load_config

_cfg = load_config().get("vector_store", {}) or {}
# CHROMA_PATH (env) wins so the test suite can isolate ChromaDB in a temp dir.
_PATH = os.getenv("CHROMA_PATH") or _cfg.get("path") or "chroma_db"
_COLLECTION = _cfg.get("collection") or "normativa"

_client = chromadb.PersistentClient(path=_PATH)
collection = _client.get_or_create_collection(_COLLECTION)


def add(ids: list[str], docs: list[str], embeddings: list[list[float]]):
    collection.add(ids=ids, documents=docs, embeddings=embeddings)


def upsert(
    ids: list[str],
    docs: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None,
):
    """Insert or overwrite documents by id, so re-indexing a law is idempotent."""
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)


def count() -> int:
    """Number of documents currently stored in the collection."""
    return collection.count()


def reset():
    """Drop and recreate the collection, to rebuild it from scratch."""
    global collection
    _client.delete_collection(_COLLECTION)
    collection = _client.get_or_create_collection(_COLLECTION)