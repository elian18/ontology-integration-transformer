import os
import chromadb

_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
collection = _client.get_or_create_collection("normativa")   # poblada en el Sprint 2


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
    _client.delete_collection("normativa")
    collection = _client.get_or_create_collection("normativa")