import os
import chromadb

_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
collection = _client.get_or_create_collection("normativa")   # se poblará en el Sprint 2

def add(ids: list[str], docs: list[str], embeddings: list[list[float]]):
    collection.add(ids=ids, documents=docs, embeddings=embeddings)