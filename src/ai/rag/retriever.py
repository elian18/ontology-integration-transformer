from __future__ import annotations

from typing import TypedDict

from src.config import load_config

from . import vector_store
from .embeddings import embed


class RetrievedArticle(TypedDict):
    number: int
    title: str
    source: str
    text: str
    distance: float


def _default_top_k() -> int:
    return int(load_config()["rag"]["top_k"])


def retrieve_articles(query: str, top_k: int | None = None) -> list[RetrievedArticle]:
    """Return the articles most similar in meaning to ``query``.

    Args:
        query: Free text to search for.
        top_k: How many articles to return. Defaults to rag.top_k from config.

    Returns:
        A list of articles (number, title, source, text and distance), ordered
        from most to least similar.
    """
    k = top_k if top_k is not None else _default_top_k()
    query_embedding = embed([query])[0]

    # Read the collection from the module (not a bound name) so it stays valid
    # even after vector_store.reset() recreates it.
    result = vector_store.collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []

    articles: list[RetrievedArticle] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        articles.append({
            "number": metadata.get("number"),
            "title": metadata.get("title", ""),
            "source": metadata.get("source", ""),
            "text": document,
            "distance": distance,
        })
    return articles


def retrieve(query: str, top_k: int | None = None) -> list[str]:
    """Backwards-compatible helper: only the article texts."""
    return [article["text"] for article in retrieve_articles(query, top_k)]