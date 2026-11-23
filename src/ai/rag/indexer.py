from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from .embeddings import embed
from .vector_store import count, upsert

if TYPE_CHECKING:
    from src.ingest.legal_segmenter import SegmentationSummary


class IndexReport(TypedDict):
    source: str
    slug: str
    n_indexed: int
    collection_count: int


def _slug(source: str) -> str:
    """Stable, filesystem-free label for a law, used as the id prefix."""
    stem = Path(source).stem if source else "law"
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "law"


def index_law(summary: "SegmentationSummary") -> IndexReport:
    """Embed and store every article of a segmented law in ChromaDB.

    Args:
        summary: Output of ``segment_articles`` (source + list of articles).

    Returns:
        A small report with the id prefix, how many articles were indexed and
        the total number of documents in the collection afterwards.
    """
    articles = summary["articles"]
    source = summary.get("source", "")
    slug = _slug(source)

    ids = [f"{slug}-art-{a['number']}" for a in articles]
    docs = [a["text"] for a in articles]
    metadatas = [
        {
            "number": a["number"],
            "title": a["title"],
            "source": source,
            "char_start": a["char_start"],
            "char_end": a["char_end"],
        }
        for a in articles
    ]

    if docs:
        embeddings = embed(docs)
        upsert(ids=ids, docs=docs, embeddings=embeddings, metadatas=metadatas)

    return {
        "source": source,
        "slug": slug,
        "n_indexed": len(ids),
        "collection_count": count(),
    }