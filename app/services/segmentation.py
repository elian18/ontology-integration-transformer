"""Adapter between the UI and the article segmenter.

The web view asks this service for a law split into articles, ready to render.
It reuses the same loaders and segmenter as the rest of the project (single
source of truth) and returns a small, display-friendly dict. It does NOT touch
ChromaDB: indexing is a back-end concern (the CLI), listing is all the view
needs, so the page stays fast and never loads the embedding model.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.config import load_config
from src.ingest.legal_segmenter import segment_articles
from src.ingest.text_loader import load_legal_text

ROOT = Path(__file__).resolve().parents[2]


def _persist(name: str, data: bytes) -> Path:
    suffix = Path(name).suffix or ".dat"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


def _short_title(title: str, limit: int = 90) -> str:
    """Caption for the list: the part before the body, trimmed for display."""
    head = title.split(".-", 1)[0].strip()
    if not head or len(head) > limit:
        head = title[:limit].rstrip()
    return head


def _preview(text: str, limit: int = 240) -> str:
    """One-line preview of the article body (whitespace collapsed)."""
    snippet = " ".join(text.split())
    return snippet if len(snippet) <= limit else snippet[:limit].rstrip() + "\u2026"


def _to_view(summary: dict) -> dict:
    """Turn a raw segmentation summary into what the view renders."""
    articles = [
        {
            "number": a["number"],
            "title": _short_title(a["title"]),
            "preview": _preview(a["text"]),
            "text": a["text"],
        }
        for a in summary["articles"]
    ]
    return {
        "source": summary["source"],
        "n_articles": summary["n_articles"],
        "articles": articles,
    }


def segment_uploaded_law(name: str, data: bytes) -> dict:
    """Segment a law uploaded by the user (bytes of a .txt or .pdf)."""
    report = load_legal_text(_persist(name, data))
    summary = segment_articles(report.text, source=name)
    return _to_view(summary)


def segment_base_law() -> dict | None:
    """Segment the project's base law (LOPDP) from the path in config.yaml.

    Returns None if the base law is not present on disk.
    """
    inputs = (load_config() or {}).get("inputs", {})
    path = Path(inputs.get("legal_text", "data/input/lopdp.pdf"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    report = load_legal_text(path)
    summary = segment_articles(report.text, source=report.path)
    return _to_view(summary)