"""Tests for the RAG indexing and semantic search (Sprint 2, S2-T05).

Runs against an isolated, throwaway ChromaDB (see conftest). Covers three
things without touching the LLM: the collection gets populated, re-indexing the
same law is idempotent, and a search returns results shaped with number/title.
Article count is checked by the ``<law>-`` id prefix, not the global count, so
the tests stay correct if several laws ever share the ``normativa`` collection.
"""

from pathlib import Path

import pytest

from src.ai.rag import vector_store
from src.ai.rag.indexer import index_law
from src.ai.rag.retriever import retrieve_articles

ROOT = Path(__file__).resolve().parents[1]
LOPDP_PATH = ROOT / "data" / "input" / "lopdp.pdf"
LOPDP_EXPECTED_ARTICLES = 77


def _synthetic_summary(source: str = "ley.txt", n: int = 5) -> dict:
    """A tiny fake segmentation, so the fast tests do not need the real PDF."""
    articles = [
        {
            "number": i,
            "title": f"Titulo {i}",
            "text": f"Art. {i}.- Cuerpo del articulo {i} sobre proteccion de datos.",
            "char_start": 0,
            "char_end": 0,
        }
        for i in range(1, n + 1)
    ]
    return {"source": source, "n_articles": n, "articles": articles}


def _ids_with_prefix(prefix: str) -> list[str]:
    ids = vector_store.collection.get()["ids"]
    return [i for i in ids if i.startswith(prefix)]


@pytest.fixture
def clean_collection():
    """Empty the isolated collection before and after each test."""
    vector_store.reset()
    yield vector_store
    vector_store.reset()


# --------------------------------------------------------------------------- #
# Indexing (fast, synthetic)                                                  #
# --------------------------------------------------------------------------- #

def test_indexing_populates_the_collection(clean_collection):
    report = index_law(_synthetic_summary(n=5))
    assert report["n_indexed"] == 5
    assert len(_ids_with_prefix("ley-art-")) == 5


def test_reindexing_same_law_is_idempotent(clean_collection):
    summary = _synthetic_summary(n=5)
    index_law(summary)
    index_law(summary)
    assert len(_ids_with_prefix("ley-art-")) == 5


def test_reset_empties_the_collection(clean_collection):
    index_law(_synthetic_summary(n=5))
    vector_store.reset()
    assert vector_store.count() == 0


def test_stored_metadata_has_number_and_title(clean_collection):
    index_law(_synthetic_summary(n=3))
    record = vector_store.collection.get(ids=["ley-art-2"], include=["metadatas"])
    metadata = record["metadatas"][0]
    assert metadata["number"] == 2
    assert metadata["title"] == "Titulo 2"


# --------------------------------------------------------------------------- #
# Search (fast, synthetic)                                                    #
# --------------------------------------------------------------------------- #

def test_search_returns_results_shaped_with_number_and_title(clean_collection):
    index_law(_synthetic_summary(n=5))
    results = retrieve_articles("proteccion de datos", top_k=3)
    assert len(results) == 3
    for article in results:
        assert isinstance(article["number"], int)
        assert "title" in article
        assert article["text"].strip() != ""


def test_search_respects_explicit_top_k(clean_collection):
    index_law(_synthetic_summary(n=5))
    assert len(retrieve_articles("articulo", top_k=2)) == 2


def test_search_on_empty_collection_returns_nothing(clean_collection):
    assert retrieve_articles("cualquier cosa", top_k=4) == []


# --------------------------------------------------------------------------- #
# Integration: the real LOPDP                                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def lopdp_summary():
    if not LOPDP_PATH.exists():
        pytest.skip(f"LOPDP no encontrada en {LOPDP_PATH}")
    from src.ingest.legal_segmenter import segment_articles
    from src.ingest.text_loader import load_legal_text
    report = load_legal_text(LOPDP_PATH)
    return segment_articles(report.text, source=report.path)


def test_lopdp_indexes_all_articles(lopdp_summary):
    vector_store.reset()
    index_law(lopdp_summary)
    assert len(_ids_with_prefix("lopdp-art-")) == LOPDP_EXPECTED_ARTICLES
    vector_store.reset()