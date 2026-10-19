"""Tests for ChromaDB collections by name (S7-T01).

Sprint 2 had ONE fixed collection ("normativa"). These tests check the generalization without
breaking it: the default collection is still "normativa", a named collection never touches it,
each folder (workspace) has its own database, and the new collections can use cosine distance
and carry their manifest. The isolated, throwaway ChromaDB of conftest is used throughout; the
"normativa" collection is only read here, never written, so test_rag/test_smoke keep their own
state.
"""
import pytest

from src.ai.rag import vector_store


@pytest.fixture
def named():
    """Two empty named collections in the (isolated) base database, removed afterwards."""
    names = ("s7_test_a", "s7_test_b")
    for n in names:
        vector_store.reset(name=n)
    yield names
    for n in names:
        vector_store.client().delete_collection(n)


def test_default_collection_is_still_normativa():
    assert vector_store.default_name() == "normativa"
    assert vector_store.collection.name == "normativa"


def test_named_collection_does_not_touch_the_default(named):
    a, _ = named
    before = vector_store.count()
    vector_store.upsert(ids=["x1"], docs=["dato"], embeddings=[[1.0, 0.0]], name=a)
    assert vector_store.count(name=a) == 1
    assert vector_store.count() == before


def test_reset_by_name_only_empties_that_collection(named):
    a, b = named
    vector_store.upsert(ids=["x1"], docs=["dato"], embeddings=[[1.0, 0.0]], name=a)
    vector_store.upsert(ids=["y1"], docs=["otro"], embeddings=[[0.0, 1.0]], name=b)
    vector_store.reset(name=a)
    assert vector_store.count(name=a) == 0
    assert vector_store.count(name=b) == 1


def test_missing_collection_counts_zero_and_is_not_created():
    assert vector_store.count(name="s7_never_created") == 0
    assert not vector_store.exists("s7_never_created")
    assert vector_store.metadata("s7_never_created") == {}


def test_space_and_manifest_are_set_when_created(named):
    a, _ = named
    vector_store.reset(name=a, space=vector_store.SPACE_COSINE,
                       metadata={"items": 2, "fingerprint": "abc"})
    meta = vector_store.metadata(a)
    assert meta["hnsw:space"] == "cosine"
    assert meta["items"] == 2 and meta["fingerprint"] == "abc"


def test_cosine_query_orders_by_similarity_and_filters_by_metadata(named):
    a, _ = named
    vector_store.reset(name=a, space=vector_store.SPACE_COSINE)
    vector_store.upsert(ids=["a", "b", "c"], docs=["a", "b", "c"],
                        embeddings=[[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]],
                        metadatas=[{"kind": "class"}, {"kind": "property"}, {"kind": "class"}],
                        name=a)
    r = vector_store.query([[1.0, 0.0]], 3, name=a)
    assert r["ids"][0] == ["a", "b", "c"]
    assert r["distances"][0][0] == pytest.approx(0.0, abs=1e-5)
    assert r["distances"][0][1] == pytest.approx(0.2, abs=1e-3)      # 1 - cos = 1 - 0.8
    only_properties = vector_store.query([[1.0, 0.0]], 3, name=a, where={"kind": "property"})
    assert only_properties["ids"][0] == ["b"]


def test_get_reads_by_id_with_embeddings(named):
    a, _ = named
    vector_store.upsert(ids=["x1"], docs=["dato"], embeddings=[[0.6, 0.8]], name=a)
    got = vector_store.get(["x1"], name=a, include=("documents", "embeddings"))
    assert got["documents"] == ["dato"]
    assert list(got["embeddings"][0]) == pytest.approx([0.6, 0.8])


def test_each_folder_has_its_own_database(tmp_path):
    ws_a, ws_b = tmp_path / "ws_a", tmp_path / "ws_b"
    vector_store.upsert(ids=["1"], docs=["art"], embeddings=[[1.0, 0.0]], name="law", path=ws_a)
    assert vector_store.count(name="law", path=ws_a) == 1
    assert vector_store.count(name="law", path=ws_b) == 0
    assert not vector_store.exists("law")                 # the base database is untouched
    assert vector_store.client_path(ws_a) == str(ws_a)