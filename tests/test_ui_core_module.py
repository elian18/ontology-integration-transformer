"""Tests for the modular-core UI service (Sprint 3, S3-T06/T07/T08)."""

import pytest

from app.services import core_module


def test_core_structure_shape():
    structure = core_module.core_structure()
    if structure is None:
        pytest.skip("OntoPriv base no disponible en data/input/")

    assert set(structure["counts"].keys()) == {"core", "profile"}
    for module in ("core", "profile"):
        c = structure["counts"][module]
        assert c["classes"] > 0 and c["properties"] > 0
        assert c["total"] == c["classes"] + c["properties"]

    assert "Principles" in structure["core_families"]
    assert "Verification" in structure["profile_families"]
    assert isinstance(structure["n_cross_refs"], int)
    assert isinstance(structure["n_flagged"], int)


def test_core_and_profile_do_not_share_general_families():
    structure = core_module.core_structure()
    if structure is None:
        pytest.skip("OntoPriv base no disponible en data/input/")
    assert "Verification" not in structure["core_families"]
    assert "Sanctions" not in structure["core_families"]


def test_build_downloads_produces_valid_rdfxml():
    downloads = core_module.build_downloads()
    if downloads is None:
        pytest.skip("OntoPriv base no disponible en data/input/")
    from rdflib import Graph
    for module in ("core", "profile"):
        info = downloads[module]
        assert info["bytes"] and info["filename"].endswith(".rdf")
        g = Graph()
        g.parse(data=info["bytes"], format="xml")
        assert len(g) == info["triples"] and info["triples"] > 0
    assert isinstance(downloads["moved_to_profile"], list)


def test_proposed_concepts_reads_the_file(tmp_path):
    import json
    payload = {
        "source": "lopdp",
        "counts": {"concepts_proposed": 2, "classes": 1, "properties": 1, "articles_processed": 3},
        "quota_exhausted": False, "processed_articles": [1, 2, 3],
        "concepts": [
            {"name": "DataSubject", "label": "titular", "type": "class", "definition": "d", "articles": [1, 4]},
            {"name": "has_x", "label": "tiene", "type": "property", "definition": "d2", "articles": [2]},
        ],
    }
    f = tmp_path / "props.json"
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    result = core_module.proposed_concepts(path=f)
    assert result["counts"]["concepts_proposed"] == 2
    assert result["concepts"][0]["articles"] == [1, 4]
    assert result["quota_exhausted"] is False


def test_proposed_concepts_missing_returns_none(tmp_path):
    assert core_module.proposed_concepts(path=tmp_path / "no_existe.json") is None