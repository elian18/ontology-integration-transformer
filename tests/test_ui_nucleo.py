"""Tests for the modular-core UI service (Sprint 3, S3-T06)."""

import pytest

from app.services import nucleo


def test_core_structure_shape():
    structure = nucleo.core_structure()
    if structure is None:
        pytest.skip("OntoPriv base no disponible en data/input/")

    assert set(structure["counts"].keys()) == {"core", "profile"}
    for module in ("core", "profile"):
        c = structure["counts"][module]
        assert c["classes"] > 0 and c["properties"] > 0
        assert c["total"] == c["classes"] + c["properties"]

    # Families land where the confirmed criterion says.
    assert "Principles" in structure["core_families"]
    assert "Verification" in structure["profile_families"]
    assert isinstance(structure["n_cross_refs"], int)
    assert isinstance(structure["n_flagged"], int)


def test_core_and_profile_do_not_share_general_families():
    structure = nucleo.core_structure()
    if structure is None:
        pytest.skip("OntoPriv base no disponible en data/input/")
    # A profile-only family must never appear as a core family.
    assert "Verification" not in structure["core_families"]
    assert "Sanctions" not in structure["core_families"]
    

def test_build_downloads_produces_valid_rdfxml():
    downloads = nucleo.build_downloads()
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