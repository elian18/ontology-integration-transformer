"""Tests for the per-entity ontology inventory (S3-T01).

Two things are checked: that the inventory reconciles exactly with Sprint 1's totals on
OntoPriv, and that the family/hierarchy logic generalizes to arbitrary ontologies (one root
and several roots), so the modular split downstream never depends on OntoPriv's shape."""
from __future__ import annotations
from pathlib import Path

import pytest

from src.ingest.ontology_loader import load_ontology, characterization_summary
from src.core.inventory import (
    build_inventory, inventory_summary,
    KIND_CLASS, KIND_OBJECT_PROP, KIND_DATA_PROP,
)

ROOT = Path(__file__).resolve().parents[1]
ONTOPRIV = ROOT / "data" / "input" / "ontopriv.rdf"

_PREFIXES = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix : <http://example.org/o#> .
"""

SINGLE_ROOT = _PREFIXES + """
:Root a owl:Class .
:A a owl:Class ; rdfs:subClassOf :Root .
:B a owl:Class ; rdfs:subClassOf :Root .
:A1 a owl:Class ; rdfs:subClassOf :A .
:p a owl:ObjectProperty ; rdfs:domain :A ; rdfs:range :B .
:d a owl:DatatypeProperty ; rdfs:domain :B .
"""

MULTI_ROOT = _PREFIXES + """
:X a owl:Class .
:Y a owl:Class .
:X1 a owl:Class ; rdfs:subClassOf :X .
"""

PUNNED = _PREFIXES + """
:Root a owl:Class .
:Fam a owl:Class ; rdfs:subClassOf :Root .
:Z a owl:Class , owl:ObjectProperty , owl:DatatypeProperty ; rdfs:subClassOf :Fam .
"""


def _load_ttl(tmp_path, text, name="o.ttl"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return load_ontology(p)


def test_inventory_reconciles_with_sprint1_on_ontopriv():
    if not ONTOPRIV.exists():
        pytest.skip("OntoPriv no disponible en data/input/ontopriv.rdf")
    report = load_ontology(ONTOPRIV)
    inv = build_inventory(report)
    s1 = characterization_summary(report)["structure"]
    for key in ("triples", "classes", "object_properties", "data_properties", "individuals"):
        assert inv.totals[key] == s1[key], f"desajuste en {key}"
    # OntoPriv has no anonymous classes, so every class becomes a listed entry.
    assert len(inv.entries_of_kind(KIND_CLASS)) == s1["classes"]
    assert inv.roots == ["Organic_Law_on_Personal_Data_Protection"]
    assert "Members_of_the_personal_data_protection_system" in inv.families


def test_single_root_uses_direct_children_as_families(tmp_path):
    inv = build_inventory(_load_ttl(tmp_path, SINGLE_ROOT))
    assert inv.roots == ["Root"]
    assert set(inv.families) == {"A", "B"}
    by_local = {e.local_name: e for e in inv.entries}
    assert by_local["A1"].family == "A"          # deep class maps to its family anchor
    assert by_local["A1"].depth == 2             # Root=0, A=1, A1=2
    assert by_local["A"].depth == 1
    assert by_local["p"].family == "A"           # property family taken from its domain
    assert by_local["p"].range == "B"
    assert by_local["d"].family == "B"


def test_multiple_roots_become_families(tmp_path):
    inv = build_inventory(_load_ttl(tmp_path, MULTI_ROOT))
    assert set(inv.roots) == {"X", "Y"}
    assert set(inv.families) == {"X", "Y"}
    by_local = {e.local_name: e for e in inv.entries}
    assert by_local["X1"].family == "X"


def test_punned_entity_keeps_all_kinds(tmp_path):
    inv = build_inventory(_load_ttl(tmp_path, PUNNED))
    z = next(e for e in inv.entries if e.local_name == "Z")
    assert set(z.kinds) == {KIND_CLASS, KIND_OBJECT_PROP, KIND_DATA_PROP}
    # It is a class, so it keeps its class family (not treated as a domain-less property).
    assert z.family == "Fam"


def test_summary_is_json_safe(tmp_path):
    import json
    inv = build_inventory(_load_ttl(tmp_path, SINGLE_ROOT))
    summary = inventory_summary(inv)
    json.dumps(summary)  # must not raise
    assert summary["totals"]["classes"] == 4
    assert set(summary["by_namespace"].keys()) == {"http://example.org/o#"}