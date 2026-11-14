"""Tests for the per-entity ontology inventory (S3-T01).

Two things are checked: that the inventory reconciles exactly with Sprint 1's totals on
OntoPriv, and that the family/hierarchy logic generalizes to arbitrary ontologies (one root
and several roots), so the modular split downstream never depends on OntoPriv's shape."""
from __future__ import annotations
from pathlib import Path

import pytest

from src.ingest.ontology_loader import (characterization_summary, load_ontology,
                                       misspelled_properties)
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

RDFS_ONLY = _PREFIXES + """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
:A a rdfs:Class .
:B a rdfs:Class .
:links a rdf:Property ; rdfs:domain :A ; rdfs:range :B .
:named a rdf:Property ; rdfs:range xsd:string .
:says a rdf:Property ; rdfs:range rdfs:Literal .
:hasStatus a rdfs:Property ; rdfs:range :B .
:both a owl:ObjectProperty, rdf:Property .
"""


def test_a_property_declared_with_rdf_property_counts(tmp_path):
    """S10-T17: DPCat declares its properties with rdfs:Property (a typo of rdf:Property);
    the DPV declares its own with rdf:Property. Both are properties now, of data when their
    range is a literal; the OWL flavor still only looks at what is typed in OWL."""
    report = _load_ttl(tmp_path, RDFS_ONLY)
    inv = build_inventory(report)
    kinds = {e.local_name: e.kinds for e in inv.entries}
    assert kinds == {"A": (KIND_CLASS,), "B": (KIND_CLASS,), "links": (KIND_OBJECT_PROP,),
                     "named": (KIND_DATA_PROP,), "says": (KIND_DATA_PROP,),
                     "hasStatus": (KIND_OBJECT_PROP,), "both": (KIND_OBJECT_PROP,)}
    by_local = {e.local_name: e for e in inv.entries}
    assert (by_local["links"].family, by_local["links"].range) == ("A", "B")
    s = characterization_summary(report)
    assert {k: s["structure"][k] for k in ("classes", "object_properties", "data_properties")} \
        == {"classes": 2, "object_properties": 3, "data_properties": 2}
    assert all(inv.totals[k] == s["structure"][k] for k in s["structure"])
    assert report.ontology_flavor == "dl-compatible"         # :both is an owl:ObjectProperty
    assert misspelled_properties(report.graph) == ["hasStatus"]
    rdfs = _load_ttl(tmp_path, RDFS_ONLY.replace(":both a owl:ObjectProperty, rdf:Property .",
                                                 ""), "rdfs.ttl")
    assert rdfs.ontology_flavor == "rdfs"
    assert misspelled_properties(_load_ttl(tmp_path, SINGLE_ROOT, "s.ttl").graph) == []
    punned = _load_ttl(tmp_path, _PREFIXES + ":C a rdfs:Class, rdfs:Property .", "p.ttl")
    assert (punned.n_class_prop_overlap, punned.ontology_flavor) == (1, "owl-full")
