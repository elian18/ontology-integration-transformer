"""Tests for materializing the modular core (S3-T04).

A synthetic ontology checks the whole contract: the core is made standalone (offenders move
to the profile), the profile owl:imports the core, individuals follow their type, and both
files are valid RDF/XML. One guarded test checks the real OntoPriv produces a self-contained
core with no dangling references."""
from __future__ import annotations
from pathlib import Path

import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import OWL

from src.ingest.ontology_loader import load_ontology
from src.core.emit import (
    materialize_modules, resolve_core_autonomy, CORE_IRI, PROFILE_IRI,
)
from src.core.split import assign_modules, CORE, PROFILE

ROOT = Path(__file__).resolve().parents[1]
ONTOPRIV = ROOT / "data" / "input" / "ontopriv.rdf"

_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix : <http://example.org/o#> .
:Root a owl:Class .
:Principles a owl:Class ; rdfs:subClassOf :Root .
:Minimization a owl:Class ; rdfs:subClassOf :Principles .
:Sanctions a owl:Class ; rdfs:subClassOf :Root .
:Fine a owl:Class ; rdfs:subClassOf :Sanctions .
:has_fine a owl:ObjectProperty ; rdfs:domain :Minimization ; rdfs:range :Fine .
:p_prof a owl:DatatypeProperty ; rdfs:domain :Fine .
:Alice a owl:NamedIndividual , :Minimization .
:Bob a owl:NamedIndividual , :Fine .
"""

_NS = "http://example.org/o#"


@pytest.fixture
def built(tmp_path):
    p = tmp_path / "o.ttl"; p.write_text(_TTL, encoding="utf-8")
    report = load_ontology(p)
    result = materialize_modules(report, out_dir=tmp_path / "out")
    core = Graph(); core.parse(result.core_path, format="xml")
    profile = Graph(); profile.parse(result.profile_path, format="xml")
    return result, core, profile


def _subjects(graph):
    return {str(s) for s in graph.subjects()}


def test_offender_moves_to_profile_and_core_is_standalone(built):
    result, core, profile = built
    assert "has_fine" in result.moved_to_profile      # it referenced Fine (profile)
    assert result.dangling_core_refs == []            # core has no reference into the profile
    assert _NS + "has_fine" not in _subjects(core)
    assert _NS + "has_fine" in _subjects(profile)


def test_core_keeps_general_concepts(built):
    result, core, profile = built
    subj = _subjects(core)
    assert _NS + "Minimization" in subj
    assert _NS + "Principles" in subj
    assert _NS + "Fine" not in subj                   # profile-only class stays out of the core


def test_profile_imports_core(built):
    result, core, profile = built
    assert (URIRef(PROFILE_IRI), OWL.imports, URIRef(CORE_IRI)) in profile


def test_individuals_follow_their_type(built):
    result, core, profile = built
    assert _NS + "Alice" in _subjects(core)           # typed Minimization (core)
    assert _NS + "Bob" in _subjects(profile)          # typed Fine (profile)


def test_both_files_are_valid_rdfxml(built):
    result, core, profile = built
    assert len(core) > 0 and len(profile) > 0
    assert Path(result.manifest_path).exists()


def test_autonomy_move_reaches_a_fixpoint(tmp_path):
    # A core property whose super-property is moved must also resolve (cascade to fixpoint).
    ttl = _TTL + ":sub_fine a owl:ObjectProperty ; rdfs:subPropertyOf :has_fine .\n"
    p = tmp_path / "o.ttl"; p.write_text(ttl, encoding="utf-8")
    report = load_ontology(p)
    split = assign_modules(report)
    module_of = {a.iri: a.module for a in split.assignments}
    final, moved = resolve_core_autonomy(report.graph, module_of)
    # No core entity may still point into the profile.
    from src.core.split import _HARD_PREDICATES
    for s, pr, o in report.graph:
        if pr in _HARD_PREDICATES and isinstance(o, URIRef):
            if final.get(str(s)) == CORE:
                assert final.get(str(o)) != PROFILE


def test_real_ontopriv_core_is_self_contained(tmp_path):
    if not ONTOPRIV.exists():
        pytest.skip("OntoPriv no disponible en data/input/ontopriv.rdf")
    report = load_ontology(ONTOPRIV)
    result = materialize_modules(report, out_dir=tmp_path / "out")
    assert result.dangling_core_refs == []
    core = Graph(); core.parse(result.core_path, format="xml")
    profile = Graph(); profile.parse(result.profile_path, format="xml")
    assert (URIRef(PROFILE_IRI), OWL.imports, URIRef(CORE_IRI)) in profile
    assert len(core) > 0 and len(profile) > 0