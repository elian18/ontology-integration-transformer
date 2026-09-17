"""Tests for the core/profile split (S3-T03).

Synthetic ontologies (using OntoPriv's real family names) exercise every rule: family-based
assignment, the verification-branch refinement, property-follows-domain, the standalone-core
cross-reference detector, and the advisory DPV flags. One guarded test checks the split
reconciles on the real OntoPriv."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from src.ingest.ontology_loader import load_ontology
from src.core.split import (
    assign_modules, split_summary, CORE, PROFILE,
)
from src.core.dpv_proximity import DpvProximityReport, ProximityEntry

ROOT = Path(__file__).resolve().parents[1]
ONTOPRIV = ROOT / "data" / "input" / "ontopriv.rdf"

_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix : <http://example.org/o#> .
:Root a owl:Class .
:Principles a owl:Class ; rdfs:subClassOf :Root .
:Minimization a owl:Class ; rdfs:subClassOf :Principles .
:Rights a owl:Class ; rdfs:subClassOf :Root .
:Rights_in_verification a owl:Class ; rdfs:subClassOf :Rights .
:Right_X a owl:Class ; rdfs:subClassOf :Rights_in_verification .
:Sanctions a owl:Class ; rdfs:subClassOf :Root .
:Fine a owl:Class ; rdfs:subClassOf :Sanctions .
:has_fine a owl:ObjectProperty ; rdfs:domain :Minimization ; rdfs:range :Fine .
:p_prof a owl:DatatypeProperty ; rdfs:domain :Fine .
:orphan a owl:ObjectProperty .
"""


@pytest.fixture
def split(tmp_path):
    p = tmp_path / "o.ttl"; p.write_text(_TTL, encoding="utf-8")
    return assign_modules(load_ontology(p))


def _by_name(split):
    return {a.local_name: a for a in split.assignments}


def test_core_family_goes_to_core(split):
    assert _by_name(split)["Minimization"].module == CORE


def test_profile_family_goes_to_profile(split):
    assert _by_name(split)["Fine"].module == PROFILE


def test_verification_branch_overrides_core_family(split):
    by = _by_name(split)
    assert by["Rights_in_verification"].module == PROFILE
    assert by["Right_X"].module == PROFILE          # descends from a verification node
    assert "verificacion" in by["Right_X"].reason


def test_property_follows_its_domain(split):
    by = _by_name(split)
    assert by["p_prof"].module == PROFILE            # domain Fine is profile
    assert by["has_fine"].module == CORE             # domain Minimization is core


def test_domainless_property_defaults_to_core_and_is_flagged(split):
    orphan = _by_name(split)["orphan"]
    assert orphan.module == CORE
    assert any("sin dominio" in f for f in orphan.flags)


def test_core_to_profile_reference_is_detected(split):
    # has_fine (core) has range Fine (profile) -> the core would not be standalone.
    refs = {(r["subject"], r["predicate"], r["object"]) for r in split.cross_refs}
    assert ("has_fine", "range", "Fine") in refs


def test_dpv_scores_raise_advisory_flags(tmp_path):
    p = tmp_path / "o.ttl"; p.write_text(_TTL, encoding="utf-8")
    report = load_ontology(p)
    # Fake proximity: a profile class scoring very high should be flagged for review.
    prox = DpvProximityReport(
        onto_path=str(p), dpv_path="x", n_dpv_concepts=1,
        entries=[ProximityEntry(
            iri="http://example.org/o#Fine", local_name="Fine",
            namespace="http://example.org/o#", kinds=("clase",), family="Sanctions",
            best_dpv_iri="d", best_dpv_label="Penalty", score=0.90)],
    )
    split = assign_modules(report, proximity=prox)
    fine = _by_name(split)["Fine"]
    assert fine.dpv_score == pytest.approx(0.90)
    assert any("cercania DPV alta" in f for f in fine.flags)


def test_summary_is_json_safe(split):
    summary = split_summary(split)
    json.dumps(summary)
    assert summary["counts"][CORE]["total"] >= 1
    assert summary["counts"][PROFILE]["total"] >= 1


def test_split_reconciles_on_ontopriv():
    if not ONTOPRIV.exists():
        pytest.skip("OntoPriv no disponible en data/input/ontopriv.rdf")
    from src.core.inventory import build_inventory
    report = load_ontology(ONTOPRIV)
    inv = build_inventory(report)
    split = assign_modules(report, inventory=inv)
    total = len(split.module(CORE)) + len(split.module(PROFILE))
    # Every named entity lands in exactly one module (punning is not double-counted).
    assert total == inv.totals["named_entities"]
    assert len(split.module(CORE)) > 0 and len(split.module(PROFILE)) > 0