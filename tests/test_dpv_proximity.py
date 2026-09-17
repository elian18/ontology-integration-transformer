"""Tests for the DPV-proximity scoring (S3-T02).

The fast tests inject a deterministic fake encoder (keyword axes), so they assert the
scoring/argmax/report logic exactly and never touch the network. One slow test exercises the
real Sprint 2 encoder end-to-end and is excluded by ``-m "not slow"``."""
from __future__ import annotations
import json
import math
from pathlib import Path

import pytest

from src.ingest.ontology_loader import load_ontology
from src.ingest.dpv_loader import load_dpv
from src.core.dpv_proximity import (
    score_against_dpv, proximity_summary, DpvProximityReport,
)

ROOT = Path(__file__).resolve().parents[1]
ONTOPRIV = ROOT / "data" / "input" / "ontopriv.rdf"

_ONTO_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix : <http://example.org/o#> .
:Root a owl:Class .
:Terminology a owl:Class ; rdfs:subClassOf :Root .
:Consent a owl:Class ; rdfs:subClassOf :Terminology .
:Security a owl:Class ; rdfs:subClassOf :Root .
:Personal_data_security a owl:Class ; rdfs:subClassOf :Security .
:Rights a owl:Class ; rdfs:subClassOf :Root .
:Right_of_access a owl:Class ; rdfs:subClassOf :Rights .
"""

_DPV_TTL = """@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dpv: <https://w3id.org/dpv#> .
dpv:Consent a rdfs:Class , skos:Concept ; skos:prefLabel "Consent"@en .
dpv:SecurityMeasure a rdfs:Class , skos:Concept ; skos:prefLabel "Security"@en .
dpv:RightOfAccess a rdfs:Class , skos:Concept ; skos:prefLabel "Right"@en .
"""

_AXES = ["consent", "security", "right", "transfer"]


def _fake_embed(texts):
    """Deterministic encoder: one axis per keyword, L2-normalized, no network."""
    vecs = []
    for t in texts:
        tl = t.lower()
        v = [1.0 if a in tl else 0.0 for a in _AXES]
        v.append(0.0 if any(v) else 1.0)          # 'other' axis avoids zero vectors
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


@pytest.fixture
def tiny(tmp_path):
    op = tmp_path / "onto.ttl"; op.write_text(_ONTO_TTL, encoding="utf-8")
    dp = tmp_path / "dpv.ttl"; dp.write_text(_DPV_TTL, encoding="utf-8")
    return load_ontology(op), load_dpv(dp)


def test_best_match_and_score_are_correct(tiny):
    report, dpv = tiny
    prox = score_against_dpv(report, dpv, embed_fn=_fake_embed, use_context=False)
    by_name = {e.local_name: e for e in prox.entries}
    assert prox.n_dpv_concepts == 3
    assert by_name["Consent"].best_dpv_label == "Consent"
    assert by_name["Consent"].score == pytest.approx(1.0, abs=1e-6)
    assert by_name["Personal_data_security"].best_dpv_label == "Security"
    assert by_name["Right_of_access"].best_dpv_label == "Right"


def test_every_entity_is_scored(tiny):
    report, dpv = tiny
    prox = score_against_dpv(report, dpv, embed_fn=_fake_embed, use_context=False)
    # 4 named classes + 2 families used as anchors = every named class gets a row.
    n_classes = report.n_classes
    assert len([e for e in prox.entries if "clase" in e.kinds]) == n_classes
    for e in prox.entries:
        assert -1.0001 <= e.score <= 1.0001


def test_context_path_runs(tiny):
    report, dpv = tiny
    prox = score_against_dpv(report, dpv, embed_fn=_fake_embed, use_context=True)
    assert len(prox.entries) > 0


def test_summary_is_json_safe(tiny):
    report, dpv = tiny
    prox = score_against_dpv(report, dpv, embed_fn=_fake_embed, use_context=False)
    summary = proximity_summary(prox)
    json.dumps(summary)
    assert summary["n_dpv_concepts"] == 3
    assert isinstance(summary["bands"], dict)
    assert isinstance(summary["family_means"], dict)


def test_empty_dpv_returns_no_entries(tmp_path):
    op = tmp_path / "onto.ttl"; op.write_text(_ONTO_TTL, encoding="utf-8")
    empty = tmp_path / "empty.ttl"
    empty.write_text("@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n", encoding="utf-8")
    report = load_ontology(op)
    dpv = load_dpv(empty)
    prox = score_against_dpv(report, dpv, embed_fn=_fake_embed)
    assert prox.entries == []


@pytest.mark.slow
def test_real_encoder_scores_ontopriv(dpv_report):
    """End-to-end with the real all-MiniLM-L6-v2 encoder (needs the model + OntoPriv)."""
    if not ONTOPRIV.exists():
        pytest.skip("OntoPriv no disponible en data/input/ontopriv.rdf")
    try:
        from src.ai.rag.embeddings import embed  # noqa: F401
    except Exception as exc:                       # pragma: no cover - entorno sin modelo
        pytest.skip(f"encoder no disponible: {exc}")
    report = load_ontology(ONTOPRIV)
    prox = score_against_dpv(report, dpv_report)   # real embed via lazy import
    assert len(prox.entries) > 0
    for e in prox.entries:
        assert -1.0001 <= e.score <= 1.0001
    by_name = {e.local_name: e for e in prox.entries}
    if "Consent" in by_name:                       # exact label match should score high
        assert by_name["Consent"].score >= 0.5
        assert "consent" in (by_name["Consent"].best_dpv_label or "").lower()