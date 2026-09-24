"""S4-T04: lexical similarity (identifiers only, never the Spanish labels)."""
import numpy as np

from src.ingest.dpv_loader import load_dpv
from src.alignment.sources import load_ai_concepts, AlignmentSources
from src.alignment.dpv_targets import build_dpv_targets, TARGET_PROPERTY
from src.alignment.lexical import (
    normalize_for_lexical, lexical_similarity, lexical_matrix, source_lexical_text,
    score_lexical, best_compatible,
)

_DPV = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dpv: <https://w3id.org/dpv#> .
dpv:DataSubject a rdfs:Class , skos:Concept ; skos:prefLabel "Data Subject"@en .
dpv:DataController a rdfs:Class , skos:Concept ; skos:prefLabel "Data Controller"@en .
dpv:SpecialCategoryPersonalData a rdfs:Class , skos:Concept ;
    skos:prefLabel "Special Category Personal Data"@en .
dpv:hasDataSubject a rdf:Property , skos:Concept ; skos:prefLabel "has data subject"@en .
dpv:hasRecipient a rdf:Property , skos:Concept ; skos:prefLabel "has recipient"@en .
"""


def _targets(tmp_path):
    f = tmp_path / "mini-dpv.ttl"
    f.write_text(_DPV, encoding="utf-8")
    return build_dpv_targets(load_dpv(f))


def _sources(concepts):
    ai = load_ai_concepts({"concepts": concepts})
    return AlignmentSources("memoria", None, concepts=ai, proposals_found=True)


def test_normalization():
    assert normalize_for_lexical("SpecialCategoriesOfPersonalData") == "special category personal data"
    assert normalize_for_lexical("Consent_in_verification") == "consent verification"
    assert normalize_for_lexical("análisis_de_riesgo") == "analisis riesgo"
    assert normalize_for_lexical("hasRecipient") == "has recipient"
    assert normalize_for_lexical("") == "" and normalize_for_lexical(None) == ""


def test_equal_and_near_equal_names_score_high():
    assert lexical_similarity("DataSubject", "Data Subject") == 1.0
    assert lexical_similarity("SpecialCategoriesOfPersonalData",
                              "Special Category Personal Data") == 1.0
    assert lexical_similarity("PersonalDataSensitive", "Sensitive Personal Data") == 1.0  # order
    assert lexical_similarity("DataControllers", "Data Controller") == 1.0               # plural


def test_partial_is_partial_and_different_is_low():
    partial = lexical_similarity("Consent_in_verification", "Consent")
    assert 0.3 < partial < 0.8
    assert lexical_similarity("Banking", "Data Controller") < 0.3
    assert lexical_similarity("", "Data Subject") == 0.0


def test_spanish_label_never_enters_the_lexical_match():
    src = _sources([
        {"name": "DataSubject", "label": "titular", "type": "class"},
        {"name": "Holder", "label": "Data Subject", "type": "class"},   # label would match, name not
    ])
    assert source_lexical_text(src.concepts[0]) == "Data Subject"
    assert source_lexical_text(src.concepts[1]) == "Holder"
    assert lexical_similarity(source_lexical_text(src.concepts[1]), "Data Subject") < 0.5


def test_matrix_shape_order_and_empty_rows():
    m = lexical_matrix(["DataSubject", "", "hasRecipient"], ["Data Subject", "has recipient"])
    assert m.shape == (3, 2)
    assert m[0, 0] == 1.0 and m[2, 1] == 1.0
    assert np.all(m[1] == 0.0)
    assert lexical_matrix([], ["x"]).shape == (0, 1)


def test_score_lexical_and_kind_compatibility(tmp_path):
    targets = _targets(tmp_path)
    src = _sources([
        {"name": "DataSubject", "label": "titular", "type": "class"},
        {"name": "DataSubject2", "label": "tiene titular", "type": "property"},
    ])
    scores = score_lexical(src, targets)
    assert scores.matrix.shape == (2, len(targets.targets))
    assert scores.score("ai:DataSubject", "https://w3id.org/dpv#DataSubject") == 1.0

    best = {r["name"]: r for r in best_compatible(src, targets, scores)}
    assert best["DataSubject"]["dpv_label"] == "Data Subject"
    # A property may only match DPV properties, even if a DPV class is spelled closer.
    prop_best = best["DataSubject2"]["dpv_iri"]
    assert next(t for t in targets.targets if t.iri == prop_best).kind == TARGET_PROPERTY
    assert best["DataSubject2"]["dpv_label"] == "has data subject"


def test_top_respects_allowed(tmp_path):
    targets = _targets(tmp_path)
    src = _sources([{"name": "DataSubject", "type": "class"}])
    scores = score_lexical(src, targets)
    j_ds = scores.target_iris.index("https://w3id.org/dpv#DataSubject")
    assert scores.top(0, k=1)[0][0] == j_ds
    others = set(range(len(targets.targets))) - {j_ds}
    assert scores.top(0, k=1, allowed=others)[0][0] != j_ds
    assert len(scores.top(0, k=3)) == 3