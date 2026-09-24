"""S4-T05: combined ranking (lexical + embeddings), top-k and kind compatibility."""
import zlib

import numpy as np
import pytest

from src.ingest.dpv_loader import load_dpv
from src.alignment.sources import load_ai_concepts, AlignmentSources, SourceConcept, ORIGIN_ONTOLOGY
from src.alignment.dpv_targets import build_dpv_targets, TARGET_CLASS, TARGET_PROPERTY
from src.alignment.lexical import normalize_for_lexical, score_lexical
from src.alignment.candidates import (
    alignment_settings, AlignmentSettings, semantic_matrix, rank_candidates, ranking_summary,
)

_DPV = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dpv: <https://w3id.org/dpv#> .
dpv:DataSubject a rdfs:Class , skos:Concept ; skos:prefLabel "Data Subject"@en ;
    skos:definition "The individual whose personal data is being processed."@en .
dpv:DataController a rdfs:Class , skos:Concept ; skos:prefLabel "Data Controller"@en ;
    skos:definition "The individual or organisation that decides the purposes of processing."@en .
dpv:PersonalData a rdfs:Class , skos:Concept ; skos:prefLabel "Personal Data"@en ;
    skos:definition "Data directly or indirectly associated with an individual."@en .
dpv:SensitivePersonalData a rdfs:Class , skos:Concept ; skos:prefLabel "Sensitive Personal Data"@en ;
    skos:definition "Personal data considered sensitive."@en ; skos:broader dpv:PersonalData .
dpv:hasDataSubject a rdf:Property , skos:Concept ; skos:prefLabel "has data subject"@en ;
    skos:definition "Indicates the data subject."@en .
dpv:hasRecipient a rdf:Property , skos:Concept ; skos:prefLabel "has recipient"@en ;
    skos:definition "Indicates the recipient of data."@en .
"""

SETTINGS = AlignmentSettings(top_k=3, weight_lexical=0.4, weight_semantic=0.6)


def fake_embed(texts):
    """Deterministic bag-of-words embedding (L2-normalized), no model needed."""
    out = []
    for text in texts:
        v = np.zeros(128)
        for tok in normalize_for_lexical(text).split():
            v[zlib.crc32(tok.encode()) % 128] += 1.0
        n = np.linalg.norm(v)
        out.append((v / n if n else v).tolist())
    return out


def _targets(tmp_path):
    f = tmp_path / "mini-dpv.ttl"
    f.write_text(_DPV, encoding="utf-8")
    return build_dpv_targets(load_dpv(f))


def _sources(concepts, extra=()):
    ai = load_ai_concepts({"concepts": concepts})
    return AlignmentSources("memoria", None, concepts=list(extra) + ai, proposals_found=True)


# --- configuration --------------------------------------------------------------------------

def test_settings_defaults_and_config_block():
    d = alignment_settings({})
    assert (d.top_k, d.weight_lexical, d.weight_semantic) == (3, 0.4, 0.6)
    s = alignment_settings({"alignment": {"top_k": 5, "weight_lexical": 2, "weight_semantic": 3}})
    assert s.top_k == 5
    assert s.weight_lexical == pytest.approx(0.4) and s.weight_semantic == pytest.approx(0.6)


def test_settings_reject_invalid_values():
    with pytest.raises(ValueError):
        alignment_settings({"alignment": {"top_k": 0}})
    with pytest.raises(ValueError):
        alignment_settings({"alignment": {"weight_lexical": 0, "weight_semantic": 0}})


# --- semantic matrix ------------------------------------------------------------------------

def test_semantic_matrix_shape_and_range():
    m = semantic_matrix(["Data Subject", "has recipient"], ["Data Subject", "Personal Data", "x"],
                        embed_fn=fake_embed)
    assert m.shape == (2, 3)
    assert m[0, 0] == pytest.approx(1.0)
    assert np.all((m >= 0) & (m <= 1))


# --- ranking --------------------------------------------------------------------------------

def test_topk_order_and_combined_score(tmp_path):
    targets = _targets(tmp_path)
    src = _sources([{"name": "DataSubject", "label": "titular", "type": "class"}])
    r = rank_candidates(src, targets, embed_fn=fake_embed, settings=SETTINGS)
    cands = r.items[0].candidates
    assert [c.rank for c in cands] == [1, 2, 3]
    assert cands[0].dpv_name == "DataSubject"
    assert [c.score for c in cands] == sorted([c.score for c in cands], reverse=True)
    for c in cands:
        assert c.score == pytest.approx(0.4 * c.lexical + 0.6 * c.semantic)


def test_kind_compatibility(tmp_path):
    targets = _targets(tmp_path)
    punned = SourceConcept(key="http://ex.org#Consent", origin=ORIGIN_ONTOLOGY, name="DataSubject",
                           kinds=("class", "property"))
    src = _sources([{"name": "DataSubject", "type": "class"},
                    {"name": "hasDataSubject", "type": "property"}], extra=[punned])
    r = rank_candidates(src, targets, embed_fn=fake_embed,
                        settings=AlignmentSettings(top_k=6, weight_lexical=0.4, weight_semantic=0.6))
    kinds_of = {it.concept.key: {c.dpv_kind for c in it.candidates} for it in r.items}
    assert kinds_of["ai:DataSubject"] == {TARGET_CLASS}
    assert kinds_of["ai:hasDataSubject"] == {TARGET_PROPERTY}
    assert kinds_of["http://ex.org#Consent"] == {TARGET_CLASS, TARGET_PROPERTY}
    assert r.get("ai:hasDataSubject").best.dpv_name == "hasDataSubject"


def test_meaning_rescues_a_name_that_does_not_match(tmp_path):
    """Lexical says nothing ('Titular' vs 'Data Subject'); the definition carries the match."""
    targets = _targets(tmp_path)
    src = _sources([{"name": "Titular", "label": "titular", "type": "class",
                     "definition": "individual whose personal data is being processed"}])
    r = rank_candidates(src, targets, embed_fn=fake_embed, settings=SETTINGS)
    best = r.items[0].best
    assert best.dpv_name == "DataSubject"
    assert best.lexical < 0.5 < best.semantic


def test_precomputed_matrices_skip_the_embedder(tmp_path):
    targets = _targets(tmp_path)
    src = _sources([{"name": "DataController", "type": "class"}])
    lex = score_lexical(src, targets)
    sem = np.zeros_like(lex.matrix)

    def must_not_run(_texts):
        raise AssertionError("no debe recalcular embeddings")

    r = rank_candidates(src, targets, embed_fn=must_not_run, settings=SETTINGS,
                        lexical=lex, semantic=sem)
    assert r.items[0].best.dpv_name == "DataController"
    with pytest.raises(ValueError):
        rank_candidates(src, targets, settings=SETTINGS, lexical=lex, semantic=sem[:, :2])


def test_summary_is_serializable(tmp_path):
    import json
    targets = _targets(tmp_path)
    src = _sources([{"name": "DataSubject", "type": "class"}])
    summary = ranking_summary(rank_candidates(src, targets, embed_fn=fake_embed, settings=SETTINGS))
    json.dumps(summary)
    assert summary["ai"]["concepts"] == 1 and summary["ai"]["with_candidates"] == 1
    assert summary["settings"]["top_k"] == 3


@pytest.mark.slow
def test_real_model_with_real_dpv(dpv_report):
    """Real multilingual encoder against the project DPV (loads the model)."""
    targets = build_dpv_targets(dpv_report)
    src = _sources([{"name": "DataSubject", "label": "titular", "type": "class",
                     "definition": "Persona natural cuyos datos personales son objeto de tratamiento."}])
    r = rank_candidates(src, targets, settings=SETTINGS)
    best = r.items[0].best
    assert best.dpv_iri == "https://w3id.org/dpv#DataSubject"
    assert 0.0 <= best.semantic <= 1.0