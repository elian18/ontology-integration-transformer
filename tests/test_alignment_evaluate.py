"""S4-T09: blind reference sample, metrics against hand labels, and weight sweep."""
import zlib

import numpy as np
import pytest

from src.ingest.dpv_loader import load_dpv
from src.alignment.sources import load_ai_concepts, AlignmentSources, SourceConcept
from src.alignment.dpv_targets import build_dpv_targets
from src.alignment.lexical import normalize_for_lexical
from src.alignment.evaluate import (
    make_reference_sample, write_reference_sample, read_reference_sample, reference_columns,
    reference_settings, evaluate, weight_sweep, find_dpv_terms, render_console, count_labelled,
)

REL = {"e": "skos:exactMatch", "c": "skos:closeMatch", "b": "skos:broadMatch",
       "n": "skos:narrowMatch", "r": "skos:relatedMatch", "x": "none"}


def _concept(key, name, origin, cands, needs=True):
    """cands: list of (dpv_name, relation code)."""
    return [{"concept_key": key, "origin": origin, "concept_name": name,
             "concept_label": None, "concept_definition": None, "rank": i,
             "dpv_name": d, "dpv_label": d, "dpv_definition": f"Definition of {d}.",
             "proposed_relation": REL[rel] if needs else None, "needs_justification": needs}
            for i, (d, rel) in enumerate(cands, start=1)]


def _eval_data():
    rows = []
    rows += _concept("o:A", "Data_subject", "ontology",
                     [("DataSubject", "e"), ("DataSubjectRight", "r"), ("Child", "n")])
    rows += _concept("o:B", "Right_of_opposition", "ontology",
                     [("DataSubjectRight", "n"), ("RightNotice", "r"), ("hasRight", "r")])
    rows += _concept("ai:C", "PortabilityRight", "ai",
                     [("RightNotice", "r"), ("DataSubjectRight", "b"), ("DigitalLiteracy", "x")])
    rows += _concept("o:D", "Banking", "ontology",
                     [("Sector", "r"), ("Finance", "r"), ("Purpose", "x")])
    rows += _concept("ai:E", "DigitalEducationRight", "ai",
                     [("DigitalLiteracy", "e"), ("RightNotice", "r"), ("Purpose", "x")])
    rows += _concept("o:F", "Weird", "ontology",
                     [("Purpose", "x"), ("Sector", "r"), ("Finance", "x")])
    rows += _concept("o:G", "Unlabelled", "ontology",
                     [("Purpose", "x"), ("Sector", "r"), ("Finance", "x")])
    rows += _concept("o:H", "Data_Controller", "ontology",
                     [("DataController", "e"), ("Sector", "r"), ("Finance", "x")])
    return {"metadata": {"settings": {"weight_lexical": 0.4}}, "rows": rows}


REFERENCE = [
    {"concept_key": "o:A", "concept_name": "Data_subject", "gold_dpv_name": "DataSubject",
     "gold_relation": "exactMatch"},
    {"concept_key": "o:B", "concept_name": "Right_of_opposition",
     "gold_dpv_name": "dpv:DataSubjectRight", "gold_relation": "broadMatch"},
    {"concept_key": "ai:C", "concept_name": "PortabilityRight",
     "gold_dpv_name": "DataSubjectRight", "gold_relation": "skos:broadMatch"},
    {"concept_key": "o:D", "concept_name": "Banking", "gold_dpv_name": "FinanceSector",
     "gold_relation": "broadMatch"},
    {"concept_key": "ai:E", "concept_name": "DigitalEducationRight", "gold_dpv_name": "none",
     "gold_relation": "none"},
    {"concept_key": "o:F", "concept_name": "Weird", "gold_dpv_name": "NONE", "gold_relation": ""},
    {"concept_key": "o:G", "concept_name": "Unlabelled", "gold_dpv_name": "", "gold_relation": ""},
    {"concept_key": "o:H", "concept_name": "Data_Controller", "gold_dpv_name": "DataController",
     "gold_relation": "equivalente"},
]


# --- metrics --------------------------------------------------------------------------------

def test_ranking_metrics():
    r = evaluate(_eval_data(), REFERENCE)
    assert (r.sample_size, r.labelled, r.unlabelled) == (8, 7, 1)
    assert r.with_gold_term == 5 and r.gold_none == 2
    assert (r.hit_at_1, r.hit_at_k) == (3, 4)                 # A, B, H first; C at rank 2
    assert r.misses == [{"concept": "Banking", "gold": "FinanceSector",
                         "candidates": ["Sector", "Finance", "Purpose"]}]
    assert r.by_origin["ontology"] == {"with_gold_term": 4, "hit_at_1": 3, "hit_at_k": 3}
    assert r.by_origin["ai"] == {"with_gold_term": 1, "hit_at_1": 0, "hit_at_k": 1}
    s = r.summary()
    assert s["hit_at_1_rate"] == 0.6 and s["hit_at_k_rate"] == 0.8


def test_type_metrics_direction_swaps_and_none():
    r = evaluate(_eval_data(), REFERENCE)
    assert (r.type_evaluated, r.type_correct) == (3, 2)       # A ok, C ok, B swapped
    assert r.direction_swaps == 1
    assert r.confusion["skos:broadMatch -> skos:narrowMatch"] == 1
    assert (r.none_respected, r.gold_none) == (1, 2)          # F yes, E had an exactMatch
    assert r.invalid_labels == ["Data_Controller: gold_relation 'equivalente'"]


def test_count_labelled():
    assert count_labelled(REFERENCE) == 7                     # 'none' counts, '' does not
    assert count_labelled([{"gold_dpv_name": "  "}, {}]) == 0


def test_unknown_concept_is_reported():
    ref = [{"concept_key": "o:ZZ", "concept_name": "Ghost", "gold_dpv_name": "DataSubject",
            "gold_relation": "exactMatch"}]
    r = evaluate(_eval_data(), ref)
    assert r.labelled == 0 and r.invalid_labels == ["Ghost: concepto no esta en el archivo"]


# --- sample ---------------------------------------------------------------------------------

def _pool():
    rows = []
    for i in range(40):
        rows += _concept(f"o:{i:02d}", f"Onto_{i:02d}", "ontology",
                         [("DataSubject", "e"), ("Child", "r"), ("Purpose", "x")])
    for i in range(12):
        rows += _concept(f"ai:{i:02d}", f"Ai{i:02d}", "ai",
                         [("DataSubjectRight", "b"), ("RightNotice", "r"), ("Purpose", "x")])
    for i in range(5):                                       # possible duplicates: not sampled
        rows += _concept(f"ai:dup{i}", f"Dup{i}", "ai",
                         [("DataSubject", "e"), ("Child", "r"), ("Purpose", "x")], needs=False)
    return {"rows": rows}


def test_sample_is_stratified_deterministic_and_blind():
    s1 = make_reference_sample(_pool(), size=30, ai_share=1 / 3, seed=42)
    s2 = make_reference_sample(_pool(), size=30, ai_share=1 / 3, seed=42)
    assert s1 == s2
    assert sum(1 for s in s1 if s["origin"] == "ai") == 10
    assert sum(1 for s in s1 if s["origin"] == "ontology") == 20
    assert not any(s["concept_name"].startswith("Dup") for s in s1)
    assert [s["sample_id"] for s in s1] == list(range(1, 31))
    assert [s["origin"] for s in s1] == ["ai"] * 10 + ["ontology"] * 20   # grouped, then by name
    first = next(s for s in s1 if s["origin"] == "ontology")
    assert set(first) == set(reference_columns(3))
    assert (first["cand1_dpv"], first["cand2_dpv"], first["cand3_dpv"]) == \
        ("DataSubject", "Child", "Purpose")
    assert not any("relation" in c and c != "gold_relation" for c in first)   # blind to the AI
    assert not any(c in first for c in ("score", "lexical", "semantic"))
    assert first["gold_dpv_name"] is None and first["gold_relation"] is None
    assert make_reference_sample(_pool(), size=30, seed=7) != s1


def test_sample_refills_when_a_stratum_is_short():
    data = {"rows": [r for r in _pool()["rows"] if not r["concept_key"].startswith("ai:0")]}
    s = make_reference_sample(data, size=30, ai_share=1 / 3, seed=1)   # only 2 AI left
    assert sum(1 for x in s if x["origin"] == "ai") == 2 and len(s) == 30


def test_write_protects_labels_and_roundtrips(tmp_path):
    sample = make_reference_sample(_pool(), size=5, seed=3)
    path = write_reference_sample(sample, tmp_path / "ref" / "muestra.csv")
    with pytest.raises(FileExistsError):
        write_reference_sample(sample, path)
    write_reference_sample(sample, path, force=True)
    back = read_reference_sample(path)
    assert len(back) == 5 and list(back[0]) == reference_columns(3)
    assert back[0]["gold_dpv_name"] == ""
    assert back[0]["cand1_dpv"] == sample[0]["cand1_dpv"]


def test_read_tolerates_excel_semicolon_and_cp1252(tmp_path):
    p = tmp_path / "excel.csv"
    p.write_bytes("concept_key;concept_name;gold_dpv_name;gold_relation\n"
                  "o:A;Educación;DataSubject;exactMatch\n".encode("cp1252"))
    rows = read_reference_sample(p)
    assert rows == [{"concept_key": "o:A", "concept_name": "Educación",
                     "gold_dpv_name": "DataSubject", "gold_relation": "exactMatch"}]


def test_reference_settings():
    s = reference_settings({})
    assert (s["size"], s["seed"]) == (30, 42)
    assert s["path"].as_posix() == "data/reference/alignment-reference-sample.csv"
    with pytest.raises(ValueError):
        reference_settings({"alignment": {"reference_ai_share": 2}})


def test_console_mentions_the_key_figures():
    text = render_console(evaluate(_eval_data(), REFERENCE),
                          sweep=[{"weight_lexical": 0.4, "weight_semantic": 0.6, "concepts": 5,
                                  "hit_at_1": 0.6, "hit_at_k": 0.8}], current_weight=0.4)
    assert "Hit@1 (el correcto sale primero): 3/5 = 60.0 %" in text
    assert "Hit@3 (el correcto esta entre los 3): 4/5 = 80.0 %" in text
    assert "Direccion invertida (broad <-> narrow): 1" in text
    assert "<- actual" in text


# --- weight sweep and DPV search ------------------------------------------------------------

_DPV = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dpv: <https://w3id.org/dpv#> .
dpv:DataSubject a rdfs:Class , skos:Concept ; skos:prefLabel "Data Subject"@en ;
    skos:definition "The individual whose personal data is being processed."@en .
dpv:Titles a rdfs:Class , skos:Concept ; skos:prefLabel "Titular Titles"@en ;
    skos:definition "Honorific names."@en .
dpv:DataController a rdfs:Class , skos:Concept ; skos:prefLabel "Data Controller"@en ;
    skos:definition "Decides the purposes of processing."@en .
dpv:hasRecipient a rdf:Property , skos:Concept ; skos:prefLabel "has recipient"@en ;
    skos:definition "Indicates the recipient of data."@en .
"""


def fake_embed(texts):
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


def test_weight_sweep_shows_the_effect_of_the_weights(tmp_path):
    targets = _targets(tmp_path)
    titular = load_ai_concepts({"concepts": [{
        "name": "Titular", "label": "titular", "type": "class",
        "definition": "The individual whose personal data is being processed"}]})
    sources = AlignmentSources("m", None, concepts=titular, proposals_found=True)
    ref = [{"concept_key": "ai:Titular", "gold_dpv_name": "DataSubject"}]
    out = weight_sweep(sources, targets, ref, embed_fn=fake_embed, weights=(0.0, 1.0), k=1)
    by_w = {r["weight_lexical"]: r for r in out}
    assert by_w[0.0]["hit_at_1"] == 1.0          # meaning finds it
    assert by_w[1.0]["hit_at_1"] == 0.0          # spelling alone picks 'Titular Titles'
    assert by_w[0.0]["concepts"] == 1


def test_weight_sweep_ignores_none_and_unlabelled(tmp_path):
    targets = _targets(tmp_path)
    sources = AlignmentSources("m", None, concepts=[
        SourceConcept(key="o:X", origin="ontology", name="Data_subject", kinds=("class",))])
    ref = [{"concept_key": "o:X", "gold_dpv_name": "none"}]
    assert weight_sweep(sources, targets, ref, embed_fn=fake_embed) == []


def test_find_dpv_terms(tmp_path):
    targets = _targets(tmp_path)
    assert {x.name for x in find_dpv_terms(targets, "data")} >= {"DataSubject",
                                                                 "DataController"}
    assert [x.name for x in find_dpv_terms(targets, "honorific")] == ["Titles"]  # by definition