"""S7-T11: the "ley + ontología" mode. The uploaded law gives each concept of the uploaded
ontology the articles that back it (semantic search in the index of the law, S7-T07), in the
column law_evidence of alignment.csv, for a person to review them; they are not written to the
ontology and the alignment of S7-T06/T08 does not change. The rule (top_k and a maximum
distance) is calibrated on GDPRtEXT, whose concepts say in which GDPR article they are
defined.

Fast tests: a fake embedder that gives one direction per keyword, a small law in .txt and a
small ontology; the end-to-end test uses the small validated base of the decider tests. The slow
test runs the real calibration (real embeddings, no LLM) and checks config.yaml against it."""
from __future__ import annotations

import csv
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from rdflib import Graph
from rdflib.namespace import DCTERMS

from src.alignment import law_evidence as lev
from src.alignment.law_evidence import (EvidenceSettings, align_with_law, calibrate, choose,
                                        cross_validate, evidence_for_workspace,
                                        evidence_settings, grid_scores, ground_truth,
                                        is_current, score, search, select)
from src.ingest.law_upload import chroma_path, prepare_law
from src.workspace.store import DONE, ERROR, KIND_LAW, KIND_ONTOLOGY, open_upload
from tests.test_alignment_bundle import _resolve
from tests.test_alignment_propose import TTL, DeciderLLM, Embed, make_env, targets  # noqa: F401

EX = "http://example.org/gdpr#"
LAW = ("Artículo 1°. Consentimiento. El consent del titular es libre.\n"
       "Artículo 2°. Supresión. El titular puede pedir la erasure de sus datos.\n"
       "Artículo 3°. Responsable. El controller decide sobre el tratamiento.\n"
       "Artículo 4°. Datos personales. Toda información personal de una persona.\n")
ONTOLOGY = f"""
@prefix ex: <{EX}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://example.org/gdpr> a owl:Ontology .
ex:Consent a owl:Class ; rdfs:label "consent" ;
    rdfs:isDefinedBy <http://purl.org/adaptcentre/openscience/resources/GDPRtEXT#article1-2> ,
                     <http://purl.org/adaptcentre/openscience/resources/GDPRtEXT#recital32> .
ex:RightToErasure a owl:Class ; rdfs:label "right to erasure" ;
    rdfs:isDefinedBy <http://purl.org/adaptcentre/openscience/resources/GDPRtEXT#article2> .
ex:Controller a owl:Class ; rdfs:label "controller" ;
    rdfs:comment "Decides on the processing of personal data."@en ;
    rdfs:isDefinedBy <http://purl.org/adaptcentre/openscience/resources/GDPRtEXT#article4> .
ex:Pizza a owl:Class ; rdfs:label "pizza" .
"""
KEYWORDS = ("consent", "erasure", "controller", "personal")


class KeywordEmbed:
    """One direction per keyword, weighed by order of appearance (1, 0.5, 0.25...); a text with
    none of them gets a direction of its own."""

    def __init__(self):
        self.calls, self.seen = 0, {}

    def __call__(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = np.zeros(32)
            low = t.lower()
            found = sorted((low.find(k), i) for i, k in enumerate(KEYWORDS) if k in low)
            for weight, (_, i) in enumerate(found):
                v[i] = 0.5 ** weight
            if not found:
                v[len(KEYWORDS) + self.seen.setdefault(t, len(self.seen)) % 20] = 1.0
            out.append(v)
        return out


def _concept(key, text):
    return type("C", (), {"key": key, "semantic_text": lambda self: text})()


@pytest.fixture
def cfg(tmp_path):
    return {"workspaces": {"dir": str(tmp_path / "ws")}, "outputs": {"dir": str(tmp_path / "out")},
            "auto": {"min_precision": 0.75}}


@pytest.fixture
def ws(cfg):
    w = open_upload({KIND_ONTOLOGY: ("gdpr.ttl", ONTOLOGY.encode("utf-8")),
                     KIND_LAW: ("ley.txt", LAW.encode("utf-8"))}, cfg=cfg)
    prepare_law(w, embed_fn=KeywordEmbed(), model_name="modelo-falso")
    return w


# ------------------------------------------------------------------ settings

def test_settings():
    assert evidence_settings({}) == EvidenceSettings() == EvidenceSettings(3, None)
    assert evidence_settings({"law_evidence": {"top_k": 1, "max_distance": 0.4}}) == \
        EvidenceSettings(1, 0.4)
    assert evidence_settings({"law_evidence": {"max_distance": None}}).max_distance is None
    for bad in ({"top_k": 0}, {"max_distance": 0}, {"max_distance": 2.5}):
        with pytest.raises(ValueError, match="bloque law_evidence"):
            evidence_settings({"law_evidence": bad})
    assert EvidenceSettings(2, 0.35).label() == "hasta 2 artículos por concepto, distancia <= 0,35"
    assert EvidenceSettings().label() == "hasta 3 artículos por concepto, sin distancia máxima"
    assert EvidenceSettings(1).label() == "hasta 1 artículo por concepto, sin distancia máxima"


def test_the_project_settings_are_valid():
    s = evidence_settings()
    assert s.top_k in lev.GRID_K and s.max_distance in lev.GRID_DISTANCE


# ------------------------------------------------------------------ search

def test_search_finds_the_closest_articles(ws):
    concepts = [_concept("a", "consent of the data subject"), _concept("b", "erasure"),
                _concept("c", "nothing to do")]
    hits = search(concepts, path=chroma_path(ws), embed_fn=KeywordEmbed(), n=2)
    assert [h["number"] for h in hits["a"]][:1] == [1] and hits["a"][0]["distance"] == 0.0
    assert hits["a"][0]["title"] == "Consentimiento"
    assert hits["b"][0]["number"] == 2 and len(hits["c"]) == 2
    assert all(h["distance"] >= 0.99 for h in hits["c"])               # unrelated: far
    assert hits["c"] == sorted(hits["c"], key=lambda h: (h["distance"], h["number"]))
    assert search([], path=chroma_path(ws)) == {}


def test_ties_go_to_the_lower_article(ws, monkeypatch):
    from src.ai.rag import vector_store
    monkeypatch.setattr(vector_store, "query", lambda *a, **kw: {
        "metadatas": [[{"number": 3, "title": "C"}, {"number": 1, "title": None}, {"number": 2}]],
        "distances": [[0.4, 0.4, 0.1]]})
    hits = search([_concept("a", "x")], path=chroma_path(ws), embed_fn=KeywordEmbed())
    assert hits["a"] == [{"number": 2, "title": "", "distance": 0.1},
                         {"number": 1, "title": "", "distance": 0.4},
                         {"number": 3, "title": "C", "distance": 0.4}]


def test_search_needs_the_index(cfg, tmp_path):
    with pytest.raises(ValueError, match="indexar la ley"):
        search([_concept("a", "x")], path=tmp_path / "vacio", embed_fn=KeywordEmbed())


def test_select():
    hits = [{"number": n, "title": "", "distance": d} for n, d in ((1, 0.2), (2, 0.4), (3, 0.6))]
    assert [h["number"] for h in select(hits, 3, None)] == [1, 2, 3]
    assert [h["number"] for h in select(hits, 2, None)] == [1, 2]
    assert [h["number"] for h in select(hits, 3, 0.4)] == [1, 2]       # the limit is included
    assert select(hits, 1, 0.1) == []


# ------------------------------------------------------------------ calibration

def test_ground_truth_reads_the_articles_of_gdprtext():
    from src.alignment.inherit import user_concepts
    import tempfile
    path = Path(tempfile.mkdtemp()) / "o.ttl"
    path.write_text(ONTOLOGY, encoding="utf-8")
    truth = ground_truth(Graph().parse(path), user_concepts(path))
    assert truth == {EX + "Consent": {1}, EX + "RightToErasure": {2}, EX + "Controller": {4}}


def test_the_real_ground_truth():
    from src.alignment.inherit import user_concepts
    from src.config import ROOT
    from src.transform.convert import load_graph
    path = ROOT / lev.CALIBRATION_ONTOLOGY
    if not path.exists():
        pytest.skip(f"Falta {lev.CALIBRATION_ONTOLOGY}")
    concepts = user_concepts(path)
    truth = ground_truth(load_graph(path), concepts)
    assert (len(concepts), len(truth), sum(map(len, truth.values()))) == (235, 80, 113)
    assert len(set().union(*truth.values())) == 38


def _rows():
    def hits(*pairs):
        return [{"number": n, "title": "", "distance": d} for n, d in pairs]
    return [{"key": "a", "truth": [1], "hits": hits((1, 0.2), (2, 0.5), (3, 0.7))},
            {"key": "b", "truth": [2], "hits": hits((5, 0.3), (2, 0.35), (1, 0.8))},
            {"key": "c", "truth": [3], "hits": hits((4, 0.45), (6, 0.5), (7, 0.9))},
            {"key": "d", "truth": [9, 8], "hits": hits((8, 0.25), (9, 0.3), (1, 0.4))}]


def test_score_and_grid():
    rows = _rows()
    assert score(rows, 1, None) == {"top_k": 1, "max_distance": None, "links": 4, "correct": 2,
                                    "covered": 2, "concepts": 4, "precision": 0.5,
                                    "coverage": 0.5}
    s = score(rows, 2, 0.35)
    assert (s["links"], s["correct"], s["covered"]) == (5, 4, 3)   # c keeps nothing
    assert score([], 1, None)["precision"] is None and score([], 1, None)["coverage"] is None
    table = grid_scores(rows)
    assert len(table) == len(lev.GRID_K) * len(lev.GRID_DISTANCE)
    assert table[0]["top_k"] == 1 and table[0]["max_distance"] == 0.30


def test_choose_follows_the_fixed_criterion():
    def rule(k, d, links, precision, coverage):
        return {"top_k": k, "max_distance": d, "links": links, "precision": precision,
                "coverage": coverage}
    table = [rule(1, 0.3, 20, 0.80, 0.40), rule(2, 0.4, 30, 0.76, 0.55),
             rule(3, None, 60, 0.50, 0.70), rule(1, 0.5, 5, 0.95, 0.10),
             rule(2, 0.5, 34, 0.76, 0.55), rule(1, None, 12, None, 0.0)]
    assert choose(table, 0.75) == table[1]                 # most coverage; tie -> fewer links
    assert choose(table, 0.9) is None                       # 0.95 has too few links
    assert choose(table, 0.9, min_links=5) == table[3]
    tie = [rule(2, 0.5, 30, 0.8, 0.5), rule(1, 0.5, 30, 0.8, 0.5), rule(1, None, 30, 0.8, 0.5),
           rule(1, 0.4, 30, 0.8, 0.5)]
    assert choose(tie, 0.75) == tie[3]                      # smaller top_k, shorter distance
    assert lev.review_rule(table) == table[2]               # for review: precision aside
    assert lev.review_rule(table, min_links=70) is None


def test_cross_validation_is_reproducible():
    rows = _rows() * 10
    rows = [{**r, "key": f"{r['key']}{i}"} for i, r in enumerate(rows)]
    a = cross_validate(rows, 0.75, repeats=5)
    assert a == cross_validate(rows, 0.75, repeats=5) == {
        "repeats": 5, "seed": lev.CV_SEED, "folds": 10, "folds_without_rule": 2,
        "precision_mean": 0.7815, "coverage_mean": 0.75, "precision_p10": 0.7037,
        "precision_p90": 0.8636}                   # measured on the other half: p10 < 75 %
    assert cross_validate(rows, 0.75, repeats=5, seed=1) != a
    assert choose(grid_scores(rows), 0.75)["precision"] == 0.8      # on all: optimistic
    none = cross_validate(rows, 0.99, repeats=3)
    assert none["folds_without_rule"] == 6 and none["precision_mean"] is None
    assert none["precision_p10"] is None and none["coverage_mean"] is None


def test_calibrate_and_console(tmp_path, cfg):
    onto, law = tmp_path / "gdprtext.ttl", tmp_path / "gdpr.txt"
    onto.write_text(ONTOLOGY, encoding="utf-8")
    law.write_text(LAW, encoding="utf-8")
    r = calibrate(onto, law, embed_fn=KeywordEmbed(), cfg=cfg)
    assert (r["concepts"], r["judged"], r["truth_links"]) == (4, 3, 3)
    assert r["hit_at"] == {"1": 2, "2": 3, "3": 3}           # Controller: art. 4 is second
    assert [len(row["hits"]) for row in r["rows"]] == [3, 3, 3]
    assert [(g["links"], g["correct"]) for g in r["grid"] if g["max_distance"] is None] == [
        (3, 2), (6, 3), (9, 3)]
    assert r["ontology"] == "gdprtext.ttl" and r["law"] == "gdpr.txt"
    assert r["min_precision"] == 0.75 and len(r["grid"]) == 24
    assert r["chosen"] is None and r["review"] is None       # fewer than 10 links
    text = lev.render_calibration(r)
    assert text.splitlines()[0] == "Evidencia de la ley: calibración (S7-T11)"
    assert "3 conceptos con artículo conocido (de 4), 3 enlaces de referencia" in text
    assert "El artículo correcto está entre los primeros: 1: 2 de 3, 2: 3 de 3, 3: 3 de 3" in text
    assert ("  Ninguna regla llega a la precisión mínima (75,0 %) con 10 enlaces o más: los "
            "artículos no entran a la ontología.") in text.splitlines()
    assert "  Ninguna regla da 10 enlaces o más para revisar." in text.splitlines()
    assert text.splitlines()[-1] == ("  Validación cruzada (200 mitades): precisión media — (p10 "
                                     "—, p90 —), cobertura media —, sin regla en 200")
    lines = lev.render_calibration({**r, "chosen": r["grid"][1], "review": r["grid"][23]})
    lines = lines.splitlines()
    assert ("  Llega a la precisión mínima: hasta 1 artículo por concepto, distancia <= 0,35 "
            "(66,7 %, 3 enlaces).") in lines
    assert ("  Para revisar en alignment.csv (la mayor cobertura con menos enlaces): hasta 3 "
            "artículos por concepto, sin distancia máxima, 9 enlaces, el correcto para 3 de 3 "
            "conceptos") in lines
    assert "  Para config.yaml: law_evidence: top_k: 3, max_distance: null" in lines
    assert lines[5].endswith(" <- precisión mínima") and lines[27].endswith(" <- para revisar")
    assert sum("<-" in line for line in lines) == 2
    assert "law_evidence: top_k: 3, max_distance: 0.60" in lev.render_calibration(
        {**r, "review": r["grid"][22]})
    with pytest.raises(FileNotFoundError, match="calibración"):
        calibrate(tmp_path / "no.ttl", law, embed_fn=KeywordEmbed(), cfg=cfg)


def test_the_rule_for_review_ignores_the_precision(tmp_path, cfg, monkeypatch):
    onto, law = tmp_path / "gdprtext.ttl", tmp_path / "gdpr.txt"
    onto.write_text(ONTOLOGY, encoding="utf-8")
    law.write_text(LAW, encoding="utf-8")
    monkeypatch.setattr(lev, "MIN_LINKS", 3)
    r = calibrate(onto, law, embed_fn=KeywordEmbed(), cfg=cfg)
    assert r["min_links"] == 3 and r["chosen"] == r["review"]           # 3 of 4 = 75 %
    assert (r["review"]["top_k"], r["review"]["max_distance"], r["review"]["links"],
            r["review"]["precision"], r["review"]["coverage"]) == (2, 0.60, 4, 0.75, 1.0)
    r = calibrate(onto, law, embed_fn=KeywordEmbed(), cfg={**cfg, "auto": {"min_precision": 0.8}})
    assert r["chosen"] is None and r["review"]["max_distance"] == 0.60


# ------------------------------------------------------------------ the workspace

def test_evidence_for_workspace(ws, cfg):
    r = evidence_for_workspace(ws, embed_fn=KeywordEmbed(), cfg=cfg)
    by = {e["name"]: e["articles"] for e in r["entries"]}
    assert [a["number"] for a in by["Consent"]][:1] == [1] and len(by["Consent"]) == 3
    assert r["settings"] == {"top_k": 3, "max_distance": None}
    assert r["law"] == "ley.txt" and r["law_articles"] == 4
    assert r["summary"]["concepts"] == 4 and r["summary"]["with_evidence"] == 4
    assert r["summary"]["links"] == 12                    # Pizza too: no maximum distance
    assert r["summary"]["articles_cited"] == len({a["number"] for v in by.values() for a in v})
    assert lev.load_result(ws) == r and is_current(r, cfg)
    assert not is_current(r, {**cfg, "law_evidence": {"top_k": 1}})
    assert not is_current({**r, "method_version": 0}, cfg) and not is_current(None, cfg)
    step = ws.step("evidence")
    assert step["status"] == DONE and step["detail"] == ("4 de 4 conceptos con artículos de la "
                                                         "ley para revisar (12 enlaces)")
    strict = {**cfg, "law_evidence": {"top_k": 1, "max_distance": 0.5}}
    r = evidence_for_workspace(ws, embed_fn=KeywordEmbed(), cfg=strict)
    assert r["summary"]["with_evidence"] == 3 and r["summary"]["links"] == 3   # not Pizza
    assert r["summary"]["articles_cited"] == 3
    assert lev.by_key(r)[EX + "Pizza"] == [] and lev.by_key(None) == {}


def test_an_error_marks_the_step(cfg, monkeypatch):
    w = open_upload({KIND_ONTOLOGY: ("gdpr.ttl", ONTOLOGY.encode("utf-8")),
                     KIND_LAW: ("ley.txt", LAW.encode("utf-8"))}, cfg=cfg)
    with pytest.raises(ValueError, match="indexar la ley"):
        evidence_for_workspace(w, embed_fn=KeywordEmbed(), cfg=cfg)   # segmented, not indexed
    assert w.step("evidence")["status"] == ERROR


def test_evidence_text():
    articles = [{"number": 17, "title": "", "distance": 0.3245},
                {"number": 4, "title": "", "distance": 0.41}]
    assert lev.evidence_text(articles) == "art. 17 (0,32); art. 4 (0,41)"
    assert lev.evidence_text([]) == ""


# ------------------------------------------------------------------ the whole mode

LAW_E2E = ("Artículo 1°. Datos personales. La información personal de una persona.\n"
           "Artículo 2°. Responsable. El controller decide sobre el tratamiento.\n"
           "Artículo 3°. Pizza. Nada que ver.\n")


@pytest.fixture
def env(tmp_path, targets):  # noqa: F811
    env = make_env(tmp_path)
    env.targets = targets
    return env


def _kw(env, **extra):
    return dict(base=env.base, targets=env.targets, embed_fn=Embed(), cfg=env.cfg,
                chroma_path=env.chroma, sleep_fn=lambda s: None, **extra)


def _csv(ws):
    with open(ws.output_dir / "alignment.csv", encoding="utf-8-sig") as handle:
        return {r["concept_name"]: r for r in csv.DictReader(handle)}


def test_align_with_law_end_to_end(env, tmp_path):
    llm, seen = DeciderLLM(), []
    ws, m, path = align_with_law("Mi GDPR.ttl", TTL.encode("utf-8"), "ley.txt",
                                 LAW_E2E.encode("utf-8"), llm=llm, progress=seen.append,
                                 **_kw(env))
    assert seen == ["Separar la ley en artículos", "Indexar los artículos de la ley",
                    "Heredar lo validado", "Buscar candidatos del DPV",
                    "Completar las etiquetas en español e inglés",
                    "Buscar en la ley los artículos de cada concepto",
                    "La IA revisa las correspondencias aceptadas",
                    "Escribir la ontología alineada", "Validar la forma (SHACL)",
                    "Validar la lógica (razonador)", "Preparar la descarga"]
    assert ws.manifest["mode"] == "ontology+law"
    assert all(s["status"] == DONE for s in ws.progress())
    e = m["law_evidence"]
    assert (e["concepts"], e["with_evidence"], e["links"], e["law"]) == (4, 4, 12, "ley.txt")
    assert e["settings"] == lev.evidence_settings(env.cfg).__dict__
    assert any(n.startswith("La columna law_evidence de alignment.csv trae los artículos de "
                            "ley.txt") for n in m["notes"])
    rows = _csv(ws)                                    # the articles: in the CSV only
    assert rows["PersonalData"]["law_evidence"].startswith("art. 1 (0,00); art. ")
    assert all(r["law_evidence"].count("art. ") == 3 for r in rows.values())
    assert set(m["modules"]) == {"alignment", "labels"}
    names = zipfile.ZipFile(path).namelist()
    assert not any("evidence" in n for n in names)
    folder = tmp_path / "z"
    zipfile.ZipFile(path).extractall(folder)
    g = _resolve(folder, m["modules"]["alignment"])
    assert not list(g.triples((None, DCTERMS.relation, None)))
    readme = (folder / "LEEME.txt").read_text(encoding="utf-8")
    assert ("  - alignment.csv: una fila por concepto con lo que pasó (ábrelo en Excel).\n"
            "    Su columna law_evidence trae, para 4 de tus conceptos, los artículos de ley.txt "
            "más cercanos (12 en total, con su distancia), encontrados por búsqueda semántica: "
            "son para que los revises; no entran a la ontología.\n") in readme
    from src.alignment.user_alignment import render_console
    assert (f"  Evidencia de ley.txt: 4 de 4 conceptos con artículos para revisar (12 enlaces, "
            f"{e['articles_cited']} artículos distintos) en alignment.csv; no entran a la "
            "ontología") in render_console(ws, m, path).splitlines()
    calls, seen = llm.calls, []
    ws2, m2, path2 = align_with_law("Mi GDPR.ttl", TTL.encode("utf-8"), "ley.txt",
                                    LAW_E2E.encode("utf-8"), llm=llm, progress=seen.append,
                                    **_kw(env))
    assert llm.calls == calls and "Buscar en la ley los artículos de cada concepto" not in seen
    assert ("en alignment.csv; no entran a la ontología (de la corrida anterior)"
            in render_console(ws2, m2, path2))                          # S7-T13


def test_other_settings_redo_only_the_evidence(env):
    from src.alignment.user_alignment import align_workspace
    ws, m, _ = align_with_law("Mi GDPR.ttl", TTL.encode("utf-8"), "ley.txt",
                              LAW_E2E.encode("utf-8"), llm=DeciderLLM(), **_kw(env))
    strict, llm, seen = ({**env.cfg, "law_evidence": {"top_k": 1, "max_distance": 0.0001}},
                         DeciderLLM(), [])
    _, m2, _ = align_workspace(ws, **{**_kw(env), "cfg": strict, "llm": llm,
                                      "progress": seen.append})
    assert "Buscar en la ley los artículos de cada concepto" in seen and llm.calls == 0
    e = m2["law_evidence"]
    assert (e["links"], e["with_evidence"], e["articles_cited"]) == (2, 2, 2)  # exact matches
    assert e["settings"] == {"top_k": 1, "max_distance": 0.0001}
    rows = _csv(ws)
    assert rows["PersonalData"]["law_evidence"] == "art. 1 (0,00)"
    assert rows["Pizza"]["law_evidence"] == ""


def test_an_ontology_alone_has_no_evidence(env):
    from src.alignment.user_alignment import align_upload
    w, m, _ = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), llm=DeciderLLM(), **_kw(env))
    assert m["law_evidence"] is None and "evidence" not in [s["step"] for s in w.progress()]
    assert all(r["law_evidence"] == "" for r in _csv(w).values())
    assert not any("law_evidence" in n for n in m["notes"])


def test_cli(env, tmp_path, monkeypatch, capsys):
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setattr("src.ai.rag.vector_store._PATH", str(tmp_path / "chroma-cli"))
    monkeypatch.setattr("src.config.load_config", lambda: env.cfg)
    import src.ai.llm_client as client
    from src.alignment import inherit as inh, propose as prop

    class Counted(DeciderLLM):
        made = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            Counted.made += 1

    monkeypatch.setattr(client, "LLMClient", Counted)
    monkeypatch.setattr(inh, "load_base", lambda **kw: env.base)
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: env.targets)
    onto, law = tmp_path / "gdpr.ttl", tmp_path / "ley.txt"
    onto.write_text(TTL, encoding="utf-8")
    law.write_text(LAW_E2E, encoding="utf-8")
    assert lev._run([str(onto), str(law)]) == 0
    out = capsys.readouterr().out
    assert "Ontología alineada (S7-T08)" in out and "Evidencia de ley.txt:" in out
    assert Counted.made > 0
    other, Counted.made = tmp_path / "otra.ttl", 0
    other.write_text(TTL + "ex:Recipient a owl:Class .\n", encoding="utf-8")
    assert lev._run([str(other), str(law), "--sin-ia"]) == 0 and Counted.made == 0
    assert lev._run([str(onto)]) == 1
    assert "indica la ontología y la ley" in capsys.readouterr().out
    assert lev._run([str(onto), str(tmp_path / "no.txt")]) == 1
    owx = tmp_path / "owx.ttl"
    owx.write_text(TTL + "ex:Owx a owl:Class .\n", encoding="utf-8")
    assert lev._run([str(owx), str(law), "--formato", "owx"]) == 1
    assert not any("owx" in p.name for p in Path(env.cfg["workspaces"]["dir"]).iterdir())
    gdprtext, gdpr = tmp_path / "gdprtext.ttl", tmp_path / "gdpr.txt"
    gdprtext.write_text(ONTOLOGY, encoding="utf-8")
    gdpr.write_text(LAW, encoding="utf-8")
    monkeypatch.setattr(lev, "CALIBRATION_ONTOLOGY", str(gdprtext))
    monkeypatch.setattr(lev, "CALIBRATION_LAW", str(gdpr))
    assert lev._run(["--calibrar"]) == 0
    out = capsys.readouterr().out
    assert "Evidencia de la ley: calibración (S7-T11)" in out and lev.CALIBRATION_FILE in out
    saved = Path(env.cfg["outputs"]["dir"]) / lev.CALIBRATION_FILE
    assert json.loads(saved.read_text(encoding="utf-8"))["judged"] == 3
    monkeypatch.setattr(lev, "CALIBRATION_LAW", str(tmp_path / "falta.txt"))
    assert lev._run(["--calibrar"]) == 1


def load_graph(path):
    from src.transform.convert import load_graph as _load
    return _load(path)


# ------------------------------------------------------------------ real files and embeddings

@pytest.mark.slow
def test_the_config_has_the_calibrated_rule(tmp_path):
    """The real calibration (GDPRtEXT + GDPR, real embeddings, no LLM, in a temporary
    workspace): no rule reaches the precision (so the articles stay in the CSV, S7-T11) and
    config.yaml has the rule for the review."""
    from src.config import ROOT, load_config
    for name in (lev.CALIBRATION_ONTOLOGY, lev.CALIBRATION_LAW):
        if not (ROOT / name).exists():
            pytest.skip(f"Falta {name}")
    cfg = {**load_config(), "workspaces": {"dir": str(tmp_path / "ws")}}
    r = calibrate(cfg=cfg)
    print("\n" + lev.render_calibration(r))
    assert (r["concepts"], r["judged"], r["truth_links"]) == (235, 80, 113)
    assert r["chosen"] is None, ("una regla ya llega a la precisión mínima: revisar la decisión "
                                 "de S7-T11 (los artículos podrían entrar a la ontología)")
    review = r["review"]
    assert evidence_settings(cfg) == EvidenceSettings(review["top_k"], review["max_distance"])