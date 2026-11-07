"""Tests for the catalog of competency questions (S9-T03).

The real catalog must be complete and its expected answers must be in the LOPDP: every
article exists, every element is written in its article and every value is in the quoted
wording. Synthetic catalogs check that each missing piece is reported in Spanish, and small
cases fix how an answer will be recognized before any SPARQL query exists."""
from __future__ import annotations

import copy
import hashlib
import re
import unicodedata
from pathlib import Path

import pytest
import yaml
from rdflib import Graph, URIRef

from src.validation import question_catalog as qc

ROOT = Path(__file__).resolve().parents[1]
LOPDP = ROOT / "data" / "input" / "lopdp.pdf"
DPV = ROOT / "vocab" / "dpv.ttl"
NUMBERS = {3: "tres", 5: "cinco", 15: "quince"}


@pytest.fixture(scope="module")
def catalog():
    return qc.load_catalog()


@pytest.fixture(scope="module")
def law():
    if not LOPDP.exists():
        pytest.skip("LOPDP no disponible en data/input/lopdp.pdf")
    from src.ingest.legal_segmenter import segment_articles
    from src.ingest.text_loader import load_legal_text
    articles = segment_articles(load_legal_text(LOPDP).text)["articles"]
    return {a["number"]: _plain(a["text"]) for a in articles}


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(text.split())


# ------------------------------------------------------------------ the real catalog

def test_the_catalog_has_15_base_questions_and_8_generic_ones(catalog):
    ids = [q.id for q in catalog.questions]
    assert ids == ([f"P{n:02}" for n in range(1, 6)] + [f"L{n:02}" for n in range(1, 11)]
                   + [f"G{n:02}" for n in range(1, 9)])
    base = [q for q in catalog.questions if q.group == qc.BASE]
    assert [q.source for q in base] == ["paper"] * 5 + ["lopdp"] * 10
    assert {q.group for q in catalog.questions if q.id.startswith("G")} == {qc.GENERIC}
    assert catalog.version == 1 and catalog.law == "LOPDP"
    assert catalog.sha256 == hashlib.sha256(qc.CATALOG_FILE.read_bytes()).hexdigest()


def test_no_question_lacks_origin_criterion_modes_or_its_expected_answer(catalog):
    for q in catalog.questions:
        assert q.source in qc.SOURCES and q.reference, q.id
        assert q.kind in qc.KINDS[q.group], q.id
        assert q.modes and set(q.modes) <= set(qc.MODES), q.id
        assert qc.criterion_text(q).startswith("Verde si"), q.id
        if q.group == qc.BASE:
            e = q.expected
            assert e.text and e.articles, q.id
            assert "art" in e.text, q.id                      # the answer cites its article
            assert all(x.articles for x in e.elements), q.id
            assert (e.value is not None) == (q.kind == qc.VALUE), q.id
        else:
            assert q.expected is None and q.looks_for, q.id
    paper = [q for q in catalog.questions if q.source == "paper"]
    assert all(set(q.paper) == set(qc.PAPER_FIELDS) for q in paper)


def test_the_expected_answers_are_written_in_the_law(catalog, law):
    for q in (q for q in catalog.questions if q.group == qc.BASE):
        e = q.expected
        assert set(e.articles) <= set(law), q.id
        if q.kind == qc.VALUE:
            text = law[e.articles[0]]
            assert _plain(e.quote) in text, q.id
            assert str(int(e.value)) in e.quote or NUMBERS[int(e.value)] in e.quote, q.id
        for element in e.elements:
            for article in element.articles:
                assert _plain(element.name) in law[article], (q.id, element.name, article)


def test_the_paper_answers_that_differ_from_the_law_say_so(catalog):
    by_id = {q.id: q for q in catalog.questions}
    assert by_id["P01"].expected.value == 3 and "5" in by_id["P01"].paper["reported"]
    assert len(by_id["P03"].expected.elements) == 6 and "Destinatario" in [
        e.name for e in by_id["P03"].expected.elements]
    assert by_id["P04"].kind == qc.ANY and by_id["P05"].expected.value == 15
    assert by_id["L01"].expected.articles == (12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23)


def test_the_generic_questions_look_for_real_dpv_terms(catalog):
    dpv = Graph().parse(DPV)
    for q in (q for q in catalog.questions if q.group == qc.GENERIC):
        if q.looks_for.startswith("dpv:") and q.looks_for != qc.ANY_DPV:
            term = URIRef("https://w3id.org/dpv#" + q.looks_for[4:])
            assert (term, None, None) in dpv, q.id
        assert q.kind == qc.AT_LEAST and q.min == 1


def test_the_modes_of_each_question(catalog):
    assert {q.modes for q in catalog.questions if q.group == qc.BASE} == {("base",)}
    every = ("base", "ontology", "law", "ontology+law")
    assert [q.id for q in catalog.questions if q.modes == every] == [f"G0{n}" for n in range(1, 8)]
    assert {q.id: q.modes for q in catalog.questions}["G08"] == ("base", "law")
    assert [len(qc.for_mode(catalog, m)) for m in every] == [23, 7, 8, 7]


def test_the_console_shows_every_question_with_its_criterion(catalog):
    text = qc.render_console(catalog)
    assert text.splitlines()[0] == (f"Catálogo de preguntas de competencia v1 (2026-11-07, "
                                    f"sha256 {catalog.sha256[:12]})")
    assert ("  23 preguntas: 15 de la base (5 del artículo de OntoPriv y 10 de la LOPDP) y 8 "
            "genéricas") in text
    assert all(f"  {q.id} {q.question}" in text for q in catalog.questions)
    assert text.count("      Criterio: Verde si") == 23
    assert ("      Respuesta esperada (11 elementos): Información, acceso, rectificación"
            in text)
    assert "      Modos: Base del sistema, Una ley" in text
    assert ("      Origen: Artículo de OntoPriv, Question 1 (Listing 1, Tabla 4); respondió: 5 "
            "(Tabla 4, sin unidad)") in text
    assert "      Nota: El artículo responde 5, que es el plazo para notificar a la Autoridad" in text
    assert "      Origen: LOPDP, art. 7" in text and "      Origen: DPV 2.3, rama dpv:Purpose" in text
    assert qc.COLORS in text and "dpv:DataSubjectRight o con un término que esté debajo" in text


def test_the_cli_prints_the_catalog_or_what_is_missing(capsys, monkeypatch, tmp_path):
    assert qc._run([]) == 0
    assert capsys.readouterr().out.startswith("Catálogo de preguntas de competencia v1")
    broken = tmp_path / "questions.yaml"
    broken.write_text("version: 1\nquestions: []\n", encoding="utf-8")
    monkeypatch.setattr(qc, "CATALOG_FILE", broken)
    assert qc._run([]) == 1
    out = capsys.readouterr().out
    assert out.startswith("El catálogo de preguntas está incompleto:")
    assert "  - el catálogo no tiene preguntas" in out


# ------------------------------------------------------------------ what check() reports

def _valid() -> dict:
    return {"version": 1, "law": "LOPDP", "law_file": "data/input/lopdp.pdf", "paper": "Artículo",
            "matching": {"relations": ["exactMatch"], "hierarchy": ["skos:broader"]},
            "questions": [
                {"id": "P01", "group": "base", "question": "¿Cuántos días?", "modes": ["base"],
                 "origin": {"source": "paper", "reference": "Q1", "original": "How long?",
                            "reported": "5", "note": "Difiere."},
                 "expected": {"text": "Tres días (art. 46).", "value": 3, "unit": "días",
                              "articles": [46], "quote": "tres días"},
                 "criterion": {"kind": "value"}},
                {"id": "L01", "group": "base", "question": "¿Qué derechos?", "modes": ["base"],
                 "origin": {"source": "lopdp", "reference": "art. 13"},
                 "expected": {"text": "Acceso (art. 13).",
                              "elements": [{"name": "Derecho de acceso", "articles": [13],
                                            "terms": ["access"]}]},
                 "criterion": {"kind": "all"}},
                {"id": "G01", "group": "generic", "question": "¿Qué conceptos?",
                 "modes": ["base", "ontology"], "origin": {"source": "dpv", "reference": "rama"},
                 "looks_for": "dpv:DataSubjectRight", "criterion": {"kind": "at_least", "min": 1}},
            ]}


def _broken(path: list, value=None, delete=False) -> dict:
    data = _valid()
    node = data
    for key in path[:-1]:
        node = node[key]
    if delete:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return data


def test_a_complete_catalog_has_no_problems():
    assert qc.check(_valid()) == []
    assert qc.check(yaml.safe_load(qc.CATALOG_FILE.read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("path, value, delete, problem", [
    (["questions", 1, "origin"], None, True, "L01: le falta el origen"),
    (["questions", 1, "origin", "reference"], "", False, "L01: le falta el origen"),
    (["questions", 0, "origin", "reported"], None, True, "P01: al origen del artículo le falta «reported»"),
    (["questions", 1, "criterion"], None, True, "L01: le falta el criterio (all o any o value)"),
    (["questions", 2, "criterion", "kind"], "all", False, "G01: le falta el criterio (at_least)"),
    (["questions", 2, "criterion", "min"], 0, False, "G01: el criterio at_least necesita un mínimo de 1 o más"),
    (["questions", 1, "modes"], [], False, "L01: le faltan los modos (base, ontology, law u ontology+law)"),
    (["questions", 2, "modes"], ["ontology", "cloud"], False, "G01: le faltan los modos (base, ontology, law u ontology+law)"),
    (["questions", 1, "modes"], ["base", "law"], False, "L01: una pregunta de la base solo aplica a la base"),
    (["questions", 1, "expected"], None, True, "L01: le falta la respuesta esperada"),
    (["questions", 1, "expected", "text"], " ", False, "L01: la respuesta esperada no tiene texto"),
    (["questions", 1, "expected", "elements"], [], False, "L01: la respuesta esperada no trae sus elementos"),
    (["questions", 1, "expected", "elements", 0, "articles"], [], False, "L01: el elemento «Derecho de acceso» no dice su artículo"),
    (["questions", 1, "expected", "elements", 0, "articles"], [0], False, "L01: el elemento «Derecho de acceso» no dice su artículo"),
    (["questions", 1, "expected", "elements", 0, "name"], "", False, "L01: el elemento 1 no tiene nombre"),
    (["questions", 1, "expected", "elements", 0, "terms"], [""], False, "L01: los términos de «Derecho de acceso» deben ser textos"),
    (["questions", 0, "expected", "articles"], None, True, "P01: la respuesta esperada no dice su artículo"),
    (["questions", 0, "expected", "value"], "tres", False, "P01: la respuesta esperada no trae su valor"),
    (["questions", 0, "expected", "quote"], None, True, "P01: la respuesta esperada no trae el texto de la ley que la respalda"),
    (["questions", 0, "expected", "unit"], "", False, "P01: la respuesta esperada no trae su unidad"),
    (["questions", 2, "expected"], {"text": "x"}, False, "G01: una pregunta genérica no lleva respuesta esperada"),
    (["questions", 2, "looks_for"], "DataSubjectRight", False, "G01: le falta qué busca (dpv:Término, dpv:* o dct:source)"),
    (["questions", 1, "question"], "Qué derechos", False, "L01: la pregunta debe ir entre ¿ y ?"),
    (["questions", 1, "group"], "extra", False, "L01: el grupo debe ser base o generic"),
    (["questions", 1, "id"], "P01", False, "P01: el identificador está repetido"),
    (["questions", 1, "id"], "", False, "la pregunta 2 no tiene identificador"),
    (["version"], 0, False, "al catálogo le falta su versión"),
    (["paper"], None, True, "al catálogo le falta «paper»"),
    (["matching", "relations"], [], False, "al catálogo le falta «matching.relations»"),
])
def test_check_reports_what_is_missing(path, value, delete, problem):
    assert problem in qc.check(_broken(path, value, delete))


def test_load_catalog_refuses_an_incomplete_file(tmp_path):
    data = _valid()
    del data["questions"][1]["origin"]
    p = tmp_path / "questions.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(qc.CatalogError, match="L01: le falta el origen"):
        qc.load_catalog(p)
    good = copy.deepcopy(_valid())
    p.write_text(yaml.safe_dump(good, allow_unicode=True), encoding="utf-8")
    loaded = qc.load_catalog(p)
    assert [q.id for q in loaded.questions] == ["P01", "L01", "G01"]
    assert loaded.questions[1].expected.articles == (13,)           # taken from its elements
    assert loaded.questions[0].expected.articles == (46,)


# ------------------------------------------------------------------ how an answer is recognized

def test_normalize_splits_names_and_drops_accents():
    assert qc.normalize("DataSubject_Right-of-Access") == "data subject right of access"
    assert qc.normalize("Autoridad de ProtecciónDatos") == "autoridad de proteccion datos"
    assert qc.normalize("  Inequívoca. ") == "inequivoca"


def test_an_element_is_named_by_whole_words():
    access = qc.Element("Derecho de acceso", (13,), ("access", "right of access"))
    assert qc.names(access, "Right_of_access")
    assert qc.names(access, "derecho de ACCESO")
    assert not qc.names(access, "Accessibility")
    assert not qc.names(qc.Element("Libre", (8,), ("free", "freely given")), "Freedom")
    assert qc.names(qc.Element("Libre", (8,), ("free", "freely given")), "FreelyGiven")


def test_an_article_counts_only_when_no_other_element_shares_it(catalog):
    by_id = {q.id: q for q in catalog.questions}
    rights, bases = by_id["L01"], by_id["L02"]
    access = next(e for e in rights.expected.elements if e.name == "Derecho de acceso")
    assert qc.own_articles(rights, access) == {13}
    assert qc.found(rights, access, ["Alguna_cosa"], ["LOPDP, art. 13"])
    assert not qc.found(rights, access, ["Alguna_cosa"], ["LOPDP, art. 14"])
    legal = next(e for e in bases.expected.elements if e.name == "Obligación legal")
    assert qc.own_articles(bases, legal) == set()                  # art. 7 is shared by all eight
    assert not qc.found(bases, legal, ["Alguna_cosa"], ["LOPDP, art. 7"])
    assert qc.found(bases, legal, ["Legal_obligation"], [])
    assert qc.cited_articles("LOPDP, art. 41") == {41}
    assert qc.cited_articles("artículo 7 y Art.13") == {7, 13}


def test_the_criterion_of_each_kind(catalog):
    by_id = {q.id: q for q in catalog.questions}
    assert qc.criterion_text(by_id["P01"]) == "Verde si la respuesta trae el valor 3 días; amarillo si no lo trae."
    assert qc.criterion_text(by_id["L01"]) == ("Verde si la respuesta trae los 11 elementos que da "
                                               "la ley; amarillo si falta alguno.")
    assert qc.criterion_text(by_id["P04"]) == ("Verde si la respuesta trae al menos uno de los 2 "
                                               "elementos que da la ley; amarillo si no trae ninguno.")
    assert qc.criterion_text(by_id["G02"]) == ("Verde si la respuesta trae al menos 1 concepto; "
                                               "amarillo si no trae ninguno.")
    assert qc.matching_text(catalog)[1] == (
        "En las genéricas cuenta un concepto alineado por exactMatch, closeMatch o broadMatch con "
        "el término del DPV o con uno que esté debajo (rdfs:subClassOf o skos:broader).")
