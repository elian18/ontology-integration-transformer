"""Tests for the competency questions answered with SPARQL (S9-T04).

Synthetic answers check how each kind of question is judged; toy graphs check the rdflib
engine and the generic queries on an ontology that is not OntoPriv. Guarded tests run the
real base: its colors, the original OntoPriv against the base, the four formats and the same
rows from Jena and from rdflib."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.validation import logic_inspector as li
from src.validation import question_catalog as qc
from src.validation import questions as qs
from src.validation.shape_inspector import GREEN, YELLOW

ROOT = Path(__file__).resolve().parents[1]
DPV = ROOT / "vocab" / "dpv.ttl"
BASE_FILES = [ROOT / "data" / "output" / name for name in
              ("ontopriv-core.rdf", "profile-ecuador-lopdp.rdf", "ontopriv-dpv-alignment.rdf")]
needs_base = pytest.mark.skipif(not all(p.exists() for p in BASE_FILES),
                                reason="Falta la base en data/output")


def _java() -> str | None:
    if not shutil.which("java"):
        return None
    done = subprocess.run(["java", "-version"], capture_output=True, text=True)
    return li.java_version(done.stderr + done.stdout)


JAVA = _java()
HAS_JENA = (JAVA is not None and int(JAVA.split(".")[0]) >= li.MIN_JAVA
            and li.jena_version(li._home(li.Settings())) is not None)
needs_jena = pytest.mark.skipif(not HAS_JENA, reason="Falta Java 21+ o Apache Jena en tools/")


@pytest.fixture(scope="module")
def catalog():
    return qc.load_catalog()


def _q(catalog, qid):
    return next(q for q in catalog.questions if q.id == qid)


def uri(v):
    return {"t": "uri", "v": v}


def lit(v, dt="http://www.w3.org/2001/XMLSchema#string", lang=None):
    return {"t": "literal", "v": v, "dt": dt, "lang": lang}


# ------------------------------------------------------------------ the queries

def test_each_question_has_its_query_and_no_query_is_left_over(catalog):
    ids = {q.id for q in catalog.questions}
    assert {p.stem for p in qs.QUERIES_DIR.glob("*.rq")} == ids
    for q in catalog.questions:
        text = qs.query_text(q.id)
        assert text.splitlines()[0] == f"# {q.id} · {q.question}"
        assert re.search(r"^SELECT DISTINCT \?concept\b", text, re.M), q.id


def test_the_generic_queries_do_not_depend_on_ontopriv(catalog):
    for q in (q for q in catalog.questions if q.group == qc.GENERIC):
        text = qs.query_text(q.id)
        assert "semanticweb.org" not in text and "onto:" not in text, q.id
        if q.looks_for.startswith("dpv:") and q.looks_for != qc.ANY_DPV:
            assert q.looks_for in text, q.id


def test_no_query_lists_its_expected_answer(catalog):
    """A query may name the ontology's classes and properties, never the answers themselves."""
    for q in (q for q in catalog.questions if q.group == qc.BASE):
        body = "\n".join(l for l in qs.query_text(q.id).splitlines() if not l.startswith("#"))
        assert "VALUES ?concept" not in body, q.id
        for e in q.expected.elements:
            assert qc.normalize(e.name) not in qc.normalize(body), (q.id, e.name)


# ------------------------------------------------------------------ judging an answer

def test_a_value_question_looks_for_its_number_in_the_texts(catalog):
    q = _q(catalog, "P01")
    answer = {"vars": ["concept", "value"],
              "rows": [[uri("http://x#notification001"), lit("3 days")]]}
    r = qs.judge(q, answer)
    assert r["color"] == GREEN and r["values"] == ["3"]
    assert r["answer"] == ["notification001 3 days"]
    five = {"vars": ["concept", "value"], "rows": [[uri("http://x#n3"), lit("5 days")]]}
    assert qs.judge(q, five)["color"] == YELLOW            # the 3 of «n3» is a name, not a value
    integer = {"vars": ["concept", "value"],
               "rows": [[uri("http://x#r"), lit("15", "http://www.w3.org/2001/XMLSchema#integer")]]}
    assert qs.judge(_q(catalog, "L05"), integer)["values"] == ["15"]


def test_a_list_question_finds_elements_by_name_or_by_their_own_article(catalog):
    q = _q(catalog, "L01")
    rows = [[uri("http://x#Right_of_access"), None, None, lit("LOPDP, art. 13", lang="es")],
            [uri("http://x#Something"), None, None, lit("LOPDP, art. 22", lang="es")]]
    r = qs.judge(q, {"vars": ["concept", "label", "match", "source"], "rows": rows})
    assert r["color"] == YELLOW
    assert r["found"] == [{"name": "Derecho de acceso", "how": "nombre"},
                          {"name": "Derecho de consulta", "how": "artículo"}]
    assert len(r["missing"]) == 9 and "Derecho a la portabilidad" in r["missing"]
    shared = _q(catalog, "L02")                            # art. 7 is shared by all eight
    r = qs.judge(shared, {"vars": ["concept", "source"],
                          "rows": [[uri("http://x#Thing"), lit("LOPDP, art. 7", lang="es")]]})
    assert r["found"] == [] and r["color"] == YELLOW


def test_all_needs_every_element_and_any_needs_one(catalog):
    members = _q(catalog, "P03")
    names = ["data_owner", "responsible", "DataProcessor", "recipient", "Control_Authority", "delegate"]
    answer = {"vars": ["concept"], "rows": [[uri(f"http://x#{n}")] for n in names]}
    r = qs.judge(members, answer)
    assert r["color"] == YELLOW and r["missing"] == ["Responsable del tratamiento"]
    answer["rows"].append([uri("http://x#DataController")])
    assert qs.judge(members, answer)["color"] == GREEN
    who = _q(catalog, "P04")
    one = {"vars": ["concept"], "rows": [[uri("http://x#supervisory_authority")]]}
    assert qs.judge(who, one)["color"] == GREEN
    assert qs.judge(who, {"vars": ["concept"], "rows": []})["color"] == YELLOW


def test_a_generic_question_counts_distinct_concepts(catalog):
    q = _q(catalog, "G02")
    rows = [[uri("http://x#A"), uri("https://w3id.org/dpv#RightOfAccess")],
            [uri("http://x#A"), uri("https://w3id.org/dpv#DataSubjectRight")]]
    r = qs.judge(q, {"vars": ["concept", "match"], "rows": rows})
    assert r["color"] == GREEN and r["count"] == 1
    assert qs.judge(q, {"vars": ["concept", "match"], "rows": []})["color"] == YELLOW


def test_a_question_whose_query_did_not_run_is_gray(catalog):
    q = _q(catalog, "G01")
    r = qs.judge(q, {"error": "QueryParseException: línea 3"})
    assert r["color"] == qs.GRAY and r["error"] == "QueryParseException: línea 3"
    assert qs.judge(q, None)["error"] == "la consulta no corrió"
    assert qs.summary([r, {"color": GREEN}]) == {GREEN: 1, YELLOW: 0, qs.GRAY: 1,
                                                 qs.NOT_APPLICABLE: 0}


def test_the_color_of_a_set_of_questions_leaves_out_what_does_not_apply():
    na = {"color": qs.GRAY, "applies": False}
    assert qs.status_of([{"color": GREEN}, na]) == GREEN
    assert qs.status_of([{"color": GREEN}, {"color": qs.GRAY}, na]) == YELLOW   # did not run
    assert qs.status_of([{"color": YELLOW, "applies": True}]) == YELLOW
    assert qs.summary([{"color": GREEN}, na]) == {GREEN: 1, YELLOW: 0, qs.GRAY: 0,
                                                  qs.NOT_APPLICABLE: 1}
    assert qs.counts_text(qs.summary([{"color": GREEN}, na]), 2) == (
        "1 en verde, 0 en amarillo y 0 en gris de las 1 que aplican; 1 no aplica")
    assert qs.counts_text({GREEN: 2, YELLOW: 0, qs.GRAY: 0, qs.NOT_APPLICABLE: 2}) == (
        "2 en verde, 0 en amarillo y 0 en gris; 2 no aplican")


def test_rows_compare_the_same_from_jena_and_from_rdflib():
    from rdflib import BNode, Literal, URIRef
    from rdflib.namespace import XSD
    jena = [[uri("http://x#a"), lit("hola"), {"t": "bnode"},
             lit("hola", "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString", "es"),
             lit("15", "http://www.w3.org/2001/XMLSchema#integer"), None]]
    mine = [[qs._cell(URIRef("http://x#a")), qs._cell(Literal("hola")), qs._cell(BNode()),
             qs._cell(Literal("hola", lang="ES")), qs._cell(Literal("15", datatype=XSD.integer)),
             qs._cell(None)]]
    assert qs.rows_of({"rows": jena}) == qs.rows_of({"rows": mine})
    assert qs.rows_of({"rows": [[lit("hola", lang="en")]]}) != qs.rows_of({"rows": [[lit("hola")]]})


# ------------------------------------------------------------------ the engine on toy graphs

TOY = """@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix dpv: <https://w3id.org/dpv#> .
@prefix ex: <http://example.org/mine#> .
ex:Rights skos:exactMatch dpv:DataSubjectRight .
ex:Interest skos:closeMatch dpv:LegitimateInterest .
ex:Customer skos:broadMatch dpv:DataSubject .
ex:Note skos:relatedMatch dpv:Purpose .
ex:Complaint skos:relatedMatch dpv:DataSubjectRight .
"""


@pytest.fixture
def toy(tmp_path):
    p = tmp_path / "mine.ttl"
    p.write_text(TOY, encoding="utf-8")
    return {"toy": [p, DPV]}


def test_the_generic_queries_answer_an_ontology_that_is_not_ontopriv(catalog, toy):
    ids = [q.id for q in qc.for_mode(catalog, "ontology")]
    used, runs = qs.run(toy, ids, engine=qs.RDFLIB)
    assert used["name"] == qs.RDFLIB and used["reason"] == "elegido con --motor"
    colors = {q.id: qs.judge(q, runs["toy"]["answers"][q.id])["color"]
              for q in qc.for_mode(catalog, "ontology")}
    assert colors == {"G01": GREEN, "G02": GREEN, "G03": GREEN, "G04": YELLOW,
                      "G05": YELLOW, "G06": GREEN, "G07": YELLOW}   # relatedMatch does not count
    g03 = runs["toy"]["answers"]["G03"]                     # LegitimateInterest is below LegalBasis
    assert qs.rows_of(g03) == {(("uri", "http://example.org/mine#Interest"),
                                ("uri", "https://w3id.org/dpv#LegitimateInterest"))}
    assert qs.judge(_q(catalog, "G01"), runs["toy"]["answers"]["G01"])["count"] == 3
    assert qs.judge(_q(catalog, "G02"), runs["toy"]["answers"]["G02"])["count"] == 1
    assert runs["toy"]["triples"] > 14000                   # the DPV is loaded with it


def test_a_broken_query_goes_gray_and_the_rest_still_run(catalog, toy, tmp_path, monkeypatch):
    folder = tmp_path / "queries"
    folder.mkdir()
    for qid in ("G01", "G02"):
        (folder / f"{qid}.rq").write_text(qs.query_text(qid), encoding="utf-8")
    (folder / "G02.rq").write_text("SELECT ?concept WHERE { ?concept ", encoding="utf-8")
    monkeypatch.setattr(qs, "QUERIES_DIR", folder)
    _, runs = qs.run(toy, ["G01", "G02"], engine=qs.RDFLIB)
    assert "rows" in runs["toy"]["answers"]["G01"] and "error" in runs["toy"]["answers"]["G02"]
    assert qs.judge(_q(catalog, "G02"), runs["toy"]["answers"]["G02"])["color"] == qs.GRAY


def test_without_java_the_engine_is_rdflib_and_says_why(toy):
    used, runs = qs.run(toy, ["G01"], java="17.0.2")
    assert used["name"] == qs.RDFLIB
    assert used["reason"] == "Java 17.0.2: Jena 6 necesita Java 21 o superior"
    assert "rows" in runs["toy"]["answers"]["G01"]


def test_a_failed_jena_falls_back_to_rdflib_unless_jena_was_asked(toy, monkeypatch):
    monkeypatch.setattr(qs.li, "choose_engine", lambda s, java=None: {
        "name": qs.JENA, "version": "6.2.0", "java": "21", "home": Path("tools/jena")})
    def broken(*args, **kwargs):
        raise RuntimeError("Jena falló: OutOfMemoryError")
    monkeypatch.setattr(qs, "_jena", broken)
    used, _ = qs.run(toy, ["G01"])
    assert used == {"name": qs.RDFLIB, "reason": "Jena falló: OutOfMemoryError",
                    "version": used["version"]}
    with pytest.raises(RuntimeError, match="OutOfMemoryError"):
        qs.run(toy, ["G01"], engine=qs.JENA)


def test_compare_lists_the_questions_whose_rows_differ():
    runs = {"a": {"answers": {"Q1": {"rows": [[uri("x")]]}, "Q2": {"rows": []},
                              "Q3": {"rows": [[uri("y")]]}}},
            "b": {"answers": {"Q1": {"rows": [[uri("x")]]}, "Q2": {"error": "e"},
                              "Q3": {"rows": [[uri("z")]]}}}}
    assert qs.compare(runs, ["Q1", "Q2", "Q3"], "a", "b") == [
        {"id": "Q2", "only_in_a": 0, "only_in_b": 0},           # no rows, but one did not run
        {"id": "Q3", "only_in_a": 1, "only_in_b": 1}]


# ------------------------------------------------------------------ the console

def _report(results, *, original=(), formats=None, engine=None):
    return {"date": "2026-11-07T00:00:00+00:00", "seconds": 1.0, "triples": 10,
            "engine": engine or {"name": qs.JENA, "version": "6.2.0", "java": "21.0.2"},
            "catalog": {"file": "src/validation/questions.yaml", "version": 1, "sha256": "a" * 64},
            "files": ["data/output/ontopriv-core.rdf"], "results": results,
            "summary": qs.summary(results),
            "original": {"files": ["data/input/ontopriv.rdf"], "differences": list(original),
                         "colors": {"P01": GREEN, "P02": YELLOW}},
            "formats": formats or {"owl": [], "turtle": [], "jsonld": []}}


def test_the_log_shows_each_question_with_its_color_and_the_engine(catalog):
    results = [qs.judge(_q(catalog, "P01"), {"vars": ["concept", "value"],
                                              "rows": [[uri("http://x#n"), lit("3 days")]]}),
               qs.judge(_q(catalog, "P02"), {"vars": ["concept", "value"],
                                              "rows": [[uri("http://x#c"), lit("Free")]]}),
               qs.judge(_q(catalog, "G01"), {"error": "sin Java"})]
    text = qs.render_log(_report(results))
    assert "  Motor: Apache Jena 6.2.0, SPARQL sin inferencia (Java 21.0.2)" in text
    assert "  Resultado: 1 en verde, 1 en amarillo y 1 en gris de 3 (1.0 s)" in text
    assert f"    [verde] P01 {_q(catalog, 'P01').question}" in text
    assert "         Valores que trae: 3 · n 3 days" in text
    assert ("         Trae 1 de 4: Libre. Falta: Específica, Informada, Inequívoca") in text
    assert "    [gris] G01 " in text and "         No corrió: sin Java" in text
    assert "    dan las mismas filas que la base." in text
    assert "    los cuatro dan las mismas filas en las 3 preguntas." in text
    rdflib = qs.render_log(_report(results, engine={"name": qs.RDFLIB, "version": "7.6.0",
                                                     "reason": "no se encontró Java (java)"}))
    assert "  Motor: rdflib 7.6.0, SPARQL sin inferencia (no se encontró Java (java))" in rdflib


def test_the_log_says_what_changed_in_the_original_and_in_a_format(catalog):
    results = [qs.judge(_q(catalog, "P01"), {"vars": ["concept"], "rows": []})]
    text = qs.render_log(_report(
        results, original=[{"id": "P02", "only_in_base": 2, "only_in_original": 0}],
        formats={"owl": [], "turtle": [{"id": "L03"}], "jsonld": []}))
    assert "    dan las mismas filas que la base en 1; cambian:" in text
    assert ("    - P02: 2 filas solo en la base, 0 solo en el original (color en el original: "
            "amarillo)") in text
    assert "    - Turtle: L03 da otras filas" in text
    assert "los cuatro dan las mismas filas" not in text


def test_the_cli_needs_base():
    assert qs._run([]) == 1


# ------------------------------------------------------------------ the real base

@needs_base
def test_the_real_base_with_rdflib(tmp_path):
    report = qs.inspect_base(engine=qs.RDFLIB, out_dir=tmp_path)
    colors = {r["id"]: r["color"] for r in report["results"]}
    assert [k for k, c in colors.items() if c == YELLOW] == ["P02", "L03", "L04", "L09", "L10"]
    assert report["summary"] == {GREEN: 18, YELLOW: 5, qs.GRAY: 0, qs.NOT_APPLICABLE: 0}
    assert report["original"]["differences"] == []
    assert report["original"]["colors"] == {"P01": GREEN, "P02": YELLOW, "P03": GREEN,
                                            "P04": GREEN, "P05": GREEN}
    assert report["formats"] == {"owl": [], "turtle": [], "jsonld": []}
    by_id = {r["id"]: r for r in report["results"]}
    assert by_id["P01"]["values"] == ["3"] and by_id["P02"]["missing"] == ["Inequívoca"]
    assert by_id["P03"]["missing"] == [] and len(by_id["P03"]["found"]) == 6
    saved = json.loads((tmp_path / qs.REPORT_FILE).read_text(encoding="utf-8"))
    assert saved["catalog"]["sha256"] == qc.load_catalog().sha256
    assert (tmp_path / qs.LOG_FILE).read_text(encoding="utf-8").startswith(
        "Preguntas de competencia de la base del sistema")


@needs_base
@needs_jena
def test_jena_and_rdflib_return_the_same_rows_on_the_base(catalog):
    sets = qs.base_files()
    ids = [q.id for q in catalog.questions]
    jena, by_jena = qs.run(sets, ids, engine=qs.JENA)
    _, by_rdflib = qs.run(sets, ids, engine=qs.RDFLIB)
    assert jena["name"] == qs.JENA
    for name in sets:
        assert by_jena[name]["triples"] == by_rdflib[name]["triples"], name
        for qid in ids:
            assert qs.rows_of(by_jena[name]["answers"][qid]) == \
                qs.rows_of(by_rdflib[name]["answers"][qid]), (name, qid)


@needs_base
@needs_jena
def test_the_cli_answers_the_base_with_jena(capsys):
    assert qs._run(["--base", "--motor", "jena"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Preguntas de competencia de la base del sistema")
    assert "  Motor: Apache Jena 6.2.0, SPARQL sin inferencia" in out
    assert "  Resultado: 18 en verde, 5 en amarillo y 0 en gris de 23" in out
