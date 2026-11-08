"""Tests for the AI that drafts the SPARQL of each competency question (S9-T08).

A fake LLM answers with the reference queries, broken ones or other ones, so the measurement, the
batches and the cache are checked without Gemini; a toy graph checks the summary the AI sees. A
guarded test asks the real AI for the 23 drafts of the base (slow)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Graph

from src.validation import query_drafter as qd
from src.validation import question_catalog as qc
from src.validation import questions as qs
from src.validation.shape_inspector import GREEN, YELLOW
from tests.test_questions import HAS_JENA

needs_jena = pytest.mark.skipif(not HAS_JENA, reason="Falta Java 21+ o Apache Jena en tools/")

ROOT = Path(__file__).resolve().parents[1]
BASE_FILES = [ROOT / "data" / "output" / name for name in
              ("ontopriv-core.rdf", "profile-ecuador-lopdp.rdf", "ontopriv-dpv-alignment.rdf")]
needs_base = pytest.mark.skipif(not all(p.exists() for p in BASE_FILES),
                                reason="Falta la base en data/output")
TOY = """
@prefix onto: <http://www.semanticweb.org/ley-organica-proteccion-datos-personales#> .
@prefix op: <http://www.semanticweb.org/franc/ontologies/2023/9/OntologiaLOPDP#> .
@prefix ex: <http://example.org/otra#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dpv: <https://w3id.org/dpv#> .
onto:Rights a owl:Class .
onto:Right_of_access a owl:Class ; rdfs:subClassOf onto:Rights ;
    skos:broadMatch dpv:DataSubjectRight ; dct:source "LOPDP, art. 13" ;
    rdfs:label "Derecho de acceso"@es .
op:Notification a owl:Class .
op:Holder a owl:Class .
op:time_of_a_vulnerability_notification a owl:DatatypeProperty ;
    rdfs:domain op:Notification ; rdfs:range xsd:string .
op:notifies a owl:ObjectProperty ; rdfs:domain [ owl:unionOf ( op:Notification onto:Rights ) ] ;
    rdfs:range op:Holder .
op:notification001 a owl:NamedIndividual , op:Notification ;
    op:time_of_a_vulnerability_notification "3 days" ; op:notifies op:holder001 .
op:holder001 a op:Holder .
ex:Thing a owl:Class .
"""


@pytest.fixture(scope="module")
def catalog():
    return qc.load_catalog()


def _q(catalog, qid):
    return next(q for q in catalog.questions if q.id == qid)


def _ids(context: str) -> list[str]:
    return [b.split(" ===", 1)[0] for b in context.split("=== ")[1:]
            if not b.startswith("RESUMEN")]


class DraftLLM:
    """Answers each question of the context: the reference query, unless ``drafts`` (sparql.v1)
    or ``drafts2`` (sparql.v2) say what."""

    def __init__(self, drafts=None, error=None, model="modelo-de-prueba", drafts2=None):
        self.drafts, self.drafts2, self.error, self.model = drafts or {}, drafts2 or {}, error, model
        self.calls, self.contexts, self.prompts = 0, [], []

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        self.calls += 1
        self.contexts.append(context)
        self.prompts.append(prompt)
        if self.error:
            raise Exception(self.error)
        drafts = self.drafts2 if prompt == qd.PROMPTS[2].text else self.drafts
        out = []
        for qid in _ids(context):
            sparql = drafts.get(qid, "reference")
            if sparql is None:                                   # left out of the answer
                continue
            out.append({"id": qid.lower() if qid == "G07" else qid,
                        "sparql": qs.query_text(qid) if sparql == "reference" else sparql})
        return "```json\n" + json.dumps({"queries": out}) + "\n```"


def _toy(tmp_path) -> Path:
    path = tmp_path / "toy.ttl"
    path.write_text(TOY, encoding="utf-8")
    return path


# ------------------------------------------------------------------ what the AI sees

def test_the_summary_is_made_from_the_triples(tmp_path):
    text = qd.summary_text([_toy(tmp_path)])
    lines = text.splitlines()
    assert "PREFIX onto: <http://www.semanticweb.org/ley-organica-proteccion-datos-personales#>" \
        in lines
    assert "PREFIX ns1: <http://example.org/otra#>" in lines                 # an unknown one
    assert "onto:Right_of_access < onto:Rights" in lines
    assert ("op:notifies: onto:Rights | op:Notification -> op:Holder; 1 usos" in lines)
    assert ("op:time_of_a_vulnerability_notification: op:Notification -> xsd:string; 1 usos"
            in lines)
    assert ("op:notification001: op:Notification; usa op:notifies, "
            "op:time_of_a_vulnerability_notification") in lines
    assert "op:holder001: op:Holder" in lines
    assert ("Los conceptos se enlazan con términos del DPV con skos:exactMatch (0), "
            "skos:closeMatch (0), skos:broadMatch (1), skos:narrowMatch (0), "
            "skos:relatedMatch (0).") in lines
    assert "Términos del DPV enlazados (1): dpv:DataSubjectRight." in lines
    assert ('dct:source guarda el artículo de la ley del que sale el concepto, en 1 conceptos '
            '(p. ej. "LOPDP, art. 13").') in lines
    assert "rdfs:label: 1 en es. skos:definition: ninguno." in lines
    assert "CIFRAS: " in text and "3 days" not in text                    # no values, no answers
    assert qd.summary_text([_toy(tmp_path)]) == text                      # the same every time


@needs_base
def test_the_summary_of_the_base_never_shows_a_reference_query():
    files = qs.base_files()[qs.BASE_SET]
    text = qd.summary_text(files[:-1])
    assert "SELECT" not in text and "WHERE" not in text
    assert "op:Notification" in text and "dpv:DataSubjectRight" in text
    for q in qc.load_catalog().questions:
        assert q.question not in text


def test_the_context_has_the_summary_and_one_block_per_question(catalog):
    text = qd.context("RESUMEN", [_q(catalog, "P01"), _q(catalog, "G02")])
    assert text == ("=== RESUMEN DE LA ONTOLOGIA ===\nRESUMEN\n\n=== P01 ===\n"
                    f"{_q(catalog, 'P01').question}\n\n=== G02 ===\n{_q(catalog, 'G02').question}")
    assert "Verde si" not in text and "criterion" not in text            # no criterion either


def test_the_prompts_ask_for_one_select_per_question():
    for version in (1, 2):
        text = qd.PROMPTS[version].text
        assert qd.PROMPTS[version].name == "sparql" and "?concept" in text and '"queries"' in text
        assert "SELECT DISTINCT" in text and "SERVICE" in text
    v1, v2 = qd.PROMPTS[1].text, qd.PROMPTS[2].text
    assert "COMO ESTA ARMADA LA ONTOLOGIA" in v2 and "COMO ESTA ARMADA" not in v1
    assert "Copia al inicio de cada consulta el bloque PREFIJOS" in v2
    assert "dpv:DataSubjectRight" not in v2 and "dpv:LegalBasis" not in v2   # no answer in it


def test_the_system_completes_the_standard_prefixes():
    fixed, added = qd.complete_prefixes(
        'PREFIX onto: <http://www.semanticweb.org/ley-organica-proteccion-datos-personales#> '
        'SELECT DISTINCT ?concept WHERE { ?concept rdfs:subClassOf onto:Rights ; dct:source ?s . }')
    assert added == ["dct", "rdfs"]
    assert fixed.splitlines()[:2] == ["PREFIX dct: <http://purl.org/dc/terms/>",
                                      "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"]
    assert fixed.endswith("dct:source ?s . }")
    text = ('PREFIX ONTO: <x#> SELECT ?concept WHERE { ?concept a onto:X ; rdfs:label "a skos:b" ;'
            " ex:p 'owl:c' . <http://e.org/a:b> ?p ?o } # xsd:string")
    fixed, added = qd.complete_prefixes(text)
    assert added == ["onto", "rdfs"]           # ONTO is another prefix; ex is not standard
    assert fixed.endswith(text)
    same = qd.complete_prefixes(qs.query_text("L01"))
    assert same == (qs.query_text("L01"), [])                        # nothing is missing


def test_the_settings_come_from_config():
    assert qd.draft_settings() == qd.DraftSettings(prompt_version=2, batch_size=8,
                                                   max_tokens=16384, query_timeout_seconds=20)
    assert qd.draft_settings({"questions": {}}) == qd.DraftSettings()
    for bad in ({"batch_size": 0}, {"prompt_version": 9}, {"query_timeout_seconds": 0},
                {"max_tokens": 0}):
        with pytest.raises(ValueError, match="questions.drafts"):
            qd.draft_settings({"questions": {"drafts": bad}})


# ------------------------------------------------------------------ asking

def test_parse_the_answer_of_the_ai():
    assert qd.parse_queries('{"queries": [{"id": "P01", "sparql": "SELECT"}]}') == [
        {"id": "P01", "sparql": "SELECT"}]
    assert qd.parse_queries('[{"id": "P01"}, 3]') == [{"id": "P01"}]
    assert qd.parse_queries('Aquí va: {"queries": [{"id": "G01"}]} listo') == [{"id": "G01"}]
    with pytest.raises(ValueError, match="lista de consultas"):
        qd.parse_queries("no hay JSON")


def test_batches_and_the_cache(catalog, tmp_path):
    questions = qc.for_mode(catalog, "base")
    llm, settings = DraftLLM(), qd.DraftSettings(batch_size=10)
    cache = tmp_path / qd.CACHE_FILE
    report = qd.ask(questions, "RESUMEN", llm=llm, settings=settings, cache_path=cache)
    assert (llm.calls, report.cached, len(report.drafts)) == (3, 0, 23)    # 10 + 10 + 3
    assert [len(_ids(c)) for c in llm.contexts] == [10, 10, 3]
    assert report.drafts["G07"] == qs.query_text("G07").strip()             # "g07" counts too
    assert llm.prompts[0] == qd.PROMPTS[1].text
    again = qd.ask(questions, "RESUMEN", llm=llm, settings=settings, cache_path=cache)
    assert llm.calls == 3 and (again.calls, again.cached) == (0, 23)
    assert again.drafts == report.drafts
    other = qd.ask(questions[:2], "OTRO RESUMEN", llm=llm, settings=settings, cache_path=cache)
    assert other.calls == 1 and other.cached == 0                # another base: asked again
    offline = qd.ask(questions, "NUEVO", llm=None, settings=settings, cache_path=cache)
    assert (offline.calls, offline.drafts) == (0, {})


def test_invalid_answers_and_errors(catalog, tmp_path):
    questions = [_q(catalog, q) for q in ("P01", "P02", "P03")]
    llm = DraftLLM({"P02": "", "P03": None})
    report = qd.ask(questions, "R", llm=llm, settings=qd.DraftSettings(), cache_path=None)
    assert sorted(report.drafts) == ["P01"] and report.invalid == 1
    waits = []
    failing = DraftLLM(error="429 RESOURCE_EXHAUSTED")
    report = qd.ask(questions, "R", llm=failing, settings=qd.DraftSettings(), cache_path=None,
                    sleep_fn=waits.append)
    assert failing.calls == 2 and len(waits) == 1 and report.calls == 1
    assert report.errors == ["429 RESOURCE_EXHAUSTED"] and report.drafts == {}
    busy = DraftLLM(error="503 UNAVAILABLE")
    qd.ask(questions, "R", llm=busy, settings=qd.DraftSettings(), cache_path=None,
           sleep_fn=waits.append)
    assert busy.calls == 2 and len(waits) == 2
    waits.clear()
    broken = DraftLLM(error="400 bad request")
    report = qd.ask(questions, "R", llm=broken, settings=qd.DraftSettings(), cache_path=None,
                    sleep_fn=waits.append)
    assert broken.calls == 1 and waits == []
    assert qd.ai_text(report.ai(True)) == ("1 llamada; 0 borradores desde la caché; "
                                           "error: 400 bad request")
    assert qd.ai_text(qd.DraftReport(cached=1).ai(False, "se pidió --sin-ia")) == (
        "no usada; se pidió --sin-ia; 1 borrador desde la caché")


def test_only_the_first_draft_of_a_question_asked_in_that_call(catalog):
    class Repeating:
        model, calls = "m", 0

        def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
            self.calls += 1
            return json.dumps({"queries": [
                {"id": "P01", "sparql": "SELECT ?concept WHERE {}"},
                {"id": "P01", "sparql": "otra"}, {"id": "L01", "sparql": "no se pidió"},
                {"id": "P02", "sparql": 7}]})
    report = qd.ask([_q(catalog, "P01"), _q(catalog, "P02")], "R", llm=Repeating(),
                    settings=qd.DraftSettings(), cache_path=None)
    assert report.drafts == {"P01": "SELECT ?concept WHERE {}"} and report.invalid == 1


# ------------------------------------------------------------------ judging a draft

def uri(v):
    return {"t": "uri", "v": v}


def lit(v):
    return {"t": "literal", "v": v, "dt": "http://www.w3.org/2001/XMLSchema#string", "lang": None}


ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"


def test_the_same_answer_with_other_columns(catalog):
    q = _q(catalog, "P02")                    # free, specific, informed, unequivocal
    reference = {"vars": ["concept", "label"],
                 "rows": [[uri(ONTO + "Free"), lit("free")], [uri(ONTO + "Specific"), None]]}
    draft = {"vars": ["concept"], "rows": [[uri(ONTO + "Specific")], [uri(ONTO + "Free")]]}
    r = qd.outcome(q, reference, draft)
    assert (r["outcome"], r["same_color"], r["same_rows"]) == (qd.SAME, True, False)
    assert r["reference"] == r["draft"] and r["reference_color"] == YELLOW
    fewer = {"vars": ["concept"], "rows": [[uri(ONTO + "Free")]]}
    r = qd.outcome(q, reference, fewer)
    assert (r["outcome"], r["same_color"]) == (qd.OTHER, True)        # yellow too, but less
    assert r["draft"].startswith("Trae 1 de 4: Libre")
    swapped = {"vars": ["concept"], "rows": [[uri(ONTO + "Free")], [uri(ONTO + "Informed")]]}
    r = qd.outcome(q, reference, swapped)                     # also 2 of 4, but other two
    assert (r["outcome"], r["same_color"]) == (qd.OTHER, True)


def test_values_and_concepts(catalog):
    p01 = _q(catalog, "P01")                  # 3 days
    reference = {"vars": ["concept", "value"], "rows": [[uri("n1"), lit("3 days")]]}
    assert qd.outcome(p01, reference, {"vars": ["x"], "rows": [[lit("3")]]})["outcome"] == qd.SAME
    r = qd.outcome(p01, reference, {"vars": ["x"], "rows": [[lit("3 o 5 días")]]})
    assert (r["outcome"], r["same_color"], r["draft_color"]) == (qd.OTHER, True, GREEN)
    g01 = _q(catalog, "G01")
    reference = {"vars": ["concept", "match"], "rows": [[uri("a"), uri("d1")], [uri("b"), uri("d2")]]}
    same = {"vars": ["concept"], "rows": [[uri("b")], [uri("a")]]}
    assert qd.outcome(g01, reference, same)["outcome"] == qd.SAME
    other = {"vars": ["concept"], "rows": [[uri("a")], [uri("c")]]}   # 2 concepts, not the same
    assert qd.outcome(g01, reference, other)["outcome"] == qd.OTHER
    renamed = {"vars": ["c"], "rows": [[uri("a")], [uri("b")]]}       # no ?concept: 0 concepts
    r = qd.outcome(g01, reference, renamed)
    assert (r["outcome"], r["draft_color"], r["same_color"]) == (qd.OTHER, YELLOW, False)


def _rows(right1, right2, ids=("P01", "P02", "P03")):
    return [{"id": i, "group": qc.BASE, "versions": {
        str(v): {qd.PREFIXED: {"outcome": qd.SAME if i in right else qd.OTHER}}
        for v, right in ((1, right1), (2, right2))}} for i in ids]


def test_the_rule_to_adopt_v2():
    assert qd.adopted(_rows({"P01"}, {"P01", "P02"})) == 2           # more, and loses none
    assert qd.adopted(_rows({"P01"}, {"P02", "P03"})) == 1           # more, but loses P01
    assert qd.adopted(_rows({"P01"}, {"P02"})) == 1                  # as many: v1 stays
    assert qd.adopted(_rows(set(), set())) == 1
    assert "con los prefijos completados" in qd.ADOPTION


def test_a_draft_that_does_not_run(catalog):
    q = _q(catalog, "G02")
    reference = {"vars": ["concept"], "rows": [[uri("a")]]}
    r = qd.outcome(q, reference, {"error": "QueryParseException: línea 3"})
    assert (r["outcome"], r["reason"]) == (qd.BROKEN, "QueryParseException: línea 3")
    assert qd.outcome(q, reference, None)["outcome"] == qd.BROKEN      # the engine gave nothing
    r = qd.outcome(q, reference, None, refused="usa SERVICE")
    assert (r["outcome"], r["reason"]) == (qd.BROKEN, "usa SERVICE")
    r = qd.outcome(q, reference, None, missing="la IA no dio una consulta")
    assert (r["outcome"], r["reason"]) == (qd.NO_DRAFT, "la IA no dio una consulta")


def test_the_table_counts_by_group():
    rows = [{"group": qc.BASE, "outcome": qd.SAME, "same_rows": True},
            {"group": qc.BASE, "outcome": qd.OTHER, "same_color": True, "same_rows": False},
            {"group": qc.GENERIC, "outcome": qd.OTHER, "same_color": False},
            {"group": qc.GENERIC, "outcome": qd.BROKEN}]
    t = qd.table(rows)
    assert t[qc.BASE] == {qd.SAME: 1, qd.OTHER: 1, qd.BROKEN: 0, qd.NO_DRAFT: 0,
                          "same_color": 1, "same_rows": 1, "questions": 2}
    assert t["total"][qd.OTHER] == 2 and t["total"]["same_color"] == 1
    assert t["total"]["questions"] == 4


def test_the_service_keyword_is_refused():
    assert qd._SERVICE.search("SELECT * WHERE { SERVICE <http://x> { ?s ?p ?o } }")
    assert qd._SERVICE.search("select * { service silent ?e { } }")
    assert not qd._SERVICE.search("SELECT ?c WHERE { ?c a op:Service . }")


def test_a_slow_draft_is_stopped(tmp_path):
    rq = tmp_path / "slow.draft.rq"
    rq.write_text("SELECT * WHERE { ?a ?b ?c . ?d ?e ?f . ?g ?h ?i . }", encoding="utf-8")
    toy = {"toy": [_toy(tmp_path)]}
    _, runs = qs.run(toy, ["slow.draft"], engine=qs.RDFLIB, paths={"slow.draft": rq},
                     query_timeout=0.05)
    assert "tardó más de 0.05 s" in runs["toy"]["answers"]["slow.draft"]["error"]
    _, runs = qs.run(toy, ["slow.draft"], engine=qs.RDFLIB, paths={"slow.draft": rq})
    assert len(runs["toy"]["answers"]["slow.draft"]["rows"]) == len(Graph().parse(
        _toy(tmp_path))) ** 3                                             # no limit: it ends
    ask = tmp_path / "ask.draft.rq"
    ask.write_text("ASK { ?s ?p ?o }", encoding="utf-8")
    _, runs = qs.run(toy, ["ask.draft"], engine=qs.RDFLIB, paths={"ask.draft": ask})
    assert "no es una consulta SELECT" in runs["toy"]["answers"]["ask.draft"]["error"]


@needs_jena
def test_jena_stops_a_slow_draft(tmp_path):
    rq = tmp_path / "slow.draft.rq"
    rq.write_text("SELECT * WHERE { ?a ?b ?c . ?d ?e ?f . ?g ?h ?i . ?j ?k ?l . ?m ?n ?o . }",
                  encoding="utf-8")
    used, runs = qs.run({"toy": [_toy(tmp_path)]}, ["slow.draft", "G01"], engine=qs.JENA,
                        paths={"slow.draft": rq}, query_timeout=1)
    assert used["name"] == qs.JENA
    assert runs["toy"]["answers"]["slow.draft"] == {"error": "la consulta tardó más de 1 s"}
    assert "rows" in runs["toy"]["answers"]["G01"]                    # the others still run


# ------------------------------------------------------------------ measuring on the base

def _without(qid: str, prefix: str) -> str:
    """The reference query of ``qid`` without one of its PREFIX lines."""
    return "\n".join(line for line in qs.query_text(qid).splitlines()
                     if not line.startswith(f"PREFIX {prefix}:"))


@needs_base
def test_measuring_the_drafts_on_the_base(tmp_path, monkeypatch):
    bad = {"P01": "SELEC ?concept WHERE {", "L03": None, "L01": _without("L01", "dct"), "G04":
           "SELECT ?concept WHERE { SERVICE <https://dbpedia.org/sparql> { ?concept ?p ?o } }",
           "G05": "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                  "SELECT DISTINCT ?concept WHERE { ?concept a owl:Class }"}
    llm = DraftLLM(bad)                                        # v2: every reference query
    (tmp_path / qd.DRAFTS_DIR).mkdir()
    (tmp_path / qd.DRAFTS_DIR / "X99.draft.rq").write_text("vieja", encoding="utf-8")
    seen = {}
    real_run = qs.run

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_run(*args, **kwargs)
    monkeypatch.setattr(qs, "run", spy)
    out = qd.measure(llm=llm, engine=qs.RDFLIB, out_dir=tmp_path)
    assert llm.calls == 6 and seen["query_timeout"] == 20        # 3 calls per prompt
    assert len(seen["paths"]) == 21 + 1 + 23                     # v1, its prefixed L01, v2
    assert "L01.v1.prefixed" in seen["paths"] and "G04.v1" not in seen["paths"]
    by_id = {r["id"]: r["versions"] for r in out["questions"]}
    assert len(by_id) == 23
    v1 = {qid: v["1"] for qid, v in by_id.items()}
    assert v1["P01"][qd.AS_WRITTEN]["outcome"] == qd.BROKEN
    assert v1["P01"][qd.PREFIXED] == v1["P01"][qd.AS_WRITTEN] and v1["P01"]["added"] == []
    assert (v1["L03"][qd.AS_WRITTEN]["outcome"], v1["L03"][qd.AS_WRITTEN]["reason"]) == (
        qd.NO_DRAFT, "la IA no dio una consulta")
    assert "SERVICE" in v1["G04"][qd.AS_WRITTEN]["reason"]
    assert v1["G05"][qd.AS_WRITTEN]["outcome"] == qd.OTHER and v1["G05"][qd.AS_WRITTEN]["same_color"]
    assert "Unresolved" in v1["L01"][qd.AS_WRITTEN]["reason"] or \
        "dct" in v1["L01"][qd.AS_WRITTEN]["reason"]
    assert (v1["L01"]["added"], v1["L01"][qd.PREFIXED]["outcome"]) == (["dct"], qd.SAME)
    assert v1["P02"][qd.AS_WRITTEN]["outcome"] == qd.SAME and v1["P02"][qd.AS_WRITTEN]["same_rows"]
    tables = out["versions"]
    assert tables["1"][qd.AS_WRITTEN]["total"] == {
        qd.SAME: 18, qd.OTHER: 1, qd.BROKEN: 3, qd.NO_DRAFT: 1, "same_color": 1, "same_rows": 18,
        "questions": 23}
    assert tables["1"][qd.PREFIXED]["total"][qd.SAME] == 19
    assert tables["1"][qd.PREFIXED]["total"][qd.BROKEN] == 2
    assert tables["2"][qd.AS_WRITTEN]["total"][qd.SAME] == 23
    assert tables["1"][qd.AS_WRITTEN][qc.BASE]["questions"] == 15
    assert (out["adopted"], out["configured"]) == (2, 2)              # config.yaml: v2
    v1_files = sorted(p.name for p in (tmp_path / qd.DRAFTS_DIR / "v1").iterdir())
    assert len(v1_files) == 22 + 1 and "L03.v1.rq" not in v1_files       # L03 had none
    assert "G04.v1.rq" in v1_files and "L01.v1.prefixed.rq" in v1_files
    assert len(list((tmp_path / qd.DRAFTS_DIR / "v2").iterdir())) == 23
    assert not (tmp_path / qd.DRAFTS_DIR / "X99.draft.rq").exists()       # the old ones go
    fixed = (tmp_path / qd.DRAFTS_DIR / "v1" / "L01.v1.prefixed.rq").read_text(encoding="utf-8")
    assert "# El sistema declaró los prefijos que la IA usó sin declarar: dct." in fixed
    first = (tmp_path / qd.DRAFTS_DIR / "v2" / "P02.v2.rq").read_text(encoding="utf-8")
    assert first.startswith("# Borrador de la IA (sparql.v2, modelo-de-prueba) para P02 · ")
    saved = json.loads((tmp_path / qd.EVALUATION_FILE).read_text(encoding="utf-8"))
    assert saved["versions"]["2"]["prompt"] == qd.PROMPTS[2].manifest()
    assert saved["rule"] == qd.RULE and saved["adoption"] == qd.ADOPTION
    assert saved["summary"]["chars"] > 10000
    text = qd.render(out)
    assert "  IA sparql.v1: 3 llamadas; 0 borradores desde la caché" in text
    assert "  IA sparql.v2: 3 llamadas; 0 borradores desde la caché" in text
    assert "v1 tal cual  v1 con prefijos  v2 tal cual  v2 con prefijos" in text
    row = {line.split("  ")[1].strip(): line.split()[-4:] for line in text.splitlines()
           if line.startswith("  Misma respuesta") or line.startswith("  No corrió")}
    assert row == {"Misma respuesta": ["18", "19", "23", "23"], "No corrió": ["3", "2", "0", "0"]}
    assert "  Sin borrador" in text and "    L01 · ¿Qué derechos tiene el titular" in text
    assert "             con prefijos (dct): [misma respuesta] Trae 11 de 11" in text
    assert "         v1: [no corrió] " in text
    assert text.endswith("  Se adopta: sparql.v2 (config.yaml dice v2)")
    again = DraftLLM(bad)                                     # a second run: all from the cache
    out = qd.measure(llm=again, engine=qs.RDFLIB, out_dir=tmp_path)
    assert again.calls == 1                                   # only L03 of v1, which had none
    assert out["versions"]["1"]["ai"]["cached"] == 22 and out["versions"]["2"]["ai"]["cached"] == 23


@needs_base
def test_v2_is_not_adopted_if_it_loses_a_question(tmp_path):
    llm = DraftLLM(drafts2={"P02": "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                                    "SELECT DISTINCT ?concept WHERE { ?concept a owl:Class }"})
    out = qd.measure(llm=llm, engine=qs.RDFLIB, out_dir=tmp_path)
    assert out["versions"]["2"][qd.PREFIXED]["total"][qd.SAME] == 22
    assert out["adopted"] == 1
    assert "  Sin borrador" not in qd.render(out)


@needs_base
def test_without_the_ai_only_the_cache_counts(tmp_path, monkeypatch):
    import src.ai.llm_client as client

    class NoKey:
        def __init__(self, *a, **k):
            raise KeyError("LLM_API_KEY")
    monkeypatch.setattr(client, "LLMClient", NoKey)
    out = qd.measure(engine=qs.RDFLIB, out_dir=tmp_path)
    for version in ("1", "2"):
        assert out["versions"][version]["ai"]["available"] is False
        assert out["versions"][version][qd.AS_WRITTEN]["total"][qd.NO_DRAFT] == 23
    first = out["questions"][0]["versions"]["1"][qd.AS_WRITTEN]
    assert first["reason"] == "Falta la variable 'LLM_API_KEY' en .env"
    qd.measure(llm=DraftLLM(), engine=qs.RDFLIB, out_dir=tmp_path)        # the cache fills up
    monkeypatch.setenv("LLM_MODEL", "modelo-de-prueba")
    out = qd.measure(use_llm=False, engine=qs.RDFLIB, out_dir=tmp_path)
    assert out["versions"]["2"]["ai"]["cached"] == 23
    assert out["versions"]["1"][qd.AS_WRITTEN]["total"][qd.SAME] == 23
    assert out["versions"]["1"]["ai"]["error"] == "se pidió --sin-ia"
    failing = qd.measure(llm=DraftLLM(error="400 bad request"), engine=qs.RDFLIB,
                         out_dir=tmp_path / "otra")
    first = failing["questions"][0]["versions"]["2"][qd.AS_WRITTEN]
    assert first["reason"] == "la llamada falló: 400 bad request"


def test_the_cli(monkeypatch, capsys):
    calls = []
    ai = {"available": False, "error": "se pidió --sin-ia", "calls": 0, "cached": 23,
          "invalid": 0, "errors": []}
    canned = {"model": "m", "seconds": 1.0,
              "engine": {"name": qs.RDFLIB, "version": "7", "reason": "elegido con --motor"},
              "summary": {"sha256": "a" * 64, "chars": 10}, "rule": qd.RULE,
              "versions": {str(v): {"prompt": qd.PROMPTS[v].manifest(), "ai": ai,
                                    qd.AS_WRITTEN: qd.table([]), qd.PREFIXED: qd.table([])}
                           for v in (1, 2)},
              "questions": [], "adoption": qd.ADOPTION, "adopted": 1, "configured": 1,
              "drafts": "data/output/sparql-drafts"}
    monkeypatch.setattr(qd, "measure", lambda cfg, **kw: calls.append(kw) or canned)
    assert qd._run(["--medir", "--sin-ia", "--motor", "rdflib"]) == 0
    assert calls == [{"use_llm": False, "engine": "rdflib"}]
    out = capsys.readouterr().out
    assert out.startswith("Borradores SPARQL de la IA sobre la base (S9-T08)")
    assert "  Se adopta: sparql.v1 (config.yaml dice v1)" in out
    assert "  Borradores en: data/output/sparql-drafts/" in out
    assert out.rstrip().endswith(qd.EVALUATION_FILE)


def test_the_cli_needs_medir(capsys):
    assert qd._run([]) == 1
    assert "indica --medir" in capsys.readouterr().out


@needs_base
@pytest.mark.slow
def test_the_real_ai_drafts_the_23_queries(tmp_path):
    """Gemini drafts the 23 queries with each prompt; how many are right is a result, not a
    condition."""
    pytest.importorskip("openai")
    import os
    if not os.environ.get("LLM_API_KEY"):
        pytest.skip("Falta LLM_API_KEY en .env")
    out = qd.measure(out_dir=tmp_path)
    for data in out["versions"].values():
        assert data["ai"]["available"] and not data["ai"]["errors"]
        assert data[qd.AS_WRITTEN]["total"]["questions"] == 23
        assert data[qd.AS_WRITTEN]["total"][qd.NO_DRAFT] == 0
