"""Tests for the AI that says why a competency question is yellow (S9-T06).

A fake LLM checks the single call, the cache and what it is shown; without AI (switched off,
no key, an error, an invalid answer) each yellow question gets the fixed sentence. A guarded
test asks the real AI about the five yellow questions of the base (slow)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.validation import question_catalog as qc
from src.validation import question_explainer as qe
from src.validation import questions as qs
from src.validation.shape_inspector import GREEN, YELLOW

ROOT = Path(__file__).resolve().parents[1]
BASE_FILES = [ROOT / "data" / "output" / name for name in
              ("ontopriv-core.rdf", "profile-ecuador-lopdp.rdf", "ontopriv-dpv-alignment.rdf")]
needs_base = pytest.mark.skipif(not all(p.exists() for p in BASE_FILES),
                                reason="Falta la base en data/output")


class ExplainLLM:
    """Explains every question of the context; ``bad`` answers are given as they are."""

    def __init__(self, bad=None, error=None, model="modelo-de-prueba"):
        self.bad, self.error, self.model, self.calls, self.contexts = bad, error, model, 0, []

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        self.calls += 1
        self.contexts.append(context)
        if self.error:
            raise Exception(self.error)
        if self.bad is not None:
            return self.bad
        ids = [block.split(" ===", 1)[0] for block in context.split("=== ")[1:]]
        return json.dumps({"explanations": [{"id": i, "explanation": f"Porque sí ({i})."}
                                            for i in ids]}, ensure_ascii=False)


@pytest.fixture(scope="module")
def catalog():
    return qc.load_catalog()


def _q(catalog, qid):
    return next(q for q in catalog.questions if q.id == qid)


def uri(v):
    return {"t": "uri", "v": v}


def lit(v):
    return {"t": "literal", "v": v, "dt": "http://www.w3.org/2001/XMLSchema#string", "lang": None}


def _answers():
    return {"G01": {"vars": ["concept", "match"],
                    "rows": [[uri("http://x#C"), uri("https://w3id.org/dpv#Consent")],
                             [uri("http://x#P"), uri("https://w3id.org/dpv#PersonalData")]]},
            "G03": {"vars": ["concept", "match"], "rows": []},
            "P01": {"vars": ["concept", "value"], "rows": [[uri("http://x#n"), lit("5 days")]]},
            "P02": {"vars": ["concept", "value"],
                    "rows": [[uri("http://x#consent"), lit("Free from defects")],
                             [uri("http://x#consent"), lit("Unequivocal, no doubt")]]}}


def _results(catalog, answers):
    return [qs.judge(_q(catalog, i), answers[i]) for i in ("G01", "G03", "P01", "P02")]


def test_one_call_explains_every_yellow_question_and_the_cache_answers_next_time(catalog,
                                                                                 tmp_path):
    answers, llm = _answers(), ExplainLLM()
    results = _results(catalog, answers)
    assert [r["color"] for r in results] == [GREEN, YELLOW, YELLOW, YELLOW]
    ai = qe.explain(catalog, results, answers, llm=llm, use_llm=True, version=1,
                    cache_path=tmp_path / qe.CACHE_FILE)
    assert llm.calls == 1 and ai == {"prompt": qe.PROMPTS[1].manifest(), "questions": 3,
                                     "available": True, "calls": 1, "cached": 0, "invalid": 0,
                                     "errors": []}
    assert "explanation" not in results[0]                       # green: nothing to explain
    assert [(r["explanation"], r["explained_by"]) for r in results[1:]] == [
        ("Porque sí (G03).", qe.AI), ("Porque sí (P01).", qe.AI), ("Porque sí (P02).", qe.AI)]
    again = _results(catalog, answers)
    ai = qe.explain(catalog, again, answers, llm=llm, use_llm=True, version=1,
                    cache_path=tmp_path / qe.CACHE_FILE)
    assert llm.calls == 1 and (ai["calls"], ai["cached"]) == (0, 3)
    assert again[1]["explanation"] == "Porque sí (G03)." and qe.ai_text(ai) == (
        "0 llamadas, 3 desde la caché")
    other = ExplainLLM(model="otro-modelo")                        # another model: asked again
    qe.explain(catalog, _results(catalog, answers), answers, llm=other, use_llm=True, version=1,
               cache_path=tmp_path / qe.CACHE_FILE)
    assert other.calls == 1
    newer = ExplainLLM()                                           # another prompt: asked again
    qe.explain(catalog, _results(catalog, answers), answers, llm=newer, use_llm=True, version=2,
               cache_path=tmp_path / qe.CACHE_FILE)
    assert newer.calls == 1


def test_what_the_ai_is_shown(catalog):
    answers, llm = _answers(), ExplainLLM()
    qe.explain(catalog, _results(catalog, answers), answers, llm=llm, use_llm=True, version=1)
    context = llm.contexts[0]
    assert [b.split(" ===", 1)[0] for b in context.split("=== ")[1:]] == ["G03", "P01", "P02"]
    assert ("Busca: conceptos alineados con dpv:LegalBasis o con un término que esté debajo\n"
            "Términos del DPV con los que la ontología sí se alinea: Consent, PersonalData\n"
            "Filas que trajo la consulta (0): ninguna") in context
    assert "Valor esperado: 3 días; valores que trajo: 5" in context
    assert "Filas que trajo la consulta (1): n 5 days" in context
    assert "Faltan: Específica (se buscaron: specific); Informada (se buscaron: informed)" in context
    assert ("Filas que trajo la consulta (2): consent Free from defects | consent Unequivocal, "
            "no doubt") in context
    q, r = _q(catalog, "P02"), qs.judge(_q(catalog, "P02"), answers["P02"])
    rows = [f"fila {n:02}" for n in range(10)]
    assert qe.body(q, r, rows, [], shown=3).endswith("fila 00 | fila 01 | fila 02")


def test_without_ai_each_yellow_question_gets_the_fixed_sentence(catalog, monkeypatch):
    answers = _answers()
    results = _results(catalog, answers)
    ai = qe.explain(catalog, results, answers)                    # direct calls never ask
    assert ai["available"] is False and ai["error"] == "IA desactivada (--sin-ia)"
    assert qe.ai_text(ai) == "no usada; IA desactivada (--sin-ia); frase fija"
    assert [(r["explanation"], r["explained_by"]) for r in results[1:]] == [
        ("La consulta no trajo conceptos alineados con dpv:LegalBasis o con un término que "
         "esté debajo.", qe.FIXED),
        ("La respuesta no trae el valor 3 días (trajo: 5).", qe.FIXED),
        ("Ninguna fila nombra a Específica, Informada, Inequívoca ni cita un artículo que solo "
         "le corresponda.", qe.FIXED)]

    def no_key(*args, **kwargs):
        raise KeyError("LLM_API_KEY")
    monkeypatch.setattr("src.ai.llm_client.LLMClient", no_key)
    ai = qe.explain(catalog, _results(catalog, answers), answers, use_llm=None)
    assert ai["available"] is False and ai["error"] == "Falta la variable 'LLM_API_KEY' en .env"
    off = qe.explain(catalog, _results(catalog, answers), answers, llm=ExplainLLM(),
                     use_llm=True, cfg={"questions": {"explain": False}})
    assert off["error"] == "desactivada en config.yaml (questions.explain)" and off["calls"] == 0


def test_without_ai_the_cache_still_answers(catalog, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "modelo-de-prueba")            # the model of the .env
    answers = _answers()
    cache = tmp_path / qe.CACHE_FILE
    qe.explain(catalog, _results(catalog, answers), answers, llm=ExplainLLM(), use_llm=True,
               cache_path=cache)
    results = _results(catalog, answers)[:3]                       # P02 is not asked now
    ai = qe.explain(catalog, results, answers, cache_path=cache)
    assert (ai["available"], ai["cached"], ai["questions"]) == (False, 2, 2)
    assert [r["explained_by"] for r in results[1:]] == [qe.AI, qe.AI]
    assert qe.ai_text(ai) == "no usada; IA desactivada (--sin-ia); 2 desde la caché"
    mixed = {**ai, "questions": 3}
    assert qe.ai_text(mixed) == ("no usada; IA desactivada (--sin-ia); 2 desde la caché; "
                                 "frase fija para el resto")


def test_an_error_or_a_bad_answer_falls_back_to_the_fixed_sentence(catalog, tmp_path):
    answers = _answers()
    broken = _results(catalog, answers)
    ai = qe.explain(catalog, broken, answers, llm=ExplainLLM(error="500 se cayó"), use_llm=True)
    assert ai["calls"] == 1 and ai["errors"] == ["500 se cayó"]
    assert qe.ai_text(ai) == "1 llamada, 0 desde la caché; error: 500 se cayó"
    assert {r.get("explained_by") for r in broken[1:]} == {qe.FIXED}
    bad = json.dumps({"explanations": [{"id": "G03", "explanation": "x" * 301},
                                       {"id": "P01", "explanation": " "},
                                       {"id": "L01", "explanation": "otra pregunta"},
                                       {"id": "p02", "explanation": "Bien."}]})
    results = _results(catalog, answers)
    ai = qe.explain(catalog, results, answers, llm=ExplainLLM(bad=bad), use_llm=True,
                    cache_path=tmp_path / qe.CACHE_FILE)
    assert ai["invalid"] == 2 and [r["explained_by"] for r in results[1:]] == [
        qe.FIXED, qe.FIXED, qe.AI]
    cached = [json.loads(line) for line in (tmp_path / qe.CACHE_FILE).read_text(
        encoding="utf-8").splitlines()]
    assert [c["question"] for c in cached] == ["P02"]              # only the valid one is kept
    ai = qe.explain(catalog, _results(catalog, answers), answers,
                    llm=ExplainLLM(bad="no es json"), use_llm=True)
    assert ai["errors"] == ["La respuesta del LLM no trae la lista de explicaciones en JSON."]


def test_a_429_waits_and_asks_once_more(catalog):
    class Busy(ExplainLLM):
        def ask(self, *args, **kwargs):
            if not self.calls:
                self.calls += 1
                raise Exception("429 RESOURCE_EXHAUSTED")
            return super().ask(*args, **kwargs)
    waits, llm, answers = [], Busy(), _answers()
    results = _results(catalog, answers)
    ai = qe.explain(catalog, results, answers, llm=llm, use_llm=True, sleep_fn=waits.append)
    from src.alignment.justify import QUOTA_WAIT_SECONDS
    assert waits == [QUOTA_WAIT_SECONDS] and ai["calls"] == 1
    assert results[1]["explained_by"] == qe.AI


def test_nothing_yellow_asks_nothing(catalog):
    answers, llm = _answers(), ExplainLLM()
    results = [qs.judge(_q(catalog, "G01"), answers["G01"])]
    ai = qe.explain(catalog, results, answers, llm=llm, use_llm=True)
    assert llm.calls == 0 and ai["questions"] == 0 and qe.ai_text(ai) == "nada que explicar"


def test_parse_explanations_accepts_fences_and_text_around():
    raw = 'Aquí va:\n```json\n{"explanations": [{"id": "G03", "explanation": "Una."}]}\n```'
    assert qe.parse_explanations(raw) == [{"id": "G03", "explanation": "Una."}]
    with pytest.raises(ValueError):
        qe.parse_explanations('{"otra": 1}')


def test_the_settings_come_from_config():
    assert qe.explain_settings({"questions": {"explain": False, "rows_shown": 5}}) == \
        qe.ExplainSettings(explain=False, max_tokens=2048, rows_shown=5)
    with pytest.raises(ValueError, match="bloque questions"):
        qe.explain_settings({"questions": {"rows_shown": 0}})


# ------------------------------------------------------------------ the base

@needs_base
def test_the_base_explains_its_yellow_questions_once(tmp_path):
    llm = ExplainLLM()
    report = qs.inspect_base(engine=qs.RDFLIB, out_dir=tmp_path, llm=llm, use_llm=True)
    yellow = [r for r in report["results"] if r["color"] == YELLOW]
    assert [r["id"] for r in yellow] == ["P02", "L03", "L04", "L09", "L10"]
    assert all(r["explained_by"] == qe.AI for r in yellow) and llm.calls == 1
    assert report["explanations"]["questions"] == 5
    log = (tmp_path / qs.LOG_FILE).read_text(encoding="utf-8")
    assert "  IA que explica las amarillas (prompt explain.v2): 1 llamada, 0 desde la caché" in log
    assert "         Por qué (IA): Porque sí (P02)." in log
    qs.inspect_base(engine=qs.RDFLIB, out_dir=tmp_path, llm=llm, use_llm=True)
    assert llm.calls == 1                                          # the cache answers


@pytest.mark.slow
@needs_base
def test_the_real_ai_explains_the_yellow_questions_of_the_base(tmp_path):
    report = qs.inspect_base(engine=qs.RDFLIB, out_dir=tmp_path, use_llm=True)
    yellow = [r for r in report["results"] if r["color"] == YELLOW]
    assert report["explanations"]["available"], report["explanations"]
    assert all(r["explained_by"] == qe.AI for r in yellow), report["explanations"]
    assert all(0 < len(r["explanation"]) <= qe.MAX_CHARS for r in yellow)
    for r in yellow:
        print(f"{r['id']}: {r['explanation']}")


# ------------------------------------------------------------------ prompt v2: the row of each element

class CitingLLM(ExplainLLM):
    """v2: cites the first row that holds one of ``words`` for each element (or a ``fake`` row)."""

    def __init__(self, words=None, fake=None, **kw):
        super().__init__(**kw)
        self.words, self.fake = words or {}, fake

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        self.calls += 1
        self.contexts.append(context)
        out = []
        for block in context.split("=== ")[1:]:
            qid = block.split(" ===", 1)[0]
            rows = dict(re.findall(r"^(F\d+): (.*)$", block, re.M))
            matches = []
            for eid, name in re.findall(r"^(E\d+): ([^(\n]+)", block, re.M):
                words = self.words.get(name.strip(), [])
                hit = next((f for f, t in rows.items()
                            if any(qc.normalize(w) in qc.normalize(t) for w in words)),
                           "ninguna")
                matches.append({"element": eid, "row": self.fake or hit})
            out.append({"id": qid, "matches": matches, "explanation": f"Porque sí ({qid})."})
        return json.dumps({"explanations": out}, ensure_ascii=False)


def test_v2_numbers_the_elements_and_the_rows(catalog):
    answers = _answers()
    r = qs.judge(_q(catalog, "P02"), answers["P02"])
    it = qe.item(_q(catalog, "P02"), r, ["consent Free from defects", "consent Unequivocal"], [],
                 version=2)
    assert it.elements == {"E1": "Específica", "E2": "Informada", "E3": "Inequívoca"}
    assert it.rows == {"F1": "consent Free from defects", "F2": "consent Unequivocal"}
    assert ("Elementos que faltan:\nE1: Específica (se buscaron: specific)\nE2: Informada "
            "(se buscaron: informed)\nE3: Inequívoca (se buscaron: unambiguous)\n"
            "Filas que trajo la consulta (2):\nF1: consent Free from defects\n"
            "F2: consent Unequivocal") in it.body
    few = qe.item(_q(catalog, "P02"), r, ["a", "b"], [], shown=1, version=2)
    assert "Filas que trajo la consulta (2, se muestran 1):\nF1: a" in few.body
    value = qe.item(_q(catalog, "P01"), qs.judge(_q(catalog, "P01"), answers["P01"]),
                    ["n 5 days"], [], version=2)
    assert value.elements == {"E1": "3 días"}
    generic = qe.item(_q(catalog, "G03"), qs.judge(_q(catalog, "G03"), answers["G03"]), [],
                      ["Consent"], version=2)
    assert generic.elements == {} and generic.body.endswith(
        "Filas que trajo la consulta (0): ninguna")


def test_v2_keeps_the_row_it_cites_and_only_if_it_exists(catalog, tmp_path):
    answers = _answers()
    results = _results(catalog, answers)
    llm = CitingLLM(words={"Inequívoca": ["unequivocal"]})
    ai = qe.explain(catalog, results, answers, llm=llm, use_llm=True, version=2,
                    cache_path=tmp_path / qe.CACHE_FILE)
    assert ai["prompt"] == qe.PROMPTS[2].manifest() and ai["invalid"] == 0
    p02 = results[3]
    assert p02["explained_by"] == qe.AI and p02["matches"] == [
        {"element": "Específica", "row": None}, {"element": "Informada", "row": None},
        {"element": "Inequívoca", "row": "consent Unequivocal, no doubt"}]
    assert results[1]["matches"] == []                             # G03: no elements
    again = _results(catalog, answers)
    qe.explain(catalog, again, answers, llm=llm, use_llm=True, version=2,
               cache_path=tmp_path / qe.CACHE_FILE)
    assert llm.calls == 1 and again[3]["matches"] == p02["matches"]   # from the cache
    invented = _results(catalog, answers)
    ai = qe.explain(catalog, invented, answers, llm=CitingLLM(fake="F99"), use_llm=True,
                    version=2)
    assert ai["invalid"] == 2 and invented[3]["explained_by"] == qe.FIXED   # P01 and P02
    assert "matches" not in invented[3]
    unknown = {"id": "P02", "matches": [{"element": "E9", "row": "F1"}], "explanation": "Una."}
    it = qe.Item("x", {"E1": "Libre"}, {"F1": "fila"})
    assert qe.check_answer(unknown, it, 2) is None
    assert qe.check_answer({"explanation": "Una.", "matches": "F1"}, it, 2) is None
    assert qe.check_answer({"explanation": "Una."}, it, 2) == {
        "explanation": "Una.", "matches": [{"element": "Libre", "row": None}]}
    assert qe.check_answer({"explanation": "Una.", "matches": [{"element": "e1",
                                                                "row": "f1"}]}, it, 2)[
        "matches"] == [{"element": "Libre", "row": "fila"}]


def test_the_ids_of_the_rows_do_not_reach_the_reader():
    assert qe.readable("Aunque la fila F4 contiene 'consent Unequivocal', la regla no lo vio.") \
        == "Aunque una fila contiene 'consent Unequivocal', la regla no lo vio."
    assert qe.readable("Las filas F1 y F2 dicen lo mismo; E1 falta (F9).") == \
        "Algunas filas dicen lo mismo; falta."
    assert qe.readable("Falta Inequívoca. La fila F3 dice 'unequivocal'.") == \
        "Falta Inequívoca. Una fila dice 'unequivocal'."
    assert qe.readable("E1 no aparece en ninguna fila.") == "No aparece en ninguna fila."
    assert qe.readable("Sin ids aquí.") == "Sin ids aquí."


def test_v2_reads_without_ids_in_the_results(catalog):
    class Talkative(CitingLLM):
        def ask(self, *a, **k):
            raw = json.loads(super().ask(*a, **k))
            for e in raw["explanations"]:
                e["explanation"] = f"Aunque la fila F2 lo dice ({e['id']}), la regla no lo vio."
            return json.dumps(raw)
    answers = _answers()
    results = _results(catalog, answers)
    qe.explain(catalog, results, answers, llm=Talkative(), use_llm=True, version=2)
    assert results[3]["explanation"] == "Aunque una fila lo dice (P02), la regla no lo vio."


def test_right_means_finding_the_row_that_holds_the_element():
    ref = {"Inequívoca": ["unequivocal"]}
    assert qe.is_right({"explained_by": qe.AI, "explanation": "Trae 'Unequivocal'."}, ref)
    assert not qe.is_right({"explained_by": qe.AI, "explanation": "No lo tiene."}, ref)
    assert not qe.is_right({"explained_by": qe.FIXED, "explanation": "unequivocal"}, ref)
    cited = {"explained_by": qe.AI, "explanation": "Dice que es inequívoco.",
             "matches": [{"element": "Inequívoca", "row": "consent Unequivocal, no doubt"}]}
    assert qe.is_right(cited, ref)
    assert not qe.is_right({**cited, "matches": [{"element": "Inequívoca", "row": None}]}, ref)


def test_the_rule_to_adopt_v2_was_fixed_before_measuring():
    assert qe.adopted({1: {"A": True, "B": False}, 2: {"A": True, "B": True}}) == 2
    assert qe.adopted({1: {"A": True, "B": False}, 2: {"A": False, "B": True}}) == 1   # tie
    assert qe.adopted({1: {"A": True, "B": False, "C": False},
                       2: {"A": False, "B": True, "C": True}}) == 1    # loses one v1 got
    assert qe.adopted({1: {"A": False}, 2: {"A": False}}) == 1


def test_the_reference_covers_the_yellow_questions_of_the_base(catalog):
    ref = qe.load_reference()
    assert sorted(ref) == ["L03", "L04", "L09", "L10", "P02"]
    for qid, elements in ref.items():
        names = {e.name for e in _q(catalog, qid).expected.elements}
        assert set(elements) <= names and all(elements.values()), qid


@needs_base
def test_measuring_v1_against_v2_on_the_base(tmp_path):
    words = {name: [w.lower() for w in ws] for e in qe.load_reference().values()
             for name, ws in e.items()}
    out = qe.measure(llm=CitingLLM(words=words), engine=qs.RDFLIB, out_dir=tmp_path)
    one, two = out["versions"][1], out["versions"][2]
    assert (one["right"], one["total"], two["right"], two["total"]) == (0, 5, 5, 5)
    assert out["adopted"] == 2 and out["configured"] == 2              # config.yaml: v2
    saved = json.loads((tmp_path / qe.EVALUATION_FILE).read_text(encoding="utf-8"))
    assert saved["adopted"] == 2
    text = qe.render_measure(out)
    assert "  explain.v1: 0 de 5 correctas · IA: 1 llamada, 0 desde la caché" in text
    assert "  explain.v2: 5 de 5 correctas · IA: 1 llamada, 0 desde la caché" in text
    assert "         Inequívoca ← consent Unequivocal, no doubt about the authorization" in text
    assert "  Se adopta: explain.v2 (config.yaml dice v2)" in text
    assert qe._run([]) == 1
