"""S10-T04: the measurement of the questions with simulated people (families of the base as
uploads, "siempre la recomendada" and "la persona", the baseline without AI and the three
conditions fixed before measuring). The LLM is a fake, except in the slow test."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.alignment import question_eval as qe
from src.alignment import user_questions as uq
from src.alignment.decider import Decision, RuleSettings
from src.alignment.reference import Reference, RefItem
from tests.test_user_questions import DPV, EX, FakeLLM, _align, _q, _remove, targets  # noqa: F401

RULE = RuleSettings()


def _decision(name, term=None, relation=None, score=None, kinds=("class",), candidates=()):
    return Decision(key=EX + name, name=name, status="decided", dpv_iri=DPV + term if term
                    else None, dpv_name=term, relation=f"skos:{relation}" if relation else None,
                    confidence="alta", score=score, none=term is None, kinds=kinds,
                    justification=f"Motivo de {name}.", candidates=[DPV + c for c in candidates])


DECISIONS = [
    _decision("EmploymentContext", "ProcessingContext", "exactMatch", 0.8,
              candidates=("ProcessingContext", "Context")),
    _decision("InfoAboutProcessing", "Processing", "exactMatch", 0.6, candidates=("Processing",)),
    _decision("contact", "hasRecipient", "exactMatch", 0.7, kinds=("property",),
              candidates=("hasRecipient",)),
    _decision("HealthData", "SensitivePersonalData", "closeMatch", 0.45,
              candidates=("SensitivePersonalData",)),
    _decision("Fine", "DataController", "exactMatch", 0.9, candidates=("DataController",)),
    _decision("Nothing")]
FAMILY = {"EmploymentContext": "A", "InfoAboutProcessing": "A", "contact": "A",
          "HealthData": "B", "Fine": "B", "Nothing": None}
CONCEPTS = {EX + n: SimpleNamespace(family=f, label=None, definition=f"Definición de {n}.")
            for n, f in FAMILY.items()}
ACCEPTED = {EX + "EmploymentContext": {"term": DPV + "ProcessingContext",
                                       "relation": "skos:exactMatch"},
            EX + "InfoAboutProcessing": {"term": DPV + "Processing",
                                         "relation": "skos:exactMatch"},
            EX + "Fine": {"term": DPV + "DataController", "relation": "skos:exactMatch"}}
OUTPUTS = {**ACCEPTED, EX + "Nothing": {"none": True}}
SUGGESTIONS = {EX + "contact": {}, EX + "HealthData": {}}
AUDITS = {EX + "EmploymentContext": {"verdict": "dudosa", "relation": "broadMatch",
                                     "explanation": "Es un caso."},
          EX + "InfoAboutProcessing": {"verdict": "incoherente", "relation": None,
                                       "explanation": "Es información."},
          EX + "Fine": {"verdict": "coherente", "relation": "exactMatch", "explanation": "Sí."}}
REFERENCE = Reference("base", [
    RefItem(EX + "EmploymentContext", "EmploymentContext", "aligned",
            {DPV + "ProcessingContext": "skos:broadMatch"}),
    RefItem(EX + "InfoAboutProcessing", "InfoAboutProcessing", "no_match"),
    RefItem(EX + "contact", "contact", "aligned", {DPV + "hasRecipient": "skos:exactMatch"}),
    RefItem(EX + "HealthData", "HealthData", "no_match"),
    RefItem(EX + "Fine", "Fine", "aligned", {DPV + "DataController": "skos:exactMatch"}),
    RefItem(EX + "Nothing", "Nothing", "no_match")])

GOOD = [{"preguntas": [_q(["D1"], ("Quitarla", [_remove("D1")])),
                       _q(["D2"], ("Más específico", [_align("D2", "broadMatch",
                                                             "ProcessingContext")])),
                       _q(["D3"], ("Sí", [_align("D3", "exactMatch", "hasRecipient")]))]},
        {"preguntas": [], "sin_pregunta": [{"duda": "D1", "motivo": "Puntaje bajo."}]}]
BAD = [{"preguntas": [_q(["D2"], ("Quitarla", [_remove("D2")]))]},
       {"preguntas": [_q(["D1"], ("Sí", [_align("D1", "closeMatch", "SensitivePersonalData")]))]}]


def _uploads():
    return qe.uploads_of(qe.entries_of(DECISIONS, ACCEPTED, SUGGESTIONS), CONCEPTS, AUDITS)


def _evaluate(targets, llm, **kw):  # noqa: F811
    return qe.evaluate(_uploads(), targets, REFERENCE, RULE, OUTPUTS, llm=llm,
                       settings=uq.AskSettings(pause_seconds=0), sleep_fn=lambda s: None, **kw)


# ------------------------------------------------------------------ the simulated uploads

def test_one_upload_per_family():
    ups = _uploads()
    assert [(u.name, [e["name"] for e in u.entries]) for u in ups] == [
        ("A", ["EmploymentContext", "InfoAboutProcessing", "contact"]),
        ("B", ["HealthData", "Fine"]), (qe.AI_GROUP, ["Nothing"])]
    a = ups[0]
    assert [e["status"] for e in a.entries] == ["accepted", "accepted", "suggested"]
    assert a.state == {EX + "EmploymentContext": {DPV + "ProcessingContext": {
        "relation": "skos:exactMatch", "dpv_name": "ProcessingContext", "decided_by": "rule",
        "version": 1}}, EX + "InfoAboutProcessing": {DPV + "Processing": {
            "relation": "skos:exactMatch", "dpv_name": "Processing", "decided_by": "rule",
            "version": 1}}}
    assert set(a.audits) == {EX + "EmploymentContext", EX + "InfoAboutProcessing"}
    assert a.info[EX + "contact"] == {"label": None, "definition": "Definición de contact."}
    assert ups[2].entries[0]["status"] == "other" and ups[2].state == {}


def test_how_close_a_concept_is_to_your_decision():
    item = REFERENCE.by_key()
    maps = lambda iri, rel: {DPV + iri: {"relation": f"skos:{rel}"}}           # noqa: E731
    ec, info = item[EX + "EmploymentContext"], item[EX + "InfoAboutProcessing"]
    assert qe.concept_value(ec, maps("ProcessingContext", "broadMatch")) == qe.RIGHT_RELATION
    assert qe.concept_value(ec, maps("ProcessingContext", "exactMatch")) == qe.RIGHT_TERM
    assert qe.concept_value(ec, {}) == qe.NOTHING
    assert qe.concept_value(ec, maps("Context", "broadMatch")) == qe.WRONG
    assert qe.concept_value(info, {}) == qe.RIGHT_RELATION
    assert qe.concept_value(info, maps("Processing", "exactMatch")) == qe.WRONG
    assert qe.concept_value(None, {}) == qe.NOTHING


def test_the_person_takes_what_is_closest_and_ties_leave_it(targets):  # noqa: F811
    up = _uploads()[0]
    doubts = uq.find_doubts(up.entries, up.state, audits=up.audits, rule=RULE)
    answer = {"preguntas": [_q(["D2"], ("Quitarla", [_remove("D2")]),
                               ("Casi lo mismo", [_align("D2", "closeMatch",
                                                         "ProcessingContext")]),
                               ("Más específico", [_align("D2", "broadMatch",
                                                          "ProcessingContext")]), rec=2),
                            _q(["D1"], ("Relacionado", [_align("D1", "relatedMatch",
                                                               "Processing")]))]}
    questions, _, _ = uq.validate([answer], doubts, up.index, up.state, targets)
    ref = REFERENCE.by_key()
    assert [q.doubts[0].name for q in questions] == ["InfoAboutProcessing", "EmploymentContext"]
    assert [qe.person_choice(q, up.state, ref) for q in questions] == [1, 2]  # 0 -> 0: leave
    tie = {"preguntas": [_q(["D2"], ("Casi lo mismo", [_align("D2", "closeMatch",
                                                             "ProcessingContext")]))]}
    q, _, _ = uq.validate([tie], doubts, up.index, up.state, targets)
    assert qe.person_choice(q[0], up.state, ref) == 1            # 2 = 2: "Dejar como está"


def test_the_fixed_questions_recommend_nothing(targets):  # noqa: F811
    up = _uploads()[0]
    doubts = uq.find_doubts(up.entries, up.state, audits=up.audits, rule=RULE)
    fixed = uq.fixed_questions(doubts, up.index, up.state, targets)
    state, changed, chosen = qe._answer(fixed, up, REFERENCE.by_key(), "recommended")
    assert chosen == [1, 2, 1] and changed == set() and state == up.state     # all "leave"
    _, changed, chosen = qe._answer(fixed, up, REFERENCE.by_key(), "person")
    assert chosen == [0, 0, 0] and len(changed) == 3


# ------------------------------------------------------------------ measuring

def test_a_good_ai_meets_the_three_conditions(targets, tmp_path):  # noqa: F811
    llm = FakeLLM(*GOOD)
    r = _evaluate(targets, llm, cache_path=tmp_path / uq.CACHE_FILE)
    assert llm.calls == 2 and r["complete"] and r["adopted"] is True
    assert r["doubts"] == 4 and r["questions"] == {
        "ai": 3, "ai_served": 3, "ai_joined": 0, "baseline": 4, "baseline_served": 3,
        "refused_options": 0, "not_asked": {uq.BY_AI: 1, uq.BY_RULE: 0, uq.BY_LIMIT: 0,
                                            uq.NO_ANSWER: 0}}
    res = r["results"]
    assert (res["rule"]["precision"], res["rule"]["coverage"]) == (0.6667, 0.6667)
    assert (res["recommended"]["precision"], res["recommended"]["coverage"]) == (1.0, 1.0)
    assert (res["person"]["relation_ok"], res["baseline_person"]["relation_ok"]) == (3, 3)
    assert [c["ok"] for c in r["conditions"]] == [True, True, True]
    again = _evaluate(targets, FakeLLM(), cache_path=tmp_path / uq.CACHE_FILE)
    assert (again["ai"]["calls"], again["ai"]["cached"], again["adopted"]) == (0, 2, True)


def test_a_bad_ai_fails_and_the_console_says_which(targets):  # noqa: F811
    r = _evaluate(targets, FakeLLM(*BAD))
    assert r["questions"]["ai"] == 2 and r["questions"]["ai_served"] == 0
    assert r["results"]["recommended"]["precision"] == 0.3333
    assert [c["ok"] for c in r["conditions"]] == [False, False, False] and r["adopted"] is False
    text = qe.render_console(r)
    assert ("  Resultado: no se adopta ask.v1 (fallan las condiciones 1, 2 y 3); las subidas "
            "hacen las preguntas fijas") in text
    assert "    2. Sirven al menos la mitad de las preguntas: no (0 de 2)" in text


def test_the_console_and_the_json(targets):  # noqa: F811
    r = _evaluate(targets, FakeLLM(*GOOD))
    lines = qe.render_console(r, examples=1).splitlines()
    assert lines[:4] == [
        "Preguntas de la IA con opciones (S10-T04) · ask.v1 · fake-model",
        "  Subidas simuladas: 3 familias de la base (6 conceptos), 4 dudas",
        "  IA: 2 llamadas, 0 desde la caché", ""]
    assert lines[5:9] == [
        "  Regla sola                                 -        -     66,7 %     66,7 %       1/2",
        "  Con IA, siempre la recomendada             3        -    100,0 %    100,0 %       3/3",
        "  Con IA, la persona                         3        3    100,0 %    100,0 %       3/3",
        "  Preguntas fijas, la persona                4        3    100,0 %    100,0 %       3/3"]
    assert "  Resultado: se adopta ask.v1 (para usarla: user_questions.ai: true)" in lines
    assert ("  Juntan varias dudas: 0 de 3 preguntas. Sin pregunta: 1 duda por la IA, 0 por la "
            "regla, 0 por el máximo; la regla dejó fuera 0 opciones") in lines
    assert lines[11:14] == [
        "    1. Siempre la recomendada no baja la precisión ni la cobertura de la regla sola: sí",
        "    2. Sirven al menos la mitad de las preguntas: sí (3 de 3)",
        "    3. Menos preguntas que las fijas, sin peor precisión ni cobertura con la persona: "
        "sí (3 preguntas con IA, 4 fijas)"]
    assert lines[-9:] == ["", "  Ejemplos (la persona elige como en tus decisiones):", "  [A]",
                          "  P1. ¿Qué hacemos?", "     a) Quitarla (recomendada, elegida)",
                          "        InfoAboutProcessing: exactMatch Processing (regla, versión 1)"
                          " -> sin correspondencia", "     b) Dejar como está",
                          "     Por qué: Porque sí.",
                          "  Detalle: data/output/user-questions-evaluation.json"]
    data = qe.to_json(r)
    assert "_rows" not in data and data["adopted"] is True
    a = data["uploads"][0]
    assert (a["name"], a["concepts"], a["doubts"], a["baseline"]) == (
        "A", 3, 3, {"questions": 3, "served": 3})
    assert [q["person"] for q in a["ai"]["questions"]] == [0, 0, 0]
    assert data["uploads"][1]["ai"]["skipped"] == [{
        "doubt": EX + "HealthData", "name": "HealthData", "kind": uq.SUGGESTED, "by": uq.BY_AI,
        "reason": "Puntaje bajo."}]
    json.dumps(data)


def test_the_conditions_at_their_limits():
    m = lambda p, c: {"precision": p, "coverage": c}                       # noqa: E731
    results = {"rule": m(0.75, 0.295), "recommended": m(0.75, 0.295), "person": m(0.8, 0.4),
               "baseline_person": m(0.8, 0.4)}
    questions = {"ai": 4, "ai_served": 2, "baseline": 5}
    assert qe.conditions(results, questions) == [True, True, True]          # equal is enough
    assert qe.conditions({**results, "recommended": m(0.7499, 0.5)}, questions)[0] is False
    assert qe.conditions({**results, "recommended": m(0.9, None)}, questions)[0] is False
    assert qe.conditions(results, {**questions, "ai_served": 1})[1] is False
    assert qe.conditions(results, {**questions, "ai": 0, "ai_served": 0})[1] is False
    assert qe.conditions(results, {**questions, "baseline": 4})[2] is False  # as many as sin IA
    assert qe.conditions({**results, "person": m(0.8, 0.39)}, questions)[2] is False


def test_without_ai_the_measurement_is_incomplete(targets):  # noqa: F811
    r = _evaluate(targets, None)
    assert not r["complete"] and r["adopted"] is None and r["ai"]["pending"] == 4
    assert "  Resultado: incompleto (faltan respuestas de la IA); vuelve a correr para seguir " \
           "desde la caché." in qe.render_console(r)


def test_a_429_stops_the_run(targets):  # noqa: F811
    r = _evaluate(targets, FakeLLM(error="429 RESOURCE_EXHAUSTED"))
    assert r["ai"]["calls"] == 1 and r["ai"]["quota_exhausted"] and r["adopted"] is None


# ------------------------------------------------------------------ inputs and CLI

def _fake_half(calls):
    def decide_half(inputs, prompt, half, **kw):
        calls.append((half, kw["budget"]))
        part = DECISIONS[:3] if half == "desarrollo" else DECISIONS[3:]
        seen = [SimpleNamespace(key=d.key, name=d.name, kinds=d.kinds, **{
            k: getattr(CONCEPTS[d.key], k) for k in ("family", "label", "definition")})
            for d in part]
        return None, seen, SimpleNamespace(decisions=part), None, "base"
    return decide_half


def test_the_inputs_come_from_the_caches(monkeypatch, targets, tmp_path):  # noqa: F811
    from src.validation import ai_auditor as au
    calls, asked = [], []
    monkeypatch.setattr("src.alignment.decider_eval.load_eval_inputs", lambda cfg: SimpleNamespace(
        references={"base": REFERENCE}, targets=targets))
    monkeypatch.setattr("src.alignment.inherit.base_paths",
                        lambda cfg: {"candidates": tmp_path / "out" / "c.json"})
    monkeypatch.setattr("src.alignment.prompt_lab.decide_half", _fake_half(calls))

    def ask_audit(items, settings=None, cache_path=None):
        asked.append((sorted(i.name for i in items), cache_path))
        return SimpleNamespace(answers={k: v for k, v in AUDITS.items()})
    monkeypatch.setattr(au, "ask_audit", ask_audit)
    uploads, t, ref, rule, outputs = qe.load_inputs({})
    assert calls == [("desarrollo", 0), ("prueba", 0)]              # never a call to the AI
    assert asked == [(["EmploymentContext", "Fine", "InfoAboutProcessing"],
                      tmp_path / "out" / "decider" / au.CACHE_FILE)]
    assert [u.name for u in uploads] == ["A", "B", qe.AI_GROUP] and ref is REFERENCE
    assert not (outputs.get(EX + "HealthData") or {}).get("term")      # 0.45 < 0.5
    assert [e["status"] for e in uploads[1].entries] == ["suggested", "accepted"]
    monkeypatch.setattr(au, "ask_audit", lambda items, **kw: SimpleNamespace(answers={}))
    with pytest.raises(ValueError, match="Faltan 3 revisiones de la IA revisora"):
        qe.load_inputs({})
    pending = [Decision(key=EX + "x", name="x", status="pending")]
    monkeypatch.setattr("src.alignment.prompt_lab.decide_half", lambda *a, **kw: (
        None, [], SimpleNamespace(decisions=pending), None, None))
    with pytest.raises(ValueError, match="Faltan 2 decisiones del decisor"):
        qe.load_inputs({})


def test_the_command(monkeypatch, targets, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "out"
    monkeypatch.setattr("src.config.load_config", lambda: {})
    monkeypatch.setattr("src.alignment.inherit.base_paths",
                        lambda cfg: {"candidates": out / "c.json"})
    monkeypatch.setattr(qe, "load_inputs", lambda cfg: (_uploads(), targets, REFERENCE, RULE,
                                                        OUTPUTS))
    answers = list(GOOD)
    monkeypatch.setattr("src.ai.llm_client.LLMClient", lambda: FakeLLM(*answers))
    monkeypatch.setattr(uq, "ask_settings", lambda cfg=None: uq.AskSettings(pause_seconds=0))
    assert qe._run(["--max-llamadas", "5"]) == 0
    text = capsys.readouterr().out
    assert "  Resultado: se adopta ask.v1" in text
    saved = json.loads((out / qe.EVALUATION_FILE).read_text(encoding="utf-8"))
    assert saved["adopted"] is True and (out / "decider" / uq.CACHE_FILE).exists()

    def no_key():
        raise KeyError("LLM_API_KEY")
    monkeypatch.setattr("src.ai.llm_client.LLMClient", no_key)
    assert qe._run([]) == 1
    assert capsys.readouterr().out == ("Error: falta LLM_API_KEY en el .env (la medición "
                                       "pregunta a la IA).\n")

    def missing(cfg):
        raise ValueError("Faltan 3 decisiones del decisor en su caché: corre primero X")
    monkeypatch.setattr(qe, "load_inputs", missing)
    assert qe._run([]) == 1
    assert capsys.readouterr().out.startswith("Error: Faltan 3 decisiones")


@pytest.mark.slow
def test_real_questions_on_one_family_of_the_base(tmp_path):
    """Real data and Gemini: the decider and the AI reviewer from their caches, one call to
    ask.v1 on the smallest family with doubts; the second run comes from the cache."""
    from src.ai.llm_client import LLMClient
    from src.config import load_config
    cfg = load_config()
    uploads, targets, _, rule, _ = qe.load_inputs(cfg)
    settings = uq.ask_settings(cfg)
    up, doubts = min(((u, uq.find_doubts(u.entries, u.state, info=u.info, audits=u.audits,
                                         rule=rule)) for u in uploads),
                     key=lambda x: (not x[1], len(x[1])))
    cache = tmp_path / uq.CACHE_FILE
    first = uq.propose_questions(doubts, up.index, up.state, targets, llm=LLMClient(),
                                 settings=settings, cache_path=cache)
    assert first.complete and first.ai["calls"] == 1
    assert all(c.ok for q in first.questions for o in q.options for c in o.checked)
    again = uq.propose_questions(doubts, up.index, up.state, targets, settings=settings,
                                 cache_path=cache)
    assert again.ai["calls"] == 0 and again.ai["cached"] == 1
    assert [q.text for q in again.questions] == [q.text for q in first.questions]
