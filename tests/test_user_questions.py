"""S10-T04: the doubts of an upload, the questions of the AI and the fixed rule over every
option. The LLM is a fake: no test calls Gemini."""
from __future__ import annotations

import json

import pytest

from src.ai.prompts import load_prompt
from src.alignment import propose as dec
from src.alignment import user_decisions as ud
from src.alignment import user_questions as uq
from src.alignment.decider import RuleSettings
from src.alignment.dpv_targets import build_dpv_targets
from src.ingest.dpv_loader import load_dpv

DPV = "https://w3id.org/dpv#"
EX = "http://example.org/onto#"
TTL = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dpv: <https://w3id.org/dpv#> .
dpv:Context a rdfs:Class , skos:Concept ; skos:prefLabel "Context"@en ;
    skos:definition "Contextually relevant information."@en .
dpv:ProcessingContext a rdfs:Class , skos:Concept ; skos:prefLabel "Processing Context"@en ;
    skos:definition "Context or conditions within which processing takes place."@en ;
    skos:broader dpv:Context .
dpv:Processing a rdfs:Class , skos:Concept ; skos:prefLabel "Processing"@en ;
    skos:definition "Operations over data."@en .
dpv:PersonalData a rdfs:Class , skos:Concept ; skos:prefLabel "Personal Data"@en ;
    skos:definition "Data associated with an individual."@en .
dpv:SensitivePersonalData a rdfs:Class , skos:Concept ;
    skos:prefLabel "Sensitive Personal Data"@en ;
    skos:definition "Personal data considered sensitive."@en ; skos:broader dpv:PersonalData .
dpv:DataController a rdfs:Class , skos:Concept ; skos:prefLabel "Data Controller"@en ;
    skos:definition "Decides the purposes of processing."@en .
dpv:Consent a rdfs:Class , skos:Concept ; skos:prefLabel "Consent"@en .
dpv:hasRecipient a rdf:Property , skos:Concept ; skos:prefLabel "has recipient"@en ;
    skos:definition "Indicates the recipient of data."@en .
dpv:hasDataSubject a rdf:Property , skos:Concept ; skos:prefLabel "has data subject"@en .
"""


@pytest.fixture
def targets(tmp_path):
    path = tmp_path / "dpv.ttl"
    path.write_text(TTL, encoding="utf-8")
    return build_dpv_targets(load_dpv(path))


def _entry(name, status, term=None, relation=None, score=None, kinds=("class",),
           candidates=()):
    return {"key": EX + name, "name": name, "kinds": list(kinds), "status": status,
            "decided_by": "rule", "dpv_iri": DPV + term if term else None, "dpv_name": term,
            "relation": f"skos:{relation}" if relation else None, "score": score,
            "confidence": "alta", "justification": f"Motivo de {name}.",
            "candidates": [DPV + c for c in candidates]}


ENTRIES = [
    _entry("EmploymentContext", dec.ACCEPTED, "ProcessingContext", "exactMatch", 0.8,
           candidates=("ProcessingContext", "Context", "Processing")),
    _entry("InfoAboutProcessing", dec.ACCEPTED, "Processing", "exactMatch", 0.6,
           candidates=("Processing", "Context")),
    _entry("contact", dec.SUGGESTED, "hasRecipient", "exactMatch", 0.7, kinds=("property",),
           candidates=("hasRecipient", "hasDataSubject")),
    _entry("HealthData", dec.SUGGESTED, "SensitivePersonalData", "closeMatch", 0.45,
           candidates=("SensitivePersonalData", "PersonalData")),
    _entry("Related", dec.SUGGESTED, "Consent", "relatedMatch", 0.9),
    _entry("Fine", dec.ACCEPTED, "DataController", "exactMatch", 0.9),
    _entry("Nothing", dec.NO_MATCH),
]
AUDITS = {EX + "EmploymentContext": {"verdict": "dudosa", "relation": "exactMatch",
                                     "ai_relation": "broadMatch",       # as in a workspace
                                     "explanation": "Es un caso laboral."},
          EX + "InfoAboutProcessing": {"verdict": "incoherente", "relation": "relatedMatch",
                                       "explanation": "Es información, no el tratamiento."},
          EX + "Fine": {"verdict": "coherente", "relation": "exactMatch", "explanation": "Sí."}}
INFO = {EX + "EmploymentContext": {"label": "Contexto laboral",
                                   "definition": "Tratamiento en el trabajo."}}
RULE = RuleSettings()


def _index():
    return {e["key"]: {"key": e["key"], "name": e["name"], "kinds": e["kinds"],
                       "alignable": True} for e in ENTRIES}


def _state():
    return {e["key"]: {e["dpv_iri"]: {"relation": e["relation"], "dpv_name": e["dpv_name"],
                                      "decided_by": "rule", "version": 1}}
            for e in ENTRIES if e["status"] == dec.ACCEPTED}


def _doubts(**kw):
    return uq.find_doubts(ENTRIES, _state(), info=INFO, audits=AUDITS, rule=RULE, **kw)


class FakeLLM:
    model = "fake-model"

    def __init__(self, *answers, error=None):
        self.answers, self.error, self.calls, self.contexts = list(answers), error, 0, []

    def ask(self, prompt, context, max_tokens=None, json_mode=False):
        self.calls += 1
        self.contexts.append(context)
        if self.error:
            raise RuntimeError(self.error)
        answer = self.answers.pop(0)
        return answer if isinstance(answer, str) else json.dumps(answer)


def _align(did, relation, term):
    return {"duda": did, "accion": "alinear", "relacion": relation, "termino": term}


def _remove(did):
    return {"duda": did, "accion": "quitar", "relacion": "", "termino": ""}


def _q(dudas, *options, rec=1, text="¿Qué hacemos?", reason="Porque sí."):
    return {"dudas": dudas, "pregunta": text, "recomendada": rec, "motivo": reason,
            "opciones": [{"texto": t, "cambios": c} for t, c in options]}


# ------------------------------------------------------------------ the doubts

def test_the_doubts_in_the_order_they_are_asked():
    doubts = _doubts()
    assert [(d.id, d.name, d.kind) for d in doubts] == [
        ("D1", "InfoAboutProcessing", uq.REVIEWED), ("D2", "EmploymentContext", uq.REVIEWED),
        ("D3", "contact", uq.SUGGESTED), ("D4", "HealthData", uq.SUGGESTED)]
    d1, d2, d3, d4 = doubts
    assert d2.audit == {"verdict": "dudosa", "relation": "broadMatch",
                        "explanation": "Es un caso laboral."}       # ai_relation of a workspace
    assert d2.current == [{"dpv_iri": DPV + "ProcessingContext", "dpv_name": "ProcessingContext",
                           "relation": "skos:exactMatch", "decided_by": "rule", "version": 1}]
    assert (d2.label, d2.definition) == ("Contexto laboral", "Tratamiento en el trabajo.")
    assert d3.why == "las propiedades nunca se alinean solas" and d3.is_property
    assert d4.why == "su puntaje (0.45) es menor que 0.5" and d4.current == []
    assert d1.audit["verdict"] == "incoherente" and d1.why == ""
    assert d4.terms() == {DPV + "SensitivePersonalData", DPV + "PersonalData"}
    short = _doubts(candidates=2, skip={EX + "InfoAboutProcessing"})
    assert [d.id for d in short] == ["D1", "D2", "D3"]                  # the person decided it
    assert short[0].candidates == [DPV + "ProcessingContext", DPV + "Context"]


def test_why_the_rule_left_a_suggestion_out():
    rule = RuleSettings(property_relations=("skos:exactMatch",))
    assert uq.left_out({"kinds": ["property"], "score": 0.7, "relation": "skos:closeMatch"},
                       rule) == "la regla no acepta esa relación"
    assert uq.left_out({"kinds": ["class"], "score": 0.9, "relation": "skos:exactMatch"},
                       rule) == "no pasó la regla"


def test_what_the_ai_sees_of_a_doubt(targets):
    terms = {t.iri: t for t in targets.targets}
    d1, d2, d3, _ = _doubts()
    assert uq.doubt_body(d2, terms).splitlines() == [
        "Concepto: EmploymentContext (clase)",
        "Etiqueta: Contexto laboral",
        "Definicion: Tratamiento en el trabajo.",
        "Estado actual: exactMatch (es lo mismo que) ProcessingContext, decidida por regla",
        "Duda: revisada. La IA revisora la marco como dudosa: Es un caso laboral. (relacion que "
        "usaria: broadMatch)",
        "Motivo de la IA del sistema: Motivo de EmploymentContext.",
        "Candidatos DPV:",
        "1. ProcessingContext | Processing Context | definicion: Context or conditions within "
        "which processing takes place. | padres en el DPV: Context",
        "2. Context | Context | definicion: Contextually relevant information. | padres en el "
        "DPV: -",
        "3. Processing | Processing | definicion: Operations over data. | padres en el DPV: -"]
    body = uq.doubt_body(d3, terms)
    assert "Concepto: contact (propiedad)" in body and "Estado actual: sin correspondencia" in body
    assert ("Duda: sugerida. La IA del sistema eligio exactMatch (es lo mismo que) hasRecipient "
            "con confianza alta, puntaje 0.70, pero la regla no la acepto: las propiedades nunca "
            "se alinean solas.") in body


def test_the_prompt_asks_for_what_the_code_reads():
    p = load_prompt("ask", 1)
    assert uq.ask_prompt().sha256 == p.sha256
    for word in ('"preguntas"', '"sin_pregunta"', '"dudas"', '"pregunta"', '"opciones"',
                 '"texto"', '"cambios"', '"duda"', '"accion"', '"relacion"', '"termino"',
                 '"recomendada"', '"motivo"', '"alinear"', '"quitar"', "dejar como esta",
                 "maximo 10 preguntas"):
        assert word in p.text, word


def test_the_settings_come_from_config():
    s = uq.ask_settings({"user_questions": {"max_questions": 4, "unknown": 1}})
    assert (s.max_questions, s.batch_size, s.prompt_version, s.ai) == (4, 30, 1,
                                                                        False)
    assert uq.ask_settings() == uq.AskSettings()                     # config.yaml = defaults
    with pytest.raises(ValueError, match="user_questions"):
        uq.ask_settings({"user_questions": {"max_questions": 0}})


# ------------------------------------------------------------------ asking the AI

def test_parse_answer():
    raw = '```json\n{"preguntas": [{"dudas": ["D1"]}, 3], "sin_pregunta": [{"duda": "D2"}]}\n```'
    assert uq.parse_answer(raw) == {"preguntas": [{"dudas": ["D1"]}],
                                    "sin_pregunta": [{"duda": "D2"}]}
    assert uq.parse_answer('Aquí va: {"preguntas": []}') == {"preguntas": [], "sin_pregunta": []}
    for bad in ("", "[]", '{"questions": []}'):
        with pytest.raises(ValueError, match="lista de preguntas"):
            uq.parse_answer(bad)


def test_one_call_per_batch_and_the_second_run_from_the_cache(targets, tmp_path):
    doubts, cache = _doubts(), tmp_path / uq.CACHE_FILE
    settings = uq.AskSettings(batch_size=3, pause_seconds=0)
    llm = FakeLLM({"preguntas": [], "sin_pregunta": []}, {"preguntas": []})
    report = uq.ask_ai(doubts, targets, llm=llm, settings=settings, cache_path=cache)
    assert (report.calls, report.cached, report.pending, len(report.answers)) == (2, 0, 0, 2)
    assert llm.contexts[0].count("=== D") == 3 and "=== D4 ===" in llm.contexts[1]
    again = uq.ask_ai(doubts, targets, llm=FakeLLM(), settings=settings, cache_path=cache)
    assert (again.calls, again.cached) == (0, 2)
    entry = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
    assert entry["doubts"] == [d.key for d in doubts[:3]]
    assert entry["prompt"]["name"] == "ask" and entry["model"] == "fake-model"


def test_without_ai_or_out_of_calls_the_doubts_wait(targets, tmp_path):
    doubts = _doubts()
    report = uq.ask_ai(doubts, targets, settings=uq.AskSettings(batch_size=3))
    assert (report.calls, report.pending) == (0, 4)
    limited = uq.ask_ai(doubts, targets, llm=FakeLLM({"preguntas": []}),
                        settings=uq.AskSettings(batch_size=3, max_calls=1, pause_seconds=0))
    assert (limited.calls, limited.pending, limited.call_limit_reached) == (1, 1, True)
    waits = []
    quota = uq.ask_ai(doubts, targets, llm=FakeLLM(error="429 RESOURCE_EXHAUSTED quota"),
                      settings=uq.AskSettings(batch_size=3, pause_seconds=0),
                      sleep_fn=waits.append)
    assert (quota.calls, quota.pending, quota.quota_exhausted) == (1, 4, True)
    assert quota.errors[0].startswith("429") and waits                  # it waited once
    broken = uq.ask_ai(doubts, targets, llm=FakeLLM("no es JSON", {"preguntas": []}),
                       settings=uq.AskSettings(batch_size=3, pause_seconds=0))
    assert (broken.calls, broken.pending, len(broken.answers)) == (2, 3, 1)


# ------------------------------------------------------------------ the fixed rule

def test_the_questions_after_the_fixed_rule(targets):
    doubts = _doubts()
    answer = {"preguntas": [
        _q(["D2"], ("Es un caso más específico", [_align("D2", "broadMatch",
                                                         "ProcessingContext")]),
           text="¿EmploymentContext es un caso de ProcessingContext?"),
        _q(["D3", "D4"],
           ("Sí, los dos", [_align("D3", "exactMatch", "hasRecipient"),
                            _align("D4", "closeMatch", "SensitivePersonalData")]),
           ("Otro término", [_align("D4", "exactMatch", "DataController")]),
           ("Mezcla", [_align("D3", "exactMatch", "PersonalData")])),
        _q(["D1", "D2"], ("Quitarla", [_remove("D1")]),
           ("Solo relacionados", [_align("D1", "relatedMatch", "Processing")]), rec=2)],
        "sin_pregunta": []}
    questions, skipped, refused = uq.validate([answer], doubts, _index(), _state(), targets)
    assert [(q.id, [d.id for d in q.doubts], len(q.options), q.recommended)
            for q in questions] == [("P1", ["D1"], 2, 1), ("P2", ["D2"], 1, 0),
                                    ("P3", ["D3", "D4"], 1, 0)]       # what is in it first
    assert skipped == []
    assert refused == [
        {"question": "¿Qué hacemos?", "option": "Otro término",
         "reason": "«DataController» no es uno de los términos que se mostraron para "
                   "«HealthData»"},
        {"question": "¿Qué hacemos?", "option": "Mezcla",
         "reason": "«PersonalData» no es uno de los términos que se mostraron para «contact»"}]
    p1 = questions[0]
    assert [c.after for c in p1.options[1].checked] == [[{
        "dpv_iri": DPV + "Processing", "dpv_name": "Processing", "relation": "skos:relatedMatch",
        "decided_by": "user", "version": None}]]
    assert p1.options[0].checked[0].after == []


def test_what_the_rule_leaves_out_and_why(targets):
    doubts = _doubts()
    answer = {"preguntas": [
        _q(["D1"], ("Igual", [_align("D1", "exactMatch", "Processing")]),
           ("Quitar", [_remove("D1")])),                              # recommended is a no-op
        _q(["D2"], ("Raro", [{"duda": "D2", "accion": "borrar"}]),
           ("Dos veces", [_align("D2", "broadMatch", "ProcessingContext"), _remove("D2")]),
           ("Ajeno", [_remove("D3")]), ("Cuarta", [_remove("D2")])),
        _q(["D3"], ("Propiedad", [_align("D3", "closeMatch", "hasRecipient")]),
           ("Repetida", [_align("D3", "closeMatch", "hasRecipient")]),
           ("Sin término", [_align("D3", "closeMatch", "")]),
           ("Sin relación", [_align("D3", "", "hasRecipient")])),
        _q(["D3"], ("Otra vez", [_align("D3", "exactMatch", "hasRecipient")])),
        _q(["D9"], ("No existe", [_remove("D9")]))],
        "sin_pregunta": [{"duda": "D4", "motivo": "El puntaje es bajo."}]}
    questions, skipped, refused = uq.validate([answer], doubts, _index(), _state(), targets)
    assert [(q.id, q.doubts[0].id, [o.text for o in q.options]) for q in questions] == [
        ("P1", "D3", ["Propiedad"])]
    assert [(s.doubt.id, s.by, s.reason) for s in skipped] == [
        ("D1", uq.BY_RULE, "la opción recomendada no pasó la regla"),
        ("D2", uq.BY_RULE, "ninguna opción pasó la regla"),
        ("D4", uq.BY_AI, "El puntaje es bajo.")]
    assert [r["reason"] for r in refused] == [
        "«InfoAboutProcessing» ya es exactMatch de «Processing»: no cambia nada.",
        "«borrar» no es alinear ni quitar",
        "cambia dos veces «EmploymentContext»",
        "cambia un concepto que no es de la pregunta",
        "repite otra opción",
        "no dice con qué término alinear «contact»"]            # the 4th option is not read


def test_the_maximum_and_what_the_ai_did_not_say(targets):
    doubts = _doubts()
    answer = {"preguntas": [_q(["D3"], ("Sí", [_align("D3", "closeMatch", "hasRecipient")])),
                            _q(["D2"], ("Sí", [_align("D2", "broadMatch", "ProcessingContext")]))]}
    questions, skipped, _ = uq.validate([answer], doubts, _index(), _state(), targets,
                                        max_questions=1)
    assert [q.doubts[0].id for q in questions] == ["D2"]
    assert [(s.doubt.id, s.by) for s in skipped] == [
        ("D1", uq.NO_ANSWER), ("D4", uq.NO_ANSWER), ("D3", uq.BY_LIMIT)]
    assert skipped[-1].reason == "quedó fuera del máximo de 1 preguntas"


def test_a_question_goes_where_its_most_urgent_doubt_goes(targets):
    answer = {"preguntas": [
        _q(["D2"], ("Más específico", [_align("D2", "broadMatch", "ProcessingContext")])),
        _q(["D3", "D1"], ("Juntas", [_align("D3", "closeMatch", "hasRecipient"),
                                     _remove("D1")]))]}
    questions, _, _ = uq.validate([answer], _doubts(), _index(), _state(), targets)
    assert [[d.id for d in q.doubts] for q in questions] == [["D3", "D1"], ["D2"]]


def test_propose_questions_end_to_end(targets, tmp_path):
    answer = {"preguntas": [_q(["D2"], ("Más específico", [_align("D2", "broadMatch",
                                                                  "ProcessingContext")]))],
              "sin_pregunta": [{"duda": d, "motivo": "Está bien."} for d in ("D1", "D3", "D4")]}
    qs = uq.propose_questions(_doubts(), _index(), _state(), targets, llm=FakeLLM(answer),
                              cache_path=tmp_path / uq.CACHE_FILE)
    assert qs.complete and qs.doubts == 4 and qs.model == "fake-model"
    assert qs.prompt == load_prompt("ask", 1).manifest()
    assert uq.summary_text(qs) == ("4 dudas, 1 pregunta (0 juntan varias dudas); sin pregunta: "
                                   "3 por la IA, 0 por la regla, 0 por el máximo")
    assert uq.render_question(qs.questions[0], chosen=0) == [
        "  P1. ¿Qué hacemos?",
        "     a) Más específico (recomendada, elegida)",
        "        EmploymentContext: exactMatch ProcessingContext (regla, versión 1) -> broadMatch "
        "ProcessingContext (usuario)",
        "     b) Dejar como está",
        "     Por qué: Porque sí."]
    j = uq.question_json(qs.questions[0], {"person": 0})
    assert (j["names"], j["kinds"], j["person"], j["options"][0]["text"]) == (
        ["EmploymentContext"], [uq.REVIEWED], 0, "Más específico")
    pending = uq.propose_questions(_doubts(), _index(), _state(), targets)
    assert not pending.complete and pending.ai["pending"] == 4 and not pending.questions


# ------------------------------------------------------------------ without AI and answering

def test_the_fixed_questions_the_uploads_ask(targets):
    qs = uq.fixed_questions(_doubts(), _index(), _state(), targets, max_questions=3)
    assert [(q.id, q.text, [o.text for o in q.options], q.recommended)
            for q in qs.questions] == [
        ("P1", "¿Qué hacemos con «InfoAboutProcessing»? La regla aceptó que es lo mismo que "
               "«Processing» y la IA revisora la marcó como incoherente.",
         ["Cambiar a: solo está relacionado con Processing (lo que propone la IA revisora)",
          "Quitar la correspondencia"], None),
        ("P2", "¿Qué hacemos con «EmploymentContext»? La regla aceptó que es lo mismo que "
               "«ProcessingContext» y la IA revisora la marcó como dudosa.",
         ["Cambiar a: es un caso más específico de ProcessingContext (lo que propone la IA "
          "revisora)", "Quitar la correspondencia"], None),
        ("P3", "¿«contact» es lo mismo que «hasRecipient» del DPV?",
         ["Sí, es lo mismo que hasRecipient"], None)]
    assert [q.reason for q in qs.questions] == [
        "La IA revisora: «Es información, no el tratamiento».",
        "La IA revisora: «Es un caso laboral».",
        "La IA lo propuso: «Motivo de contact». La regla no lo aceptó: las propiedades nunca "
        "se alinean solas."]
    assert [(s.doubt.id, s.by) for s in qs.skipped] == [("D4", uq.BY_LIMIT)]
    assert qs.prompt == {"name": "preguntas fijas"} and qs.complete and qs.refused == []
    assert uq.render_question(qs.questions[2]) == [
        "  P3. ¿«contact» es lo mismo que «hasRecipient» del DPV?",
        "     a) Sí, es lo mismo que hasRecipient",
        "        contact: sin correspondencia -> exactMatch hasRecipient (usuario)",
        "     b) Dejar como está",
        "     Por qué se pregunta: La IA lo propuso: «Motivo de contact». La regla no lo "
        "aceptó: las propiedades nunca se alinean solas."]


def test_a_fixed_option_also_goes_through_the_rule(targets):
    d1, d2, d3, _ = _doubts()
    d1.audit = {"verdict": "incoherente", "relation": "exactMatch"}       # what it already has
    d2.audit = {"verdict": "dudosa", "relation": None}
    d3.proposal = {**d3.proposal, "dpv_iri": DPV + "PersonalData", "dpv_name": "PersonalData"}
    qs = uq.fixed_questions([d1, d2, d3], _index(), _state(), targets)
    assert [[o.text for o in q.options] for q in qs.questions] == [
        ["Quitar la correspondencia"], ["Quitar la correspondencia"]]
    assert [(s.doubt.id, s.by, s.reason) for s in qs.skipped] == [
        ("D3", uq.BY_RULE, "«contact» es una propiedad y «PersonalData» no: una propiedad solo "
                           "se alinea con propiedades del DPV.")]
    assert qs.refused[0]["option"] == "Sí, es lo mismo que PersonalData"


def test_answering_changes_only_the_concepts_of_the_options(targets):
    qs = uq.fixed_questions(_doubts(), _index(), _state(), targets)
    state = _state()
    after = uq.answered(state, [qs.questions[0].options[1], qs.questions[3].options[0]])
    assert after[EX + "InfoAboutProcessing"] == {}
    assert after[EX + "HealthData"] == {DPV + "SensitivePersonalData": {
        "relation": "skos:closeMatch", "dpv_name": "SensitivePersonalData",
        "decided_by": "user", "version": None}}
    assert after[EX + "Fine"] == state[EX + "Fine"] and state == _state()   # not in place


def test_check_state_is_the_rule_of_s10_t03(targets):
    checked = ud.check_state(_index(), _state(), [
        ud.Change(EX + "Fine", ud.SET, relation="exactMatch", dpv="DataController"),
        ud.Change(EX + "Nothing", ud.SET, relation="closeMatch", dpv="hasRecipient")], targets)
    assert [c.reason for c in checked] == [
        "«Fine» ya es exactMatch de «DataController»: no cambia nada.",
        "«Nothing» es una clase y «hasRecipient» es una propiedad del DPV: una clase solo se "
        "alinea con clases del DPV."]
