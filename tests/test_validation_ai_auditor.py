from __future__ import annotations

import json
import re

import pytest

from src.alignment import decider_eval
from src.validation import ai_auditor as au
from src.validation.ai_auditor import (COHERENT, DOUBTFUL, INCOHERENT, INVALID, AuditItem,
                                       AuditSettings, ask_audit, audit_settings, body, measure,
                                       parse_reviews, validate_review)
from tests.test_alignment_propose import targets  # noqa: F401
from tests.test_prompt_lab import EX, LabLLM, env  # noqa: F401
from tests.test_user_alignment import _ws

DPV = "https://w3id.org/dpv#"
REVIEWS = {"SensitiveInfo": (INCOHERENT, "exactMatch", "Un dato sensible no es cualquier dato."),
           "PersonalData": (COHERENT, "exactMatch", "Ambos son datos personales."),
           "Doubtful": (DOUBTFUL, "closeMatch", "No hay definición."),
           "Broken": ("tal vez", "parecido", "")}


class AuditLLM:
    def __init__(self, reviews=None, error=None, model="modelo-de-prueba"):
        self.reviews = REVIEWS if reviews is None else reviews
        self.error, self.model, self.calls, self.contexts, self.prompts = error, model, 0, [], []

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        self.calls += 1
        self.contexts.append(context)
        self.prompts.append(prompt)
        if self.error:
            raise Exception(self.error)
        out = []
        for block in context.split("=== ")[1:]:
            cid = block.split(" ===", 1)[0]
            name = re.search(r"^Concepto: (\S+) \(", block, re.M).group(1)
            verdict, relation, why = self.reviews.get(name, (COHERENT, "exactMatch", "Bien."))
            out.append({"concept": cid, "verdict": verdict, "relation": relation,
                        "explanation": why})
        return json.dumps({"reviews": out}, ensure_ascii=False)


def _item(name, relation="skos:exactMatch", dpv="PersonalData", definition="Algo.",
          kinds=("class",)):
    return AuditItem(key=EX + name, name=name, kinds=kinds, label=None, definition=definition,
                     relation=relation, dpv_iri=DPV + dpv, dpv_name=dpv,
                     dpv_definition="Data about a person.", dpv_parents=["Data"])


# ------------------------------------------------------------------ the prompt and the answers

def test_the_prompt_is_ascii_and_asks_for_the_reviews():
    assert au.PROMPT.text.isascii() and au.PROMPT.version == 1
    for token in ('"reviews"', '"concept"', '"verdict"', '"relation"', '"explanation"',
                  "coherente", "dudosa", "incoherente"):
        assert token in au.PROMPT.text


def test_body_has_what_the_ai_needs():
    text = body(_item("SensitiveInfo", relation=None, definition=None))
    assert text == ("Concepto: SensitiveInfo (clase)\nEtiqueta: -\nDefinicion: -\n"
                    "Relacion propuesta: sin tipo (rdfs:seeAlso)\nTermino DPV: PersonalData\n"
                    "Definicion DPV: Data about a person.\nPadres en el DPV: Data")
    assert "Concepto: hasData (propiedad)" in body(_item("hasData", kinds=("property",)))
    assert "Relacion propuesta: closeMatch" in body(_item("X", relation="skos:closeMatch"))
    assert "\nDefinicion: a a a a a a a a a a\n" in body(_item("X", definition="a " * 300),
                                                          chars=20)


def test_parse_and_validate_the_reviews():
    raw = '```json\n{"reviews": [{"concept": "C1", "verdict": "Incoherente"}, 3]}\n```'
    assert parse_reviews(raw) == [{"concept": "C1", "verdict": "Incoherente"}]
    with pytest.raises(ValueError, match="lista de revisiones"):
        parse_reviews("no es json")
    assert validate_review({"verdict": "Incoherente", "relation": "skos:broadMatch",
                            "explanation": " Otra  cosa. "}) == {
        "verdict": INCOHERENT, "relation": "broadMatch", "explanation": "Otra cosa."}
    assert validate_review({"verdict": "quizás", "relation": "igual"}) == {
        "verdict": INVALID, "relation": None, "explanation": None}


def test_the_cache_key_changes_with_the_prompt(monkeypatch):
    from dataclasses import replace
    key = au.cache_key("Concepto: X", "m")
    assert key == au.cache_key("Concepto: X", "m") != au.cache_key("Concepto: X", "otro")
    monkeypatch.setattr(au, "PROMPT", replace(au.PROMPT, sha256="otro-prompt"))
    assert au.cache_key("Concepto: X", "m") != key


def test_settings():
    assert audit_settings({}) == AuditSettings()
    assert audit_settings({"audit": {"batch_size": 4}}).batch_size == 4
    with pytest.raises(ValueError, match="bloque audit"):
        audit_settings({"audit": {"batch_size": 0}})


# ------------------------------------------------------------------ asking

def test_ask_audit_batches_caches_and_limits(tmp_path):
    items = [_item(n) for n in ("SensitiveInfo", "PersonalData", "Doubtful", "Broken", "Other")]
    cache = tmp_path / "audit-cache.jsonl"
    llm = AuditLLM()
    report = ask_audit(items, llm=llm, settings=AuditSettings(batch_size=2, pause_seconds=0),
                       cache_path=cache)
    assert (report.calls, report.cached, llm.calls) == (3, 0, 3)
    assert {k[len(EX):]: a["verdict"] for k, a in report.answers.items()} == {
        "SensitiveInfo": INCOHERENT, "PersonalData": COHERENT, "Doubtful": DOUBTFUL,
        "Broken": INVALID, "Other": COHERENT}
    assert all(p == au.PROMPT.text for p in llm.prompts)
    assert report.ai() == {"available": True, "calls": 3, "cached": 0, "invalid": 1,
                           "quota_exhausted": False, "call_limit_reached": False, "errors": []}
    again = ask_audit(items, llm=llm, settings=AuditSettings(pause_seconds=0), cache_path=cache)
    assert (again.calls, again.cached, llm.calls) == (0, 5, 3)
    limited = ask_audit(items, llm=AuditLLM(), settings=AuditSettings(
        batch_size=2, max_calls=1, pause_seconds=0), cache_path=tmp_path / "otra.jsonl")
    assert limited.call_limit_reached and len(limited.answers) == 2
    nothing = ask_audit(items, llm=None, cache_path=tmp_path / "vacia.jsonl")
    assert nothing.calls == 0 and nothing.answers == {}


def test_a_quota_error_stops_and_other_errors_are_reported(tmp_path):
    items = [_item(n) for n in ("A", "B", "C")]
    quota = ask_audit(items, llm=AuditLLM(error="429 RESOURCE_EXHAUSTED quota"),
                      settings=AuditSettings(batch_size=1, pause_seconds=0),
                      cache_path=tmp_path / "a.jsonl", sleep_fn=lambda s: None)
    assert quota.quota_exhausted and quota.calls == 1
    broken = ask_audit(items, llm=AuditLLM(error="boom"), settings=AuditSettings(
        batch_size=2, pause_seconds=0), cache_path=tmp_path / "b.jsonl")
    assert broken.calls == 2 and broken.errors[0]["error"] == "boom"
    assert "error: boom" in au.ai_text(broken.ai())


# ------------------------------------------------------------------ measuring

def _row(name, correct, verdict=None, relation="skos:exactMatch", person="skos:exactMatch"):
    return {"correct": correct, "person": {DPV + ("PersonalData" if correct else "Other"): person},
            "item": _item(name, relation=relation), "verdict": verdict}


def test_measure_and_the_criterion_fixed_beforehand():
    rows = ([_row(f"R{i}", True) for i in range(18)] + [_row(f"W{i}", False) for i in range(6)])
    answers = {r["item"].key: {"verdict": COHERENT, "relation": "exactMatch"} for r in rows}
    for name in ("W0", "W1", "W2"):
        answers[EX + name]["verdict"] = INCOHERENT
    answers[EX + "W3"]["verdict"] = DOUBTFUL
    for name in ("R0", "R1"):
        answers[EX + name]["verdict"] = DOUBTFUL
    result = measure(rows, answers)
    one, two = result["options"]["incoherente"], result["options"]["incoherente o dudosa"]
    assert (one["caught"], one["lost"], one["precision_before"], one["precision_after"]) == (
        3, 0, 0.75, round(18 / 21, 4))
    assert one["adopted"]                         # +10,7 points, 0 right ones lost
    assert (two["caught"], two["lost"], two["precision_after"]) == (4, 2, round(16 / 18, 4))
    assert not two["adopted"]                     # loses 2 of 18 right ones: 11 % > 10 %
    assert result["adopted"] == "incoherente"
    assert result["verdicts"][INCOHERENT] == {"right": 0, "wrong": 3}
    assert result["relations"] == {"typed": 18, "decider": 18, "ai": 18}


def test_too_small_a_gain_adopts_nothing():
    rows = [_row(f"R{i}", True) for i in range(30)] + [_row(f"W{i}", False) for i in range(10)]
    answers = {EX + "W0": {"verdict": INCOHERENT, "relation": None}}
    result = measure(rows, answers)
    assert result["options"]["incoherente"]["precision_after"] == round(30 / 39, 4)
    assert result["adopted"] is None
    assert "la IA solo explica" in au.render_evaluation({
        **result, "prompt": au.PROMPT.manifest(), "ai": {"available": True, "calls": 1,
                                                          "cached": 0, "invalid": 0},
        "pending": 0, "examples": {"caught": [], "missed": [], "false_alarms": []}})


def test_evaluate_on_the_552_decisions(env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(decider_eval, "load_eval_inputs", lambda cfg=None: env.inputs)
    from tests.test_alignment_propose import Embed
    cfg = {**env.cfg, "audit": {"batch_size": 1, "max_calls": 0, "pause_seconds": 0}}
    result = au.evaluate(cfg, llm=AuditLLM(), decider_llm=LabLLM(), embed_fn=Embed(),
                         chroma_path=env.chroma, sleep_fn=lambda s: None)
    assert (result["accepted"], result["right"], result["wrong"]) == (2, 1, 1)
    assert result["ai"]["calls"] == 2 and result["pending"] == 0   # measuring needs them all
    o = result["options"]["incoherente"]
    assert (o["caught"], o["lost"], o["precision_before"], o["precision_after"]) == (
        1, 0, 0.5, 1.0)
    assert result["adopted"] == "incoherente"
    assert [e["name"] for e in result["examples"]["caught"]] == ["SensitiveInfo"]
    assert result["examples"]["caught"][0] == {
        "name": "SensitiveInfo", "dpv": "PersonalData", "relation": "broadMatch",
        "person": ["SensitivePersonalData"], "verdict": INCOHERENT,
        "explanation": "Un dato sensible no es cualquier dato."}
    out = env.cfg["outputs"]["dir"]
    saved = json.loads((__import__("pathlib").Path(out) / au.EVALUATION_FILE).read_text(
        encoding="utf-8"))
    assert saved["adopted"] == "incoherente" and saved["prompt"] == au.PROMPT.manifest()
    text = au.render_evaluation(result)
    assert text.startswith("Revisión de la IA (prompt audit.v1) de las 2 correspondencias")
    assert "  Iguales a tu decisión: 1 · distintas: 1" in text
    assert ("    incoherente: atrapa 1 de 1 distintas, quita 0 de 1 iguales; precisión "
            "50,0 % -> 100,0 % (cumple)") in text
    assert "  Veredicto: dejar fuera las que la IA marca «incoherente»" in text
    assert ("  Relación SKOS en las 1 iguales con tipo: coincide con la tuya la del decisor en 1 "
            "y la de la IA revisora en 1") in text
    assert ("    - SensitiveInfo -> PersonalData (broadMatch); tú: SensitivePersonalData; IA: "
            "incoherente. Un dato sensible no es cualquier dato.") in text
    again = au.evaluate(env.cfg, llm=AuditLLM(), decider_llm=LabLLM(), embed_fn=Embed(),
                        chroma_path=env.chroma, sleep_fn=lambda s: None)
    assert again["ai"]["calls"] == 0 and again["ai"]["cached"] == 2


# ------------------------------------------------------------------ one upload

def test_audit_of_an_upload(env, targets, monkeypatch):  # noqa: F811
    w = _ws(env, targets)
    llm = AuditLLM(reviews={"PersonalData": (DOUBTFUL, "closeMatch", "Falta la definición.")})
    result = au.audit_workspace(w, env.cfg, llm=llm, targets=targets, sleep_fn=lambda s: None)
    assert result["summary"] == {"accepted": 1, COHERENT: 0, DOUBTFUL: 1, INCOHERENT: 0,
                                 INVALID: 0, "pending": 0}
    assert result["entries"][0] == {"key": "http://example.org/gdpr#PersonalData",
                                    "name": "PersonalData", "dpv_iri": DPV + "PersonalData",
                                    "dpv_name": "PersonalData", "relation": "exactMatch",
                                    "verdict": DOUBTFUL, "ai_relation": "closeMatch",
                                    "explanation": "Falta la definición."}
    assert "Definicion: Information about an identified person." in llm.contexts[0]
    assert au.load_result(w) == result
    text = au.render_workspace(result, w.id)
    assert "  Coherentes 0 · dudosas 1 · incoherentes 0" in text
    assert "    [dudosa] PersonalData -> PersonalData (exactMatch): Falta la definición." in text
    monkeypatch.setenv("LLM_MODEL", "modelo-de-prueba")         # the model of the cache
    without = au.audit_workspace(w, env.cfg, use_llm=False, targets=targets)
    assert without["ai"]["available"] is False and without["summary"][DOUBTFUL] == 1  # cache
    (w.work_dir / au.CACHE_FILE).unlink()
    spy = AuditLLM()
    empty = au.audit_workspace(w, env.cfg, llm=spy, use_llm=False, targets=targets)
    assert spy.calls == 0 and empty["summary"]["pending"] == 1


def test_a_concept_without_definition_shows_the_one_the_law_gave(env, targets):  # noqa: F811
    from src.alignment import law_definitions as ld
    from tests.test_alignment_propose import TTL
    w = _ws(env, targets, ttl=TTL.replace(
        ' ; rdfs:comment "Information about an identified person."@en', ""))
    (w.work_dir / ld.RESULT_FILE).write_text(json.dumps({"entries": [
        {"key": "http://example.org/gdpr#PersonalData", "definition": "Dato de una persona.",
         "status": ld.LAW_CONCEPT}]}), encoding="utf-8")
    llm = AuditLLM()
    result = au.audit_workspace(w, env.cfg, llm=llm, targets=targets, sleep_fn=lambda s: None)
    assert result["summary"]["accepted"] == 1
    assert "\nDefinicion: Dato de una persona.\n" in llm.contexts[0]


def test_an_upload_not_decided_yet(env, targets, tmp_path):  # noqa: F811
    from src.workspace.store import open_ontology
    from tests.test_alignment_propose import TTL
    w = open_ontology("otra.ttl", TTL.encode("utf-8"), cfg=env.cfg)
    with pytest.raises(ValueError, match="Primero hay que decidir"):
        au.audit_workspace(w, env.cfg, llm=AuditLLM(), targets=targets)


def test_the_cli(env, targets, capsys, monkeypatch):  # noqa: F811
    assert au._run([]) == 1 and au._run(["--medir", "x"]) == 1
    assert "Error: indica un espacio de trabajo o --medir." in capsys.readouterr().out
    assert au._run(["no-existe"]) == 1
    assert "Error: No existe el espacio de trabajo no-existe." in capsys.readouterr().out
    w = _ws(env, targets)
    monkeypatch.setattr(au, "audit_workspace", lambda ws, cfg, use_llm=True: {
        "summary": {"accepted": 0, COHERENT: 0, DOUBTFUL: 0, INCOHERENT: 0, INVALID: 0,
                    "pending": 0}, "ai": {"available": use_llm, "calls": 0, "cached": 0,
                                          "invalid": 0}, "entries": []})
    assert au._run([str(w.path), "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"Revisión de la IA del espacio {w.id}: 0 correspondencias")
    assert "  IA: no usada" in out