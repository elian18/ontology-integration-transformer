from __future__ import annotations

import csv
import json
import sys
import types
import zipfile

import pytest
from types import SimpleNamespace

from src.alignment import inherit as inh
from src.alignment import propose as prop
from src.alignment import user_alignment as ua
from src.validation import ai_auditor as au
from src.validation import logic_inspector as li
from src.validation import shape_inspector as si
from src.validation import upload_validation as uv
from src.validation.shape_inspector import GREEN, RED, YELLOW
from src.validation.upload_validation import FAILED, NOT_RUN, audit_status, overall
from src.workspace.store import CACHE_DIR, DONE, ERROR, shared_cache
from tests.test_alignment_propose import TTL, DeciderLLM, Embed, env, targets  # noqa: F401

DPV = "https://w3id.org/dpv#"
TERMS = ("PersonalData", "DataSubject", "DataController", "DataSubjectRight",
         "SensitivePersonalData", "hasDataSubject", "hasRecipient", "hasEntity", "Consent",
         "ConsentRecord")                               # the last two: what the base inherits


class FlaggingLLM(DeciderLLM):
    """DeciderLLM whose reviewer marks PersonalData as incoherent."""

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        if '"reviews"' not in prompt:
            return super().ask(prompt, context, max_tokens, json_mode)
        self.calls += 1
        return json.dumps({"reviews": [{"concept": "C1", "verdict": "incoherente",
                                        "relation": "closeMatch",
                                        "explanation": "Trata de otra cosa."}]})


def _real_dpv(env):
    """The small DPV of the tests with rdfs:isDefinedBy, as the real one has."""
    path = env.cfg["inputs"]["dpv"]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("".join(f"dpv:{t} rdfs:isDefinedBy dpv: .\n" for t in TERMS))
    return env


def _align(env, targets, llm=None, **kw):  # noqa: F811
    return ua.align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), fmt="turtle", embed_fn=Embed(),
                           llm=llm or DeciderLLM(), base=env.base, targets=targets, cfg=env.cfg,
                           chroma_path=env.chroma, sleep_fn=lambda s: None, **kw)


def _csv(w) -> dict:
    with open(w.output_dir / ua.CSV_FILE, encoding="utf-8-sig") as handle:
        return {r["concept_name"]: r for r in csv.DictReader(handle)}


def test_every_ai_answer_of_an_upload_goes_to_the_shared_cache(env, targets):  # noqa: F811
    from src.alignment.decider import CACHE_FILE as DECIDER
    from src.transform.labels import CACHE_FILE as LABELS
    from src.validation.question_explainer import CACHE_FILE as EXPLAIN
    w, _, _ = _align(env, targets)
    assert not list(w.work_dir.glob("*-cache.jsonl"))           # S10-T02
    shared = {p.name for p in (w.path.parent / CACHE_DIR).glob("*.jsonl")}
    assert shared == {DECIDER, LABELS, au.CACHE_FILE, EXPLAIN}


# ------------------------------------------------------------------ the colors

def test_the_worst_color_wins_and_a_failure_is_yellow():
    assert overall([GREEN, GREEN, NOT_RUN]) == GREEN
    assert overall([GREEN, YELLOW, GREEN]) == YELLOW
    assert overall([GREEN, FAILED, NOT_RUN]) == YELLOW
    assert overall([YELLOW, RED, FAILED]) == RED


def test_the_color_of_the_ai_review():
    def s(**kw):
        return {"summary": {"accepted": 3, au.COHERENT: 0, au.DOUBTFUL: 0, au.INCOHERENT: 0,
                            au.INVALID: 0, "pending": 0, **kw}}
    assert audit_status(None) == NOT_RUN
    assert audit_status(s(**{au.COHERENT: 3})) == GREEN
    assert audit_status(s(**{au.COHERENT: 2, au.INCOHERENT: 1})) == YELLOW
    assert audit_status(s(**{au.COHERENT: 2, au.DOUBTFUL: 1})) == YELLOW
    assert audit_status(s(pending=3)) == NOT_RUN                  # the AI reviewed nothing
    assert audit_status(s(**{au.COHERENT: 2}, pending=1)) == GREEN
    assert audit_status(s(accepted=0)) == GREEN                   # nothing to review


def test_a_result_saved_in_the_sprint_8_reads_as_yellow(tmp_path):
    old = {"workspace": "w", "status": si.OLD_YELLOW,
           "inspectors": {uv.SHAPES: {"status": si.OLD_YELLOW}, uv.LOGIC: {"status": GREEN},
                          uv.AUDIT: {"status": uv.NOT_RUN}}}
    (tmp_path / uv.RESULT_FILE).write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    v = uv.load_result(SimpleNamespace(output_dir=tmp_path))
    assert v["status"] == YELLOW == "amarillo"
    assert [b["status"] for b in v["inspectors"].values()] == [YELLOW, GREEN, uv.NOT_RUN]
    assert si.current_color(RED) == RED and si.current_color(uv.FAILED) == uv.FAILED


# ------------------------------------------------------------------ each upload

def test_each_upload_is_validated_and_the_zip_says_so(env, targets):  # noqa: F811
    w, m, path = _align(_real_dpv(env), targets)
    v = uv.load_result(w)
    shapes, logic, audit = (v["inspectors"][k] for k in (uv.SHAPES, uv.LOGIC, uv.AUDIT))
    assert (shapes["status"], shapes["new"], shapes["inherited"], shapes["warnings"]) == (
        YELLOW, 0, 0, 3)                                           # 3 concepts without definition
    assert (logic["status"], logic["consistent"], logic["new"]) == (GREEN, True, 0)
    assert logic["engine"] == f"owlrl {li.version('owlrl')}, OWL 2 RL en Python"
    assert audit["status"] == GREEN and audit["summary"]["accepted"] == 1
    assert v["status"] == YELLOW
    assert w.step("shapes")["status"] == DONE and w.step("shapes")["result"] == YELLOW
    assert w.step("logic")["detail"] == ("verde: 0 inconsistencias nuevas, 0 heredadas, "
                                         "0 advertencias")
    assert w.step("audit")["detail"] == "Coherentes 1 · dudosas 0 · incoherentes 0"
    names = zipfile.ZipFile(path).namelist()
    assert uv.RESULT_FILE in names and uv.LOG_FILE in names
    readme = zipfile.ZipFile(path).read("LEEME.txt").decode("utf-8")
    assert "\nValidación: amarillo (no bloquea la descarga)\n" in readme
    assert (f"  - Forma (pySHACL {li.version('pyshacl')}): amarillo · 0 violaciones nuevas, "
            "0 heredadas, 3 advertencias\n") in readme
    assert ("  - IA revisora (audit.v1): verde · 0 incoherentes y 0 dudosas de 1 aceptada por "
            "la regla; solo explica, no cambia lo aceptado\n") in readme
    log = (w.output_dir / uv.LOG_FILE).read_text(encoding="utf-8")
    assert log.startswith(f"Validación del espacio {w.id}: amarillo\n\nInspector de forma (SHACL)")
    assert "Inspector de lógica (razonador)" in log and "Revisión de la IA del espacio" in log
    text = ua.render_console(w, m, path)
    assert "  IA de la revisión: 1 llamada, 0 desde la caché, 0 inválidas\n" in text
    assert "\n  Validación: amarillo (no bloquea la descarga)\n    Forma (pySHACL" in text
    assert "\n    Detalle: output/validation-log.txt\n" in text
    saved = (w.output_dir / uv.RESULT_FILE).read_text(encoding="utf-8")
    assert not any(si.OLD_YELLOW in t for t in (saved, readme, log, text))   # S9-T01
    row = _csv(w)["PersonalData"]
    assert (row["ai_review"], row["ai_relation"], row["ai_explanation"]) == (
        au.COHERENT, "exactMatch", "Revisión de prueba para PersonalData.")
    assert _csv(w)["DataController"]["ai_review"] == ""           # suggested: not reviewed
    assert m["ai_audit"]["accepted"] == 1 and m["ai_audit"]["ai"]["calls"] == 1
    assert m["settings"]["prompts"]["audit"] == au.PROMPT.manifest()


def test_a_term_outside_the_dpv_is_red_and_the_zip_is_still_there(env, targets):  # noqa: F811
    w, m, path = _align(env, targets)                  # the small DPV has no rdfs:isDefinedBy
    v = uv.load_result(w)
    assert v["status"] == RED and v["inspectors"][uv.SHAPES]["new"] == 3
    assert path.exists() and w.step("package")["status"] == DONE


def test_the_review_flags_and_is_reused_while_nothing_changes(env, targets):  # noqa: F811
    llm = FlaggingLLM()
    w, m, _ = _align(_real_dpv(env), targets, llm=llm)
    audit = uv.load_result(w)["inspectors"][uv.AUDIT]
    assert audit["status"] == YELLOW and audit["flagged"] == [{
        "name": "PersonalData", "dpv_name": "PersonalData", "relation": "exactMatch",
        "verdict": au.INCOHERENT, "ai_relation": "closeMatch",
        "explanation": "Trata de otra cosa."}]
    assert _csv(w)["PersonalData"]["ai_review"] == au.INCOHERENT
    assert m["summary"]["accepted_mappings"] == 1                 # it explains, it never demotes
    calls, seen = llm.calls, []
    _align(env, targets, llm=llm, progress=seen.append)
    assert llm.calls == calls and "La IA revisa las correspondencias aceptadas" not in seen
    assert "Validar la forma (SHACL)" in seen


def test_when_a_review_is_current(env, targets, monkeypatch):  # noqa: F811
    w, _, _ = _align(env, targets)
    result = au.load_result(w)
    assert au.is_current(result, w, env.cfg) and au.current_result(w) == result
    assert not au.is_current(None, w, env.cfg)
    other = {**result, "prompt": {**result["prompt"], "sha256": "otro"}}
    assert not au.is_current(other, w, env.cfg)
    stale = {**result, "decisions": "otras"}
    assert not au.is_current(stale, w, env.cfg)
    (w.work_dir / au.RESULT_FILE).write_text(json.dumps(stale), encoding="utf-8")
    assert au.current_result(w) is None                           # another set of accepted
    assert ua.write_user_alignment(w, env.cfg)["ai_audit"] is None
    assert uv.validate_workspace(w, env.cfg)["inspectors"][uv.AUDIT] == {"status": NOT_RUN}
    pending = {**result, "summary": {**result["summary"], "pending": 1}}
    assert not au.is_current(pending, w, env.cfg)                 # the AI can finish it
    assert au.is_current(pending, w, env.cfg, use_llm=False)
    off = {**env.cfg, "audit": {"enabled": False}}
    assert au.is_current(pending, w, off)


def test_without_a_key_or_switched_off_the_review_asks_nothing(env, targets,  # noqa: F811
                                                                 monkeypatch):
    w, _, _ = _align(env, targets)
    shared_cache(w, au.CACHE_FILE).unlink()
    def no_key(*a, **k):                               # .env is loaded again on import
        raise KeyError("LLM_API_KEY")
    monkeypatch.setattr("src.ai.llm_client.LLMClient", no_key)
    out = au.audit_workspace(w, env.cfg, targets=targets)
    assert out["ai"]["available"] is False and out["summary"]["pending"] == 1
    assert out["ai"]["error"] == "Falta la variable 'LLM_API_KEY' en .env"
    assert "no usada (Falta la variable 'LLM_API_KEY' en .env)" == au.ai_text(out["ai"])
    spy = DeciderLLM()
    off = au.audit_workspace(w, {**env.cfg, "audit": {"enabled": False}}, llm=spy,
                             targets=targets)
    assert spy.calls == 0 and off["ai"]["error"] == "desactivada en config.yaml (audit.enabled)"
    assert au.audit_workspace(w, env.cfg, llm=spy, use_llm=False,
                              targets=targets)["ai"]["error"] == "IA desactivada (--sin-ia)"
    assert w.step("audit")["status"] == DONE


def test_a_failed_check_never_blocks_the_download(env, targets, monkeypatch):  # noqa: F811
    def no_java(ws, cfg=None, engine=None):
        raise RuntimeError("Java no responde")

    def broken(*args, **kwargs):
        raise RuntimeError("se cayó")

    monkeypatch.setattr(li, "inspect_workspace", no_java)
    monkeypatch.setattr(au, "audit_workspace", broken)
    w, m, path = _align(_real_dpv(env), targets)
    v = uv.load_result(w)
    assert v["inspectors"][uv.LOGIC] == {"status": FAILED, "error": "Java no responde"}
    assert v["inspectors"][uv.AUDIT] == {"status": NOT_RUN}
    assert v["status"] == YELLOW and m["ai_audit"] is None
    assert w.step("logic")["status"] == ERROR and w.step("logic")["detail"] == "Java no responde"
    assert path.exists() and w.step("package")["status"] == DONE
    log = (w.output_dir / uv.LOG_FILE).read_text(encoding="utf-8")
    assert "Lógica: no se pudo correr (Java no responde)" in log
    assert "Revisión de la IA: no se hizo" in log
    lines = uv.summary_lines(v)
    assert "  Lógica: no se pudo correr (Java no responde)" in lines
    assert "  IA revisora: sin revisar · no se hizo la revisión" in lines


def test_the_text_of_each_inspector():
    block = {"status": NOT_RUN, "summary": {"accepted": 4, au.COHERENT: 0, au.DOUBTFUL: 0,
                                             au.INCOHERENT: 0, au.INVALID: 0, "pending": 4},
             "ai": {"available": False, "error": "IA desactivada (--sin-ia)"},
             "prompt": au.PROMPT.manifest()}
    assert uv.inspector_text(uv.AUDIT, block) == (
        "IA revisora (audit.v1): sin revisar · la IA no revisó las 4 aceptadas por la regla: "
        "IA desactivada (--sin-ia)")
    failed = {**block, "ai": {"available": True, "errors": ["429 cuota"]}}
    assert uv.inspector_text(uv.AUDIT, failed).endswith("aceptadas por la regla: 429 cuota")
    assert uv.inspector_text(uv.AUDIT, {**block, "ai": {"available": True}}).endswith(
        "regla: sin respuesta")
    flagged = {**block, "status": YELLOW, "summary": {**block["summary"], au.COHERENT: 1,
                                                     au.INCOHERENT: 1, au.DOUBTFUL: 1,
                                                     "pending": 1}}
    assert uv.inspector_text(uv.AUDIT, flagged) == (
        "IA revisora (audit.v1): amarillo · 1 incoherente y 1 dudosa de 4 aceptadas por la regla, "
        "1 sin revisar; solo explica, no cambia lo aceptado")
    none = {**block, "status": GREEN, "summary": {**block["summary"], "accepted": 0}}
    assert uv.inspector_text(uv.AUDIT, none) == (
        "IA revisora (audit.v1): verde · la regla no aceptó ninguna correspondencia que revisar")
    logic = {"status": YELLOW, "engine": "Apache Jena 6.2.0, OWL Micro", "new": 0,
             "inherited": 255, "warnings": 1}
    assert uv.inspector_text(uv.LOGIC, logic) == (
        "Lógica (Apache Jena 6.2.0, OWL Micro): amarillo · 0 inconsistencias nuevas, "
        "255 heredadas, 1 advertencia")


def test_cli_runs_a_workspace_again(env, targets, tmp_path, monkeypatch, capsys):  # noqa: F811
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma-cli"))
    monkeypatch.setattr("src.config.load_config", lambda: env.cfg)
    monkeypatch.setattr(inh, "load_base", lambda **kw: env.base)
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: targets)
    w, _, _ = _align(_real_dpv(env), targets)
    assert ua._run(["--espacio", w.id, "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert "  Validación: amarillo (no bloquea la descarga)\n" in out
    assert f"  IA de la revisión: {ua.REUSED} (1 llamada, 0 desde la caché" in out
    assert ua._run([]) == 1 and ua._run(["x.ttl", "--espacio", w.id]) == 1
    assert "Error: indica un archivo o --espacio." in capsys.readouterr().out
    assert ua._run(["--espacio", "no-existe"]) == 1
    assert "Error: No existe el espacio de trabajo no-existe." in capsys.readouterr().out
    (w.work_dir / au.RESULT_FILE).unlink()
    shared_cache(w, au.CACHE_FILE).unlink()
    assert ua._run(["--espacio", str(w.path), "--sin-ia"]) == 0     # the folder works too
    assert ("  IA de la revisión: no usada (IA desactivada (--sin-ia))\n"
            in capsys.readouterr().out)


def test_the_blocks_of_each_inspector():
    counts = {"nuevo": 1, "ontología": 2, "base": 3}
    shape_report = {"status": RED, "engine": {"name": "pySHACL", "version": "0.40.1"},
                    "summary": {"violations": counts, "warnings": {**counts, "nuevo": 4}},
                    "completeness": {"concepts": 1}, "seconds": 0.5}
    assert uv.shapes_block(shape_report) == {
        "status": RED, "engine": "pySHACL 0.40.1", "new": 1, "inherited": 5, "warnings": 9,
        "completeness": {"concepts": 1}, "seconds": 0.5}
    logic_report = {"status": YELLOW, "consistent": False, "seconds": 2.1,
                    "engine": {"name": li.JENA, "version": "6.2.0", "reasoner": "owl-micro",
                               "java": "24.0.2"},
                    "summary": {"errors": {**counts, "nuevo": 0}, "warnings": counts}}
    block = uv.logic_block(logic_report)
    assert block["engine"] == "Apache Jena 6.2.0, OWL Micro"
    assert (block["new"], block["inherited"], block["warnings"]) == (0, 5, 6)
    entries = [{"name": v, "dpv_name": "PersonalData", "relation": "exactMatch", "verdict": v,
                "ai_relation": None, "explanation": "x"}
               for v in (au.COHERENT, au.DOUBTFUL, au.INCOHERENT, au.INVALID, None)]
    audit = {"summary": {"accepted": 5, au.COHERENT: 1, au.DOUBTFUL: 1, au.INCOHERENT: 1,
                         au.INVALID: 1, "pending": 1}, "ai": {}, "prompt": au.PROMPT.manifest(),
             "entries": entries}
    assert [e["verdict"] for e in uv.audit_block(audit)["flagged"]] == [au.DOUBTFUL,
                                                                        au.INCOHERENT]


def test_the_fingerprint_follows_term_and_relation():
    one = [{"key": "a", "dpv_iri": DPV + "PersonalData", "relation": "skos:exactMatch"}]
    assert au.fingerprint(one) == au.fingerprint(list(one))
    assert au.fingerprint(one) != au.fingerprint([{**one[0], "relation": "skos:closeMatch"}])
    assert au.fingerprint(one) != au.fingerprint([{**one[0], "dpv_iri": DPV + "Data"}])
    assert au.fingerprint(one) != au.fingerprint([{**one[0], "key": "b"}])


def test_the_log_has_every_flagged_correspondence(env, targets, monkeypatch):  # noqa: F811
    w, _, _ = _align(_real_dpv(env), targets)
    entries = [{"key": f"k{n}", "name": f"Concepto{n}", "dpv_iri": DPV + "PersonalData",
                "dpv_name": "PersonalData", "relation": "exactMatch", "verdict": au.INCOHERENT,
                "ai_relation": None, "explanation": f"Motivo {n}."} for n in range(au.EXAMPLES + 2)]
    review = {"summary": {"accepted": len(entries), au.COHERENT: 0, au.DOUBTFUL: 0,
                          au.INCOHERENT: len(entries), au.INVALID: 0, "pending": 0},
              "ai": {"available": True, "calls": 1, "cached": 0, "invalid": 0},
              "prompt": au.PROMPT.manifest(), "entries": entries}
    monkeypatch.setattr(au, "current_result", lambda ws: review)
    uv.validate_workspace(w, env.cfg)
    log = (w.output_dir / uv.LOG_FILE).read_text(encoding="utf-8")
    assert all(f"[incoherente] Concepto{n} -> PersonalData" in log for n in range(len(entries)))

# ------------------------------------------------------------------ the competency questions (S9-T05)

def test_each_upload_answers_the_generic_questions_of_its_mode(env, targets):  # noqa: F811
    from src.validation import questions as qs
    seen = []
    w, m, path = _align(_real_dpv(env), targets, progress=seen.append)
    assert seen[-3:] == ["Validar la lógica (razonador)",
                         "Responder las preguntas de competencia", "Preparar la descarga"]
    v = uv.load_result(w)
    assert list(v["inspectors"]) == [uv.SHAPES, uv.LOGIC, uv.QUESTIONS, uv.AUDIT]
    q = v["inspectors"][uv.QUESTIONS]
    assert q["status"] == YELLOW and q["engine"] == f"rdflib {li.version('rdflib')}"
    assert q["summary"] == {GREEN: 2, YELLOW: 5, qs.GRAY: 0, qs.NOT_APPLICABLE: 1}
    assert [r["id"] for r in q["results"]] == [f"G0{n}" for n in range(1, 9)]
    g01, g08 = q["results"][0], q["results"][-1]
    assert (g01["color"], g01["applies"]) == (GREEN, True)
    assert g01["answer"] == "2 conceptos (p. ej. Consent Consent; PersonalData PersonalData)"
    assert (g08["color"], g08["applies"]) == (qs.GRAY, False)
    assert g08["answer"] == ("No aplica a «Mi ontología»: este modo no produce lo que la "
                             "pregunta necesita")
    assert w.step("questions")["detail"] == ("amarillo: 2 en verde, 5 en amarillo y 0 en gris; "
                                             "1 no aplica")
    report = qs.load_result(w)
    assert report["files"] == [m["files"]["ontology"], m["files"]["labels"],
                               m["files"]["alignment"], qs._shown(env.cfg["inputs"]["dpv"])]
    line = (f"Preguntas (rdflib {li.version('rdflib')}): amarillo · 2 en verde, 5 en amarillo "
            "y 0 en gris de las 7 que aplican; 1 no aplica")
    readme = zipfile.ZipFile(path).read("LEEME.txt").decode("utf-8")
    assert f"\n  - {line}\n" in readme and "«no aplica» que tu modo no produce" in readme
    log = (w.output_dir / uv.LOG_FILE).read_text(encoding="utf-8")
    assert f"Preguntas de competencia del espacio {w.id} (Mi ontología): amarillo" in log
    assert "    [no aplica] G08 ¿De qué artículo de la ley sale cada concepto?" in log
    assert f"\n    {line}\n" in ua.render_console(w, m, path)
    g02 = q["results"][1]                              # S9-T06: the AI says why
    assert (g02["explanation"], g02["explained_by"]) == ("Explicación de prueba para G02.", "ia")
    assert "explanation" not in g01 and "explanation" not in g08
    assert q["ai"]["calls"] == 1 and q["ai"]["questions"] == 5
    why = ("G02 ¿Qué derechos del titular representa la ontología? Por qué (IA): Explicación de "
           "prueba para G02.")
    assert f"\n      {why}\n" in readme
    assert f"\n      - {why}\n" in ua.render_console(w, m, path)
    assert "         Por qué (IA): Explicación de prueba para G02." in log


def test_a_repeated_upload_does_not_ask_the_ai_again_why(env, targets):  # noqa: F811
    from tests.test_alignment_propose import DeciderLLM as Fake
    llm = Fake()
    w, _, _ = _align(_real_dpv(env), targets, llm=llm)
    calls = llm.calls
    _align(env, targets, llm=llm)
    assert llm.calls == calls
    q = uv.load_result(w)["inspectors"][uv.QUESTIONS]
    assert (q["ai"]["calls"], q["ai"]["cached"]) == (0, 5)
    assert q["results"][1]["explained_by"] == "ia"
    off = uv.validate_workspace(w, env.cfg, llm=Fake(), use_llm=False)
    assert off["inspectors"][uv.QUESTIONS]["results"][1]["explained_by"] == "fija"


def test_a_failed_question_engine_is_gray_and_never_blocks_the_download(env, targets,  # noqa: F811
                                                                         monkeypatch):
    from app.services import align_ontology as ao
    from src.validation import questions as qs
    def no_engine(ws, cfg=None, engine=None, java=None, **ai):
        raise RuntimeError("Jena falló: OutOfMemoryError")
    monkeypatch.setattr(qs, "inspect_workspace", no_engine)
    w, m, path = _align(_real_dpv(env), targets)
    v = uv.load_result(w)
    assert v["inspectors"][uv.QUESTIONS] == {"status": FAILED,
                                             "error": "Jena falló: OutOfMemoryError"}
    assert w.step("questions")["status"] == ERROR and path.exists()
    assert w.step("package")["status"] == DONE
    assert "  Preguntas: no se pudo correr (Jena falló: OutOfMemoryError)" in uv.summary_lines(v)
    log = (w.output_dir / uv.LOG_FILE).read_text(encoding="utf-8")
    assert "Preguntas: no se pudo correr (Jena falló: OutOfMemoryError)" in log
    view = ao.validation_view(w)                       # a gray card, the download goes on
    assert [(c["key"], c["badge"]) for c in view["cards"]] == [
        (uv.SHAPES, ":yellow-badge[Amarillo]"), (uv.LOGIC, ":green-badge[Verde]"),
        (uv.QUESTIONS, ":gray-badge[No corrió]"), (uv.AUDIT, ":green-badge[Verde]")]
    assert view["cards"][2]["text"] == "No pudo correr  \nJena falló: OutOfMemoryError"
    assert view["questions"] == {"caption": "Las preguntas no pudieron correr: Jena falló: "
                                            "OutOfMemoryError. Esto no bloquea la descarga.",
                                 "rows": []}
    assert view["headline"].endswith("no pudo correr: Preguntas (SPARQL).")


def test_the_headline_counts_the_questions_to_look_at():
    from app.services import align_ontology as ao
    from src.validation import questions as qs
    block = {"status": GREEN, "summary": {"accepted": 0, au.COHERENT: 0, au.DOUBTFUL: 0,
                                           au.INCOHERENT: 0, au.INVALID: 0, "pending": 0}}
    clean = {"status": GREEN, "new": 0, "inherited": 0, "warnings": 0}
    result = {"status": YELLOW, "inspectors": {
        uv.SHAPES: clean, uv.LOGIC: clean, uv.AUDIT: block,
        uv.QUESTIONS: {"status": YELLOW, "summary": {GREEN: 5, YELLOW: 1, qs.GRAY: 2,
                                                      qs.NOT_APPLICABLE: 0}}}}
    assert ao.validation_headline(result) == (
        "Nada de lo que escribió el sistema rompe una regla. Para mirar: 1 pregunta de "
        "competencia que tu ontología no responde; 2 preguntas de competencia que no se "
        "pudieron responder.")


def test_the_questions_of_each_mode(env, tmp_path, monkeypatch):  # noqa: F811
    from src.validation import questions as qs
    ttl = tmp_path / "mine.ttl"
    ttl.write_text("@prefix dct: <http://purl.org/dc/terms/> .\n"
                   "<http://e.org/A> dct:source \"Ley 1, art. 3\"@es .\n", encoding="utf-8")
    monkeypatch.setattr(qs, "workspace_files", lambda ws, cfg=None: [ttl])
    for mode, g08 in (("ontology", False), ("law", True), ("ontology+law", False), (None, False)):
        folder = tmp_path / str(mode)
        ws = SimpleNamespace(id=f"w-{mode}", manifest={"mode": mode} if mode else {},
                             work_dir=folder / "work", path=folder)
        report = qs.inspect_workspace(ws, env.cfg)
        by_id = {r["id"]: r for r in report["results"]}
        assert report["mode"] == (mode or "ontology")
        assert by_id["G08"]["applies"] is g08
        assert all(by_id[f"G0{n}"]["applies"] for n in range(1, 8))
        assert by_id["G08"]["color"] == (GREEN if g08 else qs.GRAY)
        assert report["summary"][qs.NOT_APPLICABLE] == (0 if g08 else 1)
        assert (folder / "work" / qs.REPORT_FILE).exists()
        assert (folder / "work" / qs.LOG_FILE).read_text(encoding="utf-8").startswith(
            f"Preguntas de competencia del espacio w-{mode} ")


def test_the_questions_use_the_upload_and_not_the_copy_of_the_core(env, tmp_path):  # noqa: F811
    from src.validation import questions as qs
    out = tmp_path / "output"
    out.mkdir()
    (out / ua.MANIFEST_FILE).write_text(json.dumps({"files": {
        "ontology": "input/x.ttl", "alignment": "output/x-dpv-alignment.ttl",
        "core": "output/ontopriv-core.ttl"}}), encoding="utf-8")
    ws = SimpleNamespace(path=tmp_path, output_dir=out)
    files = qs.workspace_files(ws, env.cfg)
    assert files[:2] == [tmp_path / "input/x.ttl", tmp_path / "output/x-dpv-alignment.ttl"]
    assert len(files) == 3 and str(files[-1]) == env.cfg["inputs"]["dpv"]
    with pytest.raises(ValueError, match="falta output/alignment-manifest.json"):
        qs.workspace_files(SimpleNamespace(path=tmp_path, output_dir=tmp_path / "nada"), env.cfg)


def test_the_cli_answers_the_questions_of_a_workspace(env, targets, capsys,  # noqa: F811
                                                      monkeypatch):
    from src.validation import questions as qs
    w, _, _ = _align(_real_dpv(env), targets)
    def no_key(*args, **kwargs):                       # the CLI asks the AI: never in the tests
        raise KeyError("LLM_API_KEY")
    monkeypatch.setattr("src.ai.llm_client.LLMClient", no_key)
    assert qs._run([str(w.path), "--motor", "rdflib", "--sin-ia"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"Preguntas de competencia del espacio {w.id} (Mi ontología): ")
    assert f"  Guardado en: {w.work_dir / qs.REPORT_FILE}" in out
    assert "no usada; IA desactivada (--sin-ia); frase fija" in out
    assert "         Por qué: La consulta no trajo conceptos alineados con dpv:DataSubjectRight" in out
    assert qs._run([str(w.path), "--base"]) == 1
    assert capsys.readouterr().out == "Error: indica un espacio de trabajo o --base.\n"
