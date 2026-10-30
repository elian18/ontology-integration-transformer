from __future__ import annotations

import csv
import json
import sys
import types
import zipfile

from src.alignment import inherit as inh
from src.alignment import propose as prop
from src.alignment import user_alignment as ua
from src.validation import ai_auditor as au
from src.validation import logic_inspector as li
from src.validation import upload_validation as uv
from src.validation.shape_inspector import AMBER, GREEN, RED
from src.validation.upload_validation import FAILED, NOT_RUN, audit_status, overall
from src.workspace.store import DONE, ERROR
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


# ------------------------------------------------------------------ the colors

def test_the_worst_color_wins_and_a_failure_is_amber():
    assert overall([GREEN, GREEN, NOT_RUN]) == GREEN
    assert overall([GREEN, AMBER, GREEN]) == AMBER
    assert overall([GREEN, FAILED, NOT_RUN]) == AMBER
    assert overall([AMBER, RED, FAILED]) == RED


def test_the_color_of_the_ai_review():
    def s(**kw):
        return {"summary": {"accepted": 3, au.COHERENT: 0, au.DOUBTFUL: 0, au.INCOHERENT: 0,
                            au.INVALID: 0, "pending": 0, **kw}}
    assert audit_status(None) == NOT_RUN
    assert audit_status(s(**{au.COHERENT: 3})) == GREEN
    assert audit_status(s(**{au.COHERENT: 2, au.INCOHERENT: 1})) == AMBER
    assert audit_status(s(**{au.COHERENT: 2, au.DOUBTFUL: 1})) == AMBER
    assert audit_status(s(pending=3)) == NOT_RUN                  # the AI reviewed nothing
    assert audit_status(s(**{au.COHERENT: 2}, pending=1)) == GREEN
    assert audit_status(s(accepted=0)) == GREEN                   # nothing to review


# ------------------------------------------------------------------ each upload

def test_each_upload_is_validated_and_the_zip_says_so(env, targets):  # noqa: F811
    w, m, path = _align(_real_dpv(env), targets)
    v = uv.load_result(w)
    shapes, logic, audit = (v["inspectors"][k] for k in (uv.SHAPES, uv.LOGIC, uv.AUDIT))
    assert (shapes["status"], shapes["new"], shapes["inherited"], shapes["warnings"]) == (
        AMBER, 0, 0, 3)                                           # 3 concepts without definition
    assert (logic["status"], logic["consistent"], logic["new"]) == (GREEN, True, 0)
    assert logic["engine"] == f"owlrl {li.version('owlrl')}, OWL 2 RL en Python"
    assert audit["status"] == GREEN and audit["summary"]["accepted"] == 1
    assert v["status"] == AMBER
    assert w.step("shapes")["status"] == DONE and w.step("shapes")["result"] == AMBER
    assert w.step("logic")["detail"] == ("verde: 0 inconsistencias nuevas, 0 heredadas, "
                                         "0 advertencias")
    assert w.step("audit")["detail"] == "Coherentes 1 · dudosas 0 · incoherentes 0"
    names = zipfile.ZipFile(path).namelist()
    assert uv.RESULT_FILE in names and uv.LOG_FILE in names
    readme = zipfile.ZipFile(path).read("LEEME.txt").decode("utf-8")
    assert "\nValidación: ámbar (no bloquea la descarga)\n" in readme
    assert (f"  - Forma (pySHACL {li.version('pyshacl')}): ámbar · 0 violaciones nuevas, "
            "0 heredadas, 3 advertencias\n") in readme
    assert ("  - IA revisora (audit.v1): verde · 0 incoherentes y 0 dudosas de 1 aceptada por "
            "la regla; solo explica, no cambia lo aceptado\n") in readme
    log = (w.output_dir / uv.LOG_FILE).read_text(encoding="utf-8")
    assert log.startswith(f"Validación del espacio {w.id}: ámbar\n\nInspector de forma (SHACL)")
    assert "Inspector de lógica (razonador)" in log and "Revisión de la IA del espacio" in log
    text = ua.render_console(w, m, path)
    assert "  IA de la revisión: 1 llamadas, 0 desde la caché, 0 inválidas\n" in text
    assert "\n  Validación: ámbar (no bloquea la descarga)\n    Forma (pySHACL" in text
    assert "\n    Detalle: output/validation-log.txt\n" in text
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
    assert audit["status"] == AMBER and audit["flagged"] == [{
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
    (w.work_dir / au.CACHE_FILE).unlink()
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
    assert v["status"] == AMBER and m["ai_audit"] is None
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
    flagged = {**block, "status": AMBER, "summary": {**block["summary"], au.COHERENT: 1,
                                                     au.INCOHERENT: 1, au.DOUBTFUL: 1,
                                                     "pending": 1}}
    assert uv.inspector_text(uv.AUDIT, flagged) == (
        "IA revisora (audit.v1): ámbar · 1 incoherente y 1 dudosa de 4 aceptadas por la regla, "
        "1 sin revisar; solo explica, no cambia lo aceptado")
    none = {**block, "status": GREEN, "summary": {**block["summary"], "accepted": 0}}
    assert uv.inspector_text(uv.AUDIT, none) == (
        "IA revisora (audit.v1): verde · la regla no aceptó ninguna correspondencia que revisar")
    logic = {"status": AMBER, "engine": "Apache Jena 6.2.0, OWL Micro", "new": 0,
             "inherited": 255, "warnings": 1}
    assert uv.inspector_text(uv.LOGIC, logic) == (
        "Lógica (Apache Jena 6.2.0, OWL Micro): ámbar · 0 inconsistencias nuevas, "
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
    assert "  Validación: ámbar (no bloquea la descarga)\n" in out
    assert f"  IA de la revisión: {ua.REUSED} (1 llamadas, 0 desde la caché" in out
    assert ua._run([]) == 1 and ua._run(["x.ttl", "--espacio", w.id]) == 1
    assert "Error: indica un archivo o --espacio." in capsys.readouterr().out
    assert ua._run(["--espacio", "no-existe"]) == 1
    assert "Error: No existe el espacio de trabajo no-existe." in capsys.readouterr().out
    for name in (au.RESULT_FILE, au.CACHE_FILE):
        (w.work_dir / name).unlink()
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
    logic_report = {"status": AMBER, "consistent": False, "seconds": 2.1,
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