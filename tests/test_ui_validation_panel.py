from __future__ import annotations

import pytest

from app.services import align_ontology as svc
from src.validation import ai_auditor as au
from src.validation import upload_validation as uv
from src.validation.shape_inspector import AMBER, GREEN, RED
from src.workspace.store import load_workspace
from tests.test_alignment_propose import TTL, DeciderLLM, Embed, env, targets  # noqa: F401
from tests.test_ui_align_ontology import _boxes, cfg, page  # noqa: F401
from tests.test_validation_upload import FlaggingLLM, _real_dpv


def _align(env, targets, llm=None):  # noqa: F811
    return svc.align("Mi GDPR.ttl", TTL.encode("utf-8"), base=env.base, targets=targets,
                     embed_fn=Embed(), llm=llm or DeciderLLM(), chroma_path=env.chroma,
                     sleep_fn=lambda s: None)


@pytest.fixture
def amber(env, targets, monkeypatch):  # noqa: F811
    """An upload whose validation is amber: 3 concepts without definition, PersonalData
    marked by the AI reviewer."""
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    return _align(_real_dpv(env), targets, FlaggingLLM())


@pytest.fixture
def red(env, targets, monkeypatch):  # noqa: F811
    """The small DPV of the tests has no rdfs:isDefinedBy: its terms "are not in the DPV"."""
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    return _align(env, targets)


# ------------------------------------------------------------------ the texts

def _audit(**kw):
    summary = {"accepted": 4, au.COHERENT: 2, au.DOUBTFUL: 1, au.INCOHERENT: 1, au.INVALID: 0,
               "pending": 0, **kw.pop("summary", {})}
    return {"status": AMBER, "summary": summary, "ai": {"available": True},
            "prompt": au.PROMPT.manifest(), "flagged": [{}, {}], **kw}


def test_the_badge_of_each_status():
    assert [svc.badge(s) for s in (GREEN, AMBER, RED, uv.NOT_RUN, uv.FAILED)] == [
        ":green-badge[Verde]", ":orange-badge[Ámbar]", ":red-badge[Rojo]",
        ":gray-badge[Sin revisar]", ":gray-badge[No corrió]"]


def test_the_text_of_each_card():
    shapes = {"status": AMBER, "engine": "pySHACL 0.40.1", "new": 0, "inherited": 232,
              "warnings": 4}
    assert svc.card_text(uv.SHAPES, shapes) == ("Reglas rotas: 0 nuevas · 232 heredadas  \n"
                                                "Avisos: 4")
    assert svc.card_text(uv.LOGIC, {**shapes, "inherited": 255}).startswith(
        "Contradicciones: 0 nuevas · 255 heredadas")
    assert svc.card_caption(uv.SHAPES, shapes) == "pySHACL 0.40.1"
    assert svc.card_text(uv.LOGIC, {"status": uv.FAILED, "error": "Java no responde"}) == (
        "No pudo correr  \nJava no responde")
    assert svc.card_text(uv.AUDIT, _audit()) == ("Incoherentes: 1 · dudosas: 1  \n"
                                                 "De 4 aceptadas por la regla")
    assert svc.card_text(uv.AUDIT, _audit(summary={"pending": 2})).endswith(" · sin revisar: 2")
    assert svc.card_text(uv.AUDIT, {"status": uv.NOT_RUN}) == "No se hizo la revisión"
    assert svc.card_text(uv.AUDIT, _audit(summary={"accepted": 0})).startswith(
        "Nada que revisar")
    off = _audit(status=uv.NOT_RUN, ai={"available": False, "error": "IA desactivada (--sin-ia)"})
    assert svc.card_text(uv.AUDIT, off) == "No revisó las 4 aceptadas  \nIA desactivada (--sin-ia)"
    failed = _audit(status=uv.NOT_RUN, ai={"available": True, "errors": ["429 cuota"]})
    assert svc.card_text(uv.AUDIT, failed).endswith("\n429 cuota")
    assert svc.card_caption(uv.AUDIT, _audit()) == "prompt audit.v1 · solo explica"
    assert svc.card_caption(uv.AUDIT, {"status": uv.NOT_RUN}) == "solo explica"


def test_the_headline_says_what_to_look_at():
    shapes = {"status": AMBER, "new": 0, "inherited": 232, "warnings": 4}
    logic = {"status": GREEN, "new": 0, "inherited": 0, "warnings": 0}
    result = {"status": AMBER, "inspectors": {uv.SHAPES: shapes, uv.LOGIC: logic,
                                              uv.AUDIT: _audit()}}
    assert svc.validation_headline(result) == (
        "Nada de lo que escribió el sistema rompe una regla. Para mirar: 232 hallazgos "
        "heredados (ya estaban en tu ontología o en la base del sistema); 4 avisos "
        "(recomendaciones, como conceptos sin definición); 2 correspondencias que la IA "
        "revisora marcó para que las mires.")
    one = {**result, "inspectors": {uv.SHAPES: {**shapes, "inherited": 1, "warnings": 0},
                                    uv.LOGIC: {"status": uv.FAILED, "error": "x"},
                                    uv.AUDIT: {"status": uv.NOT_RUN}}}
    assert svc.validation_headline(one).endswith(
        "Para mirar: 1 hallazgo heredado (ya estaban en tu ontología o en la base del sistema); "
        "no pudo correr: Lógica (razonador).")
    green = {"status": GREEN, "inspectors": {uv.SHAPES: logic, uv.LOGIC: logic,
                                             uv.AUDIT: _audit(status=GREEN)}}
    assert svc.validation_headline(green) == ("Ninguna regla se rompe y el razonador no "
                                              "encontró contradicciones. La IA revisora no "
                                              "marcó nada.")
    unreviewed = {**green, "inspectors": {**green["inspectors"],
                                          uv.AUDIT: _audit(status=uv.NOT_RUN)}}
    assert svc.validation_headline(unreviewed).endswith(
        "La IA no revisó las correspondencias aceptadas.")
    nothing = {**green, "inspectors": {**green["inspectors"],
                                       uv.AUDIT: _audit(status=GREEN, summary={"accepted": 0})}}
    assert svc.validation_headline(nothing).endswith("no encontró contradicciones.")
    assert svc.validation_headline({**result, "status": RED}).startswith(
        "El sistema escribió algo que rompe una regla")


# ------------------------------------------------------------------ the panel of an upload

def test_the_panel_of_an_upload(amber):
    v = svc.view(amber)
    val = v["validation"]
    assert (val["status"], val["badge"]) == (AMBER, ":orange-badge[Ámbar]")
    assert [(c["title"], c["badge"]) for c in val["cards"]] == [
        ("Forma (SHACL)", ":orange-badge[Ámbar]"), ("Lógica (razonador)", ":green-badge[Verde]"),
        ("IA revisora", ":orange-badge[Ámbar]")]
    assert val["rules"] == [{"Regla": "Cada concepto tiene definición", "Tipo": "aviso",
                             "Nuevas": 0, "Heredadas": 3}]
    assert val["kinds"] == []
    assert val["flagged"] == [{"Concepto": "PersonalData", "Término del DPV": "PersonalData",
                               "Relación": "exactMatch", "Veredicto": au.INCOHERENT,
                               "Explicación": "Trata de otra cosa."}]
    assert list(val["flagged"][0]) == svc.FLAGGED_COLUMNS
    review = {r["Concepto"]: r["Revisión de la IA"] for r in v["rows"]}
    assert review["PersonalData"] == au.INCOHERENT and review["DataController"] == ""
    log = svc.validation_log(amber, svc.workspace_cfg()).decode("utf-8")
    assert log.startswith(f"Validación del espacio {amber}: ámbar")


def test_a_red_panel_still_lets_the_person_download(red):
    val = svc.view(red)["validation"]
    assert val["status"] == RED and val["cards"][0]["badge"] == ":red-badge[Rojo]"
    assert val["rules"][0] == {"Regla": "El término del DPV existe", "Tipo": "regla",
                               "Nuevas": 3, "Heredadas": 0}
    assert len(svc.download_options(red)) == 4


def test_the_incoherent_ones_come_first():
    def f(name, verdict):
        return {"name": name, "dpv_name": "X", "relation": None, "verdict": verdict,
                "explanation": None}
    result = {"inspectors": {uv.AUDIT: {"flagged": [f("b", au.DOUBTFUL), f("C", au.INCOHERENT),
                                                    f("a", au.INCOHERENT)]}}}
    rows = svc.flagged_rows(result)
    assert [(r["Concepto"], r["Veredicto"]) for r in rows] == [
        ("a", au.INCOHERENT), ("C", au.INCOHERENT), ("b", au.DOUBTFUL)]
    assert rows[0]["Relación"] == "" and rows[0]["Explicación"] == ""
    assert svc.flagged_rows(None) == []


def test_an_upload_from_before_the_validation(amber):
    ws = load_workspace(amber, cfg=svc.workspace_cfg())
    (ws.output_dir / uv.RESULT_FILE).unlink()
    assert svc.view(amber)["validation"] is None


def test_the_detail_counts_what_comes_from_the_base(monkeypatch):
    from src.validation import logic_inspector as li, shape_inspector as si
    rule = {"name": "Ningún texto rdf:langString sin idioma", "severity": "violación",
            "nuevo": 0, "ontología": 2, "base": 1}
    kind = {"label": "Un individuo está en dos clases disjuntas", "errors": 3, "warnings": 0,
            "nuevo": 1, "ontología": 0, "base": 2}
    monkeypatch.setattr(si, "load_result", lambda ws: {"summary": {"rules": {"R": rule}}})
    monkeypatch.setattr(li, "load_result", lambda ws: {"summary": {"kinds": {"K": kind}}})
    ok = {"status": AMBER}
    assert svc._rule_rows(None, ok) == [{"Regla": rule["name"], "Tipo": "regla", "Nuevas": 0,
                                         "Heredadas": 3}]
    assert svc._kind_rows(None, ok) == [{"Tipo": kind["label"], "Contradicciones": 3,
                                         "Avisos": 0, "Nuevas": 1, "Heredadas": 2}]


def test_a_failed_inspector_has_no_stale_detail(amber):
    ws = load_workspace(amber, cfg=svc.workspace_cfg())
    block = {"status": uv.FAILED, "error": "x"}
    assert svc._rule_rows(ws, block) == [] and svc._kind_rows(ws, block) == []


# ------------------------------------------------------------------ the page (AppTest)

def test_the_page_shows_the_panel_and_what_the_ai_marked(page, amber):
    at = page()
    at.session_state["align_ws"] = amber
    at.run()
    assert not at.exception
    text = [m.value for m in at.markdown]
    assert "**Validación**" in text
    assert any(t.startswith(":orange-badge[Ámbar] Nada de lo que escribió el sistema")
               for t in text)
    assert "**Forma (SHACL)** :orange-badge[Ámbar]" in text
    assert "**Lógica (razonador)** :green-badge[Verde]" in text
    assert "Contradicciones: 0 nuevas · 0 heredadas  \nAvisos: 0" in text
    assert {"¿Qué revisa cada uno?", "Ver el detalle"} <= set(_boxes(at))
    assert any(t.startswith("**Colores.** rojo:") for t in text)
    log = [d for d in at.download_button if d.key == "align_dl_log"]
    assert len(log) == 1 and log[0].label == "Descargar el detalle (validation-log.txt)"
    assert log[0].proto.deferred_file_id and log[0].proto.ignore_rerun   # built when clicked
    assert "**Marcadas por la IA revisora (1)**" in text                 # "Para revisar"
    frames = [list(d.value.columns) for d in at.dataframe]
    assert svc.FLAGGED_COLUMNS in frames and ["Regla", "Tipo", "Nuevas", "Heredadas"] in frames
    assert [t.label for t in at.tabs] == ["Resumen", "Conceptos", "Para revisar"]
    assert not at.warning and not at.error                  # the panel uses no alerts


def test_the_page_of_a_red_upload(page, red):
    at = page()
    at.session_state["align_ws"] = red
    at.run()
    assert not at.exception
    text = [m.value for m in at.markdown]
    assert any(t.startswith(":red-badge[Rojo] El sistema escribió algo que rompe una regla")
               for t in text)
    assert len([d for d in at.download_button if d.key != "align_dl_log"]) == 4


def test_the_page_of_an_upload_from_before_the_validation(page, amber):
    ws = load_workspace(amber, cfg=svc.workspace_cfg())
    (ws.output_dir / uv.RESULT_FILE).unlink()
    at = page()
    at.session_state["align_ws"] = amber
    at.run()
    assert not at.exception
    assert svc.NO_VALIDATION in [c.value for c in at.caption]
    assert not [d for d in at.download_button if d.key == "align_dl_log"]
