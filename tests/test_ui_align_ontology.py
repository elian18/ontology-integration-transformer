"""S6-T08 + S7-T06: the "Alinear mi ontología" page (service + the page with AppTest).

AppTest cannot drive a file uploader, so the upload itself is tested through the service
(``align``) and the page is run with the workspace id already in the session, exactly what
the page stores after an upload."""
from __future__ import annotations

import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from app.services import align_ontology as svc
from src.workspace.store import WorkspaceError
from tests.test_alignment_propose import TTL, DeciderLLM, Embed, env, targets  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(env):
    return env.cfg


@pytest.fixture
def aligned(env, targets, monkeypatch):
    """An upload already aligned (what the page has after the user drops a file)."""
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    seen = []
    ws_id = svc.align("Mi GDPR.ttl", TTL.encode("utf-8"), progress=seen.append, base=env.base,
                      targets=targets, embed_fn=Embed(), llm=DeciderLLM(),
                      chroma_path=env.chroma, sleep_fn=lambda s: None)
    return ws_id, seen


# ------------------------------------------------------------------ service

def test_align_runs_every_step_and_names_them(aligned):
    ws_id, seen = aligned
    assert ws_id.startswith("mi-gdpr-")
    assert seen == ["Heredar lo validado", "Buscar candidatos del DPV",
                    "Escribir la ontología alineada", "Preparar la descarga"]


def test_view_has_the_numbers_and_the_table(aligned):
    v = svc.view(aligned[0])
    assert v["file"] == "Mi GDPR.ttl" and v["concepts"] == 4 and v["version"] == 1
    assert v["metrics"] == {"Validadas en la base": 1, "Aceptadas automáticamente": 1,
                            "Sugeridas": 1, "Sin correspondencia": 1, "Sin propuesta": 0}
    assert (v["mappings"], v["validated_mappings"], v["accepted_mappings"], v["suggested"]) == (
        3, 2, 1, 1)
    assert v["rule"].startswith("confianza alta de la IA") and v["ai_note"] is None
    states = [r["Estado"] for r in v["rows"]]
    assert states == ["Validada en la base", "Validada en la base", "Aceptada automáticamente",
                      "Sugerida", "Sin correspondencia"]
    consent = [r for r in v["rows"] if r["Concepto"] == "Consent"]
    assert len(consent) == 2 and consent[0]["Decidido por"] == "base · mismo nombre · Consent"
    accepted = svc.filter_rows(v["rows"], "Aceptada automáticamente")[0]
    assert (accepted["Concepto"], accepted["Decidido por"], accepted["Confianza de la IA"]) == (
        "PersonalData", "regla", "alta")
    assert accepted["Justificación"] == "Decisión de prueba para PersonalData."
    assert svc.filter_rows(v["rows"], "Sugerida")[0]["Relación"] == "skos:relatedMatch"
    assert svc.filter_rows(v["rows"], "Todos") == v["rows"]


def test_download_in_each_format_without_recomputing(aligned):
    ws_id, _ = aligned
    for key, label in svc.FORMATS:
        pkg = svc.download(ws_id, key)
        assert pkg["label"] == label and pkg["file_name"] == f"mi-gdpr-alineada-{key}.zip"
        names = svc.zip_names(pkg["data"])
        assert pkg["main_file"] in names and "alignment.csv" in names
        assert (pkg["note"] is not None) == (key == "jsonld")


def test_friendly_errors(cfg, monkeypatch):
    monkeypatch.setattr(svc, "workspace_cfg", lambda: cfg)
    with pytest.raises(WorkspaceError) as exc:
        svc.align("notas.md", b"hola")
    assert svc.error_message(exc.value).startswith("Formato de ontología no soportado")
    assert "base construida" in svc.error_message(FileNotFoundError("x.json"))
    assert svc.error_message(RuntimeError("boom")) == \
        "No se pudo alinear la ontología: RuntimeError: boom"
    with pytest.raises(WorkspaceError, match="No existe"):
        svc.view("no-existe-00000000")


def test_ai_note_only_when_concepts_were_left_without_the_ai():
    assert svc._ai_note({"pending_ai": 0, "ai": {"available": False}}) is None
    off = svc._ai_note({"pending_ai": 3, "ai": {"available": False}, "threshold_without_ai": 0.75})
    assert "no estuvo disponible: a 3 conceptos" in off and "puntaje >= 0.75" in off
    quota = svc._ai_note({"pending_ai": 2, "ai": {"available": True, "quota_exhausted": True}})
    assert "cuota diaria" in quota and "lo ya decidido no se repite" in quota
    limit = svc._ai_note({"pending_ai": 2, "ai": {"available": True, "call_limit_reached": True}})
    assert "límite de llamadas" in limit


def test_without_ai_the_page_says_what_happened(env, targets, monkeypatch):
    offline = {**env.cfg, "auto": {**env.cfg["auto"], "use_llm": False}}
    monkeypatch.setattr(svc, "workspace_cfg", lambda: offline)
    v = svc.view(svc.align("Mi GDPR.ttl", TTL.encode("utf-8"), base=env.base, targets=targets,
                           embed_fn=Embed(), chroma_path=env.chroma))
    assert v["metrics"]["Aceptadas automáticamente"] == 2 and v["metrics"]["Sin propuesta"] == 1
    assert "La IA no estuvo disponible: a 3 conceptos" in v["ai_note"]
    assert {r["Decidido por"] for r in v["rows"] if r["Estado"] == "Aceptada automáticamente"} \
        == {"regla sin IA"}
    assert "Sin decidir" in {r["Estado"] for r in v["rows"]}


def test_a_workspace_of_v060_asks_to_upload_again(aligned):
    from src.alignment import user_alignment as ua
    from src.workspace.store import load_workspace
    ws = load_workspace(aligned[0], cfg=svc.workspace_cfg())
    path = ws.output_dir / ua.MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("decider")                                     # what v0.6.0 wrote
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(WorkspaceError, match="vuelve a subirlo"):
        svc.view(aligned[0])


def test_page_is_first_in_the_navigation():
    text = (ROOT / "app/app.py").read_text(encoding="utf-8")
    assert '"Tu ontología": [' in text and '"Construcción de la base": [' in text
    assert text.index("views/align_ontology.py") < text.index("views/inputs.py")


# ------------------------------------------------------------------ the page (AppTest)

@pytest.fixture
def page(cfg, monkeypatch):  # noqa: F811
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    app_dir = str(ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    view_svc = importlib.import_module("services.align_ontology")   # what the page imports
    monkeypatch.setattr(view_svc, "workspace_cfg", lambda: cfg)
    return lambda: AppTest.from_file(str(ROOT / "app/views/align_ontology.py"),
                                     default_timeout=30)


def test_page_without_upload_explains_what_to_do(page):
    at = page()
    at.run()
    assert not at.exception
    assert any("Sube un archivo para empezar" in i.value for i in at.info)
    assert not at.download_button


def test_page_shows_the_result_and_the_download(page, aligned):
    at = page()
    at.session_state["align_ws"] = aligned[0]
    at.run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Validadas en la base", "Aceptadas automáticamente",
                                            "Sugeridas", "Sin correspondencia", "Sin propuesta"]
    assert [m.value for m in at.metric] == ["1", "1", "1", "1", "0"]
    assert all(m.help for m in at.metric)                         # every number explained
    assert any("3 correspondencias en la ontología" in c.value for c in at.caption)
    assert any(c.value.startswith("Regla: confianza alta") for c in at.caption)
    assert at.radio(key="align_fmt").value == "Turtle"
    assert at.download_button[0].label == "Descargar ontología alineada (Turtle, .zip)"
    assert any("mi-gdpr-dpv-alignment.ttl" in c.value for c in at.caption)
    at.radio(key="align_fmt").set_value("JSON-LD").run()
    assert at.download_button[0].label == "Descargar ontología alineada (JSON-LD, .zip)"
    assert any("Protégé 5.6 no abre JSON-LD" in i.value for i in at.info)
    at.selectbox(key="align_filter").set_value("Validada en la base").run()
    assert len(at.dataframe[0].value) == 2


def test_page_reports_a_broken_workspace(page):
    at = page()
    at.session_state["align_ws"] = "no-existe-00000000"
    at.run()
    assert not at.exception
    assert any("No existe el espacio de trabajo" in e.value for e in at.error)


def test_upload_limit_matches_the_workspace_limit():
    import tomllib
    from src.workspace.store import workspace_settings
    toml = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))
    assert toml["server"]["maxUploadSize"] == workspace_settings()["max_upload_mb"]