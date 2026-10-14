"""S6-T08: the "Alinear mi ontología" page (service + the page with AppTest).

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
from src.alignment.inherit import load_base
from src.workspace.store import WorkspaceError
from tests.test_alignment_propose import CFG, TTL, Embed, FakeLLM, targets  # noqa: F401
from tests.test_alignment_write import _decide_all

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    return {**CFG, "workspaces": {"dir": str(tmp_path / "ws")}}


@pytest.fixture
def base(tmp_path):
    data, log = _decide_all(tmp_path)
    cand = tmp_path / "base.json"
    cand.write_text(json.dumps(data), encoding="utf-8")
    return load_base(cand, log)


@pytest.fixture
def aligned(cfg, base, targets, monkeypatch):
    """An upload already aligned (what the page has after the user drops a file)."""
    monkeypatch.setattr(svc, "workspace_cfg", lambda: cfg)
    seen = []
    ws_id = svc.align("Mi GDPR.ttl", TTL.encode("utf-8"), progress=seen.append, base=base,
                      targets=targets, embed_fn=Embed(), llm=FakeLLM())
    return ws_id, seen


# ------------------------------------------------------------------ service

def test_align_runs_every_step_and_names_them(aligned):
    ws_id, seen = aligned
    assert ws_id.startswith("mi-gdpr-")
    assert seen == ["Heredar lo validado", "Buscar candidatos del DPV",
                    "Escribir la ontología alineada", "Preparar la descarga"]


def test_view_has_the_numbers_and_the_table(aligned):
    v = svc.view(aligned[0])
    assert v["file"] == "Mi GDPR.ttl" and v["concepts"] == 4
    assert v["metrics"]["Validadas"] == 1 and v["metrics"]["Propuestas automáticas"] >= 1
    assert v["validated_mappings"] == 2 and v["threshold"] == 0.6
    states = [r["Estado"] for r in v["rows"]]
    assert states[0] == "Validada" and "Propuesta automática" in states
    consent = [r for r in v["rows"] if r["Concepto"] == "Consent"]
    assert len(consent) == 2 and consent[0]["Origen"] == "mismo nombre · Consent"
    assert svc.filter_rows(v["rows"], "Validada") == consent
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


def test_ai_note_only_when_it_matters():
    assert svc._ai_note({"available": False}, 0) is None
    assert "sin tipo SKOS" in svc._ai_note({"available": False}, 3)
    assert "cuota diaria" in svc._ai_note({"available": True, "quota_exhausted": True}, 3)
    assert svc._ai_note({"available": True}, 3) is None


def test_page_is_first_in_the_navigation():
    text = (ROOT / "app/app.py").read_text(encoding="utf-8")
    assert '"Tu ontología": [' in text and '"Construcción de la base": [' in text
    assert text.index("views/align_ontology.py") < text.index("views/inputs.py")


# ------------------------------------------------------------------ the page (AppTest)

@pytest.fixture
def page(cfg, monkeypatch):
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
    assert [m.label for m in at.metric] == ["Validadas", "Propuestas automáticas",
                                            "Sin correspondencia", "Sin propuesta"]
    assert at.metric[0].value == "1"
    assert all(m.help for m in at.metric)                         # every number explained
    assert at.radio(key="align_fmt").value == "Turtle"
    assert at.download_button[0].label == "Descargar ontología alineada (Turtle, .zip)"
    assert any("mi-gdpr-dpv-alignment.ttl" in c.value for c in at.caption)
    at.radio(key="align_fmt").set_value("JSON-LD").run()
    assert at.download_button[0].label == "Descargar ontología alineada (JSON-LD, .zip)"
    assert any("Protégé 5.6 no abre JSON-LD" in i.value for i in at.info)
    at.selectbox(key="align_filter").set_value("Validada").run()
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