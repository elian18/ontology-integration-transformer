"""S5-T07: the "Grafo alineado" tab writes the graph only when nothing is pending and offers the
files and the Protégé zip for download (service logic + the page with AppTest)."""
import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from app.services import alignment_graph as ag
from app.services import mapping_review as mr
from src.alignment.decisions import read_log
from src.alignment.write import ALIGNMENT_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV
from src.alignment.bundle import BUNDLE_FILE, CATALOG_FILE
from tests.test_alignment_write import _decide_all, ONTO, DPV

ROOT = Path(__file__).resolve().parents[1]
CFG = {"review": {"source_id": "ontopriv+lopdp", "reviewer": "Elian"}}


def _files(tmp_path, *, leave_pending=False):
    data, log = _decide_all(tmp_path, leave_pending=leave_pending)
    out = tmp_path / "output"
    out.mkdir()
    cand = out / "alignment-candidates.json"
    cand.write_text(json.dumps(data), encoding="utf-8")
    return {"candidates": cand, "log": log, "dpv": tmp_path / "dpv.ttl",
            "proposals": tmp_path / "none.json"}


def _state(paths):
    return mr.load(candidates_path=paths["candidates"], log_path=paths["log"], cfg=CFG)


# ---------------------------------------------------------------- service
def test_readiness_follows_the_review(tmp_path):
    pending = _files(tmp_path / "a", leave_pending=True)
    r = ag.readiness(_state(pending))
    assert not r["ready"] and "Faltan 1 de 3" in r["message"]
    done = _files(tmp_path / "b")
    r = ag.readiness(_state(done))
    assert r["ready"] and "Los 3 conceptos están revisados" in r["message"]


def test_write_refuses_while_pending(tmp_path):
    paths = _files(tmp_path, leave_pending=True)
    ok, msg = ag.write_graph(paths, CFG)
    assert not ok and "No se escribió el grafo" in msg
    assert ag.current_graph(paths) is None
    assert not (ag.output_dir(paths) / ALIGNMENT_FILE).exists()


def test_write_and_offer_the_files(tmp_path):
    paths = _files(tmp_path)
    ok, msg = ag.write_graph(paths, CFG)
    assert ok and "3 correspondencias SKOS" in msg and "1 conceptos nuevos" in msg
    graph = ag.current_graph(paths)
    assert set(graph["files"]) == {ALIGNMENT_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV,
                                   BUNDLE_FILE}
    names = zipfile.ZipFile(io.BytesIO(graph["files"][BUNDLE_FILE])).namelist()
    assert ALIGNMENT_FILE in names and CATALOG_FILE in names
    assert graph["bundle_missing"]                     # no core/profile in this temp folder
    assert graph["manifest"]["mappings"]["total"] == 3
    rows = ag.summary_rows(graph["manifest"])
    assert rows[0] == {"Tipo SKOS": "skos:exactMatch", "Significado": "equivalente",
                       "Correspondencias": 2}
    assert not ag.stale(_state(paths), graph["manifest"])


def test_stale_when_the_review_changes_after_writing(tmp_path):
    paths = _files(tmp_path)
    ag.write_graph(paths, CFG)
    manifest = ag.current_graph(paths)["manifest"]
    state = _state(paths)
    ok, _ = mr.apply_undo(state, ONTO + "Consent", "Elian", DPV + "ConsentRecord")
    assert ok
    assert ag.stale(_state(paths), manifest)


# ---------------------------------------------------------------- the page (AppTest)
@pytest.fixture
def page(tmp_path, monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    app_dir = str(ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    view_mr = importlib.import_module("services.mapping_review")
    view_al = importlib.import_module("services.alignment")

    def make(leave_pending=False):
        paths = _files(tmp_path, leave_pending=leave_pending)
        monkeypatch.setattr(view_mr, "default_paths", lambda: paths)
        monkeypatch.setattr(view_al, "_output_dir", lambda: ag.output_dir(paths))
        return AppTest.from_file(str(ROOT / "app/views/alignment.py"), default_timeout=30), paths
    return make


def test_tab_disables_the_button_while_pending(page):
    at, paths = page(leave_pending=True)
    at.run()
    assert not at.exception
    assert at.tabs[2].label == "Grafo alineado"
    assert at.button(key="ag_write").disabled
    assert any("Faltan 1 de 3" in i.value for i in at.info)
    assert not at.download_button


def test_write_from_the_page_and_download(page):
    at, paths = page()
    at.run()
    assert not at.button(key="ag_write").disabled
    at.button(key="ag_write").click().run()
    assert not at.exception
    assert any("Grafo escrito" in s.value for s in at.success)
    assert (ag.output_dir(paths) / ALIGNMENT_FILE).exists()
    labels = [b.label for b in at.download_button]
    assert labels == ["Descargar paquete para Protégé (.zip)", "Grafo (RDF/XML)",
                      "Manifiesto (JSON)", "Correspondencias (CSV)"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Correspondencias SKOS"] == "3" and metrics["Conceptos nuevos"] == "1"
    assert [d.target for d in read_log(paths["log"])].count("mapping") == 4   # log untouched