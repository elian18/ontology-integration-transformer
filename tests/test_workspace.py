from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from src.workspace import store
from src.workspace.store import (BASE_DIRS, DEFINITION_STEPS, DONE, ERROR, EVIDENCE_STEPS,
                                 EXTRACT_STEPS, KIND_LAW, KIND_ONTOLOGY, LAW_STEPS, MANIFEST_FILE,
                                 PENDING, PROFILE_STEPS, STEPS, WorkspaceError, check_root,
                                 list_workspaces, load_workspace, open_law, open_ontology,
                                 open_upload, safe_file_name, slugify, steps_for, workspace_id,
                                 workspace_settings)
from tests.test_transform_convert import FIXTURE_TTL

ROOT = Path(__file__).resolve().parents[1]
TTL = FIXTURE_TTL.encode("utf-8")


@pytest.fixture
def cfg(tmp_path):
    return {"workspaces": {"dir": str(tmp_path / "ws"), "max_upload_mb": 1}}


# ------------------------------------------------------------------ names and ids

def test_slugify_makes_readable_ascii_names():
    assert slugify("Ontología GDPR v2") == "ontologia-gdpr-v2"
    assert slugify("  __ ") == "carga"
    assert len(slugify("x" * 100)) == 40


def test_safe_file_name_drops_folders_and_keeps_extension():
    assert safe_file_name("..\\..\\Mi Ontología.TTL") == "mi-ontologia.ttl"
    assert safe_file_name("C:/datos/gdpr.rdf") == "gdpr.rdf"


def test_same_upload_same_id_other_content_other_id():
    a = workspace_id({KIND_ONTOLOGY: ("GDPR.ttl", TTL)})
    assert a == workspace_id({KIND_ONTOLOGY: ("GDPR.ttl", TTL)})
    assert a.startswith("gdpr-") and len(a.split("-")[-1]) == 8
    assert a != workspace_id({KIND_ONTOLOGY: ("GDPR.ttl", TTL + b"\n")})
    assert workspace_id({KIND_ONTOLOGY: ("o.ttl", TTL), KIND_LAW: ("l.txt", b"x")}) != \
        workspace_id({KIND_ONTOLOGY: ("o.ttl", TTL), KIND_LAW: ("l.txt", b"y")})


# ------------------------------------------------------------------ create and reopen

def test_open_creates_the_layout_and_manifest(cfg):
    ws = open_ontology("Mi GDPR.ttl", TTL, cfg=cfg)
    assert not ws.reused
    assert ws.path.parent == Path(cfg["workspaces"]["dir"]).resolve()
    for sub in ("input", "work", "output"):
        assert (ws.path / sub).is_dir()
    assert ws.input_path().read_bytes() == TTL
    m = json.loads((ws.path / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert m["id"] == ws.id and m["source_id"] == f"upload:{ws.id}"
    assert m["mode"] == "ontology"
    entry = m["inputs"]["ontology"]
    assert entry["original_name"] == "Mi GDPR.ttl" and entry["stored_as"] == "mi-gdpr.ttl"
    assert entry["format"] == "Turtle" and entry["sha256"] == hashlib.sha256(TTL).hexdigest()
    assert entry["characterization"]["structure"]["triples"] > 0
    assert [r["status"] for r in ws.progress()] == [PENDING] * len(STEPS)


def test_same_file_reopens_the_same_workspace(cfg):
    first = open_ontology("gdpr.ttl", TTL, cfg=cfg)
    first.mark_step("inherit", DONE, "527 heredadas")
    again = open_ontology("gdpr.ttl", TTL, cfg=cfg)
    assert again.reused and again.path == first.path
    assert again.manifest["opened"] == 2
    assert again.step("inherit")["status"] == DONE          # earlier work is kept
    other = open_ontology("gdpr.ttl", TTL + b"\n", cfg=cfg)
    assert other.path != first.path


def test_law_and_both_modes_are_accepted(cfg):
    law = open_upload({KIND_LAW: ("Ley Perú.pdf", b"%PDF-1.4 ...")}, cfg=cfg)
    assert law.manifest["mode"] == "law" and law.input_path(KIND_LAW).name == "ley-peru.pdf"
    both = open_upload({KIND_ONTOLOGY: ("o.ttl", TTL), KIND_LAW: ("l.txt", b"Art. 1")},
                       cfg=cfg)
    assert both.manifest["mode"] == "ontology+law" and both.id.startswith("o-")


def test_the_steps_depend_on_what_the_workspace_holds(cfg):
    assert steps_for("ontology") == STEPS and steps_for(None) == STEPS
    assert STEPS == ("inherit", "candidates", "propose", "labels", "write", "package")  # S7-T08
    assert LAW_STEPS == ("segment", "index") and EXTRACT_STEPS == ("extract", "accept")
    assert PROFILE_STEPS == ("profile",)
    # solo ley: the law (S7-T07), its concepts (S7-T09), the profile and its alignment (S7-T10)
    assert steps_for("law") == LAW_STEPS + EXTRACT_STEPS + PROFILE_STEPS + STEPS
    assert steps_for("ontology+law") == (LAW_STEPS + STEPS[:1] + DEFINITION_STEPS + STEPS[1:4]
                                         + EVIDENCE_STEPS + STEPS[4:])
    assert DEFINITION_STEPS == ("definitions",) and EVIDENCE_STEPS == ("evidence",)
    assert STEPS[4:] == ("write", "package")
    law = open_law("Ley.txt", b"Art. 1.- Objeto.", cfg=cfg)
    assert law.manifest["mode"] == "law"
    assert list(law.manifest["steps"]) == list(steps_for("law"))
    assert [r["label"] for r in law.progress()][:5] == [
        "Separar la ley en artículos", "Indexar los artículos de la ley",
        "Extraer los conceptos de la ley con la IA", "Aceptar los conceptos por regla",
        "Armar el perfil de la ley sobre OntoPriv-Core"]
    law.mark_step("accept", DONE, "1 aceptados")
    law.mark_step("profile", DONE, "1 nuevos")
    law.mark_step("evidence", DONE, "1 enlaces")                      # a known step
    law.mark_step("segment", DONE, "1 artículos")
    again = load_workspace(law.id, cfg=cfg)
    assert again.step("segment")["status"] == again.step("accept")["status"] == DONE
    assert again.step("profile")["status"] == DONE


# ------------------------------------------------------------------ friendly errors

@pytest.mark.parametrize("name,data,message", [
    ("vacia.ttl", b"", "está vacío"),
    ("notas.md", b"hola", "Sube la ontología en RDF/XML"),
    ("ontopriv.owx", b"<x/>", "Protégé"),
    ("rota.ttl", b"esto no es turtle {{{", "No se pudo leer la ontología"),
])
def test_bad_uploads_are_refused_without_leaving_folders(cfg, name, data, message):
    with pytest.raises(WorkspaceError, match=message):
        open_ontology(name, data, cfg=cfg)
    root = Path(cfg["workspaces"]["dir"])
    assert not root.exists() or not any(root.iterdir())


def test_too_large_and_wrong_law_format(cfg):
    with pytest.raises(WorkspaceError, match="el máximo es 1 MB"):
        open_ontology("grande.ttl", b"#" * (2 * 1024 * 1024), cfg=cfg)
    with pytest.raises(WorkspaceError, match=r"\.txt o \.pdf"):
        open_upload({KIND_LAW: ("ley.docx", b"x")}, cfg=cfg)
    with pytest.raises(WorkspaceError, match="ningún archivo"):
        open_upload({}, cfg=cfg)


# ------------------------------------------------------------------ steps and listing

def test_mark_step_persists_and_validates(cfg):
    ws = open_ontology("a.ttl", TTL, cfg=cfg)
    ws.mark_step("candidates", ERROR, "sin red", rows=0)
    reloaded = load_workspace(ws.id, cfg=cfg)
    step = reloaded.step("candidates")
    assert step["status"] == ERROR and step["detail"] == "sin red" and step["rows"] == 0
    with pytest.raises(WorkspaceError, match="Paso desconocido"):
        ws.mark_step("volar", DONE)
    with pytest.raises(WorkspaceError, match="Estado desconocido"):
        ws.mark_step("inherit", "casi")
    with pytest.raises(WorkspaceError, match="No existe"):
        load_workspace("no-existe-00000000", cfg=cfg)


def test_list_workspaces_most_recent_first(cfg):
    a = open_ontology("a.ttl", TTL, cfg=cfg)
    b = open_ontology("b.ttl", TTL + b"\n", cfg=cfg)
    time.sleep(0.01)
    a.mark_step("inherit", DONE)                      # a becomes the most recent
    rows = list_workspaces(cfg=cfg)
    assert [r["id"] for r in rows] == [a.id, b.id]
    assert rows[0]["files"] == ["a.ttl"] and rows[0]["mode"] == "ontology"


# ------------------------------------------------------------------ the base is never touched

def test_root_cannot_be_inside_the_base():
    for base in ("data/output", "data/review/sub"):
        with pytest.raises(WorkspaceError, match="base del sistema"):
            check_root(ROOT / base)
    assert check_root(ROOT / "data/workspaces") == (ROOT / "data/workspaces").resolve()


def test_default_root_is_data_workspaces_and_ignored_by_git():
    settings = workspace_settings()
    assert settings["root"] == ROOT / "data" / "workspaces"
    assert "data/workspaces/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def _snapshot() -> dict[str, str]:
    files = {}
    for base in BASE_DIRS:
        for path in (ROOT / base).rglob("*"):
            if path.is_file():
                files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def test_uploading_never_changes_the_base_files(cfg):
    before = _snapshot()
    source = ROOT / "data" / "input" / "sample.ttl"
    data = source.read_bytes() if source.exists() else TTL
    ws = open_ontology(source.name, data, cfg=cfg)
    ws.mark_step("inherit", DONE)
    assert _snapshot() == before


def test_cli_creates_then_reopens(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.config.load_config",
                        lambda: {"workspaces": {"dir": str(tmp_path / "ws")}})
    source = tmp_path / "Mi ontología.ttl"
    source.write_bytes(TTL)
    assert store._run([str(source)]) == 0
    out = capsys.readouterr().out
    assert "Espacio de trabajo (S6-T04)" in out and "(nuevo)" in out
    assert "Heredar lo validado: pendiente" in out
    assert store._run([str(source)]) == 0
    assert "reabierto: ya existía" in capsys.readouterr().out
    assert store._run([str(tmp_path / "no.ttl")]) == 1