from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.ai.rag import vector_store
from src.ingest import law_upload as lu
from src.ingest.law_upload import (ARTICLES_FILE, LAW_COLLECTION, chroma_path, current_articles,
                                   index_workspace_law, prepare_law, retrieve_law_articles,
                                   segment_workspace_law)
from src.workspace.store import (DONE, ERROR, EXTRACT_STEPS, KIND_ONTOLOGY, LAW_STEPS,
                                 WorkspaceError, open_law, open_upload)

ROOT = Path(__file__).resolve().parents[1]
LAW = (
    "LEY DE PRUEBA\n"
    "Artículo 1°. Objeto. Esta ley protege los datos personales de las personas.\n"
    "Artículo 2°. Consentimiento. El tratamiento requiere la autorización del titular.\n"
    "Artículo 3°. Sanciones. Quien incumpla la ley recibirá una multa.\n"
)


class Embed:
    """Bag of words over 64 dimensions; counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = np.zeros(64)
            for w in re.findall(r"[a-záéíóúñ]{4,}", t.lower()):
                v[int(hashlib.md5(w[:6].encode()).hexdigest(), 16) % 64] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


@pytest.fixture
def cfg(tmp_path):
    return {"workspaces": {"dir": str(tmp_path / "ws")}}


@pytest.fixture
def ws(cfg):
    return open_law("Ley de prueba.txt", LAW.encode("utf-8"), cfg=cfg)


def _prepare(ws, embed=None, **kw):
    return prepare_law(ws, embed_fn=embed or Embed(), model_name="modelo-falso", **kw)


# ------------------------------------------------------------------ the steps of a law

def test_a_law_workspace_has_the_law_steps(ws, cfg):
    assert ws.manifest["mode"] == "law"
    assert [r["step"] for r in ws.progress()][:4] == list(LAW_STEPS + EXTRACT_STEPS) == [
        "segment", "index", "extract", "accept"]                       # S7-T09, then S7-T10
    both = open_upload({KIND_ONTOLOGY: ("o.ttl", b"@prefix ex: <http://e.org/> .\n"),
                        "law": ("l.txt", LAW.encode("utf-8"))}, cfg=cfg)
    assert [r["step"] for r in both.progress()] == [
        "segment", "index", "inherit", "definitions", "candidates", "propose", "labels",
        "evidence", "write", "package"]


def test_segment_and_index_a_law(ws):
    seen, embed = [], Embed()
    prepared = _prepare(ws, embed, progress=seen.append)
    assert seen == ["Separar la ley en artículos", "Indexar los artículos de la ley"]
    r = prepared["articles"]
    assert [a["number"] for a in r["articles"]] == [1, 2, 3]
    assert [a["title"] for a in r["articles"]] == ["Objeto", "Consentimiento", "Sanciones"]
    assert (r["language"], r["style"], r["file"]) == ("es", "es", "Ley de prueba.txt")
    assert r["law_sha256"] == ws.manifest["inputs"]["law"]["sha256"]
    assert r["cleaning"]["pages"] == 1 and r["segmenter_version"] == 2
    saved = json.loads((ws.work_dir / ARTICLES_FILE).read_text(encoding="utf-8"))
    assert saved == r
    index = prepared["index"]
    assert index["built"] and index["items"] == 3
    assert vector_store.count(name=LAW_COLLECTION, path=chroma_path(ws)) == 3
    assert chroma_path(ws) == ws.path / "chroma"
    meta = vector_store.get(["art-2"], name=LAW_COLLECTION, path=chroma_path(ws),
                            include=("metadatas",))["metadatas"][0]
    assert meta["number"] == 2 and meta["title"] == "Consentimiento"
    assert meta["source"] == "Ley de prueba.txt"
    assert [s["status"] for s in ws.progress()][:2] == [DONE, DONE]
    assert ws.step("segment")["articles"] == 3 and ws.step("segment")["language"] == "es"
    assert ws.step("index")["items"] == 3


def test_the_base_database_is_never_written(ws):
    before = vector_store.exists(LAW_COLLECTION)
    _prepare(ws)
    assert vector_store.exists(LAW_COLLECTION) == before is False


def test_uploading_the_same_law_again_costs_nothing(ws, cfg):
    _prepare(ws)
    again = open_law("Ley de prueba.txt", LAW.encode("utf-8"), cfg=cfg)
    embed = Embed()
    prepared = _prepare(again, embed)
    assert again.path == ws.path and not prepared["index"]["built"] and embed.calls == 0
    other_model = prepare_law(again, embed_fn=embed, model_name="otro-modelo")
    assert other_model["index"]["built"] and embed.calls >= 1            # model changed


def test_another_segmenter_version_segments_again(ws, monkeypatch):
    first = current_articles(ws)
    path = ws.work_dir / ARTICLES_FILE
    old = {**first, "segmenter_version": 1, "n_articles": 0, "articles": []}
    path.write_text(json.dumps(old), encoding="utf-8")
    assert current_articles(ws)["n_articles"] == 3                        # redone
    assert current_articles(ws) == json.loads(path.read_text(encoding="utf-8"))


def test_retrieval_finds_the_closest_article(ws):
    embed = Embed()
    _prepare(ws, embed)
    hits = retrieve_law_articles(ws, "autorización del titular para el tratamiento", 2,
                                 embed_fn=embed)
    assert hits[0]["number"] == 2 and hits[0]["title"] == "Consentimiento"
    assert len(hits) == 2 and hits[0]["distance"] <= hits[1]["distance"]


def test_a_text_without_articles_is_refused(cfg):
    w = open_law("notas.txt", "Unas notas sin artículos numerados.".encode("utf-8"), cfg=cfg)
    with pytest.raises(WorkspaceError, match="No se encontraron artículos numerados en notas.txt"):
        segment_workspace_law(w)
    assert w.step("segment")["status"] == ERROR
    assert not (w.work_dir / ARTICLES_FILE).exists()


def test_a_workspace_without_law_is_refused(cfg):
    w = open_upload({KIND_ONTOLOGY: ("o.ttl", b"@prefix ex: <http://e.org/> .\n")}, cfg=cfg)
    with pytest.raises(WorkspaceError, match="no tiene una ley"):
        index_workspace_law(w, embed_fn=Embed(), model_name="m")


def test_console_and_cli(tmp_path, monkeypatch, capsys, cfg):
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()                                    # no real model in the tests
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setattr("src.ai.rag.retriever.embed", fake.embed)
    monkeypatch.setattr("src.config.load_config", lambda: {**cfg, "rag": {"top_k": 3}})
    path = tmp_path / "ley.txt"
    path.write_text(LAW, encoding="utf-8")
    assert lu._run([str(path), "--consulta", "multa por incumplir la ley"]) == 0
    out = capsys.readouterr().out
    assert "Ley subida (S7-T07)" in out and "idioma: español" in out
    assert "Artículos: 3 (del 1 al 3; no falta ninguno) · encabezados descartados: 0" in out
    assert "Primeros: 1. Objeto; 2. Consentimiento; 3. Sanciones" in out
    assert "(construida ahora)" in out
    assert re.search(r'Consulta "multa por incumplir la ley":\n    Art\. 3\. Sanciones', out)
    assert lu._run([str(path)]) == 0 and "(ya estaba al día)" in capsys.readouterr().out
    assert lu._run([str(tmp_path / "no.txt")]) == 1
    bad = tmp_path / "ley.docx"
    bad.write_bytes(b"x")
    assert lu._run([str(bad)]) == 1


def test_the_gdpr_end_to_end(cfg):
    # The Colombian law is checked in test_legal_segmenter and test_law_text (its PDF is slow
    # to read: pypdf repairs its header).
    name, expected = "gdpr-en.pdf", 99
    path = ROOT / "data" / "input" / "laws" / name
    if not path.exists():
        pytest.skip(f"Falta {path}")
    w = open_law(name, path.read_bytes(), cfg=cfg)
    prepared = _prepare(w)
    assert prepared["articles"]["n_articles"] == prepared["index"]["items"] == expected
    assert "Último: " in lu.render_console(w, prepared)