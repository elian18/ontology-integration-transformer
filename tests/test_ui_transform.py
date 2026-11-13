"""S10-T10: "Solo transformar" reads the uploaded ontology and gives it back in RDF/XML, OWL,
Turtle or JSON-LD with the same content: no alignment, no AI, no workspace."""
from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import BNode
from rdflib.compare import isomorphic

from app.services import align_ontology as svc
from src.transform.convert import ConversionError, load_graph, parse_bytes
from tests.test_alignment_propose import TTL, env, targets  # noqa: F401
from tests.test_ui_align_ontology import _titles, cfg, page  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
REAL = [ROOT / "data/input/gdprtext-ontology.ttl", ROOT / "data/input/ontopriv.rdf"]


@pytest.mark.parametrize("path", REAL, ids=lambda p: p.name)
def test_the_same_content_in_the_four_formats(path):
    original = load_graph(path)
    downloads = svc.transform_downloads(path.name, path.read_bytes())
    assert [d["key"] for d in downloads] == ["rdfxml", "owl", "turtle", "jsonld"]
    for d in downloads:
        back = parse_bytes(d["data"](), d["key"])
        assert len(back) == len(original), d["key"]
        if any(isinstance(x, BNode) for t in original for x in t):     # OntoPriv: SWRL rules
            assert _named(back) == _named(original), d["key"]
            assert _shapes(back) == _shapes(original), d["key"]
        else:
            assert isomorphic(back, original), d["key"]


def _named(g) -> set:
    return {t for t in g if not any(isinstance(x, BNode) for x in t)}


def _shapes(g) -> list:
    """Triples with blank nodes, the blank nodes hidden (isomorphic() takes 25 s on OntoPriv)."""
    return sorted(tuple("_" if isinstance(x, BNode) else str(x) for x in t) for t in g
                  if any(isinstance(x, BNode) for x in t))


def test_the_buttons():
    downloads = svc.transform_downloads("Mi GDPR.v2.ttl", TTL.encode("utf-8"))
    assert [d["label"] for d in downloads] == ["RDF/XML", "OWL", "Turtle", "JSON-LD"]
    assert [d["file_name"] for d in downloads] == [
        "Mi GDPR.v2.rdf", "Mi GDPR.v2.owl", "Mi GDPR.v2.ttl", "Mi GDPR.v2.jsonld"]
    assert [d["mime"] for d in downloads] == ["application/rdf+xml", "application/rdf+xml",
                                              "text/turtle", "application/ld+json"]
    assert downloads[1]["help"] == "OWL (RDF/XML)"
    assert downloads[3]["help"] == svc.TRANSFORM_JSONLD


@pytest.mark.parametrize("name, data, message", [
    ("ontopriv.owx", b"<Ontology/>", "OWL/XML (.owx) es solo de entrada"),
    ("notas.txt", b"hola", "Formato de ontología no soportado: '.txt'."),
    ("rota.ttl", b"@prefix ex: <http://x#> . ex:a ex:b", "No se pudo leer rota.ttl: "),
    ("vacia.ttl", b"# nada\n", "vacia.ttl está vacío: no tiene nada que convertir."),
])
def test_what_cannot_be_transformed(name, data, message):
    with pytest.raises(ConversionError) as err:
        svc.transform_downloads(name, data)
    assert message in svc.error_message(err.value)


class _Upload:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


@pytest.fixture
def uploaded(page, cfg, monkeypatch, tmp_path):  # noqa: F811
    """The page of "Solo transformar" with a file it can see (AppTest cannot drive an uploader);
    the AI and the workspaces would fail if anything touched them."""
    import streamlit
    import src.ai.llm_client as client

    def no_key(*a, **k):
        raise KeyError("LLM_API_KEY")
    monkeypatch.setattr(client, "LLMClient", no_key)
    upload = {}
    monkeypatch.setattr(streamlit, "file_uploader",
                        lambda *a, **k: _Upload(*upload["file"]) if upload else None)
    return page, upload


def _open(page):
    at = page()
    at.run()
    at.button(key="align_path_transform").click().run()
    return at


def test_the_page_gives_the_four_downloads(uploaded, cfg):  # noqa: F811
    page, upload = uploaded
    at = _open(page)
    assert not at.download_button and not at.success
    upload["file"] = ("Mi GDPR.ttl", TTL.encode("utf-8"))
    at.run()
    assert not at.exception and not at.error
    assert at.success[0].proto.title == svc.TRANSFORM_READY
    assert at.success[0].value == "Mi GDPR.ttl"
    buttons = [d for d in at.download_button if d.key.startswith("transform_dl_")]
    assert [d.label for d in buttons] == ["RDF/XML", "OWL", "Turtle", "JSON-LD"]
    assert all(d.proto.deferred_file_id and d.proto.ignore_rerun for d in buttons)
    assert svc.TRANSFORM_CAPTION in [c.value for c in at.caption]
    assert not list(Path(cfg["workspaces"]["dir"]).glob("*"))       # no workspace
    at.button(key="align_home").click().run()
    assert _titles(at) == [svc.START_TITLE] and not at.download_button


def test_the_page_says_why_a_file_cannot_be_read(uploaded):
    page, upload = uploaded
    at = _open(page)
    upload["file"] = ("ontopriv.owx", b"<Ontology/>")
    at.run()
    assert not at.exception and not at.download_button and not at.success
    assert at.error[0].value.startswith("OWL/XML (.owx) es solo de entrada")
