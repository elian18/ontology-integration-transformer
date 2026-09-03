from pathlib import Path
import pytest
from app.services import ingest

ROOT = Path(__file__).resolve().parents[1]
ONTO = ROOT / "data/input/ontopriv.rdf"
PDF = ROOT / "data/input/lopdp.pdf"


@pytest.mark.skipif(not ONTO.exists(), reason="OntoPriv (RDF/XML) no está en data/input/")
def test_adapter_ontology_from_bytes():
    r = ingest.ingest_ontology("ontopriv.rdf", ONTO.read_bytes())
    assert r.loaded
    assert r.n_classes == 175
    assert r.ontology_flavor == "owl-full"


@pytest.mark.skipif(not ONTO.exists(), reason="OntoPriv (RDF/XML) no está en data/input/")
def test_adapter_summary_shape():
    s = ingest.ontology_summary("ontopriv.rdf", ONTO.read_bytes())
    assert s["flavor"] == "owl-full"
    assert "structure" in s and "owl_full_signals" in s


@pytest.mark.skipif(not PDF.exists(), reason="LOPDP PDF no está en data/input/")
def test_adapter_legal_text_from_bytes():
    t = ingest.ingest_legal_text("lopdp.pdf", PDF.read_bytes())
    assert t.source == "pdf"
    assert t.n_chars > 100_000



def test_base_ontology_loads():
    r = ingest.base_ontology()
    assert r is not None
    assert r.n_classes == 175
    assert r.ontology_flavor == "owl-full"


def test_base_legal_text_loads():
    t = ingest.base_legal_text()
    assert t is not None
    assert t.source == "pdf"
    assert t.candidate_articles == 77