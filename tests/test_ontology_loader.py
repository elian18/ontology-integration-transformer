from pathlib import Path
import pytest
from src.ingest.ontology_loader import load_ontology, characterization_summary

ROOT = Path(__file__).resolve().parents[1]
ONTO = ROOT / "data/input/ontopriv.rdf"


@pytest.mark.skipif(not ONTO.exists(), reason="OntoPriv (RDF/XML) no está en data/input/")
def test_ontopriv_loads():
    r = load_ontology(ONTO)
    assert r.loaded
    assert r.source_format == "rdf/xml"
    assert r.n_classes > 0


@pytest.mark.skipif(not ONTO.exists(), reason="OntoPriv (RDF/XML) no está en data/input/")
def test_ontopriv_is_owl_full():
    r = load_ontology(ONTO)
    assert r.ontology_flavor == "owl-full"
    assert r.is_owl_full
    assert r.n_op_dp_overlap > 0 and r.n_class_prop_overlap > 0


@pytest.mark.skipif(not ONTO.exists(), reason="OntoPriv (RDF/XML) no está en data/input/")
def test_characterization_summary_shape():
    s = characterization_summary(load_ontology(ONTO))
    assert s["flavor"] == "owl-full"
    assert s["is_owl_full"] is True
    assert set(s["structure"]) == {"triples", "classes", "object_properties",
                                    "data_properties", "individuals"}
    assert set(s["owl_full_signals"]) == {"object_and_data_props", "class_and_property"}


def test_owx_is_rejected_with_clear_message():
    # rdflib no parsea OWL/XML: debe fallar con un mensaje claro, no reventar feo.
    with pytest.raises(ValueError, match="OWL/XML"):
        load_ontology(ROOT / "data/input/ontopriv.owx")



def test_loader_reads_turtle_and_jsonld(tmp_path):
    # La misma ontología en otros formatos debe cargar igual (mismo nº de clases).
    import rdflib
    base = load_ontology(ONTO)
    g = base.graph

    ttl = tmp_path / "onto.ttl"
    jsonld = tmp_path / "onto.jsonld"
    g.serialize(destination=str(ttl), format="turtle")
    g.serialize(destination=str(jsonld), format="json-ld")

    r_ttl = load_ontology(ttl)
    r_jsonld = load_ontology(jsonld)

    assert r_ttl.source_format == "turtle"
    assert r_jsonld.source_format == "json-ld"
    assert r_ttl.n_classes == base.n_classes
    assert r_jsonld.n_classes == base.n_classes