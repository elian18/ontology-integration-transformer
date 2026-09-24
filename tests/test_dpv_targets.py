"""S4-T03: the DPV side of the alignment (DPV terms only, concept/property marked)."""
import rdflib
from rdflib.namespace import RDF, OWL, SKOS

from src.ingest.dpv_loader import load_dpv
from src.core.dpv_proximity import collect_dpv_concepts
from src.alignment.dpv_targets import (
    build_dpv_targets, DPV_NAMESPACE, TARGET_CLASS, TARGET_PROPERTY,
)

_TTL = """
@prefix dpv: <https://w3id.org/dpv#> .
@prefix dcat: <http://www.w3.org/ns/dcat#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

dpv:PersonalData a rdfs:Class , skos:Concept ;
    skos:prefLabel "Personal Data"@en ;
    skos:definition "Data directly or indirectly associated with a person."@en .
dpv:SensitivePersonalData a rdfs:Class , skos:Concept ;
    skos:prefLabel "Sensitive Personal Data"@en ;
    skos:definition "Personal data that is sensitive."@en ;
    skos:broader dpv:PersonalData ; rdfs:subClassOf dpv:PersonalData .
dpv:DataSubject a rdfs:Class , skos:Concept ; skos:prefLabel "Data Subject"@en .
dpv:hasEntity a rdf:Property , skos:Concept ; skos:prefLabel "has entity"@en .
dpv:hasRecipient a rdf:Property , skos:Concept ;
    skos:prefLabel "has recipient"@en ; rdfs:subPropertyOf dpv:hasEntity .
dcat:Resource skos:prefLabel "dcat:Resource" .
"""


def _targets(tmp_path):
    f = tmp_path / "mini-dpv.ttl"
    f.write_text("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n" + _TTL,
                 encoding="utf-8")
    return build_dpv_targets(load_dpv(f))


def test_only_dpv_namespace_and_external_reported(tmp_path):
    t = _targets(tmp_path)
    assert all(x.iri.startswith(DPV_NAMESPACE) for x in t.targets)
    assert t.external == ["http://www.w3.org/ns/dcat#Resource"]
    assert t.counts()["total"] == 5


def test_kinds_are_marked(tmp_path):
    t = _targets(tmp_path)
    assert {x.name for x in t.of_kind(TARGET_PROPERTY)} == {"hasEntity", "hasRecipient"}
    assert {x.name for x in t.of_kind(TARGET_CLASS)} == {
        "PersonalData", "SensitivePersonalData", "DataSubject"}


def test_labels_definitions_parents_and_texts(tmp_path):
    t = {x.name: x for x in _targets(tmp_path).targets}
    spd = t["SensitivePersonalData"]
    assert spd.label == "Sensitive Personal Data"
    assert spd.parents == ["PersonalData"]                     # broader + subClassOf, no dupes
    assert spd.semantic_text().startswith("Sensitive Personal Data. Personal data")
    assert t["hasRecipient"].parents == ["hasEntity"]
    assert t["DataSubject"].definition is None
    assert t["DataSubject"].semantic_text() == "Data Subject"


def test_compatible_targets_by_source_kind(tmp_path):
    t = _targets(tmp_path)
    assert {x.kind for x in t.compatible_with(("class",))} == {TARGET_CLASS}
    assert {x.kind for x in t.compatible_with(("individual",))} == {TARGET_CLASS}
    assert {x.kind for x in t.compatible_with(("property",))} == {TARGET_PROPERTY}
    assert len(t.compatible_with(("class", "property"))) == len(t.targets)   # punned


def test_sprint3_collect_is_reused_unchanged(tmp_path):
    f = tmp_path / "mini-dpv.ttl"
    f.write_text("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n" + _TTL,
                 encoding="utf-8")
    dpv = load_dpv(f)
    iris, _ = collect_dpv_concepts(dpv)
    t = build_dpv_targets(dpv)
    assert len(iris) == len(t.targets) + len(t.external)


def test_real_dpv(dpv_report):
    """Against the project's DPV (conftest fixture): consistent counts, nothing lost."""
    t = build_dpv_targets(dpv_report)
    g = dpv_report.graph
    labelled = {s for s in g.subjects(SKOS.prefLabel, None)
                if isinstance(s, rdflib.URIRef) and str(s).startswith(DPV_NAMESPACE)}
    props = {s for s in labelled if any((s, RDF.type, k) in g
                                        for k in (RDF.Property, OWL.ObjectProperty,
                                                  OWL.DatatypeProperty))}
    c = t.counts()
    assert c["total"] == len(labelled)
    assert c["properties"] == len(props)
    assert c["classes"] + c["properties"] == c["total"]
    assert len({x.iri for x in t.targets}) == c["total"]