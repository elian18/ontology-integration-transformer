from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS

from src.publish import base_doc as bd
from src.transform.merge import removed

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "output"
needs_base = pytest.mark.skipif(not all((OUTPUT / f"{m}.ttl").exists() for m in bd.MODULES),
                                reason="Falta la base en data/output")
PREFIXES = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix dpv: <https://w3id.org/dpv#> .
@prefix gdprtext: <http://purl.org/adaptcentre/openscience/ontologies/GDPRtEXT#> .
@prefix bridge: <http://www.semanticweb.org/profiles/ontopriv-gdpr-alignment#> .
@prefix lopdp: <http://www.semanticweb.org/ley-organica-proteccion-datos-personales#> .
@prefix onto: <http://www.semanticweb.org/franc/ontologies/2023/9/OntologiaLOPDP#> .
"""
CORE = PREFIXES + """
<http://www.semanticweb.org/ontopriv-core> a owl:Ontology .
lopdp:Data_Owner a owl:Class ; lopdp:description "The person the data is about." .
lopdp:Controller a owl:Class, owl:ObjectProperty .
lopdp:Plain a owl:Class .
"""
PROFILE = PREFIXES + """
<http://www.semanticweb.org/profiles/ecuador-lopdp> a owl:Ontology ;
    owl:imports <http://www.semanticweb.org/ontopriv-core> .
onto:Boys_and_girls a owl:Class ; rdfs:label "niñas y niños"@es .
"""
ALIGNMENT = PREFIXES + """
<http://www.semanticweb.org/profiles/ecuador-lopdp-dpv-alignment> a owl:Ontology ;
    owl:imports <http://www.semanticweb.org/profiles/ecuador-lopdp> .
lopdp:Data_Owner skos:broadMatch dpv:Person ; skos:exactMatch dpv:DataSubject ;
    dcterms:source "LOPDP, art. 31"@es, "LOPDP, art. 4"@es .
lopdp:Controller skos:closeMatch dpv:DataController .
onto:Boys_and_girls skos:exactMatch dpv:Child .
"""
BRIDGE = PREFIXES + """
<http://www.semanticweb.org/profiles/ontopriv-gdpr-alignment> a owl:Ontology ;
    owl:imports <http://www.semanticweb.org/profiles/ecuador-lopdp-dpv-alignment> .
lopdp:Data_Owner skos:closeMatch gdprtext:DataSubject .
[] a owl:Axiom ; owl:annotatedSource lopdp:Data_Owner ; owl:annotatedProperty skos:closeMatch ;
    owl:annotatedTarget gdprtext:DataSubject ; bridge:chainedBy dpv:DataSubject .
"""
LOPDP = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
ONTO = "http://www.semanticweb.org/franc/ontologies/2023/9/OntologiaLOPDP#"


def _modules() -> list[Graph]:
    return [Graph().parse(data=text, format="turtle")
            for text in (CORE, PROFILE, ALIGNMENT, BRIDGE)]


def _texts(g: Graph, term: str, p=RDFS.comment) -> dict:
    return {o.language: str(o) for o in g.objects(URIRef(term), p)}


@pytest.fixture
def doc():
    return bd.build(_modules())


def test_one_header_with_the_authorship_and_the_credit_to_ontopriv(doc):
    g, _ = doc
    head = URIRef(bd.DOC_IRI)
    assert set(g.subjects(RDF.type, OWL.Ontology)) == {head}
    assert not list(g.triples((None, OWL.imports, None)))          # one file, nothing to fetch
    creator = g.value(head, DCTERMS.creator)
    assert str(g.value(creator, bd.SCHEMA.name)) == "Elian Alexander Gavilanes Miranda"
    affiliation = g.value(creator, bd.SCHEMA.affiliation)
    assert str(g.value(affiliation, bd.SCHEMA.name)) == "Escuela Politécnica Nacional (EPN)"
    assert g.value(head, DCTERMS.license) == URIRef(bd.LICENSE)
    assert g.value(head, DCTERMS.source) == URIRef("https://github.com/kelvinojedaepn/OntoPriv")
    assert str(g.value(head, OWL.versionInfo)) == "1.1.0"
    assert set(_texts(g, bd.DOC_IRI, DCTERMS.title)) == {"es", "en"}
    description = _texts(g, bd.DOC_IRI, DCTERMS.description)
    assert "Basada en OntoPriv" in description["es"] and "Based on OntoPriv" in description["en"]
    assert "D. A. Flores (2025)" in description["es"]


def test_each_concept_says_its_correspondences_closest_first_in_both_languages(doc):
    g, summary = doc
    owner = _texts(g, LOPDP + "Data_Owner")
    assert owner["es"] == ("The person the data is about. Correspondencias con el DPV: es igual "
                           "a dpv:DataSubject; es un tipo de dpv:Person. Puente con el GDPR:"
                           " se parece a gdprtext:DataSubject (por dpv:DataSubject).")
    assert owner["en"] == ("The person the data is about. Correspondences with the DPV: is the "
                           "same as dpv:DataSubject; is a kind of dpv:Person. Bridge with the "
                           "GDPR: is close to gdprtext:DataSubject (through dpv:DataSubject).")
    assert not _texts(g, LOPDP + "Plain")                    # nothing to say: no sentence
    assert (summary.terms, summary.described, summary.with_dpv, summary.with_gdpr) == (4, 3, 3, 1)


def test_a_term_without_name_gets_a_readable_one(doc):
    g, summary = doc
    assert g.value(URIRef(LOPDP + "Data_Owner"), RDFS.label) == Literal("Data Owner")
    assert list(g.objects(URIRef(ONTO + "Boys_and_girls"), RDFS.label)) == [
        Literal("niñas y niños", lang="es")]                              # its own name stays
    assert summary.labelled == 3


def test_the_table_lists_every_linked_concept_even_a_class_that_is_also_a_property(doc):
    g, _ = doc
    lines = _texts(g, bd.DOC_IRI, DCTERMS.description)["es"].split("\n")
    rows = [line for line in lines if line.startswith("| [")]
    assert rows == [
        "| [Controller](#Controller) | se parece a dpv:DataController |  |  |",
        "| [Data Owner](#Data_Owner) | es igual a dpv:DataSubject; es un tipo de dpv:Person | "
        "se parece a gdprtext:DataSubject | art. 4, art. 31 |",
        f"| [niñas y niños](#{ONTO}Boys_and_girls) | es igual a dpv:Child |  |  |"]
    assert "| Concept | DPV | GDPR | Articles of the LOPDP |" in _texts(
        g, bd.DOC_IRI, DCTERMS.description)["en"]


def test_nothing_of_the_base_is_left_out():
    modules = _modules()                                 # the same blank nodes on both sides
    g, _ = bd.build(modules)
    left_out = set().union(*(set(m) for m in modules)) - set(g)
    assert left_out == removed(modules, keep=bd.DOC_IRI)             # only headers and imports
    assert all(p in (RDF.type, OWL.imports) for _, p, _ in left_out)


def test_a_missing_module_is_said_in_spanish(tmp_path):
    with pytest.raises(FileNotFoundError, match="faltan módulos de la base"):
        bd.write(tmp_path, tmp_path / "doc.ttl")


@needs_base
def test_the_real_base_is_documented_without_touching_its_modules(tmp_path, capsys):
    files = sorted(OUTPUT.glob("*.*"))
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    out = tmp_path / "ontopriv-base.ttl"
    assert bd._run(["--salida", str(out)]) == 0
    assert {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files} == before
    printed = capsys.readouterr().out
    assert printed.startswith("Documento de la base para WIDOCO (S11-T05)")
    assert f"Guardado en: {out}" in printed
    g = Graph().parse(out, format="turtle")
    owner = _texts(g, LOPDP + "Data_Owner")
    assert "es igual a dpv:DataSubject" in owner["es"] and "gdprtext:DataSubject" in owner["es"]
    table = _texts(g, bd.DOC_IRI, DCTERMS.description)["es"]
    linked = set(bd.dpv_links(g)) | set(bd.gdpr_links(g))
    assert table.count("\n| [") == len(linked) > 100
