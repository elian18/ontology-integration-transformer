"""S6-T01: conversion between RDF/XML, Turtle and JSON-LD keeps the exact same graph.

The fixture reproduces, in a few triples, every case that broke plain rdflib on the real files:
a ``"15"^^xsd:decimal`` (Turtle rewrote it as 15.0), a list typed ``rdf:List`` like OntoPriv's
SWRL rules (JSON-LD dropped the type), blank nodes (owl:Restriction, owl:Axiom), es/en labels,
typed xsd:string and integers (JSON-LD turned them into native JSON values)."""
from __future__ import annotations

import json

import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import RDF, XSD

from src.transform import convert
from src.transform.convert import (ConversionError, OUTPUT_FORMATS, convert_file,
                                   formats_for, input_format, jsonld_context, load_graph,
                                   output_format, parse_bytes, serialize, used_prefixes)

FIXTURE_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix swrl: <http://www.w3.org/2003/11/swrl#> .
@prefix ley-organica: <http://example.org/ley-organica#> .
@prefix unused: <http://example.org/never-used#> .

<http://example.org/ley-organica> a owl:Ontology ;
    owl:imports <http://example.org/core> .

ley-organica:Consent a owl:Class ;
    rdfs:label "Consentimiento"@es, "Consent"@en ;
    rdfs:comment "Texto con \\"comillas\\" y tildes: protección"^^xsd:string ;
    skos:exactMatch <https://w3id.org/dpv#Consent> ;
    rdfs:subClassOf [ a owl:Restriction ;
        owl:onProperty ley-organica:hasPurpose ;
        owl:someValuesFrom ley-organica:Purpose ] .

ley-organica:hasPurpose a owl:ObjectProperty, owl:DatatypeProperty .

ley-organica:Controller ley-organica:deadline "15"^^xsd:decimal ;
    ley-organica:days "007"^^xsd:integer ;
    ley-organica:ratio "1.5E0"^^xsd:double ;
    ley-organica:active "true"^^xsd:boolean .

ley-organica:Rule a swrl:Imp ;
    swrl:arguments _:l1 .
_:l1 a rdf:List ; rdf:first "1"^^xsd:integer ; rdf:rest _:l2 .
_:l2 a rdf:List ; rdf:first ley-organica:Consent ; rdf:rest rdf:nil .

ley-organica:Union owl:unionOf ( ley-organica:Consent ley-organica:Controller ) .

[] a owl:Axiom ;
    owl:annotatedSource ley-organica:Consent ;
    owl:annotatedProperty skos:exactMatch ;
    owl:annotatedTarget <https://w3id.org/dpv#Consent> ;
    rdfs:comment "Propuesta automática, no validada"@es .
"""

EX = Namespace("http://example.org/ley-organica#")


@pytest.fixture
def graph() -> Graph:
    g = Graph()
    g.parse(data=FIXTURE_TTL, format="turtle")
    return g


@pytest.fixture
def source_file(tmp_path, graph):
    path = tmp_path / "fixture.rdf"
    path.write_bytes(graph.serialize(format="xml", encoding="utf-8"))
    return path


# ------------------------------------------------------------------ formats

def test_output_format_accepts_keys_and_aliases():
    assert output_format("turtle").extension == ".ttl"
    assert output_format("TTL").key == "turtle"
    assert output_format("json-ld").key == "jsonld"
    assert output_format("rdf/xml").key == "rdfxml"
    assert output_format("owl").extension == ".owl"
    assert output_format("owl").rdflib_name == "xml"      # .owl is RDF/XML


def test_unknown_output_format_explains_in_spanish():
    with pytest.raises(ConversionError, match="Formato de salida no soportado"):
        output_format("owx")


def test_owx_is_input_only_with_protege_hint():
    with pytest.raises(ConversionError, match="Protégé"):
        input_format("ontopriv.owx")
    with pytest.raises(ConversionError, match="no soportado"):
        input_format("ley.pdf")
    assert input_format("x.jsonld") == "json-ld"
    assert input_format("x.OWL") == "xml"


def test_load_graph_reports_missing_file(tmp_path):
    with pytest.raises(ConversionError, match="No existe"):
        load_graph(tmp_path / "nada.ttl")


# ------------------------------------------------------------------ round trip

@pytest.mark.parametrize("key", list(OUTPUT_FORMATS))
def test_round_trip_keeps_the_same_graph(graph, key):
    back = parse_bytes(serialize(graph, key), key)
    assert len(back) == len(graph)
    assert isomorphic(graph, back)


def test_chain_rdfxml_turtle_jsonld_rdfxml(graph):
    current = graph
    for key in ("rdfxml", "turtle", "jsonld", "rdfxml"):
        current = parse_bytes(serialize(current, key), key)
    assert isomorphic(graph, current)


def test_turtle_keeps_decimal_text(graph):
    text = serialize(graph, "turtle").decode("utf-8")
    back = parse_bytes(text.encode("utf-8"), "turtle")
    assert (EX.Controller, EX.deadline, Literal("15", datatype=XSD.decimal)) in back
    assert "15.0" not in text                       # plain rdflib wrote 15.0
    assert (EX.Controller, EX.days, Literal("007", datatype=XSD.integer)) in back


def test_jsonld_keeps_typed_list_nodes(graph):
    back = parse_bytes(serialize(graph, "jsonld"), "jsonld")
    typed = list(back.subjects(RDF.type, RDF.List))
    assert len(typed) == 2                          # plain rdflib dropped both


def test_jsonld_keeps_exact_literal_text(graph):
    back = parse_bytes(serialize(graph, "jsonld"), "jsonld")
    assert (EX.Controller, EX.days, Literal("007", datatype=XSD.integer)) in back
    assert (EX.Controller, EX.ratio, Literal("1.5E0", datatype=XSD.double)) in back
    comments = [o for o in back.objects(EX.Consent, URIRef(
        "http://www.w3.org/2000/01/rdf-schema#comment"))]
    assert comments and comments[0].datatype == XSD.string


def test_jsonld_has_a_context_with_only_used_prefixes(graph):
    doc = json.loads(serialize(graph, "jsonld").decode("utf-8"))
    ctx = doc["@context"]
    assert ctx["ley-organica"] == "http://example.org/ley-organica#"
    assert ctx["dpv"] == "https://w3id.org/dpv#"            # bound because it is used
    assert "unused" not in ctx and "brick" not in ctx
    assert "@graph" in doc


def test_used_prefixes_skips_unused_and_xml(graph):
    prefixes = used_prefixes(graph)
    assert "unused" not in prefixes and "xml" not in prefixes and "" not in prefixes
    assert prefixes["owl"] == "http://www.w3.org/2002/07/owl#"
    assert jsonld_context(graph) == used_prefixes(graph)


def test_known_prefix_is_bound_only_when_used():
    g = Graph()
    g.add((EX.A, RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class")))
    text = serialize(g, "turtle").decode("utf-8")
    assert "dpv:" not in text
    g.add((EX.A, URIRef("http://www.w3.org/2004/02/skos/core#closeMatch"),
           URIRef("https://w3id.org/dpv#Purpose")))
    text = serialize(g, "turtle").decode("utf-8")
    assert "dpv:Purpose" in text


def test_owl_imports_is_kept_as_an_iri(graph):
    for key in OUTPUT_FORMATS:
        back = parse_bytes(serialize(graph, key), key)
        imports = list(back.objects(URIRef("http://example.org/ley-organica"),
                                    URIRef("http://www.w3.org/2002/07/owl#imports")))
        assert imports == [URIRef("http://example.org/core")]


# ------------------------------------------------------------------ files and CLI

def test_convert_file_writes_next_to_source(source_file):
    result = convert_file(source_file, "turtle")
    assert result.target == source_file.with_suffix(".ttl")
    assert result.target.exists()
    assert result.complete and result.triples_in == result.triples_out
    assert result.source_label == "RDF/XML" and result.target_label == "Turtle"


def test_convert_file_to_another_folder_and_stem(source_file, tmp_path):
    out = tmp_path / "exports"
    result = convert_file(source_file, "jsonld", out_dir=out, stem="mi-ontologia")
    assert result.target == out / "mi-ontologia.jsonld"
    assert isomorphic(load_graph(source_file), load_graph(result.target))


def test_convert_file_never_overwrites_the_source(source_file):
    with pytest.raises(ConversionError, match="no se sobrescribe"):
        convert_file(source_file, "rdfxml")


def test_all_skips_the_source_own_extension(source_file):
    keys = [f.key for f in formats_for("all", source_file)]
    assert keys == ["owl", "turtle", "jsonld"]
    assert [f.key for f in formats_for("all", "x.ttl")] == ["rdfxml", "owl", "jsonld"]


def test_cli_converts_and_reports_in_spanish(source_file, capsys):
    code = convert._run([str(source_file), "--to", "all"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Conversión de formatos (S6-T01)" in out
    assert "fixture.ttl" in out and "fixture.jsonld" in out and "fixture.owl" in out
    assert "ninguna tripleta perdida" in out


def test_cli_reports_errors_without_traceback(tmp_path, capsys):
    code = convert._run([str(tmp_path / "ontopriv.owx")])
    assert code == 1
    assert "Error:" in capsys.readouterr().out