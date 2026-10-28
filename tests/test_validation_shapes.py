from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

import pyshacl
import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, SH, SKOS

from src.alignment import user_alignment as ua
from src.alignment.sources import KNOWN_VOCAB_NAMESPACES
from src.validation import shapes as sh
from src.validation.shapes import VIOLATION, WARNING, Finding, check, dpv_terms, rules
from tests.test_alignment_bundle import _resolve
from tests.test_alignment_export import _DPV
from tests.test_alignment_propose import targets  # noqa: F401
from tests.test_law_definitions import LawLLM, _align, env  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
DPV = "https://w3id.org/dpv#"
EX = "http://example.org/gdpr#"
PREFIXES = f"""
@prefix ex: <{EX}> .
@prefix dpv: <{DPV}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix al: <http://example.org/gdpr-dpv-alignment#> .
dpv:Consent rdfs:isDefinedBy dpv: .
dpv:DataController rdfs:isDefinedBy dpv: .
dpv:PersonalData rdfs:isDefinedBy dpv: .
"""
SHAPES = sh.load_shapes()


def found(ttl: str) -> list[Finding]:
    return check(Graph().parse(data=PREFIXES + ttl, format="turtle"), SHAPES)


def by_rule(ttl: str) -> Counter:
    return Counter(f.rule for f in found(ttl))


def messages(ttl: str, rule: str) -> list[str]:
    return [f.message for f in found(ttl) if f.rule == rule]


# ------------------------------------------------------------------ the file and its catalog

def test_the_file_is_valid_shacl():
    conforms, _, _ = pyshacl.validate(Graph(), shacl_graph=SHAPES, meta_shacl=True,
                                      advanced=True)
    assert conforms


def test_the_catalog():
    catalog = rules(SHAPES)
    assert [r.id for r in catalog] == [
        "DpvTermShape", "MappingConflictShape", "AlignedConceptShape", "MappingAnnotationShape",
        "LabelShape", "DefinitionShape", "LawDefinitionShape", "LangStringShape", "ImportShape",
        "SpanishLabelShape", "EnglishLabelShape", "DefinitionCompletenessShape"]
    assert [r.group for r in catalog][::4] == ["Correspondencias con el DPV",
                                               "Etiquetas y definiciones", "Paquete"]
    assert {r.severity for r in catalog if r.group == "Completitud"} == {WARNING}
    assert {r.severity for r in catalog if r.group != "Completitud"} == {VIOLATION}
    assert all(r.name and r.description.endswith(".") for r in catalog)


def test_the_cli_shows_the_catalog(capsys):
    assert sh._run([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Reglas de forma (SHACL) de src/validation/shapes.ttl: 12 reglas")
    assert "\nCompletitud\n  10. Cada concepto tiene etiqueta en español (advertencia)" in out
    assert "   1. El término del DPV existe (violación)" in out


def test_an_ontology_with_nothing_wrong_conforms():
    assert found("""
        <http://example.org/gdpr> a owl:Ontology .
        ex:Consent a owl:Class ; rdfs:label "Consentimiento"@es, "Consent"@en ;
            skos:definition "Autorización del titular."@es ; skos:exactMatch dpv:Consent .
        """) == []


# ------------------------------------------------------------------ mappings to the DPV

def test_a_term_that_is_not_in_the_dpv():
    ttl = """
        ex:Consent a owl:Class ; skos:exactMatch dpv:Consentt .
        ex:Controller a owl:Class ; rdfs:seeAlso dpv:DataControler .
        ex:Data a owl:Class ; skos:closeMatch dpv:PersonalData ;
            skos:exactMatch <http://example.org/other#Data> ; rdfs:seeAlso "texto" .
        """
    result = [f for f in found(ttl) if f.rule == "DpvTermShape"]
    assert sorted(f.value for f in result) == [DPV + "Consentt", DPV + "DataControler"]
    assert {f.message for f in result} == {"Apunta a un término que no está en el DPV."}


@pytest.mark.parametrize("first,second,clash", [
    ("exactMatch", "broadMatch", True), ("exactMatch", "narrowMatch", True),
    ("exactMatch", "relatedMatch", True), ("relatedMatch", "broadMatch", True),
    ("relatedMatch", "narrowMatch", True), ("exactMatch", "closeMatch", False),
    ("broadMatch", "narrowMatch", False), ("closeMatch", "relatedMatch", False)])
def test_skos_relations_that_contradict_each_other(first, second, clash):
    ttl = f"ex:Consent a owl:Class ; skos:{first} dpv:Consent ; skos:{second} dpv:Consent ."
    result = [f for f in found(ttl) if f.rule == "MappingConflictShape"]
    assert len(result) == clash
    if clash:
        assert result[0].value == DPV + "Consent" and "SKOS declara incompatibles" in (
            result[0].message)


def test_different_terms_never_clash():
    assert not by_rule("""
        ex:Consent a owl:Class ; skos:exactMatch dpv:Consent ; skos:broadMatch dpv:PersonalData .
        """)["MappingConflictShape"]


def test_the_aligned_concept_is_declared():
    assert messages("ex:Ghost skos:exactMatch dpv:Consent .", "AlignedConceptShape") == [
        "Tiene una correspondencia pero no está declarado en la ontología."]
    assert not by_rule("ex:Consent a owl:NamedIndividual ; skos:exactMatch dpv:Consent .")[
        "AlignedConceptShape"]


ANNOTATED = """
    ex:Consent a owl:Class ; skos:exactMatch dpv:Consent .
    [] a owl:Axiom ; owl:annotatedSource ex:Consent ; owl:annotatedProperty skos:exactMatch ;
       owl:annotatedTarget {target} ; {extra} .
    """


def _annotation(target="dpv:Consent", extra='al:decidedBy "regla"@es ; owl:versionInfo "1"'):
    return [f.message for f in found(ANNOTATED.format(target=target, extra=extra))
            if f.rule == "MappingAnnotationShape"]


def test_a_complete_annotation_passes():
    assert _annotation() == []
    assert _annotation(extra='al:decidedBy "regla sin IA" ; owl:versionInfo "2"') == []


def test_an_annotation_without_who_or_version():
    who = "La correspondencia no dice quién la decidió (base, regla o regla sin IA)."
    assert _annotation(extra='owl:versionInfo "1"') == [who]
    assert _annotation(extra='al:decidedBy "una persona"@es ; owl:versionInfo "1"') == [who]
    assert _annotation(extra='al:decidedBy "base"@es') == [
        "La correspondencia no dice su versión."]


def test_an_annotation_of_a_mapping_that_is_not_there():
    assert _annotation(target="dpv:PersonalData") == [
        "Anota una correspondencia que no está en el módulo."]


def test_annotations_of_other_things_are_left_alone():
    assert _annotation(target="<http://example.org/other#Consent>", extra='rdfs:comment "x"') == []


# ------------------------------------------------------------------ labels and definitions

def test_labels_and_definitions_need_text_and_language():
    ttl = """
        ex:A a owl:Class ; rdfs:label "sin idioma", "Bien"@es, "Fine"@en-GB, " "@en ;
            skos:definition "Una definición."@es, "untagged" .
        """
    assert sorted(messages(ttl, "LabelShape")) == ["La etiqueta está vacía.",
                                                   "La etiqueta no tiene idioma."]
    assert messages(ttl, "DefinitionShape") == ["La definición no tiene idioma."]


def test_a_lang_string_without_language():
    ttl = """
        ex:person a owl:NamedIndividual ;
            ex:activity "Debe facilitar el acceso."^^rdf:langString ; ex:note "Bien"@es ;
            ex:code "A1" .
        """
    result = [f for f in found(ttl) if f.rule == "LangStringShape"]
    assert len(result) == 1 and result[0].focus == EX + "person"
    assert result[0].message == ("Su valor de activity está marcado como rdf:langString pero no "
                                 "tiene idioma.")


LAW_DEFINITION = """
    ex:Consent a owl:Class {definition} {own} .
    [] a owl:Axiom ; owl:annotatedSource ex:Consent ; owl:annotatedProperty skos:definition ;
       owl:annotatedTarget "{target}"@es ; dct:source "{source}" ; rdfs:comment "Origen."@es .
    """


def _law(target="Autorización del titular.", source="ley.pdf, art. 12", own="",
         definition='; skos:definition "Autorización del titular."@es'):
    ttl = LAW_DEFINITION.format(target=target, source=source, own=own, definition=definition)
    return [f.message for f in found(ttl) if f.rule == "LawDefinitionShape"]


def test_a_definition_taken_from_the_law():
    assert _law() == []
    article = ("La definición no dice de qué artículo de la ley sale (dct:source «<ley>, "
               "art. N»).")
    assert _law(source="ley.pdf") == [article]
    assert _law(source="ley.pdf, art. doce") == [article]
    orphan = ["Anota una definición que no está en la ontología."]
    assert _law(target="Otra definición.") == orphan
    assert _law(definition="") == orphan
    assert _law(own='; rdfs:comment "Consent of the data subject."@en') == [
        "El concepto ya tenía una definición: la de la ley solo va donde falta."]


# ------------------------------------------------------------------ the package

def test_every_import_is_in_the_package():
    ttl = """
        <http://example.org/gdpr-dpv-alignment> a owl:Ontology ;
            owl:imports <http://example.org/gdpr>, <http://example.org/gdpr-labels> .
        <http://example.org/gdpr> a owl:Ontology ; owl:versionIRI <http://example.org/gdpr/1> .
        <http://example.org/gdpr-definitions> a owl:Ontology ;
            owl:imports <http://example.org/gdpr/1> .
        """
    assert messages(ttl, "ImportShape") == [
        "Importa http://example.org/gdpr-labels, que no viene en el paquete."]


# ------------------------------------------------------------------ completeness (warnings)

def test_completeness_of_classes_and_properties():
    ttl = """
        ex:Bare a owl:Class .
        ex:hasPart a owl:ObjectProperty ; skos:prefLabel "tiene parte"@es-EC ;
            rdfs:label "has part"@EN ; rdfs:comment "Une una parte con su todo."@es .
        ex:Untagged a owl:Class ; rdfs:label "untagged" ; rdfs:comment "  " .
        ex:OnlyEnglish a rdfs:Class ; rdfs:label "Only English"@en ; skos:definition "A."@en .
        ex:hasValue a owl:DatatypeProperty ; rdfs:label "tiene valor"@es, "has value"@en .
        """
    result = [f for f in found(ttl) if f.severity == WARNING]
    assert sorted((f.focus[len(EX):], f.message) for f in result) == [
        ("Bare", "No tiene definición."), ("Bare", "No tiene etiqueta en español."),
        ("Bare", "No tiene etiqueta en inglés."), ("OnlyEnglish", "No tiene etiqueta en español."),
        ("Untagged", "No tiene definición."), ("Untagged", "No tiene etiqueta en español."),
        ("Untagged", "No tiene etiqueta en inglés."), ("hasValue", "No tiene definición.")]
    assert {f.rule for f in result} == {"SpanishLabelShape", "EnglishLabelShape",
                                        "DefinitionCompletenessShape"}


def test_completeness_leaves_out_the_standard_vocabularies_and_blank_nodes():
    declared = "\n".join(f"<{ns}Thing{i}> a owl:Class ." for i, ns in
                         enumerate(KNOWN_VOCAB_NAMESPACES))
    union = "[] a owl:Class ; owl:unionOf ( ex:A ex:B ) ."
    assert not [f for f in found(declared + union) if f.severity == WARNING]
    other = found("<http://example.org/dpv-like#Thing> a owl:Class .")
    assert len([f for f in other if f.severity == WARNING]) == 3


# ------------------------------------------------------------------ what the system writes

def test_rule_of_goes_up_from_a_property_shape():
    shape = next(SHAPES.objects(URIRef(sh.SHAPES_NS + "LabelShape"), SH.property))
    assert sh.rule_of(SHAPES, shape) == "LabelShape"
    assert sh.rule_of(SHAPES, URIRef(sh.SHAPES_NS + "ImportShape")) == "ImportShape"


def _terms_of_the_test_dpv() -> Graph:
    dpv, terms = Graph().parse(data=_DPV, format="turtle"), Graph()
    for term in [*dpv.subjects(RDF.type, SKOS.Concept), URIRef(DPV + "Consent"),
                 URIRef(DPV + "ConsentRecord")]:          # the last two: terms of the test base
        terms.add((term, RDFS.isDefinedBy, URIRef(DPV)))
    return terms


def test_the_package_of_ontology_and_law_only_inherits_a_label(env, tmp_path):  # noqa: F811
    run = _align(env, LawLLM())
    folder = tmp_path / "z"
    zipfile.ZipFile(run.zip).extractall(folder)
    data = _resolve(folder, run.manifest["modules"]["alignment"]) + _terms_of_the_test_dpv()
    assert set(run.manifest["modules"]) == {"alignment", "labels", "definitions"}
    result = [(f.rule, f.focus[len(EX):], f.message) for f in check(data, SHAPES)]
    assert result == [
        ("DefinitionCompletenessShape", "Pizza", "No tiene definición."),   # the law lacks it
        ("LabelShape", "Erasure", "La etiqueta no tiene idioma.")]          # "erasure" of TTL


def test_dpv_terms_reads_the_markers(tmp_path):
    path = tmp_path / "dpv.ttl"
    path.write_text(PREFIXES + "dpv:Other a skos:Concept .", encoding="utf-8")
    terms = dpv_terms(path)
    assert len(terms) == 3 and (URIRef(DPV + "Consent"), RDFS.isDefinedBy, URIRef(DPV)) in terms


BASE = [ROOT / "data" / "output" / name for name in
        ("ontopriv-core.rdf", "profile-ecuador-lopdp.rdf", "ontopriv-dpv-alignment.rdf")]


@pytest.mark.skipif(not all(p.exists() for p in BASE), reason="Falta la base en data/output")
def test_the_base_only_inherits_the_lang_string_of_ontopriv(capsys):
    assert sh._run([str(p) for p in BASE]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Reglas de forma sobre 3 archivos")
    assert "  [violación] Ningún texto rdf:langString sin idioma: 1\n" in out
    assert "       responsible_for_treatment: Su valor de activity_within_right_of_access" in out
    for rule in rules(SHAPES)[:7] + rules(SHAPES)[8:9]:
        assert f"  [ok] {rule.name}: 0\n" in out


def test_the_cli_reports_a_file_it_cannot_read(tmp_path, capsys):
    assert sh._run([str(tmp_path / "no.ttl")]) == 1
    assert capsys.readouterr().out.startswith("Error: No existe el archivo")


def test_the_alignment_module_writer_satisfies_the_rules():
    inherited = {"inherited": [{"key": EX + "Consent", "name": "Consent", "method": "same_name",
                                "base_iri": "http://base#Consent", "base_name": "Consent",
                                "status": "aligned",
                                "mappings": [{"relation": "skos:exactMatch",
                                              "dpv_iri": DPV + "Consent",
                                              "reviewer": "Elian Gavilanes",
                                              "decided_at": "2026-10-07T10:00:00"}]}]}
    decisions = {"alignment_version": 1, "summary": {"rule": "confianza alta"},
                 "threshold_without_ai": 0.75,
                 "entries": [{"key": EX + "Data", "name": "Data", "status": "accepted",
                              "dpv_iri": DPV + "PersonalData", "relation": "skos:closeMatch",
                              "decided_by": "rule", "score": 0.81, "version": 1,
                              "justification": "Son datos personales."},
                             {"key": EX + "Who", "name": "Who", "status": "accepted",
                              "dpv_iri": DPV + "DataController", "relation": None,
                              "decided_by": "rule_without_ai", "score": 0.9, "version": 1}]}
    module = ua.alignment_graph(inherited, decisions, iri="http://example.org/gdpr-dpv-alignment",
                                onto_iri="http://example.org/gdpr", name="gdpr.ttl",
                                created="2026-10-28T00:00:00+00:00")
    data = module + Graph().parse(data=PREFIXES + """
        <http://example.org/gdpr> a owl:Ontology .
        ex:Consent a owl:Class . ex:Data a owl:Class . ex:Who a owl:Class .
        """, format="turtle")
    assert [f for f in check(data, SHAPES) if f.severity == VIOLATION] == []