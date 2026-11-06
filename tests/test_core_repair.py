"""Tests for the two repairs of the base modules (S9-T02).

Synthetic graphs check each rule: several domains or ranges become one union, a property
typed both ways keeps the type its use shows, ranges that no longer fit go, and a property
that lives in the other module is left alone. An owlrl toy shows the union removes the
contradiction that several domains cause. Guarded tests run the real OntoPriv: the counts,
the input file untouched and the same entities with and without the repairs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, XSD

from src.core import repair as rp
from src.core.emit import materialize_modules, render_console
from src.ingest.ontology_loader import load_ontology

ROOT = Path(__file__).resolve().parents[1]
ONTOPRIV = ROOT / "data" / "input" / "ontopriv.rdf"
NS = "http://example.org/o#"
PREFIXES = f"""@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <{NS}> .
"""


def _graph(ttl: str) -> Graph:
    return Graph().parse(data=PREFIXES + ttl, format="turtle")


def _u(name: str) -> URIRef:
    return URIRef(NS + name)


def _union(graph: Graph, node) -> list:
    return list(Collection(graph, graph.value(node, OWL.unionOf)))


# ------------------------------------------------------------------ several domains / ranges

def test_several_domains_become_one_union():
    g = _graph(":A a owl:Class . :B a owl:Class . :C a owl:Class .\n"
               ":p a owl:ObjectProperty ; rdfs:domain :B , :A ; rdfs:range :C .\n"
               ":q a owl:ObjectProperty ; rdfs:domain :A .\n")
    changes = rp.repair_module(g, rp.kept_types(g))
    domains = list(g.objects(_u("p"), RDFS.domain))
    assert len(domains) == 1 and isinstance(domains[0], BNode)
    assert (domains[0], RDF.type, OWL.Class) in g
    assert _union(g, domains[0]) == [_u("A"), _u("B")]                  # sorted
    assert list(g.objects(_u("q"), RDFS.domain)) == [_u("A")]           # one domain: untouched
    assert list(g.objects(_u("p"), RDFS.range)) == [_u("C")]
    assert changes == [{"rule": rp.SEVERAL_DOMAINS, "property": "p", "iri": NS + "p",
                        "classes": ["A", "B"]}]
    assert rp.counts(changes) == {rp.SEVERAL_DOMAINS: 1, rp.SEVERAL_RANGES: 0, rp.DOUBLE_TYPE: 0}


def test_several_ranges_union_classes_or_datatypes_but_never_both():
    g = _graph(":A a owl:Class . :B a owl:Class .\n"
               ":p a owl:ObjectProperty ; rdfs:range :A , :B .\n"
               ":d a owl:DatatypeProperty ; rdfs:range xsd:string , xsd:date .\n"
               ":m a owl:ObjectProperty ; rdfs:range :A , xsd:string .\n")
    changes = rp.repair_module(g, rp.kept_types(g))
    (classes,) = g.objects(_u("p"), RDFS.range)
    assert (classes, RDF.type, OWL.Class) in g and _union(g, classes) == [_u("A"), _u("B")]
    (data,) = g.objects(_u("d"), RDFS.range)
    assert (data, RDF.type, RDFS.Datatype) in g and (data, RDF.type, OWL.Class) not in g
    assert _union(g, data) == [XSD.date, XSD.string]
    assert set(g.objects(_u("m"), RDFS.range)) == {_u("A"), XSD.string}  # no union exists
    assert [(c["rule"], c["property"]) for c in changes] == [(rp.SEVERAL_RANGES, "d"),
                                                             (rp.SEVERAL_RANGES, "p")]


def test_owlrl_no_longer_puts_the_individual_in_two_disjoint_classes():
    from owlrl import OWLRL_Semantics
    ttl = (":A a owl:Class ; owl:disjointWith :B . :B a owl:Class .\n"
           ":p a owl:ObjectProperty ; rdfs:domain :A , :B .\n"
           ":x a owl:NamedIndividual , :A ; :p :y .\n")

    def errors(graph):
        closure = OWLRL_Semantics(graph, False, False, rdfs=False)
        closure.closure()
        return closure.error_messages
    before = _graph(ttl)
    assert any("disjoint" in m.lower() for m in errors(before))       # x would also be a B
    after = _graph(ttl)
    rp.repair_module(after, rp.kept_types(after))
    assert errors(after) == []


# ------------------------------------------------------------------ double type

def test_a_double_typed_property_keeps_the_type_its_use_shows():
    g = _graph(":A a owl:Class .\n"
               ":obj a owl:ObjectProperty , owl:DatatypeProperty .\n"
               ":text a owl:ObjectProperty , owl:DatatypeProperty .\n"
               ":both a owl:ObjectProperty , owl:DatatypeProperty .\n"
               ":unused a owl:ObjectProperty , owl:DatatypeProperty .\n"
               ":parent a owl:ObjectProperty , owl:DatatypeProperty .\n"
               ":child a owl:DatatypeProperty ; rdfs:subPropertyOf :parent .\n"
               ":grand a owl:ObjectProperty , owl:DatatypeProperty .\n"
               ":kid a owl:ObjectProperty , owl:DatatypeProperty ; rdfs:subPropertyOf :grand .\n"
               ":kidkid a owl:DatatypeProperty ; rdfs:subPropertyOf :kid .\n"
               ":kidkid2 a owl:DatatypeProperty ; rdfs:subPropertyOf :kid .\n"
               ":x a :A ; :obj :y ; :text \"hola\" , \"chao\" ; :both :y , \"hola\" ; "
               ":kidkid \"uno\" .\n")
    d = rp.kept_types(g)
    assert set(d) == {_u(n) for n in ("obj", "text", "both", "unused", "parent", "grand", "kid")}
    kept = {str(p)[len(NS):]: v["kept"] for p, v in d.items()}
    assert kept == {"obj": OWL.ObjectProperty, "text": OWL.DatatypeProperty,
                    "both": OWL.ObjectProperty, "unused": OWL.ObjectProperty,
                    "parent": OWL.DatatypeProperty,     # no values; its sub-property is data
                    "grand": OWL.DatatypeProperty,      # the values of the whole family count
                    "kid": OWL.DatatypeProperty}
    assert {k: d[_u("text")][k] for k in ("resources", "texts", "sub_properties")} == {
        "resources": 0, "texts": 2, "sub_properties": 0}
    assert d[_u("both")]["resources"] == 1 and d[_u("both")]["texts"] == 1
    assert d[_u("grand")]["sub_properties"] == 3 and d[_u("grand")]["texts"] == 1
    changes = rp.repair_module(g, d)
    for p, v in d.items():
        types = set(g.objects(p, RDF.type)) & {OWL.ObjectProperty, OWL.DatatypeProperty}
        assert types == {v["kept"]}
    text = next(c for c in changes if c["property"] == "text")
    assert text == {"rule": rp.DOUBLE_TYPE, "property": "text", "iri": NS + "text",
                    "kept": "propiedad de datos", "removed": "propiedad de objeto",
                    "evidence": {"resources": 0, "texts": 2, "sub_properties": 0},
                    "removed_ranges": []}
    assert rp.counts(changes)[rp.DOUBLE_TYPE] == 7


def test_ranges_that_no_longer_fit_the_kept_type_go():
    g = _graph(":A a owl:Class .\n"
               ":p a owl:ObjectProperty , owl:DatatypeProperty ; rdfs:range xsd:string .\n"
               ":t a owl:ObjectProperty , owl:DatatypeProperty ; rdfs:range :A , xsd:string .\n"
               ":x :p :y ; :t \"hola\" .\n")
    changes = {c["property"]: c for c in rp.repair_module(g, rp.kept_types(g))}
    assert list(g.objects(_u("p"), RDFS.range)) == []                   # a resource, no text
    assert changes["p"]["removed_ranges"] == ["string"]
    assert list(g.objects(_u("t"), RDFS.range)) == [XSD.string]         # a text, no class
    assert changes["t"]["removed_ranges"] == ["A"]


def test_a_property_in_the_other_module_is_left_alone():
    whole = _graph(":p a owl:ObjectProperty , owl:DatatypeProperty . :x :p \"hola\" .\n")
    decisions = rp.kept_types(whole)               # decided on the whole ontology
    module = _graph(":A a owl:Class .\n")
    assert rp.repair_module(module, decisions) == [] and len(module) == 1
    assert rp.kept_types(_graph(":p a owl:ObjectProperty .\n")) == {}


# ------------------------------------------------------------------ emit

_TTL = PREFIXES + """
:Root a owl:Class .
:Principles a owl:Class ; rdfs:subClassOf :Root .
:Minimization a owl:Class ; rdfs:subClassOf :Principles .
:Purpose a owl:Class ; rdfs:subClassOf :Principles .
:has_note a owl:ObjectProperty , owl:DatatypeProperty ; rdfs:domain :Minimization , :Purpose .
:Alice a owl:NamedIndividual , :Minimization ; :has_note "una nota" .
"""


def test_emit_repairs_both_modules_and_says_so(tmp_path):
    p = tmp_path / "o.ttl"
    p.write_text(_TTL, encoding="utf-8")
    before = p.read_bytes()
    result = materialize_modules(load_ontology(p), out_dir=tmp_path / "out")
    assert p.read_bytes() == before                                     # the input stays
    core = Graph().parse(result.core_path, format="xml")
    prop = _u("has_note")
    assert set(core.objects(prop, RDF.type)) == {OWL.DatatypeProperty}
    (domain,) = core.objects(prop, RDFS.domain)
    assert _union(core, domain) == [_u("Minimization"), _u("Purpose")]
    assert [(r["rule"], r["module"]) for r in result.repairs] == [
        (rp.DOUBLE_TYPE, "core"), (rp.SEVERAL_DOMAINS, "core")]
    saved = json.loads(Path(result.repairs_path).read_text(encoding="utf-8"))
    assert saved == {"rules": rp.RULES, "counts": rp.counts(result.repairs),
                     "changes": result.repairs}
    assert Path(result.repairs_path).name == rp.REPAIRS_FILE
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["repairs"] == {rp.SEVERAL_DOMAINS: 1, rp.SEVERAL_RANGES: 0,
                                   rp.DOUBLE_TYPE: 1}
    text = render_console(result)
    assert "  Reparaciones (no tocan el archivo de OntoPriv):" in text
    assert ("    - 1 propiedades con varios dominios: ahora uno, la unión de esas clases "
            "(P19 de OOPS!)") in text
    assert ("    - 1 propiedades de objeto y de datos a la vez: se quedan con un tipo (0 de "
            "objeto, 1 de datos)") in text
    assert f"    Detalle: {result.repairs_path}" in text


def test_emit_without_repair_writes_the_split_as_before(tmp_path):
    p = tmp_path / "o.ttl"
    p.write_text(_TTL, encoding="utf-8")
    result = materialize_modules(load_ontology(p), out_dir=tmp_path / "out", repair=False)
    core = Graph().parse(result.core_path, format="xml")
    assert len(list(core.objects(_u("has_note"), RDFS.domain))) == 2
    assert result.repairs == [] and result.repairs_path == ""
    assert not (tmp_path / "out" / rp.REPAIRS_FILE).exists()
    assert json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))["repairs"] is None
    assert "Reparaciones" not in render_console(result)


# ------------------------------------------------------------------ the real OntoPriv

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if not ONTOPRIV.exists():
        pytest.skip("OntoPriv no disponible en data/input/ontopriv.rdf")
    sha = hashlib.sha256(ONTOPRIV.read_bytes()).hexdigest()
    folder = tmp_path_factory.mktemp("real")
    repaired = materialize_modules(load_ontology(ONTOPRIV), out_dir=folder / "with")
    plain = materialize_modules(load_ontology(ONTOPRIV), out_dir=folder / "without",
                                repair=False)
    assert hashlib.sha256(ONTOPRIV.read_bytes()).hexdigest() == sha    # never touched
    return repaired, plain


def test_real_ontopriv_counts(real):
    repaired, _ = real
    assert rp.counts(repaired.repairs) == {rp.SEVERAL_DOMAINS: 84, rp.SEVERAL_RANGES: 35,
                                           rp.DOUBLE_TYPE: 34}
    double = [r for r in repaired.repairs if r["rule"] == rp.DOUBLE_TYPE]
    assert [r["property"] for r in double if r["kept"] == "propiedad de datos"] == [
        "Right_to_suspension_of_treatment"]
    assert [(r["property"], r["removed_ranges"]) for r in double if r["removed_ranges"]] == [
        ("are_destined_to", ["string"])]


def test_real_ontopriv_keeps_its_entities_and_ends_with_one_domain_and_one_type(real):
    repaired, plain = real
    for kind in ("core_path", "profile_path"):
        g = Graph().parse(getattr(repaired, kind), format="xml")
        g0 = Graph().parse(getattr(plain, kind), format="xml")
        named = {s for s in g.subjects() if isinstance(s, URIRef)}
        assert named == {s for s in g0.subjects() if isinstance(s, URIRef)}
        for p in named:
            assert len(list(g.objects(p, RDFS.domain))) <= 1
            assert len(list(g.objects(p, RDFS.range))) <= 1
            assert not ({(p, RDF.type, OWL.ObjectProperty), (p, RDF.type, OWL.DatatypeProperty)}
                        <= set(g.triples((p, RDF.type, None))))
    assert repaired.counts["core"]["seeds"] == plain.counts["core"]["seeds"]
    assert repaired.moved_to_profile == plain.moved_to_profile
