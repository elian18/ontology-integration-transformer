# Two general repairs of the base modules, applied after the core/profile split (S9-T02).
# OntoPriv's input file is never touched; every change is recorded with its reason.
from __future__ import annotations

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, XSD

REPAIRS_FILE = "core-repairs.json"
SEVERAL_DOMAINS, SEVERAL_RANGES, DOUBLE_TYPE = "several_domains", "several_ranges", "double_type"
RULES = {SEVERAL_DOMAINS: "P19 (OOPS!): varios rdfs:domain pasan a uno, la unión de esas clases",
         SEVERAL_RANGES: "P19 (OOPS!): varios rdfs:range pasan a uno, la unión de esas clases",
         DOUBLE_TYPE: "Propiedad de objeto y de datos a la vez (no se permite en OWL 2 DL): se "
                      "queda con el tipo que muestra su uso"}
KINDS = {OWL.ObjectProperty: "propiedad de objeto", OWL.DatatypeProperty: "propiedad de datos"}


def _local(uri) -> str:
    s = str(uri)
    return s.rsplit("#", 1)[1] if "#" in s else s.rsplit("/", 1)[-1]


def _is_datatype(graph: Graph, node) -> bool:
    return (isinstance(node, URIRef) and (str(node).startswith(str(XSD)) or node == RDFS.Literal
                                          or node == RDF.langString)) \
        or (node, RDF.type, RDFS.Datatype) in graph


def _family(graph: Graph, prop: URIRef) -> set:
    """The property and all its sub-properties (transitively)."""
    seen, todo = {prop}, [prop]
    while todo:
        for sub in graph.subjects(RDFS.subPropertyOf, todo.pop()):
            if sub not in seen:
                seen.add(sub)
                todo.append(sub)
    return seen


def kept_types(ontology: Graph) -> dict[URIRef, dict]:
    """For each property declared with both types: the one to keep and the evidence."""
    out = {}
    for prop in sorted(set(ontology.subjects(RDF.type, OWL.ObjectProperty)), key=str):
        if not isinstance(prop, URIRef) or (prop, RDF.type, OWL.DatatypeProperty) not in ontology:
            continue
        family = _family(ontology, prop)
        values = [o for p in family for o in ontology.objects(None, p)]
        resources = sum(1 for o in values if isinstance(o, (URIRef, BNode)))
        texts = sum(1 for o in values if isinstance(o, Literal))
        sub_types = {t for p in family - {prop} for t in ontology.objects(p, RDF.type)
                     if t in KINDS}
        only_data = (texts and not resources) or (not values and sub_types == {OWL.DatatypeProperty})
        kept = OWL.DatatypeProperty if only_data else OWL.ObjectProperty
        out[prop] = {"kept": kept, "resources": resources, "texts": texts,
                     "sub_properties": len(family) - 1}
    return out


def _single_type(module: Graph, decisions: dict) -> list[dict]:
    changes = []
    for prop, d in decisions.items():
        if (prop, RDF.type, OWL.ObjectProperty) not in module:
            continue                                  # the property lives in the other module
        kept = d["kept"]
        dropped = OWL.DatatypeProperty if kept == OWL.ObjectProperty else OWL.ObjectProperty
        module.remove((prop, RDF.type, dropped))
        unfit = [r for r in module.objects(prop, RDFS.range)
                 if _is_datatype(module, r) == (kept == OWL.ObjectProperty)]
        for r in unfit:
            module.remove((prop, RDFS.range, r))
        changes.append({"rule": DOUBLE_TYPE, "property": _local(prop), "iri": str(prop),
                        "kept": KINDS[kept], "removed": KINDS[dropped],
                        "evidence": {k: d[k] for k in ("resources", "texts", "sub_properties")},
                        "removed_ranges": sorted(_local(r) for r in unfit)})
    return changes


def _union_of_several(module: Graph, predicate, rule: str) -> list[dict]:
    changes = []
    for prop in sorted({s for s in module.subjects(predicate, None) if isinstance(s, URIRef)},
                       key=str):
        classes = sorted(module.objects(prop, predicate), key=str)
        data = {_is_datatype(module, c) for c in classes}
        if len(classes) < 2 or len(data) > 1:
            continue                                  # a class and a datatype have no union
        for c in classes:
            module.remove((prop, predicate, c))
        union, members = BNode(), BNode()
        Collection(module, members, classes)
        module.add((union, RDF.type, RDFS.Datatype if data == {True} else OWL.Class))
        module.add((union, OWL.unionOf, members))
        module.add((prop, predicate, union))
        changes.append({"rule": rule, "property": _local(prop), "iri": str(prop),
                        "classes": [_local(c) for c in classes]})
    return changes


def repair_module(module: Graph, decisions: dict) -> list[dict]:
    """Apply both repairs to one module (in place); ``decisions`` come from kept_types()."""
    changes = _single_type(module, decisions)
    changes += _union_of_several(module, RDFS.domain, SEVERAL_DOMAINS)
    changes += _union_of_several(module, RDFS.range, SEVERAL_RANGES)
    return changes


def counts(changes: list[dict]) -> dict:
    return {rule: sum(1 for c in changes if c["rule"] == rule) for rule in RULES}
