# PROV-O building blocks shared by the base (S10-T06) and each upload (S10-T07): stable IRIs,
# agents, activities and the records of what entered an ontology. PROV-O is referenced by IRI,
# never imported. A record about one triple is a named owl:Axiom (owl:annotatedSource, Property
# and Target), as SSSOM writes a mapping: Protégé shows it as the annotations of that triple,
# and the module it describes stays untouched.
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

PROV = Namespace("http://www.w3.org/ns/prov#")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
PROV_NS = "http://www.w3.org/ns/prov#"          # PROV-O, W3C Recommendation of 30 April 2013


def digest(*parts: str) -> str:
    """16 hex characters of the SHA-256 of the parts: the same record always gets the same IRI."""
    return hashlib.sha256(" ".join(parts).encode("utf-8")).hexdigest()[:16]


def slug(text: str) -> str:
    """'Elian Gavilanes' -> 'elian-gavilanes' (ASCII, for local names)."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-") or "x"


def when(value) -> Literal | None:
    """An xsd:dateTime from an ISO text as the logs write it; None when it is not one."""
    if not value:
        return None
    text = str(value)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return Literal(text, datatype=XSD.dateTime) if "T" in text else None


class People:
    """The names a person signed with in a log -> one prov:Person (S10-T06: "Elian" and
    "Elian Gavilanes" are the same reviewer). ``aliases``: {person: [other names]}."""

    def __init__(self, aliases: dict | None = None):
        self.aliases = {str(k): [str(n) for n in (v or [])] for k, v in (aliases or {}).items()}
        self.canonical = {n: k for k, names in self.aliases.items() for n in names}

    def name(self, signed: str) -> str:
        signed = (signed or "").strip()
        return self.canonical.get(signed, signed)

    def nicks(self, name: str) -> list[str]:
        return self.aliases.get(name, [])


class Recorder:
    """The provenance graph of one module, with helpers that keep every node an IRI (no blank
    nodes: the four formats then hold exactly the same triples)."""

    def __init__(self, iri: str, prefix: str, people: People | None = None):
        self.iri = iri
        self.ns = Namespace(iri + "#")
        self.people = people or People()
        self.g = Graph()
        for p, n in ((prefix, self.ns), ("prov", PROV), ("foaf", FOAF), ("dct", DCTERMS),
                     ("skos", SKOS), ("owl", OWL)):
            self.g.bind(p, n)

    def node(self, local: str) -> URIRef:
        return self.ns[local]

    def add(self, s, p, o) -> None:
        if o is not None:
            self.g.add((s, p, o))

    def texts(self, node, label: str | None, comment: str | None = None) -> None:
        if label:
            self.g.add((node, RDFS.label, Literal(label, lang="es")))
        if comment:
            self.g.add((node, RDFS.comment, Literal(comment, lang="es")))

    def header(self, label: str, comment: str, references=()) -> URIRef:
        onto = URIRef(self.iri)
        self.g.add((onto, RDF.type, OWL.Ontology))
        self.texts(onto, label, comment)
        for ref in (PROV_NS, *references):
            self.g.add((onto, DCTERMS.references, URIRef(ref)))
        self.g.add((onto, OWL.versionInfo, Literal("1")))
        return onto

    # ------------------------------------------------------------------ agents
    def agent(self, local: str, label: str, comment: str | None = None,
              kind=PROV.SoftwareAgent) -> URIRef:
        node = self.node(local)
        self.g.add((node, RDF.type, kind))
        self.texts(node, label, comment)
        return node

    def person(self, signed: str) -> URIRef:
        name = self.people.name(signed)
        node = self.node("person-" + slug(name))
        if (node, RDF.type, PROV.Person) not in self.g:
            self.agent("person-" + slug(name), name, kind=PROV.Person)
            self.g.add((node, FOAF.name, Literal(name)))
            for nick in self.people.nicks(name):
                self.g.add((node, FOAF.nick, Literal(nick)))
        return node

    # ------------------------------------------------------------------ activities and plans
    def plan(self, local: str, label: str, comment: str | None = None,
             identifier: str | None = None) -> URIRef:
        node = self.entity(local, label, comment, PROV.Plan)
        self.add(node, DCTERMS.identifier, Literal(identifier) if identifier else None)
        return node

    def activity(self, local: str, label: str, comment: str, *, agents=(), used=(),
                 informed_by=(), started=None, ended=None, plan=None) -> URIRef:
        """One step of the pipeline. With ``plan``, each agent's association says it followed
        it (prov:qualifiedAssociation + prov:hadPlan): the rule or the prompt."""
        node = self.node(local)
        self.g.add((node, RDF.type, PROV.Activity))
        self.texts(node, label, comment)
        for a in agents:
            self.g.add((node, PROV.wasAssociatedWith, a))
            if plan is not None:
                assoc = self.node(f"{local}-association-{str(a).rsplit('#', 1)[-1]}")
                self.g.add((assoc, RDF.type, PROV.Association))
                self.g.add((assoc, PROV.agent, a))
                self.g.add((assoc, PROV.hadPlan, plan))
                self.g.add((node, PROV.qualifiedAssociation, assoc))
        for e in used:
            self.g.add((node, PROV.used, e))
        for other in informed_by:
            self.g.add((node, PROV.wasInformedBy, other))
        self.add(node, PROV.startedAtTime, when(started))
        self.add(node, PROV.endedAtTime, when(ended))
        return node

    # ------------------------------------------------------------------ entities
    def entity(self, local: str, label: str | None, comment: str | None = None,
               *types) -> URIRef:
        node = self.node(local)
        self.g.add((node, RDF.type, PROV.Entity))
        for t in types:
            self.g.add((node, RDF.type, t))
        self.texts(node, label, comment)
        return node

    def generated(self, entity, activity, by=None, at=None) -> None:
        """prov:wasGeneratedBy, and who and when when they are known."""
        self.g.add((entity, PROV.wasGeneratedBy, activity))
        self.add(entity, PROV.wasAttributedTo, by)
        self.add(entity, PROV.generatedAtTime, when(at))

    def axiom(self, node, s, p, o) -> None:
        """The record is about the triple (s p o), which lives in another module."""
        self.g.add((node, RDF.type, OWL.Axiom))
        self.g.add((node, OWL.annotatedSource, URIRef(s)))
        self.g.add((node, OWL.annotatedProperty, URIRef(p)))
        self.g.add((node, OWL.annotatedTarget, URIRef(o)))

    def annotation_property(self, local: str, labels: dict, comment: str) -> URIRef:
        node = self.node(local)
        self.g.add((node, RDF.type, OWL.AnnotationProperty))
        for lang, text in labels.items():
            self.g.add((node, RDFS.label, Literal(text, lang=lang)))
        self.g.add((node, RDFS.comment, Literal(comment, lang="es")))
        return node

    def file(self, local: str, label: str, sha256: str | None = None,
             comment: str | None = None) -> URIRef:
        node = self.entity(local, label, comment)
        self.add(node, DCTERMS.identifier, Literal("sha256:" + sha256) if sha256 else None)
        return node


# --------------------------------------------------------------------------- reading

def records_about(g: Graph, term: str) -> list[URIRef]:
    """The records of a term: the triples it is the subject or the object of (correspondences,
    pairs; also the replaced ones, rdf:Statement) and the records whose dct:subject it is
    (concepts, repairs), in IRI order."""
    t = URIRef(term)
    found = set()
    for p in (OWL.annotatedSource, OWL.annotatedTarget, RDF.subject, RDF.object,
              DCTERMS.subject):
        found |= set(g.subjects(p, t))
    return sorted(found, key=str)


def _text(g: Graph, node, p) -> str | None:
    value = next(iter(sorted(g.objects(node, p), key=str)), None) if node is not None else None
    return str(value) if value is not None else None


def _name(g: Graph, node) -> str | None:
    if node is None:
        return None
    return _text(g, node, RDFS.label) or str(node)


def describe(g: Graph, record) -> dict:
    """What a record says, as plain values (for the console and the web, S10-T08): who and
    when, its sources, what it came from and what came from it, and the notes about it (the
    explanation of the reviewing AI of a pair)."""
    def one(p):
        return next(iter(sorted(g.objects(record, p), key=str)), None)

    notes = [describe(g, n) for n in sorted(g.subjects(DCTERMS.subject, record), key=str)]
    at = one(PROV.generatedAtTime)
    return {"iri": str(record), "label": _name(g, record),
            "comment": _text(g, record, RDFS.comment),
            "activity": _name(g, one(PROV.wasGeneratedBy)),
            "by": _name(g, one(PROV.wasAttributedTo)), "at": str(at) if at else None,
            "sources": sorted(_name(g, s) for s in g.objects(record, PROV.hadPrimarySource)),
            "derived_from": sorted(_name(g, s) for s in g.objects(record, PROV.wasDerivedFrom)),
            "led_to": sorted(_name(g, s) for s in g.subjects(PROV.wasDerivedFrom, record)),
            "notes": notes}
