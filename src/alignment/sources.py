"""Build the single list of SOURCE concepts that the alignment compares against the DPV.

Sprint 4 aligns two sources (agreed design, flows A and B):
- Flow A: every named class and property of OntoPriv (from the Sprint 3 inventory).
- Flow B: the concepts the AI proposed from the law text (``PROPOSALS_FILE``, Sprint 3).
This module only GATHERS them into one uniform shape; it computes no similarity and makes no
decision. Later tasks consume it: lexical similarity (S4-T04), embeddings (S4-T05) and the
"already in OntoPriv?" check for the AI concepts (S4-T06).

Decisions encoded here:
- Kinds are normalized to the alignment vocabulary: ``class`` / ``property`` (object and data
  properties both become ``property``; the DPV only has one property kind). An OntoPriv entity
  punned as class AND property (OWL Full) keeps both kinds, so it is later compared with DPV
  concepts and DPV properties alike.
- Terms that live in a known standard vocabulary namespace (DPV, SKOS, FOAF, Dublin Core,
  PROV, ODRL, schema.org, OWL/RDF/RDFS/XSD) are "borrowed": they are already standard, so they
  are reported apart and not aligned.
- Named individuals: measured in S4-T02, OntoPriv's 181 individuals are example instances
  (``consentimiento001``, ``banking001``...: one per class, no label, no comment). An example
  record is not a concept, so by default they are NOT aligned (their class is); they are
  counted and reported. ``include_individuals=True`` turns them on for another ontology.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import rdflib
from rdflib.namespace import RDF, RDFS, OWL, SKOS

from src.ingest.ontology_loader import OntologyReport
from src.core.inventory import (build_inventory, OntologyInventory, KIND_CLASS,
                                KIND_OBJECT_PROP, KIND_DATA_PROP)
from src.core.dpv_proximity import _humanize, _trim
from src.core.extract import PROPOSALS_FILE

ORIGIN_ONTOLOGY = "ontology"
ORIGIN_AI = "ai"

ALIGN_CLASS = "class"
ALIGN_PROPERTY = "property"
ALIGN_INDIVIDUAL = "individual"

# Namespaces of standard vocabularies: entities here are borrowed, not aligned.
KNOWN_VOCAB_NAMESPACES = (
    "https://w3id.org/dpv",
    "http://www.w3.org/2004/02/skos/core#",
    "http://xmlns.com/foaf/0.1/",
    "http://purl.org/dc/terms/",
    "http://purl.org/dc/elements/1.1/",
    "http://www.w3.org/ns/prov#",
    "http://www.w3.org/ns/odrl/2/",
    "http://schema.org/",
    "https://schema.org/",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2001/XMLSchema#",
)

_KIND_MAP = {
    KIND_CLASS: ALIGN_CLASS,
    KIND_OBJECT_PROP: ALIGN_PROPERTY,
    KIND_DATA_PROP: ALIGN_PROPERTY,
}


@dataclass
class SourceConcept:
    """One concept to align, from OntoPriv (flow A) or from the AI proposals (flow B)."""
    key: str                         # unique id: the IRI (ontology) or "ai:<name>" (AI)
    origin: str                      # ontology | ai
    name: str                        # local name / AI identifier (English CamelCase for AI)
    kinds: tuple[str, ...]           # any of class / property / individual
    label: str | None = None         # rdfs:label/skos:prefLabel (ontology) or Spanish label (AI)
    definition: str | None = None    # rdfs:comment/skos:definition (ontology) or AI definition
    iri: str | None = None
    namespace: str | None = None
    family: str | None = None        # ontology family, or the class of an individual
    articles: list[int] = field(default_factory=list)   # provenance (AI concepts)

    def name_text(self) -> str:
        """Humanized identifier ('DataSubject' -> 'Data Subject'): input of the lexical match."""
        return _humanize(self.name)

    def semantic_text(self) -> str:
        """Identifier + label + trimmed definition: input of the embeddings (any language)."""
        parts = [self.name_text()]
        if self.label and self.label.strip().lower() != parts[0].lower():
            parts.append(self.label.strip())
        if self.definition:
            parts.append(_trim(self.definition))
        return ". ".join(parts)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "origin": self.origin,
            "name": self.name,
            "kinds": list(self.kinds),
            "label": self.label,
            "definition": self.definition,
            "iri": self.iri,
            "namespace": self.namespace,
            "family": self.family,
            "articles": self.articles,
        }


@dataclass
class AlignmentSources:
    """Every source concept plus what was left out and why."""
    ontology_path: str
    proposals_path: str | None
    concepts: list[SourceConcept] = field(default_factory=list)
    borrowed: list[str] = field(default_factory=list)            # IRIs from standard vocabularies
    skipped_individuals: list[str] = field(default_factory=list)  # local names (not aligned)
    individuals_total: int = 0
    proposals_found: bool = False

    def of_origin(self, origin: str) -> list[SourceConcept]:
        return [c for c in self.concepts if c.origin == origin]

    def counts(self) -> dict:
        onto = self.of_origin(ORIGIN_ONTOLOGY)
        ai = self.of_origin(ORIGIN_AI)
        return {
            "total": len(self.concepts),
            "ontology": len(onto),
            "ai": len(ai),
            "ontology_classes": sum(1 for c in onto if ALIGN_CLASS in c.kinds),
            "ontology_properties": sum(1 for c in onto if ALIGN_PROPERTY in c.kinds),
            "ontology_class_and_property": sum(
                1 for c in onto if ALIGN_CLASS in c.kinds and ALIGN_PROPERTY in c.kinds),
            "ontology_individuals": sum(1 for c in onto if ALIGN_INDIVIDUAL in c.kinds),
            "ai_classes": sum(1 for c in ai if ALIGN_CLASS in c.kinds),
            "ai_properties": sum(1 for c in ai if ALIGN_PROPERTY in c.kinds),
            "borrowed": len(self.borrowed),
            "individuals_total": self.individuals_total,
            "individuals_skipped": len(self.skipped_individuals),
            "proposals_found": self.proposals_found,
        }


def _local(uri) -> str:
    s = str(uri)
    return s.rsplit("#", 1)[1] if "#" in s else s.rsplit("/", 1)[-1]


def _namespace(uri) -> str:
    s = str(uri)
    return s.rsplit("#", 1)[0] + "#" if "#" in s else s.rsplit("/", 1)[0] + "/"


def _literal(graph: rdflib.Graph, subject, *predicates) -> str | None:
    for predicate in predicates:
        for obj in graph.objects(subject, predicate):
            if isinstance(obj, rdflib.Literal) and str(obj).strip():
                return str(obj).strip()
    return None


def is_borrowed(iri: str) -> bool:
    """True when the IRI belongs to a known standard vocabulary (so it is not aligned)."""
    return any(iri.startswith(ns) for ns in KNOWN_VOCAB_NAMESPACES)


def _ontology_concepts(report: OntologyReport, inventory: OntologyInventory):
    g = report.graph
    concepts, borrowed = [], []
    for e in inventory.entries:
        if is_borrowed(e.iri):
            borrowed.append(e.iri)
            continue
        kinds = tuple(sorted({_KIND_MAP[k] for k in e.kinds if k in _KIND_MAP}))
        node = rdflib.URIRef(e.iri)
        concepts.append(SourceConcept(
            key=e.iri,
            origin=ORIGIN_ONTOLOGY,
            name=e.local_name,
            kinds=kinds,
            label=_literal(g, node, RDFS.label, SKOS.prefLabel),
            definition=_literal(g, node, RDFS.comment, SKOS.definition),
            iri=e.iri,
            namespace=e.namespace,
            family=e.family,
        ))
    return concepts, borrowed


def _individuals(report: OntologyReport, inventory: OntologyInventory,
                 include: bool) -> tuple[list[SourceConcept], list[str], int]:
    """Named individuals NOT already in the inventory (punned ones come in as class/property)."""
    g = report.graph
    in_inventory = {e.iri for e in inventory.entries}
    individuals = sorted({s for s in g.subjects(RDF.type, OWL.NamedIndividual)
                          if isinstance(s, rdflib.URIRef)}, key=str)
    extra = [i for i in individuals if str(i) not in in_inventory and not is_borrowed(str(i))]
    if not include:
        return [], [_local(i) for i in extra], len(individuals)
    concepts = []
    for ind in extra:
        types = sorted(_local(t) for t in g.objects(ind, RDF.type)
                       if isinstance(t, rdflib.URIRef) and t != OWL.NamedIndividual)
        concepts.append(SourceConcept(
            key=str(ind),
            origin=ORIGIN_ONTOLOGY,
            name=_local(ind),
            kinds=(ALIGN_INDIVIDUAL,),
            label=_literal(g, ind, RDFS.label, SKOS.prefLabel),
            definition=_literal(g, ind, RDFS.comment, SKOS.definition),
            iri=str(ind),
            namespace=_namespace(ind),
            family=types[0] if types else None,
        ))
    return concepts, [], len(individuals)


def default_proposals_path() -> Path:
    """Where S3-T05 writes the AI proposals (``outputs.dir`` in config.yaml, else data/output)."""
    from src.config import load_config
    out_dir = load_config().get("outputs", {}).get("dir", "data/output")
    return Path(out_dir) / PROPOSALS_FILE


def load_ai_concepts(proposals) -> list[SourceConcept]:
    """Turn the AI proposals (a path or the already-parsed dict) into source concepts."""
    if isinstance(proposals, (str, Path)):
        data = json.loads(Path(proposals).read_text(encoding="utf-8"))
    else:
        data = proposals or {}
    concepts, seen = [], set()
    for raw in data.get("concepts", []):
        name = str(raw.get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        ctype = str(raw.get("type") or "class").strip().lower()
        kind = ALIGN_PROPERTY if ctype == "property" else ALIGN_CLASS
        articles = []
        for a in raw.get("articles", []) or []:
            try:
                articles.append(int(a))
            except (TypeError, ValueError):
                continue
        concepts.append(SourceConcept(
            key=f"ai:{name}",
            origin=ORIGIN_AI,
            name=name,
            kinds=(kind,),
            label=(str(raw.get("label") or "").strip() or None),
            definition=(str(raw.get("definition") or "").strip() or None),
            articles=articles,
        ))
    return concepts


def build_alignment_sources(report: OntologyReport, proposals=None,
                            inventory: OntologyInventory | None = None,
                            include_individuals: bool = False) -> AlignmentSources:
    """Gather OntoPriv entities (flow A) and the AI proposals (flow B) into one list.

    ``proposals`` may be a path, an already-parsed dict, or None (use the default output path;
    if that file does not exist yet, flow B is simply empty and ``proposals_found`` is False).
    """
    if report.graph is None:
        raise ValueError("El reporte de la ontologia no trae grafo cargado.")
    if inventory is None:
        inventory = build_inventory(report)

    onto, borrowed = _ontology_concepts(report, inventory)
    inds, skipped, n_inds = _individuals(report, inventory, include_individuals)

    proposals_path: str | None = None
    ai: list[SourceConcept] = []
    found = False
    if proposals is None:
        default = default_proposals_path()
        if default.exists():
            proposals = default
    if isinstance(proposals, (str, Path)):
        proposals_path = str(proposals)
        if Path(proposals).exists():
            ai = load_ai_concepts(proposals)
            found = True
    elif isinstance(proposals, dict):
        ai = load_ai_concepts(proposals)
        found = True

    return AlignmentSources(
        ontology_path=report.path,
        proposals_path=proposals_path,
        concepts=onto + inds + ai,
        borrowed=borrowed,
        skipped_individuals=skipped,
        individuals_total=n_inds,
        proposals_found=found,
    )


def sources_summary(sources: AlignmentSources) -> dict:
    """Serializable summary (safe for JSON, the CLI and the web)."""
    return {
        "ontology_path": sources.ontology_path,
        "proposals_path": sources.proposals_path,
        "counts": sources.counts(),
        "kinds_combinations": {"+".join(k): n for k, n in Counter(
            c.kinds for c in sources.of_origin(ORIGIN_ONTOLOGY)).most_common()},
        "borrowed": sources.borrowed,
        "skipped_individuals": sources.skipped_individuals,
        "concepts": [c.as_dict() for c in sources.concepts],
    }


def render_console(sources: AlignmentSources) -> str:
    c = sources.counts()
    lines = [
        "Conceptos a alinear con el DPV:",
        f"  Ontologia: {sources.ontology_path}",
        (f"  Flujo A (OntoPriv): {c['ontology']} entidades "
         f"({c['ontology_classes']} con tipo clase, {c['ontology_properties']} con tipo propiedad, "
         f"{c['ontology_class_and_property']} son ambas)"),
    ]
    if c["ontology_individuals"]:
        lines.append(f"    + {c['ontology_individuals']} individuos incluidos")
    if sources.proposals_found:
        lines.append(f"  Flujo B (propuestos por la IA): {c['ai']} conceptos "
                     f"({c['ai_classes']} clases, {c['ai_properties']} propiedades)")
    else:
        lines.append("  Flujo B (propuestos por la IA): no se encontro el archivo de propuestas "
                     f"({PROPOSALS_FILE}); corre primero la extraccion del Sprint 3.")
    lines.append(f"  Total a alinear: {c['total']}")
    lines.append(f"  Terminos prestados de vocabularios estandar (no se alinean): {c['borrowed']}")
    if c["individuals_skipped"]:
        ejemplos = ", ".join(sources.skipped_individuals[:3])
        lines.append(f"  Individuos de ejemplo no alineados: {c['individuals_skipped']} "
                     f"de {c['individuals_total']} (p. ej. {ejemplos}); se alinea su clase")
    return "\n".join(lines)


def _run() -> None:
    """Demo: gather the sources from the base ontology in config.yaml and the AI proposals."""
    from src.config import load_config
    from src.ingest.ontology_loader import load_ontology

    onto_path = load_config().get("inputs", {}).get("ontology", "data/input/ontopriv.rdf")
    if not Path(onto_path).exists():
        print(f"No se encontro la ontologia base en '{onto_path}'.")
        return
    print(render_console(build_alignment_sources(load_ontology(onto_path))))


if __name__ == "__main__":
    _run()