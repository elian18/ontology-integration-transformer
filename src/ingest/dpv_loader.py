"""Load the DPV (Data Privacy Vocabulary) into memory as the alignment reference.

The DPV is an RDF vocabulary (SKOS/RDFS), so rdflib reads it directly. In Sprint 5 its
concepts are matched against OntoPriv; here we only load it and report its size."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL, SKOS


@dataclass
class DpvReport:
    path: str
    n_triples: int
    n_concepts: int
    n_classes: int
    n_properties: int
    n_labeled: int
    graph: Graph = field(default=None, repr=False)

    @property
    def loaded(self) -> bool:
        return self.graph is not None and self.n_triples > 0


def load_dpv(path: str | Path, fmt: str | None = None) -> DpvReport:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"DPV no encontrado: {path}. Descárgalo a vocab/ (tarea S1-T05)."
        )
    if fmt is None:
        fmt = {".ttl": "turtle", ".rdf": "xml", ".owl": "xml",
               ".jsonld": "json-ld", ".json": "json-ld", ".n3": "n3"}.get(path.suffix.lower(), "turtle")

    graph = Graph()
    graph.parse(str(path), format=fmt)

    concepts = set(graph.subjects(RDF.type, SKOS.Concept))
    classes = set(graph.subjects(RDF.type, OWL.Class)) | set(graph.subjects(RDF.type, RDFS.Class))
    properties = (set(graph.subjects(RDF.type, RDF.Property))
                  | set(graph.subjects(RDF.type, OWL.ObjectProperty))
                  | set(graph.subjects(RDF.type, OWL.DatatypeProperty)))
    labeled = set(graph.subjects(SKOS.prefLabel, None)) | set(graph.subjects(RDFS.label, None))

    return DpvReport(
        path=str(path),
        n_triples=len(graph),
        n_concepts=len(concepts),
        n_classes=len(classes),
        n_properties=len(properties),
        n_labeled=len(labeled),
        graph=graph,
    )