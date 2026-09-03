"""Load a base ontology, report its structure, and characterize its OWL flavor.

Structure is read with rdflib, which parses every accepted format and does not choke on
OWL Full constructs (unlike owlready2, whose Python-class model raises a metaclass conflict
when an entity is declared as several incompatible types at once).

Accepted interchange formats: RDF/XML, Turtle, JSON-LD.
OWL/XML (.owx) is OntoPriv's native format; convert it once to RDF/XML with Protégé, since
rdflib does not parse OWL/XML."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

import rdflib
from rdflib.namespace import RDF, RDFS, OWL

_RDFLIB_FORMATS = {".rdf": "xml", ".owl": "xml", ".xml": "xml",
                   ".ttl": "turtle", ".jsonld": "json-ld", ".json": "json-ld"}
_SOURCE_FORMATS = {".rdf": "rdf/xml", ".owl": "rdf/xml", ".xml": "rdf/xml",
                   ".ttl": "turtle", ".jsonld": "json-ld", ".json": "json-ld"}


@dataclass
class OntologyReport:
    path: str
    source_format: str
    sha256: str
    n_triples: int
    n_classes: int
    n_object_props: int
    n_data_props: int
    n_individuals: int
    n_op_dp_overlap: int         # props typed ObjectProperty AND DatatypeProperty (OWL Full)
    n_class_prop_overlap: int    # entities typed Class AND Property (illegal punning, OWL Full)
    ontology_flavor: str         # "owl-full" | "dl-compatible" | "rdfs" | "unknown"
    flavor_detail: str
    graph: rdflib.Graph = field(default=None, repr=False)

    @property
    def loaded(self) -> bool:
        return self.graph is not None

    @property
    def is_owl_full(self) -> bool:
        return self.n_op_dp_overlap > 0 or self.n_class_prop_overlap > 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _parse(path: Path) -> tuple[rdflib.Graph, str]:
    suffix = path.suffix.lower()
    if suffix == ".owx":
        raise ValueError(
            "OWL/XML (.owx) no lo parsea rdflib. Exporta OntoPriv a RDF/XML desde Protégé "
            "y usa ese archivo como semilla."
        )
    if suffix not in _RDFLIB_FORMATS:
        raise ValueError(f"Formato de ontología no soportado: {suffix}")
    graph = rdflib.Graph()
    graph.parse(str(path), format=_RDFLIB_FORMATS[suffix])
    return graph, _SOURCE_FORMATS.get(suffix, "unknown")


def load_ontology(path: str | Path, characterize: bool = True) -> OntologyReport:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Ontología no encontrada: {path}")

    graph, source_format = _parse(path)

    classes = set(graph.subjects(RDF.type, OWL.Class)) | set(graph.subjects(RDF.type, RDFS.Class))
    obj_props = set(graph.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(graph.subjects(RDF.type, OWL.DatatypeProperty))
    individuals = set(graph.subjects(RDF.type, OWL.NamedIndividual))
    props = obj_props | data_props | set(graph.subjects(RDF.type, RDF.Property))

    op_dp_overlap = obj_props & data_props
    class_prop_overlap = classes & props

    if characterize:
        flavor, detail = _characterize(classes, obj_props, data_props,
                                       op_dp_overlap, class_prop_overlap)
    else:
        flavor, detail = "unknown", "Caracterización desactivada."

    return OntologyReport(
        path=str(path),
        source_format=source_format,
        sha256=_sha256(path),
        n_triples=len(graph),
        n_classes=len(classes),
        n_object_props=len(obj_props),
        n_data_props=len(data_props),
        n_individuals=len(individuals),
        n_op_dp_overlap=len(op_dp_overlap),
        n_class_prop_overlap=len(class_prop_overlap),
        ontology_flavor=flavor,
        flavor_detail=detail,
        graph=graph,
    )


def normalize_to_rdfxml(graph: rdflib.Graph, out_path: str | Path) -> Path:
    """Save the parsed graph as canonical RDF/XML (rdflib serializer)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(out_path), format="xml")
    return out_path


def _characterize(classes, obj_props, data_props, op_dp_overlap, class_prop_overlap):
    """Report the OWL flavor. OWL Full can be flagged reliably; strict OWL 2 DL is never
    claimed (that needs a full profile validator we deliberately do not include)."""
    if op_dp_overlap or class_prop_overlap:
        return "owl-full", (
            f"OWL Full: {len(op_dp_overlap)} propiedades objeto+datos y "
            f"{len(class_prop_overlap)} entidades clase+propiedad (punning)."
        )
    if obj_props or data_props:
        return "dl-compatible", "Sin señales de OWL Full (candidata a OWL 2 DL, no confirmada)."
    if classes:
        return "rdfs", "Solo clases/jerarquías RDFS, sin propiedades OWL."
    return "unknown", "No se detectaron constructos OWL característicos."



def characterization_summary(report: "OntologyReport") -> dict:
    """Reusable, serializable characterization of an ontology.

    Returns a plain dict (safe for JSON, the CLI and the web) with the OWL flavor,
    its evidence, and the structural counts. Does not confirm the OWL 2 DL profile."""
    return {
        "path": report.path,
        "source_format": report.source_format,
        "flavor": report.ontology_flavor,
        "flavor_detail": report.flavor_detail,
        "is_owl_full": report.is_owl_full,
        "structure": {
            "triples": report.n_triples,
            "classes": report.n_classes,
            "object_properties": report.n_object_props,
            "data_properties": report.n_data_props,
            "individuals": report.n_individuals,
        },
        "owl_full_signals": {
            "object_and_data_props": report.n_op_dp_overlap,
            "class_and_property": report.n_class_prop_overlap,
        },
    }