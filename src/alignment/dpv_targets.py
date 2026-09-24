"""Build the TARGET side of the alignment: the DPV concepts and properties to compare against.

The source side (OntoPriv entities + AI proposals) is S4-T02. This module prepares the DPV
side in the same uniform shape, reusing Sprint 3's ``collect_dpv_concepts`` unchanged (its
signature and behaviour are kept; the extra filtering happens here).

Decisions encoded here (measured on the public DPV 2.3 ``dpv.ttl``):
- ``collect_dpv_concepts`` returns every subject with a ``skos:prefLabel`` (1123). Seven of them
  are NOT DPV terms but terms of other vocabularies that the file embeds (``dct:accessRights``,
  ``dcat:Resource``, ``foaf:page``...). Only the DPV namespace is kept (1116); the rest are
  reported as external.
- The DPV has two kinds of terms: properties (``rdf:Property``, e.g. ``dpv:hasRecipient``; 144)
  and concepts (every other term, all ``rdfs:Class`` + ``skos:Concept``; 972). OntoPriv classes
  and individuals are compared with DPV concepts; OntoPriv properties with DPV properties.
- Each target keeps its direct DPV parents (``skos:broader`` / ``rdfs:subClassOf`` /
  ``rdfs:subPropertyOf``): the AI uses them later (S4-T08) to tell a broader match from an
  exact one.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import rdflib
from rdflib.namespace import RDF, RDFS, OWL, SKOS

from src.ingest.dpv_loader import DpvReport
from src.core.dpv_proximity import collect_dpv_concepts, _humanize, _trim

DPV_NAMESPACE = "https://w3id.org/dpv#"

TARGET_CLASS = "class"
TARGET_PROPERTY = "property"

# Which DPV target kinds each source kind (S4-T02 vocabulary) may be compared with.
COMPATIBLE_TARGETS = {
    "class": {TARGET_CLASS},
    "individual": {TARGET_CLASS},
    "property": {TARGET_PROPERTY},
}

_PROPERTY_TYPES = (RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty)
_PARENT_PREDICATES = (SKOS.broader, RDFS.subClassOf, RDFS.subPropertyOf)


@dataclass
class DpvTarget:
    """One DPV term to compare against."""
    iri: str
    name: str                       # local name, e.g. 'SensitivePersonalData'
    kind: str                       # class | property
    label: str                      # skos:prefLabel (English)
    definition: str | None = None   # skos:definition (English)
    parents: list[str] = field(default_factory=list)   # direct DPV parents (local names)

    def name_text(self) -> str:
        """Label, or the humanized local name: input of the lexical match."""
        return self.label or _humanize(self.name)

    def semantic_text(self) -> str:
        """Label + trimmed definition: input of the embeddings."""
        if self.definition:
            return f"{self.name_text()}. {_trim(self.definition)}"
        return self.name_text()

    def as_dict(self) -> dict:
        return {"iri": self.iri, "name": self.name, "kind": self.kind, "label": self.label,
                "definition": self.definition, "parents": self.parents}


@dataclass
class DpvTargets:
    """Every DPV target plus what was left out."""
    dpv_path: str
    targets: list[DpvTarget] = field(default_factory=list)
    external: list[str] = field(default_factory=list)   # labelled terms of other vocabularies

    def of_kind(self, kind: str) -> list[DpvTarget]:
        return [t for t in self.targets if t.kind == kind]

    def compatible_with(self, source_kinds) -> list[DpvTarget]:
        """Targets a source with these kinds may be compared with (punned = union)."""
        allowed: set[str] = set()
        for k in source_kinds:
            allowed |= COMPATIBLE_TARGETS.get(k, set())
        return [t for t in self.targets if t.kind in allowed]

    def counts(self) -> dict:
        return {
            "total": len(self.targets),
            "classes": len(self.of_kind(TARGET_CLASS)),
            "properties": len(self.of_kind(TARGET_PROPERTY)),
            "with_definition": sum(1 for t in self.targets if t.definition),
            "external_excluded": len(self.external),
        }


def _local(uri) -> str:
    s = str(uri)
    return s.rsplit("#", 1)[1] if "#" in s else s.rsplit("/", 1)[-1]


def _literal(graph: rdflib.Graph, subject, predicate) -> str | None:
    for obj in graph.objects(subject, predicate):
        if isinstance(obj, rdflib.Literal) and str(obj).strip():
            return str(obj).strip()
    return None


def _kind(graph: rdflib.Graph, node) -> str:
    if any((node, RDF.type, t) in graph for t in _PROPERTY_TYPES):
        return TARGET_PROPERTY
    return TARGET_CLASS


def _parents(graph: rdflib.Graph, node, namespace: str) -> list[str]:
    found = []
    for predicate in _PARENT_PREDICATES:
        for obj in graph.objects(node, predicate):
            if isinstance(obj, rdflib.URIRef) and str(obj).startswith(namespace):
                name = _local(obj)
                if name not in found:
                    found.append(name)
    return found


def build_dpv_targets(dpv: DpvReport, namespace: str = DPV_NAMESPACE) -> DpvTargets:
    """Turn the loaded DPV into comparable targets (DPV namespace only, kind marked)."""
    iris, _texts = collect_dpv_concepts(dpv)          # Sprint 3, reused as is
    g = dpv.graph
    targets, external = [], []
    for iri in iris:
        if not iri.startswith(namespace):
            external.append(iri)
            continue
        node = rdflib.URIRef(iri)
        targets.append(DpvTarget(
            iri=iri,
            name=_local(iri),
            kind=_kind(g, node),
            label=_literal(g, node, SKOS.prefLabel) or _literal(g, node, RDFS.label) or "",
            definition=_literal(g, node, SKOS.definition) or _literal(g, node, RDFS.comment),
            parents=_parents(g, node, namespace),
        ))
    return DpvTargets(dpv_path=dpv.path, targets=targets, external=external)


def targets_summary(targets: DpvTargets) -> dict:
    """Serializable summary (safe for JSON, the CLI and the web)."""
    return {
        "dpv_path": targets.dpv_path,
        "counts": targets.counts(),
        "external": targets.external,
        "targets": [t.as_dict() for t in targets.targets],
    }


def render_console(targets: DpvTargets) -> str:
    c = targets.counts()
    lines = [
        "Terminos del DPV para la alineacion:",
        f"  DPV: {targets.dpv_path}",
        f"  Terminos propios del DPV: {c['total']} "
        f"({c['classes']} conceptos, {c['properties']} propiedades)",
        f"  Con definicion: {c['with_definition']}",
        f"  Terminos de otros vocabularios excluidos: {c['external_excluded']}",
    ]
    for iri in targets.external[:10]:
        lines.append(f"    - {iri}")
    top_parents = Counter(p for t in targets.targets for p in t.parents).most_common(5)
    if top_parents:
        lines.append("  Conceptos padre mas frecuentes: "
                     + ", ".join(f"{name} ({n})" for name, n in top_parents))
    return "\n".join(lines)


def _run() -> None:
    """Demo: prepare the DPV named in config.yaml as alignment targets."""
    from pathlib import Path
    from src.config import load_config
    from src.ingest.dpv_loader import load_dpv

    dpv_path = load_config().get("inputs", {}).get("dpv", "vocab/dpv.ttl")
    if not Path(dpv_path).exists():
        print(f"No se encontro el DPV en '{dpv_path}' (revisa 'inputs.dpv' en config.yaml).")
        return
    print(render_console(build_dpv_targets(load_dpv(dpv_path))))


if __name__ == "__main__":
    _run()