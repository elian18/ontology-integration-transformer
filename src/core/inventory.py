"""Per-entity inventory of a loaded ontology: the input the modular split needs.

Sprint 1's ``load_ontology`` reports TOTALS (how many classes/properties, the OWL flavor).
The modular core (Sprint 3) needs a finer breakdown: for every named class and property,
which top-level family it belongs to, where it sits in the class hierarchy, and which
namespace it comes from. Those three signals are what the core/profile split (S3-T03) and
the DPV-proximity scoring (S3-T02) consume.

This module builds that breakdown on top of the graph already parsed by ``load_ontology``;
it never re-reads the file. It is generic: it works for any ontology that ``load_ontology``
accepts, not only OntoPriv. Anonymous (blank-node) classes and restrictions are counted for
parity with Sprint 1 but are not partitionable, so no per-entity row is emitted for them.

Family rule (works for one-root and many-root ontologies alike):
- If the ontology has a single named root class, its direct subclasses are the families, and
  every class maps to whichever of them is its ancestor (a direct child maps to itself; the
  root itself maps to "(raiz)").
- If there are several roots, the roots themselves are the families.
A property's family is taken from its ``rdfs:domain`` class when it has one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, Counter

import rdflib
from rdflib.namespace import RDF, RDFS, OWL

from src.ingest.ontology_loader import OntologyReport

# Labels used both in the entity rows and in the Spanish console output.
KIND_CLASS = "clase"
KIND_OBJECT_PROP = "propiedad_objeto"
KIND_DATA_PROP = "propiedad_datos"

_NO_FAMILY = "(sin familia)"
_ROOT_FAMILY = "(raiz)"
_NO_DOMAIN = "(sin dominio)"


@dataclass
class EntityEntry:
    """One named class or property, with the signals the modular split needs."""
    iri: str
    local_name: str
    namespace: str
    kinds: tuple[str, ...]          # any of clase / propiedad_objeto / propiedad_datos (>1 = punning)
    family: str                     # top-level grouping (local name) or a "(...)" marker
    parent: str | None = None       # direct named super-class / super-property (local name)
    depth: int | None = None        # class only: edges from its root (root = 0)
    domain: str | None = None       # property only: rdfs:domain local name
    range: str | None = None        # property only: rdfs:range local name

    def as_dict(self) -> dict:
        return {
            "iri": self.iri,
            "local_name": self.local_name,
            "namespace": self.namespace,
            "kinds": list(self.kinds),
            "family": self.family,
            "parent": self.parent,
            "depth": self.depth,
            "domain": self.domain,
            "range": self.range,
        }


@dataclass
class OntologyInventory:
    """Full per-entity inventory plus reconciliation totals."""
    path: str
    source_format: str
    entries: list[EntityEntry] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)          # local names of root classes
    families: list[str] = field(default_factory=list)       # local names of the family anchors
    max_depth: int = 0
    # Totals computed exactly as Sprint 1 does (named + anonymous), for parity checking.
    totals: dict = field(default_factory=dict)

    def entries_of_kind(self, kind: str) -> list[EntityEntry]:
        return [e for e in self.entries if kind in e.kinds]


def _local(uri) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[1]
    return s.rsplit("/", 1)[-1]


def _namespace(uri) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[0] + "#"
    return s.rsplit("/", 1)[0] + "/"


def _first_named(values):
    for v in values:
        if isinstance(v, rdflib.URIRef):
            return v
    return None


def build_inventory(report: OntologyReport) -> OntologyInventory:
    """Build the per-entity inventory from an already-loaded ontology report."""
    if report.graph is None:
        raise ValueError("El reporte no trae grafo cargado; llama a load_ontology primero.")
    g = report.graph

    # Same sets Sprint 1 uses, so the totals reconcile exactly.
    classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    individuals = set(g.subjects(RDF.type, OWL.NamedIndividual))

    # Only named (URIRef) entities are partitionable; anonymous nodes are counted, not listed.
    named_classes = {c for c in classes if isinstance(c, rdflib.URIRef)}

    # Named subclass edges only (skip restrictions / anonymous class expressions).
    parent_of: dict = defaultdict(set)
    children_of: dict = defaultdict(list)
    for sub, sup in g.subject_objects(RDFS.subClassOf):
        if isinstance(sub, rdflib.URIRef) and isinstance(sup, rdflib.URIRef):
            parent_of[sub].add(sup)
            children_of[sup].append(sub)

    roots = sorted((c for c in named_classes if not parent_of.get(c)), key=_local)
    single_root = roots[0] if len(roots) == 1 else None
    if single_root is not None:
        family_anchors = set(children_of.get(single_root, []))
    else:
        family_anchors = set(roots)

    def family_of_class(cls):
        if cls == single_root:
            return _ROOT_FAMILY
        node, seen = cls, set()
        while node is not None and node not in seen:
            seen.add(node)
            if node in family_anchors:
                return _local(node)
            parents = parent_of.get(node)
            node = _first_named(parents) if parents else None
        return _NO_FAMILY

    def depth_of_class(cls):
        depth, node, seen = 0, cls, set()
        while node is not None and node not in seen and parent_of.get(node):
            seen.add(node)
            node = _first_named(parent_of.get(node))
            depth += 1
        return depth

    # Kinds per IRI (a single IRI may carry several kinds under OWL Full punning).
    kinds_of: dict = defaultdict(set)
    for c in named_classes:
        kinds_of[c].add(KIND_CLASS)
    for p in obj_props:
        if isinstance(p, rdflib.URIRef):
            kinds_of[p].add(KIND_OBJECT_PROP)
    for p in data_props:
        if isinstance(p, rdflib.URIRef):
            kinds_of[p].add(KIND_DATA_PROP)

    entries: list[EntityEntry] = []
    for iri, kinds in kinds_of.items():
        is_class = KIND_CLASS in kinds
        is_prop = bool(kinds & {KIND_OBJECT_PROP, KIND_DATA_PROP})

        parent_local = None
        depth = None
        family = _NO_FAMILY
        domain_local = None
        range_local = None

        if is_class:
            sup = _first_named(parent_of.get(iri, ()))
            parent_local = _local(sup) if sup is not None else None
            depth = depth_of_class(iri)
            family = family_of_class(iri)
        if is_prop:
            sup_prop = _first_named(g.objects(iri, RDFS.subPropertyOf))
            dom = _first_named(g.objects(iri, RDFS.domain))
            rng = _first_named(g.objects(iri, RDFS.range))
            domain_local = _local(dom) if dom is not None else None
            range_local = _local(rng) if rng is not None else None
            # Punned class+property keeps its class parent/family; pure property uses domain.
            if not is_class:
                parent_local = _local(sup_prop) if sup_prop is not None else None
                family = family_of_class(dom) if dom is not None else _NO_DOMAIN

        entries.append(EntityEntry(
            iri=str(iri),
            local_name=_local(iri),
            namespace=_namespace(iri),
            kinds=tuple(sorted(kinds)),
            family=family,
            parent=parent_local,
            depth=depth,
            domain=domain_local,
            range=range_local,
        ))

    entries.sort(key=lambda e: (e.family, e.local_name))
    max_depth = max((e.depth for e in entries if e.depth is not None), default=0)

    return OntologyInventory(
        path=report.path,
        source_format=report.source_format,
        entries=entries,
        roots=[_local(r) for r in roots],
        families=sorted(_local(a) for a in family_anchors),
        max_depth=max_depth,
        totals={
            "triples": report.n_triples,
            "classes": len(classes),
            "object_properties": len(obj_props),
            "data_properties": len(data_props),
            "individuals": len(individuals),
            "named_entities": len(entries),
        },
    )


def inventory_summary(inv: OntologyInventory) -> dict:
    """Serializable summary (safe for JSON, the CLI and the web)."""
    by_ns: dict = defaultdict(lambda: Counter())
    for e in inv.entries:
        for k in e.kinds:
            by_ns[e.namespace][k] += 1
    by_family = Counter(e.family for e in inv.entries if KIND_CLASS in e.kinds)
    return {
        "path": inv.path,
        "source_format": inv.source_format,
        "totals": inv.totals,
        "roots": inv.roots,
        "families": inv.families,
        "max_depth": inv.max_depth,
        "by_namespace": {ns: dict(c) for ns, c in by_ns.items()},
        "classes_by_family": dict(by_family.most_common()),
        "entries": [e.as_dict() for e in inv.entries],
    }


def render_console(inv: OntologyInventory) -> str:
    """Human-readable Spanish summary for the console (not the full per-entity list)."""
    t = inv.totals
    lines = [
        f"Inventario de la ontologia: {inv.path}",
        f"  Formato de origen: {inv.source_format}",
        (f"  Entidades nombradas: {t['named_entities']} "
         f"(clases {t['classes']}, prop. objeto {t['object_properties']}, "
         f"prop. datos {t['data_properties']}, individuos {t['individuals']})"),
        f"  Raices: {', '.join(inv.roots) or '(ninguna)'}",
        f"  Profundidad maxima de la jerarquia: {inv.max_depth}",
        f"  Familias de primer nivel: {len(inv.families)}",
    ]
    by_family = Counter(e.family for e in inv.entries if KIND_CLASS in e.kinds)
    for fam, n in by_family.most_common():
        lines.append(f"    - {fam}: {n} clases")
    by_ns = defaultdict(lambda: Counter())
    for e in inv.entries:
        for k in e.kinds:
            by_ns[e.namespace][k] += 1
    lines.append("  Por namespace:")
    for ns, c in by_ns.items():
        lines.append(f"    - {ns}: {c[KIND_CLASS]} clases, "
                     f"{c[KIND_OBJECT_PROP]} prop. objeto, {c[KIND_DATA_PROP]} prop. datos")
    return "\n".join(lines)


def _run() -> None:
    """Demo entry point: inventory the base ontology named in config.yaml.

    Reads the path through src.config so this module depends only on src/ (app/ adapts
    src/, never the other way around)."""
    from pathlib import Path
    from src.config import load_config
    from src.ingest.ontology_loader import load_ontology

    path = load_config().get("inputs", {}).get("ontology", "data/input/ontopriv.rdf")
    if not Path(path).exists():
        print(f"No se encontro la ontologia base en '{path}' "
              f"(revisa 'inputs.ontology' en config.yaml).")
        return
    inv = build_inventory(load_ontology(path))
    print(render_console(inv))


if __name__ == "__main__":
    _run()