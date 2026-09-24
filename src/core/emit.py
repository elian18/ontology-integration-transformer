"""Materialize the split into two standalone RDF/XML files: the core and the profile.

Takes the assignment from S3-T03 and writes:
- ``ontopriv-core.rdf``      : the reusable, jurisdiction-neutral core (must be self-contained).
- ``profile-ecuador-lopdp.rdf``: the Ecuador (LOPDP) profile, which ``owl:imports`` the core.
- ``core-manifest.json``     : a small manifest with IRIs, counts and what was moved.

Before writing, the core is made STANDALONE: any core entity that still points into the
profile (a core -> profile hard reference from S3-T03) is moved to the profile. Moving down is
always safe because the profile imports the core, so its references to core entities resolve.
The move is applied to a fixpoint (moving one entity can expose another) and every moved entity
is reported, so the human can review what left the core.

Each named class, property and individual ends up in exactly one file; the profile file is the
only one that references the core. Files are RDF/XML (the Sprint 3 output format); Turtle and
JSON-LD come later, in the transformation sprints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import rdflib
from rdflib import Graph, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL

from src.ingest.ontology_loader import OntologyReport
from src.core.split import ModularSplit, assign_modules, CORE, PROFILE, _HARD_PREDICATES

CORE_IRI = "http://www.semanticweb.org/ontopriv-core"
PROFILE_IRI = "http://www.semanticweb.org/profiles/ecuador-lopdp"
CORE_FILE = "ontopriv-core.rdf"
PROFILE_FILE = "profile-ecuador-lopdp.rdf"
MANIFEST_FILE = "core-manifest.json"


@dataclass
class MaterializeResult:
    core_path: str
    profile_path: str
    manifest_path: str
    core_iri: str
    profile_iri: str
    moved_to_profile: list[str] = field(default_factory=list)   # local names moved for autonomy
    counts: dict = field(default_factory=dict)
    dangling_core_refs: list[dict] = field(default_factory=list)  # should be empty after cleaning


def _local(uri) -> str:
    s = str(uri)
    return s.rsplit("#", 1)[1] if "#" in s else s.rsplit("/", 1)[-1]


def resolve_core_autonomy(graph: Graph, module_of: dict) -> tuple[dict, list[str]]:
    """Move every core entity that references the profile into the profile, to a fixpoint."""
    module_of = dict(module_of)
    moved: list[str] = []
    changed = True
    while changed:
        changed = False
        for s, p, o in graph:
            if (p in _HARD_PREDICATES and isinstance(s, URIRef) and isinstance(o, URIRef)
                    and module_of.get(str(s)) == CORE and module_of.get(str(o)) == PROFILE):
                module_of[str(s)] = PROFILE
                moved.append(str(s))
                changed = True
    return module_of, moved


def _describe(graph: Graph, seeds: set[str]) -> Graph:
    """Concise bounded description: every triple about each seed, following blank nodes."""
    out = Graph()
    for prefix, ns in graph.namespaces():
        out.bind(prefix, ns)
    seen_bnodes: set = set()

    def add_subject(subj):
        for p, o in graph.predicate_objects(subj):
            out.add((subj, p, o))
            if isinstance(o, BNode) and o not in seen_bnodes:
                seen_bnodes.add(o)
                add_subject(o)

    for iri in seeds:
        add_subject(URIRef(iri))
    return out


def _assign_individuals(graph: Graph, module_of: dict) -> dict:
    """Place each named individual in its type's module (profile wins if types disagree)."""
    module_of = dict(module_of)
    for ind in graph.subjects(RDF.type, OWL.NamedIndividual):
        if not isinstance(ind, URIRef) or str(ind) in module_of:
            continue
        modules = {module_of.get(str(t)) for t in graph.objects(ind, RDF.type)
                   if isinstance(t, URIRef) and str(t) in module_of}
        module_of[str(ind)] = PROFILE if PROFILE in modules else CORE
    return module_of


def _project_namespaces(split: ModularSplit) -> set[str]:
    return {a.namespace for a in split.assignments}


def _dangling_core_refs(core_graph: Graph, project_ns: set[str]) -> list[dict]:
    """Hard references from the core to a project entity not defined in the core file."""
    defined = {str(s) for s in core_graph.subjects()}
    out = []
    for s, p, o in core_graph:
        if (p in _HARD_PREDICATES and isinstance(o, URIRef)
                and any(str(o).startswith(ns) for ns in project_ns)
                and str(o) not in defined):
            out.append({"subject": _local(s), "predicate": _local(p), "object": _local(o)})
    return out


def _new_ontology_header(module_graph: Graph, onto_iri: str, imports: str | None) -> None:
    module_graph.add((URIRef(onto_iri), RDF.type, OWL.Ontology))
    if imports:
        module_graph.add((URIRef(onto_iri), OWL.imports, URIRef(imports)))


def materialize_modules(report: OntologyReport, split: ModularSplit | None = None,
                        out_dir: str | Path = "data/output",
                        core_iri: str = CORE_IRI, profile_iri: str = PROFILE_IRI,
                        auto_move: bool = True) -> MaterializeResult:
    """Write the core and profile RDF/XML files from the split, with a standalone core."""
    if report.graph is None:
        raise ValueError("El reporte de la ontologia no trae grafo cargado.")
    g = report.graph
    if split is None:
        split = assign_modules(report)

    module_of = {a.iri: a.module for a in split.assignments}
    module_of = _assign_individuals(g, module_of)
    moved: list[str] = []
    if auto_move:
        module_of, moved_iris = resolve_core_autonomy(g, module_of)
        moved = sorted(_local(m) for m in dict.fromkeys(moved_iris))

    core_seeds = {iri for iri, m in module_of.items() if m == CORE}
    profile_seeds = {iri for iri, m in module_of.items() if m == PROFILE}

    core_graph = _describe(g, core_seeds)
    profile_graph = _describe(g, profile_seeds)
    _new_ontology_header(core_graph, core_iri, imports=None)
    _new_ontology_header(profile_graph, profile_iri, imports=core_iri)

    project_ns = _project_namespaces(split)
    dangling = _dangling_core_refs(core_graph, project_ns)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core_path = out_dir / CORE_FILE
    profile_path = out_dir / PROFILE_FILE
    core_graph.serialize(destination=str(core_path), format="xml")
    profile_graph.serialize(destination=str(profile_path), format="xml")

    counts = {
        "core": {"seeds": len(core_seeds), "triples": len(core_graph)},
        "profile": {"seeds": len(profile_seeds), "triples": len(profile_graph)},
    }
    manifest = {
        "core_iri": core_iri,
        "profile_iri": profile_iri,
        "core_file": CORE_FILE,
        "profile_file": PROFILE_FILE,
        "profile_imports_core": True,
        "moved_to_profile_for_autonomy": moved,
        "counts": counts,
        "dangling_core_refs": dangling,
    }
    manifest_path = out_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return MaterializeResult(
        core_path=str(core_path), profile_path=str(profile_path), manifest_path=str(manifest_path),
        core_iri=core_iri, profile_iri=profile_iri, moved_to_profile=moved,
        counts=counts, dangling_core_refs=dangling,
    )


def render_console(result: MaterializeResult) -> str:
    c = result.counts
    lines = [
        "Nucleo modular emitido (RDF/XML):",
        f"  Nucleo:  {result.core_path}  ({c['core']['seeds']} entidades, {c['core']['triples']} tripletas)",
        f"  Perfil:  {result.profile_path}  ({c['profile']['seeds']} entidades, {c['profile']['triples']} tripletas)",
        f"  El perfil importa el nucleo (owl:imports {result.core_iri})",
        f"  Movidas al perfil para dejar el nucleo autonomo: {len(result.moved_to_profile)}",
    ]
    for name in result.moved_to_profile[:15]:
        lines.append(f"    - {name}")
    if result.dangling_core_refs:
        lines.append(f"  AVISO: quedan {len(result.dangling_core_refs)} referencias colgadas en el nucleo:")
        for ref in result.dangling_core_refs[:10]:
            lines.append(f"    - {ref['subject']} --{ref['predicate']}--> {ref['object']}")
    else:
        lines.append("  Nucleo autonomo: sin referencias colgadas.")
    lines.append(f"  Manifiesto: {result.manifest_path}")
    return "\n".join(lines)


def _run() -> None:
    """Demo: build the modular core from the base ontology named in config.yaml."""
    from src.config import load_config
    from src.ingest.ontology_loader import load_ontology

    cfg = load_config()
    onto_path = cfg.get("inputs", {}).get("ontology", "data/input/ontopriv.rdf")
    out_dir = cfg.get("outputs", {}).get("dir", "data/output")
    if not Path(onto_path).exists():
        print(f"No se encontro la ontologia base en '{onto_path}'.")
        return
    result = materialize_modules(load_ontology(onto_path), out_dir=out_dir)
    print(render_console(result))


if __name__ == "__main__":
    _run()