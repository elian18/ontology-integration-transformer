"""Assign every OntoPriv entity to the reusable core or the jurisdiction profile.

This realizes the split agreed with the design. The primary axis is the conceptual role of
each top-level family (structural and stable); the DPV proximity from S3-T02 is advisory only
(it flags entities whose score contradicts their assignment, for human review). The split
keeps the core STANDALONE: the profile may reference the core (it will owl:import it in
S3-T04), but the core must not reference the profile, or it stops being reusable for a law
that has no ontology of its own. Core -> profile references are detected and reported.

No files are written here (that is S3-T04). This module only decides and explains, and every
assignment is a proposal the human can override later in the web (T06/T08).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, defaultdict

import rdflib
from rdflib.namespace import RDFS

from src.ingest.ontology_loader import OntologyReport
from src.core.inventory import (
    build_inventory, OntologyInventory, KIND_CLASS,
)
from src.core.dpv_proximity import DpvProximityReport

CORE = "core"
PROFILE = "profile"

# Confirmed with the design (see the sprint discussion).
CORE_FAMILIES = {
    "Members_of_the_personal_data_protection_system",
    "Principles",
    "Rights",
    "Processing",
    "Terminology",
}
PROFILE_FAMILIES = {
    "Verification",
    "Personal_data_security",
    "Transfer_or_communication",
    "Sanctions",
    "Violations",
    "Corrective_measures",
    "Collection",
}

_ROOT_FAMILY = "(raiz)"                 # matches src.core.inventory
_VERIFICATION_HINT = "verification"     # franc-namespace verification sub-branches -> profile
_HARD_PREDICATES = {RDFS.subClassOf, RDFS.subPropertyOf, RDFS.domain, RDFS.range}

# Advisory DPV thresholds (only to raise review flags, never to decide the split).
_DPV_HIGH = 0.75
_DPV_LOW = 0.35


@dataclass
class Assignment:
    iri: str
    local_name: str
    namespace: str
    kinds: tuple[str, ...]
    family: str
    module: str                         # core | profile
    reason: str
    dpv_score: float | None = None
    flags: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "iri": self.iri,
            "local_name": self.local_name,
            "namespace": self.namespace,
            "kinds": list(self.kinds),
            "family": self.family,
            "module": self.module,
            "reason": self.reason,
            "dpv_score": None if self.dpv_score is None else round(self.dpv_score, 4),
            "flags": list(self.flags),
        }


@dataclass
class ModularSplit:
    onto_path: str
    assignments: list[Assignment] = field(default_factory=list)
    cross_refs: list[dict] = field(default_factory=list)   # core -> profile hard references

    def module(self, name: str) -> list[Assignment]:
        return [a for a in self.assignments if a.module == name]

    def flagged(self) -> list[Assignment]:
        return [a for a in self.assignments if a.flags]

    def counts(self) -> dict:
        out: dict = {}
        for name in (CORE, PROFILE):
            group = self.module(name)
            out[name] = {
                "total": len(group),
                "classes": sum(1 for a in group if KIND_CLASS in a.kinds),
                "properties": sum(1 for a in group if KIND_CLASS not in a.kinds),
            }
        return out


def _local(uri) -> str:
    s = str(uri)
    return s.rsplit("#", 1)[1] if "#" in s else s.rsplit("/", 1)[-1]


def _first_named(values):
    for v in values:
        if isinstance(v, rdflib.URIRef):
            return v
    return None


def assign_modules(report: OntologyReport,
                   inventory: OntologyInventory | None = None,
                   proximity: DpvProximityReport | None = None) -> ModularSplit:
    """Assign every named entity to the core or the profile, and report cross-references."""
    if report.graph is None:
        raise ValueError("El reporte de la ontologia no trae grafo cargado.")
    g = report.graph
    if inventory is None:
        inventory = build_inventory(report)

    # Named subclass parent map, to walk ancestors for the verification refinement.
    parent_of: dict = defaultdict(set)
    for sub, sup in g.subject_objects(RDFS.subClassOf):
        if isinstance(sub, rdflib.URIRef) and isinstance(sup, rdflib.URIRef):
            parent_of[sub].add(sup)

    def descends_from_verification(iri) -> bool:
        node, seen = iri, set()
        while node is not None and node not in seen:
            seen.add(node)
            if _VERIFICATION_HINT in _local(node).lower():
                return True
            node = _first_named(parent_of.get(node, ()))
        return False

    score_of = {e.iri: e.score for e in proximity.entries} if proximity else {}

    # Pass 1: classes (a punned class+property is decided here, as a class).
    class_module: dict = {}
    assignments: list[Assignment] = []
    by_iri = {rdflib.URIRef(e.iri): e for e in inventory.entries}

    for iri, entry in by_iri.items():
        if KIND_CLASS not in entry.kinds:
            continue
        fam = entry.family
        if fam in PROFILE_FAMILIES:
            module, reason = PROFILE, "familia especifica de la jurisdiccion"
        elif fam in CORE_FAMILIES:
            if descends_from_verification(iri):
                module, reason = PROFILE, "rama de verificacion (perfil LOPDP)"
            else:
                module, reason = CORE, "familia general (nivel DPV/GDPR)"
        elif fam == _ROOT_FAMILY:
            module, reason = CORE, "raiz comun (ancla del nucleo)"
        else:
            module, reason = CORE, "sin familia; nucleo por defecto"
        class_module[str(iri)] = module
        assignments.append(Assignment(
            iri=str(iri), local_name=entry.local_name, namespace=entry.namespace,
            kinds=entry.kinds, family=fam, module=module, reason=reason,
        ))

    # Pass 2: pure properties (no class kind) follow their rdfs:domain when it is a class.
    for iri, entry in by_iri.items():
        if KIND_CLASS in entry.kinds:
            continue
        dom = _first_named(g.objects(iri, RDFS.domain))
        if dom is not None and str(dom) in class_module:
            module = class_module[str(dom)]
            reason = f"por su dominio {_local(dom)}"
            flags: tuple[str, ...] = ()
        else:
            fam = entry.family
            if fam in PROFILE_FAMILIES:
                module, reason, flags = PROFILE, "familia del dominio (perfil)", ()
            elif fam in CORE_FAMILIES:
                module, reason, flags = CORE, "familia del dominio (nucleo)", ()
            else:
                module, reason, flags = CORE, "propiedad sin dominio; nucleo por defecto", \
                    ("revisar: propiedad sin dominio",)
        assignments.append(Assignment(
            iri=str(iri), local_name=entry.local_name, namespace=entry.namespace,
            kinds=entry.kinds, family=entry.family, module=module, reason=reason, flags=flags,
        ))

    module_of = {a.iri: a.module for a in assignments}

    # Advisory DPV flags + the "sin familia" review note.
    for a in assignments:
        extra: list[str] = list(a.flags)
        a.dpv_score = score_of.get(a.iri)
        if a.family in ("(sin familia)",):
            extra.append("revisar: sin familia (nucleo por defecto)")
        if a.dpv_score is not None:
            if a.module == PROFILE and a.dpv_score >= _DPV_HIGH:
                extra.append(f"revisar: cercania DPV alta ({a.dpv_score:.2f}) para un perfil")
            elif a.module == CORE and a.dpv_score < _DPV_LOW:
                extra.append(f"revisar: cercania DPV baja ({a.dpv_score:.2f}) para el nucleo")
        a.flags = tuple(extra)

    # Cross-references: a standalone core must not point INTO the profile.
    cross_refs: list[dict] = []
    for s, p, o in g:
        if (p in _HARD_PREDICATES and isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef)
                and module_of.get(str(s)) == CORE and module_of.get(str(o)) == PROFILE):
            cross_refs.append({"subject": _local(s), "predicate": _local(p), "object": _local(o)})

    return ModularSplit(onto_path=report.path, assignments=assignments, cross_refs=cross_refs)


def split_summary(split: ModularSplit) -> dict:
    """Serializable summary (safe for JSON, the CLI and the web)."""
    core_fams = Counter(a.family for a in split.module(CORE) if KIND_CLASS in a.kinds)
    prof_fams = Counter(a.family for a in split.module(PROFILE) if KIND_CLASS in a.kinds)
    return {
        "onto_path": split.onto_path,
        "counts": split.counts(),
        "core_families": dict(core_fams.most_common()),
        "profile_families": dict(prof_fams.most_common()),
        "n_flagged": len(split.flagged()),
        "n_cross_refs": len(split.cross_refs),
        "cross_refs": split.cross_refs,
        "assignments": [a.as_dict() for a in split.assignments],
    }


def render_console(split: ModularSplit) -> str:
    c = split.counts()
    lines = [
        f"Division nucleo/perfil: {split.onto_path}",
        (f"  Nucleo:  {c[CORE]['total']} entidades "
         f"({c[CORE]['classes']} clases, {c[CORE]['properties']} propiedades)"),
        (f"  Perfil:  {c[PROFILE]['total']} entidades "
         f"({c[PROFILE]['classes']} clases, {c[PROFILE]['properties']} propiedades)"),
    ]
    core_fams = Counter(a.family for a in split.module(CORE) if KIND_CLASS in a.kinds)
    prof_fams = Counter(a.family for a in split.module(PROFILE) if KIND_CLASS in a.kinds)
    lines.append("  Familias en el nucleo:")
    for fam, n in core_fams.most_common():
        lines.append(f"    - {fam}: {n} clases")
    lines.append("  Familias en el perfil:")
    for fam, n in prof_fams.most_common():
        lines.append(f"    - {fam}: {n} clases")
    lines.append(f"  Referencias nucleo->perfil (rompen la autonomia del nucleo): {len(split.cross_refs)}")
    for ref in split.cross_refs[:10]:
        lines.append(f"    - {ref['subject']} --{ref['predicate']}--> {ref['object']}")
    flagged = split.flagged()
    lines.append(f"  Entidades marcadas para revision: {len(flagged)}")
    for a in flagged[:10]:
        lines.append(f"    - {a.local_name} [{a.module}]: {'; '.join(a.flags)}")
    return "\n".join(lines)


def _run() -> None:
    """Demo: split the base ontology named in config.yaml (DPV proximity optional)."""
    from pathlib import Path
    from src.config import load_config
    from src.ingest.ontology_loader import load_ontology

    onto_path = load_config().get("inputs", {}).get("ontology", "data/input/ontopriv.rdf")
    if not Path(onto_path).exists():
        print(f"No se encontro la ontologia base en '{onto_path}'.")
        return
    split = assign_modules(load_ontology(onto_path))
    print(render_console(split))


if __name__ == "__main__":
    _run()