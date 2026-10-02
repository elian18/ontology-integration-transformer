"""Human decisions on the AI-proposed concepts (S5-T02): which ones enter the ontology.

Agreed in Sprint 4: before any DPV correspondence is approved, a person decides on every concept
the AI extracted from the law (93 for the LOPDP):
- a concept marked ``new`` is APPROVED (it enters the profile as a new entity, with the type the
  person confirms) or REJECTED (it is discarded);
- a concept marked ``possible_duplicate`` is confirmed as DUPLICATE (it already exists in
  OntoPriv: it is not created, and its law article is later attached to the OntoPriv entity),
  or, if the person sees it is NOT the same thing, it is approved or rejected like a new one.
- a concept marked ``new`` can ALSO be a duplicate the automatic check missed (e.g. the AI named
  it in English and OntoPriv in another wording: PrincipleOfLawfulness vs Juridicity). The person
  then marks it as duplicate and names the OntoPriv entity (``same_as``).

The decision is written to the S5-T01 log; nothing here touches the ontology (that is S5-T06).
A new concept gets the IRI ``<profile IRI>#<Name>``; the profile IRI is a parameter (the Ecuador
profile by default), so a law uploaded in Sprints 6-7 gets its own profile. A concept whose name
is already used by an OntoPriv entity cannot be approved as new (two entities with the same name
would be confusing): it must be confirmed as duplicate or rejected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.core.emit import PROFILE_IRI
from src.alignment.sources import ORIGIN_AI, ORIGIN_ONTOLOGY, ALIGN_PROPERTY
from src.alignment.duplicates import STATUS_DUPLICATE, STATUS_NEW
from src.alignment.decisions import (
    Decision, make_decision, append_decision, latest_decisions, snapshot_from_row,
    decision_key, TARGET_CONCEPT, ENTITY_KINDS, DECISION_APPROVED, DECISION_REJECTED,
    DECISION_DUPLICATE, DECISION_PENDING, DECISION_LABELS,
)

# What a person can do with an AI concept.
ACTION_APPROVE = "approve"          # enters as a new concept (type confirmed by the person)
ACTION_REJECT = "reject"            # discarded
ACTION_DUPLICATE = "duplicate"      # it already exists in OntoPriv (possible duplicates only)
ACTION_UNDO = "undo"                # back to "por validar"
_ACTION_TO_DECISION = {
    ACTION_APPROVE: DECISION_APPROVED,
    ACTION_REJECT: DECISION_REJECTED,
    ACTION_DUPLICATE: DECISION_DUPLICATE,
    ACTION_UNDO: DECISION_PENDING,
}

ENTITY_KIND_LABELS = {
    "class": "clase",
    "object_property": "propiedad de objeto",
    "datatype_property": "propiedad de datos",
}

_LOCAL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class AIConcept:
    """One AI-proposed concept, as the person reviews it."""
    key: str                    # "ai:<Name>"
    name: str
    label: str | None
    definition: str | None
    articles: list[int]
    duplicate_status: str       # "possible_duplicate" | "new"
    duplicate_of: str | None    # IRI of the similar OntoPriv entity
    duplicate_of_name: str | None
    duplicate_score: float | None
    duplicate_reason: str | None
    suggested_kind: str         # "class" | "object_property" (the person confirms or changes it)
    iri: str                    # IRI it would get in the profile
    name_clash: str | None      # IRI of an OntoPriv entity with the same name, if any
    row: dict = field(default_factory=dict, repr=False)   # best candidate row (for the snapshot)

    @property
    def is_possible_duplicate(self) -> bool:
        return self.duplicate_status == STATUS_DUPLICATE


def concept_iri(profile_iri: str, name: str) -> str:
    """IRI of a new concept inside the profile: ``<profile>#<Name>``."""
    name = (name or "").strip()
    if not _LOCAL_NAME.match(name):
        raise ValueError(f"Nombre no valido para un IRI: {name!r} (solo letras, numeros y _).")
    return profile_iri.rstrip("#/") + "#" + name


def suggested_entity_kind(kinds) -> str:
    """Type proposed to the person, from the kind deduced from the name in Sprint 4.

    A property cannot tell object from datatype by its name; object property is suggested and
    the person changes it if the value is a literal (a date, a text, a number)."""
    return "object_property" if ALIGN_PROPERTY in (kinds or []) else "class"


def ontology_names(data: dict) -> dict[str, str]:
    """Lower-cased local name -> IRI of every OntoPriv entity in the candidates file."""
    out: dict[str, str] = {}
    for r in data.get("rows", []):
        if r.get("origin") == ORIGIN_ONTOLOGY and r.get("concept_name"):
            out.setdefault(r["concept_name"].lower(), r["concept_key"])
    return out


def _module_name(iri: str) -> str:
    """Last segment of the entity's namespace: OntoPriv mixes two namespaces
    ('ley-organica-proteccion-datos-personales' and 'OntologiaLOPDP'), and both can hold an
    entity with the same name (e.g. two 'Confidentiality')."""
    ns = iri.rsplit("#", 1)[0] if "#" in iri else iri.rsplit("/", 1)[0]
    return ns.rstrip("/").rsplit("/", 1)[-1]


def ontology_entities(data: dict) -> list[dict]:
    """OntoPriv entities a person can pick as 'the same as' an AI concept, sorted by name.

    Each one carries its kinds and module so two entities with the same name can be told apart
    (a class vs a datatype property, or the same name in each OntoPriv namespace)."""
    seen: dict[str, dict] = {}
    for r in data.get("rows", []):
        if r.get("origin") == ORIGIN_ONTOLOGY and r["concept_key"] not in seen:
            seen[r["concept_key"]] = {"key": r["concept_key"], "name": r.get("concept_name"),
                                      "family": r.get("concept_family"),
                                      "kinds": list(r.get("concept_kinds") or []),
                                      "module": _module_name(r["concept_key"])}
    return sorted(seen.values(), key=lambda e: ((e["name"] or "").lower(), e["module"]))


def ai_concepts(data: dict, profile_iri: str = PROFILE_IRI) -> list[AIConcept]:
    """The AI concepts of a candidates file, in file order (one per concept, best candidate)."""
    names = ontology_names(data)
    best: dict[str, dict] = {}
    for r in data.get("rows", []):
        if r.get("origin") != ORIGIN_AI:
            continue
        key = r["concept_key"]
        if key not in best or (r.get("rank") or 99) < (best[key].get("rank") or 99):
            best[key] = r
    out = []
    for key, r in best.items():
        name = r.get("concept_name") or key.split(":", 1)[-1]
        out.append(AIConcept(
            key=key, name=name, label=r.get("concept_label"),
            definition=r.get("concept_definition"),
            articles=list(r.get("concept_articles") or []),
            duplicate_status=r.get("duplicate_status") or STATUS_NEW,
            duplicate_of=r.get("duplicate_of"), duplicate_of_name=r.get("duplicate_of_name"),
            duplicate_score=r.get("duplicate_score"), duplicate_reason=r.get("duplicate_reason"),
            suggested_kind=suggested_entity_kind(r.get("concept_kinds")),
            iri=concept_iri(profile_iri, name),
            name_clash=names.get(name.lower()),
            row=r,
        ))
    return out


def concept_decisions(records: list[Decision], source_id: str) -> dict[str, Decision]:
    """concept_key -> current decision about that AI concept (withdrawn ones excluded)."""
    return {d.concept_key: d for d in latest_decisions(records, source_id).values()
            if d.target == TARGET_CONCEPT}


def decide(concept: AIConcept, action: str, *, reviewer: str, source_id: str,
           log_path: str | Path, entity_kind: str | None = None, same_as: str | None = None,
           ontology_keys: set[str] | None = None, note: str | None = None,
           metadata: dict | None = None, decided_at: str | None = None) -> Decision:
    """Record a person's decision about one AI concept in the log and return it.

    ``same_as`` (duplicate only): the OntoPriv entity it equals; for a possible duplicate it
    defaults to the one the automatic check found. ``ontology_keys``: if given, ``same_as`` must
    be one of them (protects against a typo or a stale page)."""
    if action not in _ACTION_TO_DECISION:
        raise ValueError(f"Accion desconocida: {action!r}.")
    if action == ACTION_DUPLICATE:
        same_as = same_as or concept.duplicate_of
        if not same_as:
            raise ValueError(f"Elige la entidad de OntoPriv a la que equivale '{concept.name}'.")
        if ontology_keys is not None and same_as not in ontology_keys:
            raise ValueError(f"La entidad elegida no es parte de OntoPriv: {same_as}")
    else:
        same_as = None
    if action == ACTION_APPROVE:
        if entity_kind not in ENTITY_KINDS:
            raise ValueError("Confirma el tipo del concepto antes de aprobarlo "
                             f"({', '.join(ENTITY_KIND_LABELS.values())}).")
        if concept.name_clash:
            raise ValueError(f"Ya existe en OntoPriv una entidad llamada '{concept.name}' "
                             f"({concept.name_clash}). Marcalo como 'ya existe en OntoPriv' "
                             f"o descartalo.")
    else:
        entity_kind = None
    snapshot = snapshot_from_row(concept.row, metadata)
    snapshot["proposed_iri"] = concept.iri
    decision = make_decision(
        source_id=source_id, target=TARGET_CONCEPT, concept_key=concept.key,
        decision=_ACTION_TO_DECISION[action], reviewer=reviewer, entity_kind=entity_kind,
        same_as=same_as, note=note, snapshot=snapshot, decided_at=decided_at,
    )
    append_decision(decision, log_path)
    return decision


def concept_progress(concepts: list[AIConcept], current: dict[str, Decision]) -> dict:
    """How many AI concepts are still pending and how the decided ones went."""
    out = {"total": len(concepts), "pending": 0, DECISION_APPROVED: 0, DECISION_REJECTED: 0,
           DECISION_DUPLICATE: 0}
    for c in concepts:
        d = current.get(c.key)
        out[d.decision if d else "pending"] += 1
    return out


def review_rows(concepts: list[AIConcept], current: dict[str, Decision]) -> list[dict]:
    """Flat rows for the web (S5-T03): the concept plus its current decision in Spanish."""
    rows = []
    for c in concepts:
        d = current.get(c.key)
        rows.append({
            "key": c.key, "name": c.name, "label": c.label, "definition": c.definition,
            "articles": c.articles, "mark": c.duplicate_status,
            "duplicate_of_name": c.duplicate_of_name, "duplicate_score": c.duplicate_score,
            "suggested_kind": c.suggested_kind, "iri": c.iri, "name_clash": c.name_clash,
            "status": d.decision if d else DECISION_PENDING,
            "status_label": DECISION_LABELS[d.decision if d else DECISION_PENDING],
            "entity_kind": d.entity_kind if d else None,
            "same_as": d.same_as if d else None,
            "reviewer": d.reviewer if d else None,
            "decided_at": d.decided_at if d else None,
        })
    return rows


def approved_new_concepts(concepts: list[AIConcept], current: dict[str, Decision]) -> list[dict]:
    """What S5-T06 writes into the profile: the concepts a person approved as new."""
    out = []
    for c in concepts:
        d = current.get(c.key)
        if d and d.decision == DECISION_APPROVED:
            out.append({"key": c.key, "name": c.name, "iri": c.iri, "label": c.label,
                        "definition": c.definition, "articles": c.articles,
                        "entity_kind": d.entity_kind, "reviewer": d.reviewer,
                        "decided_at": d.decided_at})
    return out


def confirmed_duplicates(concepts: list[AIConcept], current: dict[str, Decision]) -> list[dict]:
    """AI concepts confirmed as already in OntoPriv: their articles go to the OntoPriv entity
    the PERSON named (``same_as``), which may differ from the one the automatic check found."""
    out = []
    for c in concepts:
        d = current.get(c.key)
        if d and d.decision == DECISION_DUPLICATE:
            out.append({"key": c.key, "name": c.name, "duplicate_of": d.same_as or c.duplicate_of,
                        "detected_duplicate_of": c.duplicate_of, "articles": c.articles,
                        "reviewer": d.reviewer, "decided_at": d.decided_at})
    return out


def excluded_from_mapping(concepts: list[AIConcept], current: dict[str, Decision]) -> set[str]:
    """AI concepts that get NO DPV correspondence: rejected ones and confirmed duplicates
    (a duplicate inherits the alignment of its OntoPriv entity)."""
    return {c.key for c in concepts
            if (d := current.get(c.key)) and d.decision in (DECISION_REJECTED, DECISION_DUPLICATE)}


def render_console(concepts: list[AIConcept], current: dict[str, Decision]) -> str:
    p = concept_progress(concepts, current)
    dup = sum(1 for c in concepts if c.is_possible_duplicate)
    clash = sum(1 for c in concepts if c.name_clash)
    return "\n".join([
        f"Conceptos propuestos por la IA: {p['total']} ({dup} posibles duplicados de OntoPriv, "
        f"{p['total'] - dup} nuevos)",
        f"  Por validar: {p['pending']}",
        f"  Aprobados como nuevos: {p[DECISION_APPROVED]}",
        f"  Descartados: {p[DECISION_REJECTED]}",
        f"  Ya existen en OntoPriv: {p[DECISION_DUPLICATE]}",
        f"  Con el mismo nombre que una entidad de OntoPriv (no se pueden aprobar como nuevos): "
        f"{clash}",
    ])


def _run() -> None:
    from src.config import load_config
    from src.alignment.export import load_candidate_file, CANDIDATES_JSON
    from src.alignment.decisions import review_settings, decisions_path, read_log

    cfg = load_config()
    out_dir = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    path = out_dir / CANDIDATES_JSON
    if not path.exists():
        print(f"No hay archivo de candidatos ({path}). Corre antes: py -m src.alignment.export")
        return
    settings = review_settings(cfg)
    concepts = ai_concepts(load_candidate_file(path))
    current = concept_decisions(read_log(decisions_path(cfg)), settings["source_id"])
    print(render_console(concepts, current))


if __name__ == "__main__":
    _run()