"""Human decisions on the DPV correspondences (S5-T04): which SKOS mappings enter the graph.

Who is reviewed:
- every OntoPriv entity of the candidates file (flow A), and
- the AI concepts a person APPROVED as new in S5-T03 (flow B). AI concepts that were rejected or
  confirmed as duplicates get no correspondence of their own (a duplicate inherits the alignment
  of its OntoPriv entity); AI concepts still pending must be decided first (S5-T03).

What a person can do, per concept:
- APPROVE a DPV candidate with the SKOS type THEY choose (the AI type is only a suggestion: the
  S4-T09 measurement showed it is right 25 % of the time). More than one candidate can be
  approved (e.g. a closeMatch and a relatedMatch).
- REJECT a candidate.
- Approve a DPV term found with the SEARCH (outside the top-3; Hit@3 was 84 %), as long as its
  kind is compatible (class/individual <-> DPV concept, property <-> DPV property).
- Mark the concept as NO MATCH (no DPV counterpart). Only possible while no mapping of the
  concept is approved; approving a mapping later withdraws the "no match".
- UNDO any of the above (back to "por validar").

A concept counts as reviewed when it has at least one approved mapping or a "no match". Every
decision goes to the S5-T01 log with a snapshot of how the candidate was produced; the aligned
graph itself is written in S5-T06 from ``approved_mappings``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.core.emit import PROFILE_IRI
from src.alignment.sources import ORIGIN_AI, ORIGIN_ONTOLOGY
from src.alignment.dpv_targets import COMPATIBLE_TARGETS
from src.alignment.justify import RELATIONS
from src.alignment.ai_concepts import ai_concepts, concept_decisions
from src.alignment.decisions import (
    Decision, make_decision, append_decision, latest_decisions, snapshot_from_row,
    TARGET_MAPPING, MAPPING_RELATIONS, DECISION_APPROVED, DECISION_REJECTED,
    DECISION_NO_MATCH, DECISION_PENDING,
)

# Review status of a concept.
STATUS_PENDING = "pending"          # nothing decided yet (or only rejections)
STATUS_ALIGNED = "aligned"          # at least one approved mapping
STATUS_NO_MATCH = "no_match"        # the person said the DPV has no counterpart
STATUS_LABELS = {STATUS_PENDING: "por validar", STATUS_ALIGNED: "alineado",
                 STATUS_NO_MATCH: "sin correspondencia"}

# Candidate (row) state inside a concept.
ROW_PENDING, ROW_APPROVED, ROW_REJECTED = "pending", "approved", "rejected"

# Entity kind confirmed in S5-T03 -> alignment kind (S4-T02 vocabulary).
_ENTITY_TO_ALIGN = {"class": "class", "object_property": "property",
                    "datatype_property": "property"}


@dataclass
class MappingConcept:
    """One concept to align, with its DPV candidates (best first)."""
    key: str                      # concept_key in the candidates file / the log
    iri: str                      # subject of the SKOS triple
    origin: str                   # "ontology" | "ai"
    name: str
    label: str | None
    definition: str | None
    family: str | None
    kinds: list[str]              # alignment kinds ("class", "property", ...)
    articles: list[int]
    candidates: list[dict] = field(default_factory=list, repr=False)

    def candidate(self, dpv_iri: str) -> dict | None:
        return next((r for r in self.candidates if r.get("dpv_iri") == dpv_iri), None)


# ------------------------------------------------------------------ who is reviewed
def mapping_concepts(data: dict, records: list[Decision], source_id: str,
                     profile_iri: str = PROFILE_IRI) -> list[MappingConcept]:
    """Concepts in the mapping review: approved AI concepts first, then OntoPriv by family."""
    rows_by_key: dict[str, list[dict]] = {}
    for r in data.get("rows", []):
        rows_by_key.setdefault(r["concept_key"], []).append(r)
    for rows in rows_by_key.values():
        rows.sort(key=lambda r: r.get("rank") or 99)

    approved_ai = {}
    concept_state = concept_decisions(records, source_id)
    for c in ai_concepts(data, profile_iri):
        d = concept_state.get(c.key)
        if d and d.decision == DECISION_APPROVED:
            approved_ai[c.key] = (c, d)

    out: list[MappingConcept] = []
    for key, (c, d) in approved_ai.items():
        out.append(MappingConcept(
            key=key, iri=c.iri, origin=ORIGIN_AI, name=c.name, label=c.label,
            definition=c.definition, family=None,
            kinds=[_ENTITY_TO_ALIGN.get(d.entity_kind, "class")], articles=c.articles,
            candidates=rows_by_key.get(key, [])))
    onto = []
    for key, rows in rows_by_key.items():
        first = rows[0]
        if first.get("origin") != ORIGIN_ONTOLOGY:
            continue
        onto.append(MappingConcept(
            key=key, iri=key, origin=ORIGIN_ONTOLOGY, name=first.get("concept_name") or key,
            label=first.get("concept_label"), definition=first.get("concept_definition"),
            family=first.get("concept_family"), kinds=list(first.get("concept_kinds") or []),
            articles=list(first.get("concept_articles") or []), candidates=rows))
    onto.sort(key=lambda m: ((m.family or "~").lower(), m.name.lower()))
    return out + onto


def mapping_decisions(records: list[Decision], source_id: str) -> dict[str, dict[str, Decision]]:
    """concept_key -> {dpv_iri ("" = no match): current decision}."""
    out: dict[str, dict[str, Decision]] = {}
    for d in latest_decisions(records, source_id).values():
        if d.target == TARGET_MAPPING:
            out.setdefault(d.concept_key, {})[d.dpv_iri] = d
    return out


def concept_status(concept: MappingConcept, current: dict[str, dict[str, Decision]]) -> str:
    decided = current.get(concept.key, {})
    if any(d.decision == DECISION_APPROVED for d in decided.values()):
        return STATUS_ALIGNED
    if "" in decided and decided[""].decision == DECISION_NO_MATCH:
        return STATUS_NO_MATCH
    return STATUS_PENDING


# ------------------------------------------------------------------ DPV search
def compatible_kinds(concept_kinds) -> set[str]:
    """DPV target kinds a concept may be mapped to (punned = union)."""
    allowed: set[str] = set()
    for k in concept_kinds or []:
        allowed |= COMPATIBLE_TARGETS.get(k, set())
    return allowed


def search_dpv(targets, text: str, concept_kinds, limit: int = 15) -> list:
    """DPV terms matching ``text`` (name, label, then definition) of a compatible kind."""
    t = (text or "").strip().lower()
    if not t:
        return []
    allowed = compatible_kinds(concept_kinds)
    pool = [x for x in targets.targets if x.kind in allowed]
    hits = [x for x in pool if t in x.name.lower() or t in (x.label or "").lower()]
    if len(hits) < limit:
        hits += [x for x in pool if x not in hits and t in (x.definition or "").lower()]
    return hits[:limit]


# ------------------------------------------------------------------ decisions
def _record(concept: MappingConcept, decision: str, *, source_id: str, log_path, reviewer: str,
            dpv_iri: str = "", relation: str | None = None, snapshot: dict | None = None,
            note: str | None = None, decided_at: str | None = None) -> Decision:
    d = make_decision(source_id=source_id, target=TARGET_MAPPING, concept_key=concept.key,
                      dpv_iri=dpv_iri, decision=decision, relation=relation, reviewer=reviewer,
                      note=note, snapshot=snapshot, decided_at=decided_at)
    append_decision(d, log_path)
    return d


def _snapshot(concept: MappingConcept, dpv_iri: str, metadata: dict | None,
              target=None) -> dict:
    row = concept.candidate(dpv_iri)
    if row is not None:
        snap = snapshot_from_row(row, metadata)
        snap["found_by"] = "candidates"
    else:
        snap = snapshot_from_row(None, metadata)
        snap["found_by"] = "search"
        if target is not None:
            snap.update({"dpv_name": target.name, "dpv_kind": target.kind})
    snap["subject_iri"] = concept.iri
    return snap


def approve(concept: MappingConcept, dpv_iri: str, relation: str, *, reviewer: str,
            source_id: str, log_path, current: dict[str, dict[str, Decision]] | None = None,
            target=None, metadata: dict | None = None, note: str | None = None,
            decided_at: str | None = None) -> Decision:
    """Approve one correspondence with the SKOS type the person chose.

    ``dpv_iri`` must be one of the concept's candidates or, if found with the search, its
    ``target`` (a DpvTarget) must be given so its kind can be checked. If the concept was marked
    "no match", that mark is withdrawn first."""
    if relation not in MAPPING_RELATIONS:
        raise ValueError("Elige el tipo SKOS de la correspondencia antes de aprobarla.")
    row = concept.candidate(dpv_iri)
    if row is None:
        if target is None or target.iri != dpv_iri:
            raise ValueError("El termino del DPV no es candidato de este concepto; buscalo "
                             "en el DPV para aprobarlo.")
        if target.kind not in compatible_kinds(concept.kinds):
            kind = "una propiedad" if "property" in concept.kinds else "una clase"
            raise ValueError(f"'{target.name}' es de un tipo incompatible: '{concept.name}' es "
                             f"{kind} y solo se alinea con terminos del DPV del mismo tipo.")
    decided = (current or {}).get(concept.key, {})
    if "" in decided and decided[""].decision == DECISION_NO_MATCH:
        _record(concept, DECISION_PENDING, source_id=source_id, log_path=log_path,
                reviewer=reviewer, note="se aprobo una correspondencia", decided_at=decided_at)
    return _record(concept, DECISION_APPROVED, source_id=source_id, log_path=log_path,
                   reviewer=reviewer, dpv_iri=dpv_iri, relation=relation, note=note,
                   snapshot=_snapshot(concept, dpv_iri, metadata, target),
                   decided_at=decided_at)


def reject(concept: MappingConcept, dpv_iri: str, *, reviewer: str, source_id: str, log_path,
           metadata: dict | None = None, note: str | None = None,
           decided_at: str | None = None) -> Decision:
    """Discard one candidate (or a previously approved searched term)."""
    return _record(concept, DECISION_REJECTED, source_id=source_id, log_path=log_path,
                   reviewer=reviewer, dpv_iri=dpv_iri, note=note,
                   snapshot=_snapshot(concept, dpv_iri, metadata), decided_at=decided_at)


def mark_no_match(concept: MappingConcept, *, reviewer: str, source_id: str, log_path,
                  current: dict[str, dict[str, Decision]] | None = None,
                  metadata: dict | None = None, note: str | None = None,
                  decided_at: str | None = None) -> Decision:
    """The DPV has no counterpart for this concept (only if nothing is approved)."""
    decided = (current or {}).get(concept.key, {})
    approved = [d for d in decided.values() if d.decision == DECISION_APPROVED]
    if approved:
        raise ValueError(f"'{concept.name}' ya tiene {len(approved)} correspondencia(s) "
                         f"aprobada(s); deshazlas antes de marcar 'sin correspondencia'.")
    snap = snapshot_from_row(concept.candidates[0] if concept.candidates else None, metadata)
    snap["ai_relations"] = [r.get("proposed_relation") for r in concept.candidates]
    snap["subject_iri"] = concept.iri
    return _record(concept, DECISION_NO_MATCH, source_id=source_id, log_path=log_path,
                   reviewer=reviewer, note=note, snapshot=snap, decided_at=decided_at)


def undo(concept: MappingConcept, dpv_iri: str = "", *, reviewer: str, source_id: str,
         log_path, decided_at: str | None = None) -> Decision:
    """Back to 'por validar' (``dpv_iri`` empty = withdraw the 'no match')."""
    return _record(concept, DECISION_PENDING, source_id=source_id, log_path=log_path,
                   reviewer=reviewer, dpv_iri=dpv_iri, decided_at=decided_at)


# ------------------------------------------------------------------ views of the state
def row_states(concept: MappingConcept, current: dict[str, dict[str, Decision]]) -> list[dict]:
    """Each candidate of the concept with its current state (plus approved searched terms)."""
    decided = current.get(concept.key, {})
    out = []
    for r in concept.candidates:
        d = decided.get(r.get("dpv_iri"))
        out.append({"row": r, "dpv_iri": r.get("dpv_iri"), "found_by": "candidates",
                    "state": d.decision if d else ROW_PENDING,
                    "relation": d.relation if d else None, "decision": d})
    known = {r.get("dpv_iri") for r in concept.candidates}
    for iri, d in decided.items():
        if iri and iri not in known and d.decision in (DECISION_APPROVED, DECISION_REJECTED):
            out.append({"row": None, "dpv_iri": iri, "found_by": "search", "state": d.decision,
                        "relation": d.relation, "decision": d,
                        "dpv_name": d.snapshot.get("dpv_name")})
    return out


def mapping_progress(concepts: list[MappingConcept],
                     current: dict[str, dict[str, Decision]]) -> dict:
    """Counts for the progress bar and the CHANGELOG."""
    status = Counter(concept_status(c, current) for c in concepts)
    by_origin = Counter((c.origin, concept_status(c, current)) for c in concepts)
    relations = Counter(m["relation"] for m in approved_mappings(concepts, current))
    return {
        "total": len(concepts),
        "pending": status.get(STATUS_PENDING, 0),
        "aligned": status.get(STATUS_ALIGNED, 0),
        "no_match": status.get(STATUS_NO_MATCH, 0),
        "reviewed": status.get(STATUS_ALIGNED, 0) + status.get(STATUS_NO_MATCH, 0),
        "ontology_pending": by_origin.get((ORIGIN_ONTOLOGY, STATUS_PENDING), 0),
        "ai_pending": by_origin.get((ORIGIN_AI, STATUS_PENDING), 0),
        "mappings": sum(relations.values()),
        "relations": {r: relations.get(r, 0) for r in MAPPING_RELATIONS},
    }


def next_pending(concepts: list[MappingConcept], current: dict[str, dict[str, Decision]],
                 after_key: str | None = None) -> str | None:
    """Key of the next concept still 'por validar' after ``after_key`` (wraps around)."""
    keys = [c.key for c in concepts]
    start = keys.index(after_key) + 1 if after_key in keys else 0
    for c in concepts[start:] + concepts[:start]:
        if concept_status(c, current) == STATUS_PENDING:
            return c.key
    return None


def approved_mappings(concepts: list[MappingConcept],
                      current: dict[str, dict[str, Decision]]) -> list[dict]:
    """What S5-T06 writes: one SKOS triple per approved correspondence."""
    out = []
    for c in concepts:
        for iri, d in current.get(c.key, {}).items():
            if iri and d.decision == DECISION_APPROVED:
                row = c.candidate(iri)
                out.append({
                    "concept_key": c.key, "subject_iri": c.iri, "origin": c.origin,
                    "concept_name": c.name, "dpv_iri": iri,
                    "dpv_name": (row or {}).get("dpv_name") or d.snapshot.get("dpv_name"),
                    "relation": d.relation, "ai_relation": (row or {}).get("proposed_relation"),
                    "found_by": "candidates" if row else "search",
                    "justification": (row or {}).get("justification"),
                    "evidence_article": (row or {}).get("evidence_article"),
                    "reviewer": d.reviewer, "decided_at": d.decided_at, "note": d.note,
                })
    return out


def ai_agreement(concepts: list[MappingConcept],
                 current: dict[str, dict[str, Decision]]) -> dict:
    """How often the person kept the AI's SKOS type, over everything reviewed (extends the
    30-concept measurement of S4-T09 to the whole review)."""
    typed = [m for m in approved_mappings(concepts, current)
             if m["ai_relation"] in MAPPING_RELATIONS]
    same = sum(1 for m in typed if m["ai_relation"] == m["relation"])
    no_match = [c for c in concepts if concept_status(c, current) == STATUS_NO_MATCH]
    ai_none = sum(1 for c in no_match
                  if c.candidates and all(r.get("proposed_relation") == "none"
                                          for r in c.candidates))
    return {
        "approved_with_ai_type": len(typed),
        "same_type_as_ai": same,
        "type_agreement": round(same / len(typed), 4) if typed else None,
        "no_match": len(no_match),
        "no_match_ai_also_none": ai_none,
    }


def relation_label(relation: str | None) -> str:
    return RELATIONS.get(relation, "-") if relation else "-"


def render_console(concepts: list[MappingConcept],
                   current: dict[str, dict[str, Decision]]) -> str:
    p = mapping_progress(concepts, current)
    a = ai_agreement(concepts, current)
    ai_total = sum(1 for c in concepts if c.origin == ORIGIN_AI)
    lines = [
        f"Revision de correspondencias con el DPV: {p['total']} conceptos "
        f"({p['total'] - ai_total} de OntoPriv, {ai_total} nuevos de la IA)",
        f"  Revisados: {p['reviewed']} ({p['aligned']} alineados, {p['no_match']} sin "
        f"correspondencia) · por validar: {p['pending']}",
        f"  Correspondencias aprobadas: {p['mappings']}",
    ]
    for rel, n in p["relations"].items():
        if n:
            lines.append(f"    - {rel} ({RELATIONS[rel]}): {n}")
    if a["type_agreement"] is not None:
        lines.append(f"  Tipo SKOS igual al propuesto por la IA: {a['same_type_as_ai']} de "
                     f"{a['approved_with_ai_type']} ({a['type_agreement'] * 100:.1f} %)")
    return "\n".join(lines)


def _run() -> None:
    from src.config import load_config
    from src.alignment.export import load_candidate_file, CANDIDATES_JSON
    from src.alignment.decisions import review_settings, decisions_path, read_log

    cfg = load_config()
    path = Path(cfg.get("outputs", {}).get("dir", "data/output")) / CANDIDATES_JSON
    if not path.exists():
        print(f"No hay archivo de candidatos ({path}). Corre antes: py -m src.alignment.export")
        return
    source_id = review_settings(cfg)["source_id"]
    records = read_log(decisions_path(cfg))
    concepts = mapping_concepts(load_candidate_file(path), records, source_id)
    print(render_console(concepts, mapping_decisions(records, source_id)))


if __name__ == "__main__":
    _run()