"""Assisted batch review of the DPV correspondences (S5-T08).

Why it exists: reviewing 552 concepts one by one, the reviewer noticed he was following the AI
suggestion (automation bias: 86 % agreement with an AI type measured right 25 % of the time,
S4-T09). He chose an ASSISTED review: the project assistant proposed a decision for every
concept still "por validar", with a short reason, in ``data/review/assistant-proposals.json``;
the person reviews them as tables, family by family, and confirms or changes them.

What this module guarantees (the honest provenance of the review):
- A proposal is NOT a decision. Nothing reaches the decisions log (S5-T01) until a person
  confirms it; the decision is written with that person's name, through the same
  ``approve`` / ``mark_no_match`` of S5-T04 (so the same checks apply: SKOS type, compatible
  kind, a searched term must exist in the DPV).
- Every confirmed decision keeps, in its snapshot, ``proposed_by: "assistant"``, the original
  proposal and whether the person accepted it as is or changed it; the note says
  "propuesta del asistente confirmada" (or "cambiada"). The graph manifest and the thesis can
  therefore separate what the person decided alone from what they confirmed.
- Only concepts still "por validar" are offered: a person's own decision always wins.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.alignment.decisions import MAPPING_RELATIONS, DECISION_APPROVED, DECISION_NO_MATCH
from src.alignment.dpv_targets import DpvTarget
from src.alignment.mapping_review import (
    MappingConcept, approve, mark_no_match, concept_status, STATUS_PENDING,
)

PROPOSALS_FILE = "assistant-proposals.json"
DEFAULT_PROPOSALS_PATH = "data/review/" + PROPOSALS_FILE

PROPOSAL_APPROVE = "approve"        # the assistant proposes one or more SKOS mappings
PROPOSAL_NO_MATCH = "no_match"      # the assistant proposes "sin correspondencia"
PROPOSAL_LABELS = {PROPOSAL_APPROVE: "alinear", PROPOSAL_NO_MATCH: "sin correspondencia"}

# What the person does with each row of the table.
ACTION_ACCEPT = "accept"
ACTION_NO_MATCH = "no_match"
ACTION_SKIP = "skip"
ACTION_LABELS = {ACTION_ACCEPT: "Aceptar", ACTION_NO_MATCH: "Sin correspondencia",
                 ACTION_SKIP: "Omitir"}

# How the confirmed decision relates to the proposal (stored in the snapshot).
CONFIRM_ACCEPTED = "accepted"                  # as proposed
CONFIRM_CHANGED_TYPE = "changed_type"          # same DPV term, another SKOS type
CONFIRM_CHANGED_TO_NO_MATCH = "changed_to_no_match"   # proposed mappings, person said no match

CONFIRMED_NOTE = "propuesta del asistente confirmada"
CHANGED_NOTE = "propuesta del asistente cambiada por el revisor"
PROPOSED_BY = "assistant"


@dataclass
class Proposal:
    """The assistant's proposal for one concept (``mappings`` empty = no match)."""
    concept_key: str
    concept_name: str
    decision: str
    reason: str
    mappings: list[dict] = field(default_factory=list)
    origin: str | None = None
    family: str | None = None
    concept_label: str | None = None

    def mapping(self, dpv_iri: str) -> dict | None:
        return next((m for m in self.mappings if m["dpv_iri"] == dpv_iri), None)


# ------------------------------------------------------------------ the proposals file
def proposals_path(cfg: dict | None = None) -> Path:
    """``review.proposals_file`` of config.yaml (default data/review/assistant-proposals.json),
    relative to the project root."""
    from src.config import ROOT
    if cfg is None:
        from src.config import load_config
        cfg = load_config() or {}
    path = Path(((cfg or {}).get("review") or {}).get("proposals_file")
                or DEFAULT_PROPOSALS_PATH)
    return path if path.is_absolute() else ROOT / path


def parse_proposals(data: dict) -> list[Proposal]:
    """Check and read the proposals (errors in Spanish, with the concept that fails)."""
    out, seen = [], set()
    for i, p in enumerate(data.get("proposals") or [], start=1):
        key = p.get("concept_key")
        if not key:
            raise ValueError(f"La propuesta {i} no tiene concept_key.")
        if key in seen:
            raise ValueError(f"El concepto '{key}' tiene mas de una propuesta.")
        seen.add(key)
        decision = p.get("decision")
        mappings = p.get("mappings") or []
        if decision not in PROPOSAL_LABELS:
            raise ValueError(f"Propuesta de '{key}': decision desconocida '{decision}'.")
        if (decision == PROPOSAL_APPROVE) != bool(mappings):
            raise ValueError(f"Propuesta de '{key}': 'approve' necesita correspondencias y "
                             f"'no_match' no debe tenerlas.")
        for m in mappings:
            if m.get("relation") not in MAPPING_RELATIONS:
                raise ValueError(f"Propuesta de '{key}': tipo SKOS no valido "
                                 f"'{m.get('relation')}'.")
            if not m.get("dpv_iri") or m.get("found_by") not in ("candidates", "search"):
                raise ValueError(f"Propuesta de '{key}': falta dpv_iri o found_by.")
        out.append(Proposal(concept_key=key, concept_name=p.get("concept_name") or key,
                            decision=decision, reason=p.get("reason") or "",
                            mappings=[dict(m) for m in mappings], origin=p.get("origin"),
                            family=p.get("family"), concept_label=p.get("concept_label")))
    return out


def load_proposals(path) -> dict | None:
    """``{"metadata": ..., "proposals": [Proposal]}`` or None when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"metadata": data.get("metadata") or {}, "proposals": parse_proposals(data)}


# ------------------------------------------------------------------ what is still open
def open_proposals(proposals: list[Proposal], concepts: dict[str, MappingConcept],
                   current: dict) -> list[Proposal]:
    """Proposals whose concept is in the review and still 'por validar'."""
    return [p for p in proposals if p.concept_key in concepts
            and concept_status(concepts[p.concept_key], current) == STATUS_PENDING]


def proposal_rows(proposals: list[Proposal]) -> list[dict]:
    """One table row per proposed mapping, or one row for a proposed 'no match'."""
    rows = []
    for p in proposals:
        base = {"concept_key": p.concept_key, "concept_name": p.concept_name,
                "concept_label": p.concept_label or "", "family": p.family,
                "origin": p.origin, "proposal": p.decision, "reason": p.reason}
        if not p.mappings:
            rows.append({**base, "dpv_iri": "", "dpv_name": "", "dpv_label": "",
                         "dpv_definition": "", "relation": None, "found_by": None})
        for m in p.mappings:
            rows.append({**base, "dpv_iri": m["dpv_iri"], "dpv_name": m.get("dpv_name") or "",
                         "dpv_label": m.get("dpv_label") or "",
                         "dpv_definition": m.get("dpv_definition") or "",
                         "relation": m["relation"], "found_by": m["found_by"]})
    return rows


# ------------------------------------------------------------------ confirming
def _target(m: dict) -> DpvTarget:
    """The searched DPV term, rebuilt from the proposal (``approve`` checks its kind)."""
    return DpvTarget(iri=m["dpv_iri"], name=m.get("dpv_name") or m["dpv_iri"].rsplit("#", 1)[-1],
                     kind=m.get("dpv_kind") or "", label=m.get("dpv_label") or "")


def _proposal_snapshot(proposal: Proposal, proposals_metadata: dict | None,
                       confirmation: str, left_out: list[str] | None = None) -> dict:
    meta = proposals_metadata or {}
    snap = {
        "proposed_by": PROPOSED_BY,
        "proposal": {"decision": proposal.decision, "reason": proposal.reason,
                     "mappings": [{"dpv_iri": m["dpv_iri"], "relation": m["relation"],
                                   "found_by": m["found_by"]} for m in proposal.mappings]},
        "proposal_assistant": meta.get("assistant"),
        "proposal_created_at": meta.get("created_at"),
        "assistant_confirmation": confirmation,
    }
    if left_out:                       # proposed mappings the person did not accept
        snap["not_accepted_from_proposal"] = list(left_out)
    return snap


def confirm(proposal: Proposal, concept: MappingConcept, action: str, *, reviewer: str,
            source_id: str, log_path, current: dict | None = None,
            accepted: list[str] | None = None, relations: dict[str, str] | None = None,
            metadata: dict | None = None, proposals_metadata: dict | None = None,
            decided_at: str | None = None) -> list:
    """Record the person's answer to one proposal; returns the decisions written.

    ``action``: ACTION_ACCEPT (the proposal; for a mapping proposal, the DPV terms in
    ``accepted`` — default all — with the type in ``relations`` — default the proposed one),
    ACTION_NO_MATCH (the concept has no DPV counterpart) or ACTION_SKIP (nothing)."""
    if action not in ACTION_LABELS:
        raise ValueError(f"Accion desconocida '{action}'.")
    if action == ACTION_SKIP:
        return []
    if concept.key != proposal.concept_key:
        raise ValueError("La propuesta no corresponde a este concepto.")
    if concept_status(concept, current or {}) != STATUS_PENDING:
        raise ValueError(f"'{concept.name}' ya fue revisado; su decision no se cambia desde "
                         f"las propuestas (usa la tarjeta del concepto).")
    common = dict(reviewer=reviewer, source_id=source_id, log_path=log_path, current=current,
                  metadata=metadata, decided_at=decided_at)
    if action == ACTION_NO_MATCH or proposal.decision == PROPOSAL_NO_MATCH:
        how = (CONFIRM_ACCEPTED if proposal.decision == PROPOSAL_NO_MATCH
               else CONFIRM_CHANGED_TO_NO_MATCH)
        note = CONFIRMED_NOTE if how == CONFIRM_ACCEPTED else f"{CHANGED_NOTE}: sin correspondencia"
        return [mark_no_match(concept, note=note,
                              extra_snapshot=_proposal_snapshot(proposal, proposals_metadata, how),
                              **common)]
    iris = [m["dpv_iri"] for m in proposal.mappings] if accepted is None else accepted
    if not iris:
        raise ValueError(f"'{concept.name}': elige al menos una correspondencia para aceptar.")
    left_out = [m["dpv_iri"] for m in proposal.mappings if m["dpv_iri"] not in iris]
    written = []
    for iri in iris:
        m = proposal.mapping(iri)
        if m is None:
            raise ValueError(f"'{concept.name}': dpv:{iri.rsplit('#', 1)[-1]} no esta en la "
                             f"propuesta.")
        relation = (relations or {}).get(iri) or m["relation"]
        how = CONFIRM_ACCEPTED if relation == m["relation"] else CONFIRM_CHANGED_TYPE
        note = (CONFIRMED_NOTE if how == CONFIRM_ACCEPTED
                else f"{CHANGED_NOTE}: tipo {m['relation']} -> {relation}")
        target = _target(m) if m["found_by"] == "search" else None
        written.append(approve(concept, iri, relation, target=target, note=note,
                               extra_snapshot=_proposal_snapshot(proposal, proposals_metadata,
                                                                 how, left_out),
                               **common))
    return written


def confirm_rows(rows: list[dict], proposals: dict[str, Proposal],
                 concepts: dict[str, MappingConcept], *, reviewer: str, source_id: str,
                 log_path, current: dict | None = None, metadata: dict | None = None,
                 proposals_metadata: dict | None = None,
                 decided_at: str | None = None) -> dict:
    """Confirm the edited table: ``rows`` have concept_key, dpv_iri, relation and action.

    Rows are grouped per concept: all "Omitir" = nothing; any "Sin correspondencia" = the
    concept has no match (contradicts an "Aceptar" in the same concept -> error for that
    concept only); otherwise the accepted mappings with the type in the row."""
    if not reviewer.strip():
        raise ValueError("Escribe el nombre del revisor antes de confirmar.")
    by_concept: dict[str, list[dict]] = {}
    for r in rows:
        by_concept.setdefault(r["concept_key"], []).append(r)
    summary = {"concepts": 0, "decisions": 0, "skipped": 0, "accepted": 0, "changed": 0,
               "errors": []}
    for key, group in by_concept.items():
        actions = {r["action"] for r in group}
        proposal, concept = proposals.get(key), concepts.get(key)
        name = concept.name if concept else key
        if actions == {ACTION_SKIP}:
            summary["skipped"] += 1
            continue
        if proposal is None or concept is None:
            summary["errors"].append((name, "no hay propuesta o el concepto ya no esta en la "
                                            "revision."))
            continue
        try:
            if ACTION_NO_MATCH in actions:
                accepting = [r for r in group if r["action"] == ACTION_ACCEPT and r["dpv_iri"]]
                if accepting:
                    raise ValueError("una fila dice 'Aceptar' y otra 'Sin correspondencia'.")
                written = confirm(proposal, concept, ACTION_NO_MATCH, reviewer=reviewer,
                                  source_id=source_id, log_path=log_path, current=current,
                                  metadata=metadata, proposals_metadata=proposals_metadata,
                                  decided_at=decided_at)
            else:
                acc = [r for r in group if r["action"] == ACTION_ACCEPT]
                written = confirm(proposal, concept, ACTION_ACCEPT, reviewer=reviewer,
                                  source_id=source_id, log_path=log_path, current=current,
                                  accepted=[r["dpv_iri"] for r in acc if r["dpv_iri"]] or None,
                                  relations={r["dpv_iri"]: r.get("relation") for r in acc
                                             if r["dpv_iri"]},
                                  metadata=metadata, proposals_metadata=proposals_metadata,
                                  decided_at=decided_at)
        except ValueError as exc:
            summary["errors"].append((name, str(exc)))
            continue
        summary["concepts"] += 1
        summary["decisions"] += len(written)
        if all(_as_proposed(d) for d in written):
            summary["accepted"] += 1
        else:
            summary["changed"] += 1
    return summary


# ------------------------------------------------------------------ provenance metrics
def _as_proposed(decision) -> bool:
    """The person kept the proposal as it was (same type, nothing left out)."""
    return (decision.snapshot.get("assistant_confirmation") == CONFIRM_ACCEPTED
            and not decision.snapshot.get("not_accepted_from_proposal"))


def review_provenance(concepts: list[MappingConcept], current: dict,
                      proposals: list[Proposal] | None = None) -> dict:
    """Who decided each reviewed concept: the person alone, or the person confirming an
    assistant proposal (as is, or changed). For the manifest (S5-T06) and the thesis."""
    by = Counter()
    how = Counter()
    for c in concepts:
        if concept_status(c, current) == STATUS_PENDING:
            continue
        final = [d for d in current.get(c.key, {}).values()
                 if d.decision in (DECISION_APPROVED, DECISION_NO_MATCH)]
        assisted = [d for d in final if d.snapshot.get("proposed_by") == PROPOSED_BY]
        if not assisted:
            by["person"] += 1
            continue
        by["assistant_confirmed"] += 1
        how["accepted" if all(_as_proposed(d) for d in assisted) else "changed"] += 1
    keys = {c.key: c for c in concepts}
    open_count = len(open_proposals(proposals, keys, current)) if proposals is not None else None
    return {"reviewed_by_person": by.get("person", 0),
            "confirmed_from_assistant": by.get("assistant_confirmed", 0),
            "accepted_as_proposed": how.get("accepted", 0),
            "changed_by_person": how.get("changed", 0),
            "open_proposals": open_count}


def family_counts(proposals: list[Proposal]) -> list[tuple[str, int, int]]:
    """(family, proposals to align, proposals of no match), for the console and the page."""
    c = Counter((p.family or "", p.decision) for p in proposals)
    fams = sorted({f for f, _ in c})
    return [(f, c.get((f, PROPOSAL_APPROVE), 0), c.get((f, PROPOSAL_NO_MATCH), 0)) for f in fams]


def render_console(open_: list[Proposal], provenance: dict, metadata: dict) -> str:
    lines = [
        "Propuestas del asistente (S5-T08) - no son decisiones hasta que una persona las "
        "confirma",
        f"  Propuestas:   {metadata.get('assistant') or '-'} ({metadata.get('created_at') or '-'})",
        f"  Abiertas (concepto aun por validar): {len(open_)}",
        f"  Conceptos revisados por la persona sola: {provenance['reviewed_by_person']}",
        f"  Conceptos confirmados desde una propuesta: {provenance['confirmed_from_assistant']}"
        f" (sin cambios {provenance['accepted_as_proposed']}, cambiados "
        f"{provenance['changed_by_person']})",
    ]
    if open_:
        lines.append("  Abiertas por familia (alinear / sin correspondencia):")
        for fam, n_app, n_none in family_counts(open_):
            lines.append(f"    {fam or '(conceptos nuevos de la IA)':<48} {n_app:>4} / {n_none:>4}")
        lines.append("  Confirmalas en la web: Alineacion DPV > Confirmar propuestas del asistente")
    return "\n".join(lines)


def _run() -> None:
    from src.config import load_config
    from src.alignment.export import load_candidate_file, CANDIDATES_JSON
    from src.alignment.decisions import review_settings, decisions_path, read_log
    from src.alignment.mapping_review import mapping_concepts, mapping_decisions

    cfg = load_config()
    path = Path(cfg.get("outputs", {}).get("dir", "data/output")) / CANDIDATES_JSON
    loaded = load_proposals(proposals_path(cfg))
    if loaded is None:
        print(f"No hay archivo de propuestas ({proposals_path(cfg)}).")
        return
    if not path.exists():
        print(f"No hay archivo de candidatos ({path}). Corre antes: py -m src.alignment.export")
        return
    source_id = review_settings(cfg)["source_id"]
    records = read_log(decisions_path(cfg))
    concepts = mapping_concepts(load_candidate_file(path), records, source_id)
    current = mapping_decisions(records, source_id)
    open_ = open_proposals(loaded["proposals"], {c.key: c for c in concepts}, current)
    print(render_console(open_, review_provenance(concepts, current, loaded["proposals"]),
                         loaded["metadata"]))


if __name__ == "__main__":
    _run()