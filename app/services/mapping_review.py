"""Bridge for the review card of the "Alineación DPV" view (Sprint 5, S5-T05) and for the
batch confirmation of the assistant proposals (S5-T08).

Reads the candidates file (Sprint 4), the human decisions log (S5-T01), the assistant
proposals (S5-T08) and, for the search outside the top-3, the DPV; records the person's
decisions through ``src.alignment.mapping_review`` (S5-T04) and
``src.alignment.assistant_review`` (S5-T08). All the logic the page needs lives here so it can
be tested without Streamlit; the view only draws widgets.
"""
from __future__ import annotations

from pathlib import Path

from src.config import load_config, ROOT
from src.alignment.export import load_candidate_file, CANDIDATES_JSON
from src.alignment.justify import RELATIONS
from src.alignment.decisions import (review_settings, decisions_path, read_log,
                                     MAPPING_RELATIONS, DECISION_APPROVED, DECISION_REJECTED)
from src.alignment.mapping_review import (
    mapping_concepts, mapping_decisions, concept_status, row_states, mapping_progress,
    next_pending as _next_pending, ai_agreement, search_dpv, approve, reject, mark_no_match,
    undo, STATUS_LABELS, STATUS_PENDING, STATUS_ALIGNED, STATUS_NO_MATCH,
)
from src.alignment.assistant_review import (
    proposals_path, load_proposals, open_proposals, proposal_rows, confirm_rows,
    review_provenance, ACTION_LABELS, ACTION_ACCEPT, ACTION_SKIP,
)

ORIGIN_LABELS = {"ontology": "OntoPriv", "ai": "IA (nuevo)"}
STATUS_FILTERS = {"Por validar": STATUS_PENDING, "Alineados": STATUS_ALIGNED,
                  "Sin correspondencia": STATUS_NO_MATCH, "Todos": None}
ORIGIN_FILTERS = {"Todos": None, "IA (nuevos)": "ai", "OntoPriv": "ontology"}
ROW_LABELS = {"pending": "por validar", DECISION_APPROVED: "aprobado",
              DECISION_REJECTED: "descartado"}


def default_paths() -> dict:
    """Where the page reads and writes (from config.yaml)."""
    cfg = load_config() or {}
    out = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    out = out if out.is_absolute() else ROOT / out
    dpv = Path((cfg.get("inputs") or {}).get("dpv", "vocab/dpv.ttl"))
    return {"candidates": out / CANDIDATES_JSON, "log": decisions_path(cfg),
            "dpv": dpv if dpv.is_absolute() else ROOT / dpv, "proposals": proposals_path(cfg)}


def load_dpv_targets(path=None):
    """DPV terms for the search (slow to parse: the view caches it)."""
    from src.ingest.dpv_loader import load_dpv
    from src.alignment.dpv_targets import build_dpv_targets
    p = Path(path) if path else default_paths()["dpv"]
    return build_dpv_targets(load_dpv(p)) if p.exists() else None


def load(candidates_path=None, log_path=None, cfg: dict | None = None) -> dict | None:
    """Everything the page shows, or None when the candidates file does not exist yet."""
    paths = default_paths()
    cand = Path(candidates_path) if candidates_path else paths["candidates"]
    log = Path(log_path) if log_path else paths["log"]
    if not cand.exists():
        return None
    settings = review_settings(cfg)
    data = load_candidate_file(cand)
    records = read_log(log)
    concepts = mapping_concepts(data, records, settings["source_id"])
    current = mapping_decisions(records, settings["source_id"])
    return {
        "settings": settings,
        "metadata": data.get("metadata", {}),
        "log_path": log,
        "concepts": concepts,
        "by_key": {c.key: c for c in concepts},
        "current": current,
        "progress": mapping_progress(concepts, current),
        "agreement": ai_agreement(concepts, current),
    }


# ------------------------------------------------------------------ lists and labels
def families(state: dict) -> list[str]:
    return sorted({c.family for c in state["concepts"] if c.family})


def status_of(state: dict, key: str) -> str:
    return concept_status(state["by_key"][key], state["current"])


def filter_concepts(state: dict, status: str | None = STATUS_PENDING, origin: str | None = None,
                    family: str | None = None, query: str = "") -> list:
    """``status``/``origin``/``family`` = None means all. ``query`` is searched in the concept
    name and label and in the names of its DPV candidates."""
    q = query.strip().lower()
    out = []
    for c in state["concepts"]:
        if status and status_of(state, c.key) != status:
            continue
        if origin and c.origin != origin:
            continue
        if family and c.family != family:
            continue
        if q:
            text = " ".join([c.name or "", c.label or ""] +
                            [r.get("dpv_name") or "" for r in c.candidates]).lower()
            if q not in text:
                continue
        out.append(c)
    return out


def option_label(state: dict, concept) -> str:
    """Text of each concept in the selector."""
    label = f" · {concept.label}" if concept.label else ""
    where = concept.family or ORIGIN_LABELS.get(concept.origin, concept.origin)
    return f"{concept.name}{label}  ({where})  [{STATUS_LABELS[status_of(state, concept.key)]}]"


def relation_options() -> dict[str, str]:
    """Spanish label -> SKOS code, for the type selector (no default: the person chooses)."""
    return {f"{code.split(':', 1)[1]} · {RELATIONS[code]}": code for code in MAPPING_RELATIONS}


def relation_text(code: str | None) -> str:
    if not code:
        return "sin tipo"
    if code == "none":
        return "sin correspondencia"
    return f"{code.split(':', 1)[-1]} · {RELATIONS.get(code, code)}"


def candidate_cards(state: dict, key: str) -> list[dict]:
    """The candidates of one concept, shaped for the card (plus approved searched terms)."""
    concept = state["by_key"][key]
    cards = []
    for s in row_states(concept, state["current"]):
        r = s["row"] or {}
        d = s["decision"]
        cards.append({
            "dpv_iri": s["dpv_iri"],
            "dpv_name": r.get("dpv_name") or s.get("dpv_name") or s["dpv_iri"].rsplit("#", 1)[-1],
            "dpv_label": r.get("dpv_label"),
            "dpv_definition": r.get("dpv_definition"),
            "dpv_parents": r.get("dpv_parents") or [],
            "rank": r.get("rank"),
            "score": r.get("score"), "lexical": r.get("lexical"), "semantic": r.get("semantic"),
            "ai_relation": r.get("proposed_relation"),
            "ai_text": relation_text(r.get("proposed_relation")) if r else None,
            "justification": r.get("justification"),
            "evidence_article": r.get("evidence_article"),
            "found_by": s["found_by"],
            "state": s["state"],
            "state_label": ROW_LABELS.get(s["state"], s["state"]),
            "relation": s["relation"],
            "reviewer": d.reviewer if d else None,
        })
    return cards


def next_pending(state: dict, after_key: str | None = None) -> str | None:
    return _next_pending(state["concepts"], state["current"], after_key)


def search(state: dict, targets, key: str, text: str, limit: int = 15) -> list:
    """DPV terms compatible with the concept, excluding its own candidates."""
    concept = state["by_key"][key]
    own = {r.get("dpv_iri") for r in concept.candidates}
    return [t for t in search_dpv(targets, text, concept.kinds, limit + len(own))
            if t.iri not in own][:limit]


# ------------------------------------------------------------------ actions
def _run(fn, *args, **kwargs) -> tuple[bool, str | None]:
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def apply_approve(state: dict, key: str, dpv_iri: str, relation: str | None, reviewer: str,
                  target=None, note: str | None = None) -> tuple[bool, str]:
    concept = state["by_key"][key]
    ok, err = _run(approve, concept, dpv_iri, relation, reviewer=reviewer,
                   source_id=state["settings"]["source_id"], log_path=state["log_path"],
                   current=state["current"], target=target, metadata=state["metadata"],
                   note=note)
    name = dpv_iri.rsplit("#", 1)[-1]
    return ok, err or f"Aprobado: {concept.name} → dpv:{name} ({relation_text(relation)})."


def apply_reject(state: dict, key: str, dpv_iri: str, reviewer: str) -> tuple[bool, str]:
    concept = state["by_key"][key]
    ok, err = _run(reject, concept, dpv_iri, reviewer=reviewer,
                   source_id=state["settings"]["source_id"], log_path=state["log_path"],
                   metadata=state["metadata"])
    return ok, err or f"Descartado: dpv:{dpv_iri.rsplit('#', 1)[-1]} para {concept.name}."


def apply_no_match(state: dict, key: str, reviewer: str,
                   note: str | None = None) -> tuple[bool, str]:
    concept = state["by_key"][key]
    ok, err = _run(mark_no_match, concept, reviewer=reviewer,
                   source_id=state["settings"]["source_id"], log_path=state["log_path"],
                   current=state["current"], metadata=state["metadata"], note=note)
    return ok, err or f"'{concept.name}' marcado como sin correspondencia en el DPV."


def apply_undo(state: dict, key: str, reviewer: str, dpv_iri: str = "") -> tuple[bool, str]:
    concept = state["by_key"][key]
    ok, err = _run(undo, concept, dpv_iri, reviewer=reviewer,
                   source_id=state["settings"]["source_id"], log_path=state["log_path"])
    what = f"dpv:{dpv_iri.rsplit('#', 1)[-1]}" if dpv_iri else "'sin correspondencia'"
    return ok, err or f"Deshecho: {what} de {concept.name} vuelve a 'por validar'."


# ------------------------------------------------------------------ assistant proposals (S5-T08)
NEW_AI_FAMILY = "Conceptos nuevos de la IA"
NO_MATCH_TEXT = "— sin correspondencia en el DPV —"
_ACTION_CODES = {label: code for code, label in ACTION_LABELS.items()}


def load_assistant(state: dict, path=None) -> dict | None:
    """The assistant proposals still open (concept 'por validar'), or None if there is no file."""
    p = Path(path) if path else default_paths()["proposals"]
    loaded = load_proposals(p)
    if loaded is None:
        return None
    open_ = open_proposals(loaded["proposals"], state["by_key"], state["current"])
    return {"path": p, "metadata": loaded["metadata"], "proposals": loaded["proposals"],
            "open": open_, "by_key": {x.concept_key: x for x in open_},
            "provenance": review_provenance(state["concepts"], state["current"],
                                            loaded["proposals"])}


def proposal_family(proposal) -> str:
    return proposal.family or NEW_AI_FAMILY


def proposal_families(assist: dict) -> list[tuple[str, int]]:
    """(family, open proposals) in the order of the review (new AI concepts first)."""
    out: dict[str, int] = {}
    for p in assist["open"]:
        fam = proposal_family(p)
        out[fam] = out.get(fam, 0) + 1
    return list(out.items())


def proposal_table(state: dict, assist: dict, family: str, prefill: bool = False) -> list[dict]:
    """Rows of the editable table for one family. "Acción" starts as "Omitir" unless the
    person ticks "marcar todas como Aceptar"; "Tipo SKOS" starts as the proposed type."""
    labels = {code: label for label, code in relation_options().items()}
    action = ACTION_LABELS[ACTION_ACCEPT] if prefill else ACTION_LABELS[ACTION_SKIP]
    rows = []
    props = [p for p in assist["open"] if proposal_family(p) == family]
    for r in proposal_rows(props):
        if r["dpv_iri"]:
            dpv = f"dpv:{r['dpv_name']}"
            if r["found_by"] == "search":
                dpv += " (búsqueda)"
        else:
            dpv = NO_MATCH_TEXT
        concept = state["by_key"].get(r["concept_key"])
        rows.append({
            "Concepto": r["concept_name"],
            "Etiqueta": r["concept_label"],
            "Definición del concepto": (concept.definition if concept else None) or "",
            "Propuesta del asistente": dpv,
            "Definición en el DPV": r["dpv_definition"],
            "Tipo SKOS": labels.get(r["relation"]) if r["relation"] else None,
            "Razón": r["reason"],
            "Acción": action,
            "_key": r["concept_key"],
            "_dpv": r["dpv_iri"],
        })
    return rows


def action_options() -> list[str]:
    return list(_ACTION_CODES)


def count_actions(rows: list[dict]) -> dict[str, int]:
    """How many rows of the (edited) table have each action label."""
    out = {label: 0 for label in _ACTION_CODES}
    for r in rows:
        if r.get("Acción") in out:
            out[r["Acción"]] += 1
    return out


def confirm_table(state: dict, assist: dict, rows: list[dict],
                  reviewer: str) -> tuple[bool, str, dict | None]:
    """Write the decisions of the edited table (rows as returned by the data editor)."""
    codes = relation_options()
    batch = []
    for r in rows:
        action = _ACTION_CODES.get(r.get("Acción"), ACTION_SKIP)
        batch.append({"concept_key": r["_key"], "dpv_iri": r["_dpv"] or "",
                      "relation": codes.get(r.get("Tipo SKOS")) if r["_dpv"] else None,
                      "action": action})
    if all(b["action"] == ACTION_SKIP for b in batch):
        return False, "No hay filas con 'Aceptar' o 'Sin correspondencia'.", None
    missing = [r["Concepto"] for r, b in zip(rows, batch)
               if b["action"] == ACTION_ACCEPT and b["dpv_iri"] and not b["relation"]]
    if missing:
        return False, ("Elige el tipo SKOS en las filas que vas a aceptar: "
                       + ", ".join(sorted(set(missing))) + "."), None
    try:
        summary = confirm_rows(batch, assist["by_key"], state["by_key"], reviewer=reviewer,
                               source_id=state["settings"]["source_id"],
                               log_path=state["log_path"], current=state["current"],
                               metadata=state["metadata"],
                               proposals_metadata=assist["metadata"])
    except ValueError as exc:
        return False, str(exc), None
    msg = (f"Registradas {summary['decisions']} decisión(es) de {summary['concepts']} "
           f"concepto(s) a nombre de {reviewer}: {summary['accepted']} tal como se "
           f"propusieron y {summary['changed']} con cambios.")
    if summary["errors"]:
        msg += " No se registraron: " + "; ".join(f"{n} ({e})" for n, e in summary["errors"])
    return summary["concepts"] > 0, msg, summary