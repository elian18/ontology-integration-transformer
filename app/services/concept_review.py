"""Bridge for the "Aprobar conceptos IA" view (Sprint 5, S5-T03).

Reads the candidates file (Sprint 4) and the human decisions log (S5-T01), and records the
person's decision on each AI-proposed concept through ``src.alignment.ai_concepts`` (S5-T02).
All the logic the page needs (state, filters, next pending, applying a decision) lives here so
it can be tested without Streamlit; the view only draws widgets.
"""
from __future__ import annotations

from pathlib import Path

from src.config import load_config, ROOT
from src.alignment.export import load_candidate_file, CANDIDATES_JSON
from src.alignment.decisions import (review_settings, decisions_path, read_log,
                                     DECISION_PENDING, DECISION_APPROVED, DECISION_REJECTED,
                                     DECISION_DUPLICATE)
from src.alignment.ai_concepts import (ai_concepts, ontology_entities, concept_decisions,
                                       concept_progress, review_rows, decide, ENTITY_KIND_LABELS,
                                       ACTION_APPROVE, ACTION_REJECT, ACTION_DUPLICATE,
                                       ACTION_UNDO)

MARK_LABELS = {"new": "nuevo", "possible_duplicate": "posible duplicado de OntoPriv"}
STATUS_FILTERS = {
    "Por validar": DECISION_PENDING,
    "Aprobados": DECISION_APPROVED,
    "Descartados": DECISION_REJECTED,
    "Ya existen en OntoPriv": DECISION_DUPLICATE,
    "Todos": None,
}
MARK_FILTERS = {"Todas": None, "Nuevos": "new", "Posibles duplicados": "possible_duplicate"}
ACTION_MESSAGES = {
    ACTION_APPROVE: "aprobado como concepto nuevo",
    ACTION_REJECT: "descartado",
    ACTION_DUPLICATE: "marcado como 'ya existe en OntoPriv'",
    ACTION_UNDO: "devuelto a 'por validar'",
}


def default_paths() -> dict:
    """Where the page reads and writes (from config.yaml)."""
    cfg = load_config() or {}
    out = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    out = out if out.is_absolute() else ROOT / out
    return {"candidates": out / CANDIDATES_JSON, "log": decisions_path(cfg)}


def load(candidates_path=None, log_path=None, cfg: dict | None = None) -> dict | None:
    """Everything the page shows, or None when the candidates file does not exist yet."""
    paths = default_paths()
    cand = Path(candidates_path) if candidates_path else paths["candidates"]
    log = Path(log_path) if log_path else paths["log"]
    if not cand.exists():
        return None
    settings = review_settings(cfg)
    data = load_candidate_file(cand)
    concepts = ai_concepts(data)
    current = concept_decisions(read_log(log), settings["source_id"])
    families = ontology_families(data)
    rows = review_rows(concepts, current)
    entities = ontology_entities(data)
    names = {e["key"]: e["name"] for e in entities}
    for r, c in zip(rows, concepts):
        r["duplicate_of_family"] = families.get(c.duplicate_of) if c.duplicate_of else None
        r["same_as_name"] = names.get(r["same_as"]) if r.get("same_as") else None
    return {
        "settings": settings,
        "metadata": data.get("metadata", {}),
        "log_path": log,
        "concepts": {c.key: c for c in concepts},
        "ontology": entities,
        "rows": rows,
        "progress": concept_progress(concepts, current),
    }


def ontology_families(data: dict) -> dict[str, str]:
    """OntoPriv entity IRI -> its family (OntoPriv has no definitions to show, the family
    gives the person context when judging a possible duplicate)."""
    return {r["concept_key"]: r.get("concept_family") for r in data.get("rows", [])
            if r.get("origin") == "ontology" and r.get("concept_family")}


def filter_rows(rows: list[dict], status: str | None = DECISION_PENDING,
                mark: str | None = None, query: str = "") -> list[dict]:
    """``status``: a decision code or None (all). ``mark``: 'new' | 'possible_duplicate' | None.
    ``query``: searched in the name, the Spanish label and the article numbers."""
    q = query.strip().lower()
    out = []
    for r in rows:
        if status and r["status"] != status:
            continue
        if mark and r["mark"] != mark:
            continue
        if q:
            text = " ".join([r["name"] or "", r["label"] or "",
                             " ".join(str(a) for a in r["articles"])]).lower()
            if q not in text:
                continue
        out.append(r)
    return out


def next_pending(rows: list[dict], after_key: str | None = None) -> str | None:
    """Key of the next concept still 'por validar' after ``after_key`` (wraps around)."""
    keys = [r["key"] for r in rows]
    start = keys.index(after_key) + 1 if after_key in keys else 0
    for r in rows[start:] + rows[:start]:
        if r["status"] == DECISION_PENDING:
            return r["key"]
    return None


def option_label(row: dict) -> str:
    """Text of each concept in the selector."""
    label = f" · {row['label']}" if row.get("label") else ""
    return f"{row['name']}{label}  [{row['status_label']}]"


_KIND_WORDS = {"class": "clase", "property": "propiedad", "individual": "individuo"}


def entity_label(entity: dict) -> str:
    """Text of each OntoPriv entity in the 'ya existe en OntoPriv' picker: name, family, type
    and module, so two entities with the same name can be told apart."""
    kinds = "/".join(_KIND_WORDS.get(k, k) for k in entity.get("kinds") or [])
    parts = [p for p in (entity.get("family"), kinds, entity.get("module")) if p]
    return f"{entity['name']}  ({' · '.join(parts)})" if parts else entity["name"]


def kind_options() -> dict[str, str]:
    """Spanish label -> stored code of the entity type."""
    return {label: code for code, label in ENTITY_KIND_LABELS.items()}


def apply(state: dict, concept_key: str, action: str, reviewer: str,
          entity_kind: str | None = None, note: str | None = None,
          same_as: str | None = None) -> tuple[bool, str]:
    """Record the decision; returns (ok, Spanish message for the page). ``same_as``: the
    OntoPriv entity picked for 'ya existe en OntoPriv'."""
    concept = state["concepts"].get(concept_key)
    if concept is None:
        return False, f"No se encontro el concepto {concept_key}."
    try:
        decide(concept, action, reviewer=reviewer, source_id=state["settings"]["source_id"],
               log_path=state["log_path"], entity_kind=entity_kind, note=note,
               same_as=same_as, ontology_keys={e["key"] for e in state["ontology"]},
               metadata=state["metadata"])
    except ValueError as exc:
        return False, str(exc)
    return True, f"'{concept.name}' {ACTION_MESSAGES[action]}."