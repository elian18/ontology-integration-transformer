"""Bridge for the alignment view (Sprint 4, S4-T10): DPV candidates and their measurement.

Reads the files written by the Sprint 4 pipeline (it never recomputes anything):
- ``alignment-candidates.json`` (S4-T07/T08): one row per (concept, DPV candidate), with the
  similarity scores, the SKOS type proposed by the AI and its justification.
- ``alignment-evaluation.json`` (S4-T09): Hit@1 / Hit@k and the AI type accuracy measured
  against the hand-labelled sample.
Everything is still "por validar": approving candidates is Sprint 5. The view only reads the
dicts returned here, and the filtering lives here so it can be tested without Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.alignment.export import CANDIDATES_JSON
from src.alignment.evaluate import EVALUATION_FILE
from src.alignment.justify import RELATIONS, RELATION_UNTYPED

ORIGIN_LABELS = {"ontology": "OntoPriv", "ai": "IA"}
DUPLICATE_LABELS = {"possible_duplicate": "ya existe en OntoPriv", "new": "nuevo"}
PENDING_LABEL = "sin tipo aun"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_dir() -> Path:
    out = Path(load_config().get("outputs", {}).get("dir", "data/output"))
    return out if out.is_absolute() else _root() / out


def relation_label(relation) -> str:
    """Stored SKOS value -> Spanish label for the web ('skos:broadMatch' -> 'el DPV es...')."""
    if relation is None:
        return PENDING_LABEL
    if relation == RELATION_UNTYPED:
        return "sin tipo valido"
    return RELATIONS.get(relation, str(relation))


def candidates(path=None) -> dict | None:
    """Candidates file as a dict for the view, or None when the pipeline has not run yet."""
    p = Path(path) if path is not None else _output_dir() / CANDIDATES_JSON
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = data.get("rows", [])
    return {
        "metadata": data.get("metadata", {}),
        "counts": data.get("counts", {}),
        "rows": rows,
        "relations": sorted({relation_label(r.get("proposed_relation")) for r in rows}),
    }


def evaluation(path=None) -> dict | None:
    """Measurement against the hand-labelled sample (S4-T09), or None if not run yet."""
    p = Path(path) if path is not None else _output_dir() / EVALUATION_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def filter_rows(rows: list[dict], origin: str | None = None, duplicate: str | None = None,
                relations: list[str] | None = None, query: str = "",
                best_only: bool = False) -> list[dict]:
    """Filter the candidate rows the way the view offers it.

    ``origin``: 'ontology' | 'ai' | None. ``duplicate``: 'possible_duplicate' | 'new' | None
    (only meaningful for AI rows). ``relations``: Spanish labels to keep (empty = all).
    ``query``: text searched in the concept name/label and the DPV name/label.
    ``best_only``: keep only the rank-1 candidate of each concept."""
    q = query.strip().lower()
    out = []
    for r in rows:
        if origin and r.get("origin") != origin:
            continue
        if duplicate and r.get("duplicate_status") != duplicate:
            continue
        if best_only and r.get("rank") != 1:
            continue
        if relations and relation_label(r.get("proposed_relation")) not in relations:
            continue
        if q:
            text = " ".join(str(r.get(k) or "") for k in
                            ("concept_name", "concept_label", "dpv_name", "dpv_label")).lower()
            if q not in text:
                continue
        out.append(r)
    return out


def table_rows(rows: list[dict]) -> list[dict]:
    """Rows shaped for st.dataframe (Spanish headers, Scrumban order first)."""
    return [{
        "Concepto": r.get("concept_name"),
        "Candidato DPV": r.get("dpv_label") or r.get("dpv_name"),
        "Similitud": r.get("score"),
        "Tipo propuesto": relation_label(r.get("proposed_relation")),
        "Puesto": r.get("rank"),
        "Origen": ORIGIN_LABELS.get(r.get("origin"), r.get("origin")),
        "Marca (IA)": DUPLICATE_LABELS.get(r.get("duplicate_status"), ""),
        "Etiqueta": r.get("concept_label") or "",
        "Justificación": r.get("justification") or "",
        "Artículo": r.get("evidence_article"),
        "Léxico": r.get("lexical"),
        "Semántico": r.get("semantic"),
    } for r in rows]