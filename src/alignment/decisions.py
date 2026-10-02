"""Human review decisions log (S5-T01): who decided what, when, and from which evidence.

Every human decision of Sprint 5 (approve or reject an AI-proposed concept, approve or reject a
DPV correspondence, mark a concept as having no DPV counterpart) is APPENDED as one JSON line
to ``data/review/alignment-decisions.jsonl``. The file is tracked in git: it is human work that
cannot be regenerated, unlike ``alignment-candidates.json`` (which ``export --force`` rebuilds).

Rules:
- Append-only. A change of mind is a NEW line; the latest line for the same key wins, so the
  history stays in the file for auditing (and for PROV-O in Sprint 10).
- Key = (source_id, target, concept_key, dpv_iri). ``source_id`` names the input that produced
  the candidates (``ontopriv+lopdp`` for the base case), so decisions about another law or
  ontology (Sprints 6-7) never mix with these.
- Each line carries a SNAPSHOT of how the proposal was produced (rank, scores, the AI's type and
  justification, the evidence article, LLM model, prompt version, embedding model), so the
  decision keeps its provenance even if the candidates file is regenerated.
- ``pending`` withdraws a previous decision (the key goes back to "por validar").
- ``same_as`` (concept decisions only): the OntoPriv entity a person says an AI concept already
  is. The automatic duplicate check (S4-T06) misses some (e.g. names in another language), so
  the person names the entity; its law article is later attached to it (S5-T06).

Codes are stored in English; console and web show them in Spanish (``DECISION_LABELS``).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from src.alignment.justify import RELATIONS

DECISIONS_FILE = "alignment-decisions.jsonl"
DEFAULT_DECISIONS_PATH = "data/review/" + DECISIONS_FILE
DEFAULT_SOURCE_ID = "ontopriv+lopdp"

# What the decision is about.
TARGET_CONCEPT = "concept"      # an AI-proposed concept: does it enter the ontology?
TARGET_MAPPING = "mapping"      # a (concept, DPV term) correspondence
TARGETS = (TARGET_CONCEPT, TARGET_MAPPING)

# Decision codes.
DECISION_APPROVED = "approved"      # concept: enters as a new concept / mapping: SKOS triple
DECISION_REJECTED = "rejected"      # concept: discarded / mapping: candidate discarded
DECISION_DUPLICATE = "duplicate"    # concept only: it already exists in OntoPriv
DECISION_NO_MATCH = "no_match"      # mapping only (dpv_iri empty): no DPV counterpart
DECISION_PENDING = "pending"        # withdraws the previous decision for this key

ALLOWED_DECISIONS = {
    TARGET_CONCEPT: (DECISION_APPROVED, DECISION_REJECTED, DECISION_DUPLICATE, DECISION_PENDING),
    TARGET_MAPPING: (DECISION_APPROVED, DECISION_REJECTED, DECISION_NO_MATCH, DECISION_PENDING),
}

DECISION_LABELS = {
    DECISION_APPROVED: "aprobado",
    DECISION_REJECTED: "descartado",
    DECISION_DUPLICATE: "ya existe en OntoPriv",
    DECISION_NO_MATCH: "sin correspondencia",
    DECISION_PENDING: "por validar",
}

# SKOS types a person can approve (the AI's "none" is a no_match decision, not a relation).
MAPPING_RELATIONS = tuple(r for r in RELATIONS if r.startswith("skos:"))

# Entity type a person confirms when approving a new concept (S5-T02).
ENTITY_KINDS = ("class", "object_property", "datatype_property")

# Candidate-row fields copied into the snapshot (how the proposal was produced).
_ROW_SNAPSHOT = ("concept_name", "concept_label", "concept_articles", "origin",
                 "duplicate_status", "duplicate_of", "duplicate_score",
                 "rank", "dpv_name", "dpv_kind", "lexical", "semantic", "score",
                 "proposed_relation", "justification", "evidence_article")


@dataclass(frozen=True)
class Decision:
    """One line of the log."""
    source_id: str
    target: str
    concept_key: str
    decision: str
    reviewer: str
    decided_at: str
    dpv_iri: str = ""
    relation: str | None = None
    entity_kind: str | None = None
    same_as: str | None = None
    note: str | None = None
    snapshot: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return decision_key(self.source_id, self.target, self.concept_key, self.dpv_iri)

    def to_dict(self) -> dict:
        return asdict(self)


def decision_key(source_id: str, target: str, concept_key: str,
                 dpv_iri: str | None = "") -> tuple[str, str, str, str]:
    return (source_id, target, concept_key, dpv_iri or "")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_decision(*, source_id: str, target: str, concept_key: str, decision: str,
                  reviewer: str, dpv_iri: str | None = "", relation: str | None = None,
                  entity_kind: str | None = None, same_as: str | None = None,
                  note: str | None = None, snapshot: dict | None = None,
                  decided_at: str | None = None) -> Decision:
    """Validate and build a decision. Raises ``ValueError`` with a Spanish message."""
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise ValueError("Falta el nombre del revisor: es obligatorio para registrar una decision.")
    if not (source_id or "").strip():
        raise ValueError("Falta el identificador de la entrada (source_id).")
    if not (concept_key or "").strip():
        raise ValueError("Falta el concepto (concept_key).")
    if target not in TARGETS:
        raise ValueError(f"Objetivo desconocido: {target!r} (validos: {', '.join(TARGETS)}).")
    if decision not in ALLOWED_DECISIONS[target]:
        raise ValueError(f"La decision {decision!r} no aplica a {target!r} "
                         f"(validas: {', '.join(ALLOWED_DECISIONS[target])}).")
    dpv_iri = dpv_iri or ""

    if target == TARGET_CONCEPT:
        if dpv_iri:
            raise ValueError("Una decision sobre un concepto no lleva termino del DPV.")
        if relation:
            raise ValueError("Una decision sobre un concepto no lleva tipo SKOS.")
        if decision == DECISION_APPROVED and entity_kind not in ENTITY_KINDS:
            raise ValueError("Para aprobar un concepto nuevo hay que confirmar su tipo "
                             f"({', '.join(ENTITY_KINDS)}).")
        if decision != DECISION_APPROVED and entity_kind:
            raise ValueError("El tipo de entidad solo se registra al aprobar el concepto.")
        if decision == DECISION_DUPLICATE and not (same_as or "").strip():
            raise ValueError("Indica a que entidad de OntoPriv equivale el concepto.")
        if decision != DECISION_DUPLICATE and same_as:
            raise ValueError("La entidad equivalente de OntoPriv solo se registra al marcar "
                             "'ya existe en OntoPriv'.")
    else:
        if entity_kind:
            raise ValueError("Una correspondencia no lleva tipo de entidad.")
        if same_as:
            raise ValueError("Una correspondencia no lleva entidad equivalente de OntoPriv.")
        if decision == DECISION_NO_MATCH:
            if dpv_iri or relation:
                raise ValueError("'Sin correspondencia' se registra para el concepto entero, "
                                 "sin termino del DPV ni tipo SKOS.")
        elif not dpv_iri:
            raise ValueError("Falta el termino del DPV (dpv_iri) de la correspondencia.")
        if decision == DECISION_APPROVED and relation not in MAPPING_RELATIONS:
            raise ValueError("Para aprobar una correspondencia hay que elegir el tipo SKOS "
                             f"({', '.join(MAPPING_RELATIONS)}).")
        if decision != DECISION_APPROVED and relation:
            raise ValueError("El tipo SKOS solo se registra al aprobar la correspondencia.")

    return Decision(source_id=source_id.strip(), target=target, concept_key=concept_key.strip(),
                    decision=decision, reviewer=reviewer, decided_at=decided_at or _utc_now(),
                    dpv_iri=dpv_iri, relation=relation, entity_kind=entity_kind,
                    same_as=(same_as or "").strip() or None,
                    note=(note or "").strip() or None, snapshot=dict(snapshot or {}))


def snapshot_from_row(row: dict | None, metadata: dict | None = None) -> dict:
    """How a proposal was produced: candidate-row fields + run metadata (models, prompt)."""
    snap = {k: row.get(k) for k in _ROW_SNAPSHOT if row and k in row}
    metadata = metadata or {}
    justification = metadata.get("justification") or {}
    snap["embedding_model"] = metadata.get("embedding_model")
    snap["llm_model"] = justification.get("llm_model")
    snap["prompt_version"] = justification.get("prompt_version")
    snap["candidates_generated_at"] = metadata.get("generated_at")
    return snap


def append_decision(decision: Decision, path: str | Path) -> Path:
    """Append one decision as a JSON line (creates the folder and file if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(decision.to_dict(), ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
    return path


def read_log(path: str | Path) -> list[Decision]:
    """All decisions in file order. A missing file is an empty log; a damaged line is an error
    (silently skipping it would lose a human decision)."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[Decision] = []
    known = set(Decision.__dataclass_fields__)
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                out.append(Decision(**{k: v for k, v in data.items() if k in known}))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Linea {number} danada en {path.name}: {exc}") from exc
    return out


def latest_decisions(records: list[Decision], source_id: str | None = None
                     ) -> dict[tuple[str, str, str, str], Decision]:
    """Current state: the last decision per key, without the withdrawn ones (``pending``)."""
    current: dict[tuple[str, str, str, str], Decision] = {}
    for d in records:
        if source_id is not None and d.source_id != source_id:
            continue
        current[d.key] = d
    return {k: d for k, d in current.items() if d.decision != DECISION_PENDING}


def count_decisions(current: dict[tuple[str, str, str, str], Decision]) -> dict:
    """Counts of the current decisions by target and code (for console, web and CHANGELOG)."""
    counts = Counter((d.target, d.decision) for d in current.values())
    return {target: {code: counts.get((target, code), 0)
                     for code in ALLOWED_DECISIONS[target] if code != DECISION_PENDING}
            for target in TARGETS}


def review_settings(cfg: dict | None = None) -> dict:
    """The ``review`` block of config.yaml with defaults."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config() or {}
    block = (cfg or {}).get("review") or {}
    return {
        "source_id": block.get("source_id") or DEFAULT_SOURCE_ID,
        "decisions_file": block.get("decisions_file") or DEFAULT_DECISIONS_PATH,
        "reviewer": block.get("reviewer") or "",
    }


def decisions_path(cfg: dict | None = None) -> Path:
    """Absolute path of the decisions log (relative paths are taken from the project root)."""
    from src.config import ROOT
    path = Path(review_settings(cfg)["decisions_file"])
    return path if path.is_absolute() else ROOT / path


def render_console(records: list[Decision], source_id: str) -> str:
    current = latest_decisions(records, source_id)
    counts = count_decisions(current)
    reviewers = sorted({d.reviewer for d in current.values()})
    c, m = counts[TARGET_CONCEPT], counts[TARGET_MAPPING]
    lines = [
        f"Registro de decisiones humanas (entrada: {source_id})",
        f"  Lineas en el registro: {sum(1 for d in records if d.source_id == source_id)} "
        f"(incluye cambios de opinion)",
        f"  Decisiones vigentes: {len(current)}",
        f"  Conceptos de la IA: {c[DECISION_APPROVED]} aprobados, {c[DECISION_REJECTED]} "
        f"descartados, {c[DECISION_DUPLICATE]} ya existen en OntoPriv",
        f"  Correspondencias: {m[DECISION_APPROVED]} aprobadas, {m[DECISION_REJECTED]} "
        f"descartadas, {m[DECISION_NO_MATCH]} conceptos sin correspondencia",
        f"  Revisores: {', '.join(reviewers) if reviewers else '(ninguno todavia)'}",
    ]
    return "\n".join(lines)


def _run() -> None:
    settings = review_settings()
    path = decisions_path()
    if not path.exists():
        print(f"Aun no hay decisiones registradas ({path}).")
        return
    print(render_console(read_log(path), settings["source_id"]))
    print(f"  Archivo: {path}")


if __name__ == "__main__":
    _run()