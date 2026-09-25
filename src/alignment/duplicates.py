"""Flow B, first step: is each AI-proposed concept already in OntoPriv under another name?

The AI read the LOPDP and proposed concepts (Sprint 3). Some of them already exist in OntoPriv
('DigitalEducationRight' vs OntoPriv's 'Right_to_digital_education'), others are new. This
module compares every AI concept with every OntoPriv entity of a compatible kind, with the same
two signals used against the DPV (S4-T04 lexical on identifiers + S4-T05 meaning on
name/label/definition), and marks it:
- ``possible_duplicate`` when either rule holds:
  * ``score``: its best OntoPriv match reaches ``alignment.duplicate_threshold`` (0.75), or
  * ``same_name``: a compatible OntoPriv entity has (almost) the same identifier, lexical >=
    ``alignment.duplicate_name_match`` (0.95). Measured on the LOPDP run: identical names
    ('Consent' = 'Consent', 'RightsExercise' = 'Exercise_of_rights') stayed below 0.75 only
    because the AI side carries a long Spanish definition and the OntoPriv side a bare English
    name, which lowers the cosine. Inside one domain ontology the same name is strong evidence.
- ``new``: neither rule holds.
It is only a MARK for the human: the three closest OntoPriv entities are kept for every AI
concept, and the approval happens in Sprint 5. The thresholds are provisional (config.yaml) and
the console shows how many concepts sit near the cut. The cut favours precision: a concept
wrongly marked "new" only costs AI quota (it gets justified against the DPV), while one wrongly
marked "duplicate" would skip that justification.

Agreed use downstream (S4-T07/T08): DPV candidates are kept for all AI concepts (they are
cheap), but the AI only spends quota justifying the ``new`` ones; a possible duplicate
inherits the alignment of its OntoPriv entity (flow A).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.alignment.sources import AlignmentSources, SourceConcept, ORIGIN_AI, ORIGIN_ONTOLOGY
from src.alignment.lexical import lexical_matrix, source_lexical_text
from src.alignment.candidates import alignment_settings, AlignmentSettings

STATUS_DUPLICATE = "possible_duplicate"
STATUS_NEW = "new"
REASON_SCORE = "score"
REASON_SAME_NAME = "same_name"
DEFAULT_DUPLICATE_THRESHOLD = 0.75
DEFAULT_NAME_MATCH = 0.95
MATCHES_KEPT = 3


def _alignment_value(cfg: dict | None, key: str, default: float) -> float:
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    value = float(((cfg or {}).get("alignment", {}) or {}).get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"alignment.{key} debe estar entre 0 y 1.")
    return value


def duplicate_threshold(cfg: dict | None = None) -> float:
    """``alignment.duplicate_threshold`` from config.yaml (default 0.75), within [0, 1]."""
    return _alignment_value(cfg, "duplicate_threshold", DEFAULT_DUPLICATE_THRESHOLD)


def name_match_threshold(cfg: dict | None = None) -> float:
    """``alignment.duplicate_name_match`` from config.yaml (default 0.95), within [0, 1]."""
    return _alignment_value(cfg, "duplicate_name_match", DEFAULT_NAME_MATCH)


@dataclass
class OntologyMatch:
    """One OntoPriv entity close to an AI concept."""
    key: str
    name: str
    family: str | None
    lexical: float
    semantic: float
    score: float

    def as_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "family": self.family,
                "lexical": round(self.lexical, 4), "semantic": round(self.semantic, 4),
                "score": round(self.score, 4)}


@dataclass
class DuplicateCheck:
    """An AI concept, its closest OntoPriv entities and the resulting mark."""
    concept: SourceConcept
    status: str
    matches: list[OntologyMatch] = field(default_factory=list)
    reason: str | None = None          # score | same_name (only for possible duplicates)

    @property
    def best(self) -> OntologyMatch | None:
        return self.matches[0] if self.matches else None

    def as_dict(self) -> dict:
        return {"key": self.concept.key, "name": self.concept.name, "label": self.concept.label,
                "status": self.status, "reason": self.reason,
                "matches": [m.as_dict() for m in self.matches]}


@dataclass
class DuplicateReport:
    threshold: float
    settings: AlignmentSettings
    name_match: float = DEFAULT_NAME_MATCH
    checks: list[DuplicateCheck] = field(default_factory=list)

    def of_status(self, status: str) -> list[DuplicateCheck]:
        return [c for c in self.checks if c.status == status]

    def status_of(self, key: str) -> str | None:
        return next((c.status for c in self.checks if c.concept.key == key), None)

    def reason_of(self, key: str) -> str | None:
        return next((c.reason for c in self.checks if c.concept.key == key), None)

    def bands(self) -> dict:
        """How many AI concepts fall in each band of best-match score (to review the cut)."""
        edges = [(0.85, "muy cerca (>= 0.85)"), (0.75, "cerca (0.75-0.85)"),
                 (0.65, "zona gris (0.65-0.75)"), (0.0, "lejos (< 0.65)")]
        out = {name: 0 for _, name in edges}
        for c in self.checks:
            s = c.best.score if c.best else 0.0
            for low, name in edges:
                if s >= low:
                    out[name] += 1
                    break
        return out


def _embed(texts: list[str], embed_fn) -> np.ndarray:
    if embed_fn is None:                       # lazy import: fast tests never load the model
        from src.ai.rag.embeddings import embed as embed_fn
    return np.asarray(embed_fn(texts), dtype=float)


def find_duplicates(sources: AlignmentSources, embed_fn=None,
                    settings: AlignmentSettings | None = None,
                    threshold: float | None = None,
                    vectors: dict[str, np.ndarray] | None = None,
                    name_match: float | None = None) -> DuplicateReport:
    """Compare every AI concept with the compatible OntoPriv entities and mark it.

    ``vectors`` (key -> embedding of ``semantic_text()``) can be passed to reuse embeddings
    already computed for the DPV ranking; missing ones are computed here."""
    if settings is None:
        settings = alignment_settings()
    if threshold is None:
        threshold = duplicate_threshold()
    if name_match is None:
        name_match = name_match_threshold()

    ai = sources.of_origin(ORIGIN_AI)
    onto = [c for c in sources.of_origin(ORIGIN_ONTOLOGY)
            if {"class", "property"} & set(c.kinds)]      # individuals are not duplicate targets
    report = DuplicateReport(threshold=threshold, settings=settings, name_match=name_match)
    if not ai:
        return report
    if not onto:
        report.checks = [DuplicateCheck(concept=c, status=STATUS_NEW) for c in ai]
        return report

    lex = lexical_matrix([source_lexical_text(c) for c in ai],
                         [source_lexical_text(c) for c in onto])

    vectors = dict(vectors or {})
    missing = [c for c in ai + onto if c.key not in vectors]
    if missing:
        new_vecs = _embed([c.semantic_text() for c in missing], embed_fn)
        vectors.update({c.key: v for c, v in zip(missing, new_vecs)})
    a = np.vstack([vectors[c.key] for c in ai])
    o = np.vstack([vectors[c.key] for c in onto])
    sem = np.clip(a @ o.T, 0.0, 1.0)
    combined = settings.weight_lexical * lex + settings.weight_semantic * sem

    onto_kinds = [set(c.kinds) for c in onto]
    onto_order = np.arange(len(onto))
    for i, concept in enumerate(ai):
        allowed = np.array([bool(set(concept.kinds) & k) for k in onto_kinds])
        idx = onto_order[allowed]
        if idx.size == 0:
            report.checks.append(DuplicateCheck(concept=concept, status=STATUS_NEW))
            continue
        # Best by combined score (ties: lexical, then order).
        by_score = [int(j) for j in idx[np.lexsort((idx, -lex[i, idx], -combined[i, idx]))]]
        chosen = by_score[:MATCHES_KEPT]
        status, reason = STATUS_NEW, None
        if combined[i, chosen[0]] >= threshold:
            status, reason = STATUS_DUPLICATE, REASON_SCORE
        else:
            # Same-name rule: the most similar identifier (ties: combined score, then order).
            by_name = int(idx[np.lexsort((idx, -combined[i, idx], -lex[i, idx]))][0])
            if lex[i, by_name] >= name_match:
                status, reason = STATUS_DUPLICATE, REASON_SAME_NAME
                chosen = [by_name] + [j for j in chosen if j != by_name][:MATCHES_KEPT - 1]
        matches = [OntologyMatch(key=onto[j].key, name=onto[j].name, family=onto[j].family,
                                 lexical=float(lex[i, j]), semantic=float(sem[i, j]),
                                 score=float(combined[i, j])) for j in chosen]
        report.checks.append(DuplicateCheck(concept=concept, status=status, matches=matches,
                                            reason=reason))
    return report


def duplicates_summary(report: DuplicateReport) -> dict:
    """Serializable summary (safe for JSON, the CLI and the web)."""
    return {
        "threshold": report.threshold,
        "name_match": report.name_match,
        "possible_duplicates": len(report.of_status(STATUS_DUPLICATE)),
        "by_same_name": sum(1 for c in report.checks if c.reason == REASON_SAME_NAME),
        "new": len(report.of_status(STATUS_NEW)),
        "bands": report.bands(),
        "checks": [c.as_dict() for c in report.checks],
    }


def render_console(report: DuplicateReport, near: int = 8) -> str:
    dups = report.of_status(STATUS_DUPLICATE)
    new = report.of_status(STATUS_NEW)
    n_same = sum(1 for c in dups if c.reason == REASON_SAME_NAME)
    lines = [
        "¿Ya existe en OntoPriv? (conceptos propuestos por la IA; marca provisional, "
        "la decision es humana en el Sprint 5)",
        (f"  Umbral: {report.threshold:.2f} o mismo nombre (lexico >= {report.name_match:.2f}) | "
         f"posibles duplicados: {len(dups)} ({n_same} por mismo nombre) | nuevos: {len(new)}"),
        "  Distribucion del mejor parecido:",
    ]
    for band, n in report.bands().items():
        lines.append(f"    - {band}: {n}")
    lines.append("  Posibles duplicados:")
    for c in sorted(dups, key=lambda x: (x.reason != REASON_SCORE, -x.best.score)):
        b = c.best
        tag = " (mismo nombre)" if c.reason == REASON_SAME_NAME else ""
        lines.append(f"    - {c.concept.name} ({c.concept.label}) = {b.name} "
                     f"{b.score:.2f} [lex {b.lexical:.2f} | sem {b.semantic:.2f}]{tag}")
    close_new = sorted((c for c in new if c.best), key=lambda x: -x.best.score)[:near]
    if close_new:
        lines.append("  Nuevos mas cercanos al umbral (revisar):")
        for c in close_new:
            b = c.best
            lines.append(f"    - {c.concept.name} ({c.concept.label}) ~ {b.name} "
                         f"{b.score:.2f} [lex {b.lexical:.2f} | sem {b.semantic:.2f}]")
    return "\n".join(lines)


def _run() -> None:
    """Demo: mark the AI proposals as possible duplicates of OntoPriv or new (loads the model)."""
    from pathlib import Path
    from src.config import load_config
    from src.ingest.ontology_loader import load_ontology
    from src.alignment.sources import build_alignment_sources

    onto_path = load_config().get("inputs", {}).get("ontology", "data/input/ontopriv.rdf")
    if not Path(onto_path).exists():
        print(f"No se encontro la ontologia base en '{onto_path}'.")
        return
    sources = build_alignment_sources(load_ontology(onto_path))
    if not sources.proposals_found:
        print("No se encontro el archivo de conceptos propuestos por la IA (Sprint 3).")
        return
    print("Calculando embeddings (puede tardar un poco)...")
    print(render_console(find_duplicates(sources)))


if __name__ == "__main__":
    _run()