"""Build and write the alignment candidates file (JSON + CSV): the Sprint 4 deliverable.

Joins both flows agreed for Sprint 4 into ONE table:
- Flow A: every OntoPriv entity (S4-T02) with its top-k DPV candidates (S4-T05).
- Flow B: every AI-proposed concept with its top-k DPV candidates AND its mark against OntoPriv
  (``possible_duplicate`` / ``new``, S4-T06).
One row per (concept, DPV candidate). Every row starts as ``review_status = "pending"``
(shown as "por validar"): nothing here is approved; that is Sprint 5.

Three columns are left EMPTY on purpose and filled by S4-T08 (the AI): ``proposed_relation``
(SKOS mapping type), ``justification`` and ``evidence_article`` (the law article shown to the
AI as evidence). ``needs_justification`` says which rows S4-T08 must
send to the AI: all OntoPriv rows and the AI concepts marked ``new``; a possible duplicate
inherits the alignment of its OntoPriv entity, so its rows are kept but not sent.

The embeddings are computed ONCE for sources + DPV targets and reused both for the DPV ranking
and for the duplicate check, so the model runs a single time. The JSON is self-contained
(concept and DPV definitions, DPV parents, law articles), so S4-T08 and the web view can work
from it without reloading the ontology or the DPV.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.alignment.sources import AlignmentSources, ORIGIN_AI, ORIGIN_ONTOLOGY
from src.alignment.dpv_targets import DpvTargets
from src.alignment.candidates import (alignment_settings, AlignmentSettings, rank_candidates,
                                      CandidateRanking)
from src.alignment.duplicates import (find_duplicates, duplicate_threshold, name_match_threshold,
                                      DuplicateReport, STATUS_DUPLICATE, STATUS_NEW)

CANDIDATES_JSON = "alignment-candidates.json"
CANDIDATES_CSV = "alignment-candidates.csv"
REVIEW_PENDING = "pending"          # shown as "por validar" in console and web

COLUMNS = [
    "concept_key", "origin", "concept_name", "concept_label", "concept_kinds",
    "concept_family", "concept_definition", "concept_articles",
    "duplicate_status", "duplicate_of", "duplicate_of_name", "duplicate_score",
    "duplicate_reason",
    "rank", "dpv_iri", "dpv_name", "dpv_label", "dpv_kind", "dpv_definition", "dpv_parents",
    "lexical", "semantic", "score",
    "needs_justification", "proposed_relation", "justification", "evidence_article",
    "review_status",
]
_LIST_COLUMNS = ("concept_kinds", "concept_articles", "dpv_parents")
_INT_COLUMNS = ("rank", "evidence_article")     # nullable ints: CSV shows 29, not 29.0


@dataclass
class CandidateTable:
    """All candidate rows plus the metadata needed to reproduce them."""
    rows: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    ranking: CandidateRanking | None = field(default=None, repr=False)
    duplicates: DuplicateReport | None = field(default=None, repr=False)

    def counts(self) -> dict:
        concepts: dict[str, dict] = {}
        for r in self.rows:
            concepts.setdefault(r["concept_key"], r)
        firsts = list(concepts.values())
        return {
            "rows": len(self.rows),
            "concepts": len(firsts),
            "ontology": sum(1 for r in firsts if r["origin"] == ORIGIN_ONTOLOGY),
            "ai": sum(1 for r in firsts if r["origin"] == ORIGIN_AI),
            "ai_possible_duplicates": sum(1 for r in firsts
                                          if r["duplicate_status"] == STATUS_DUPLICATE),
            "ai_new": sum(1 for r in firsts if r["duplicate_status"] == STATUS_NEW),
            "concepts_to_justify": sum(1 for r in firsts if r["needs_justification"]),
            "rows_to_justify": sum(1 for r in self.rows if r["needs_justification"]),
        }


def compute_vectors(sources: AlignmentSources, targets: DpvTargets,
                    embed_fn=None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Embed every source and every DPV target in ONE call; returns (source vectors, targets)."""
    if embed_fn is None:                       # lazy import: fast tests never load the model
        from src.ai.rag.embeddings import embed as embed_fn
    src_texts = [c.semantic_text() for c in sources.concepts]
    tgt_texts = [t.semantic_text() for t in targets.targets]
    if not src_texts + tgt_texts:
        return {}, np.zeros((0, 0))
    vecs = np.asarray(embed_fn(src_texts + tgt_texts), dtype=float)
    src_vecs = {c.key: vecs[i] for i, c in enumerate(sources.concepts)}
    return src_vecs, vecs[len(src_texts):]


def build_candidate_table(sources: AlignmentSources, targets: DpvTargets, embed_fn=None,
                          settings: AlignmentSettings | None = None,
                          threshold: float | None = None,
                          name_match: float | None = None) -> CandidateTable:
    """Rank DPV candidates for every concept, mark the AI concepts, and flatten into rows."""
    if settings is None:
        settings = alignment_settings()
    if threshold is None:
        threshold = duplicate_threshold()
    if name_match is None:
        name_match = name_match_threshold()

    src_vecs, tgt_matrix = compute_vectors(sources, targets, embed_fn)
    if sources.concepts and targets.targets:
        s = np.vstack([src_vecs[c.key] for c in sources.concepts])
        semantic = np.clip(s @ tgt_matrix.T, 0.0, 1.0)
    else:
        semantic = np.zeros((len(sources.concepts), len(targets.targets)))

    ranking = rank_candidates(sources, targets, settings=settings, semantic=semantic)
    duplicates = find_duplicates(sources, settings=settings, threshold=threshold,
                                 name_match=name_match, vectors=src_vecs)
    dup_by_key = {c.concept.key: c for c in duplicates.checks}
    target_by_iri = {t.iri: t for t in targets.targets}

    rows: list[dict] = []
    for item in ranking.items:
        c = item.concept
        check = dup_by_key.get(c.key)
        best_dup = check.best if (check and check.status == STATUS_DUPLICATE) else None
        needs = c.origin == ORIGIN_ONTOLOGY or (check is not None and check.status == STATUS_NEW)
        for cand in item.candidates:
            t = target_by_iri[cand.dpv_iri]
            rows.append({
                "concept_key": c.key,
                "origin": c.origin,
                "concept_name": c.name,
                "concept_label": c.label,
                "concept_kinds": list(c.kinds),
                "concept_family": c.family,
                "concept_definition": c.definition,
                "concept_articles": list(c.articles),
                "duplicate_status": check.status if check else None,
                "duplicate_of": best_dup.key if best_dup else None,
                "duplicate_of_name": best_dup.name if best_dup else None,
                "duplicate_score": round(best_dup.score, 4) if best_dup else None,
                "duplicate_reason": check.reason if check else None,
                "rank": cand.rank,
                "dpv_iri": cand.dpv_iri,
                "dpv_name": cand.dpv_name,
                "dpv_label": cand.dpv_label,
                "dpv_kind": cand.dpv_kind,
                "dpv_definition": t.definition,
                "dpv_parents": list(t.parents),
                "lexical": round(cand.lexical, 4),
                "semantic": round(cand.semantic, 4),
                "score": round(cand.score, 4),
                "needs_justification": bool(needs),
                "proposed_relation": None,
                "justification": None,
                "evidence_article": None,
                "review_status": REVIEW_PENDING,
            })

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ontology_path": sources.ontology_path,
        "proposals_path": sources.proposals_path,
        "dpv_path": targets.dpv_path,
        "embedding_model": os.environ.get("EMBEDDING_MODEL"),
        "settings": {
            "top_k": settings.top_k,
            "weight_lexical": round(settings.weight_lexical, 4),
            "weight_semantic": round(settings.weight_semantic, 4),
            "duplicate_threshold": threshold,
            "duplicate_name_match": name_match,
        },
    }
    return CandidateTable(rows=rows, metadata=metadata, ranking=ranking, duplicates=duplicates)


def _csv_value(column: str, value):
    if column in _LIST_COLUMNS:
        return "|".join(str(v) for v in (value or []))
    return value


def write_candidate_files(table: CandidateTable, out_dir: str | Path) -> tuple[Path, Path]:
    """Write the JSON (metadata + counts + rows) and the CSV (rows only, Excel-friendly)."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / CANDIDATES_JSON
    csv_path = out_dir / CANDIDATES_CSV

    payload = {"metadata": table.metadata, "counts": table.counts(), "columns": COLUMNS,
               "rows": table.rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    flat = [{col: _csv_value(col, row.get(col)) for col in COLUMNS} for row in table.rows]
    frame = pd.DataFrame(flat, columns=COLUMNS)
    for col in _INT_COLUMNS:
        frame[col] = frame[col].astype("Int64")
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")   # BOM: Excel shows accents
    return json_path, csv_path


def load_candidate_file(path: str | Path) -> dict:
    """Read a candidates JSON written by ``write_candidate_files``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def justified_rows(path: str | Path) -> int:
    """How many rows of an existing candidates file already carry the AI's type (S4-T08).

    Used to refuse overwriting AI work (quota already spent) unless ``--force`` is given."""
    p = Path(path)
    if not p.exists():
        return 0
    try:
        rows = load_candidate_file(p).get("rows", [])
    except (OSError, json.JSONDecodeError):
        return 0
    return sum(1 for r in rows if r.get("proposed_relation") is not None)


def render_console(table: CandidateTable, json_path=None, csv_path=None) -> str:
    c = table.counts()
    s = table.metadata.get("settings", {})
    lines = [
        "Archivo de candidatos de alineacion con el DPV (todo queda 'por validar'):",
        (f"  Conceptos: {c['concepts']} (OntoPriv {c['ontology']}, IA {c['ai']}: "
         f"{c['ai_possible_duplicates']} posibles duplicados, {c['ai_new']} nuevos)"),
        f"  Filas: {c['rows']} (hasta {s.get('top_k')} candidatos DPV por concepto)",
        (f"  A justificar con la IA en S4-T08: {c['concepts_to_justify']} conceptos "
         f"({c['rows_to_justify']} filas); los posibles duplicados heredan la alineacion "
         f"de OntoPriv"),
        f"  Modelo de embeddings: {table.metadata.get('embedding_model') or '(no definido)'}",
    ]
    if json_path:
        lines.append(f"  JSON: {json_path}")
    if csv_path:
        lines.append(f"  CSV:  {csv_path}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> None:
    """Build the candidates for OntoPriv + AI proposals and write them (loads the model).

    Refuses to overwrite a file that already holds AI justifications unless ``--force``."""
    import sys
    from src.config import load_config
    from src.ingest.ontology_loader import load_ontology
    from src.ingest.dpv_loader import load_dpv
    from src.alignment.sources import build_alignment_sources
    from src.alignment.dpv_targets import build_dpv_targets

    cfg = load_config()
    inputs = cfg.get("inputs", {})
    onto_path = inputs.get("ontology", "data/input/ontopriv.rdf")
    dpv_path = inputs.get("dpv", "vocab/dpv.ttl")
    out_dir = cfg.get("outputs", {}).get("dir", "data/output")
    if not Path(onto_path).exists() or not Path(dpv_path).exists():
        print("Falta la ontologia base o el DPV (revisa 'inputs' en config.yaml).")
        return
    argv = sys.argv[1:] if argv is None else argv
    already = justified_rows(Path(out_dir) / CANDIDATES_JSON)
    if already and "--force" not in argv:
        print(f"El archivo de candidatos ya tiene {already} filas justificadas por la IA "
              f"(S4-T08). Regenerarlo las borraria.\n"
              f"Si de verdad quieres empezar de cero: py -m src.alignment.export --force")
        return
    sources = build_alignment_sources(load_ontology(onto_path))
    if not sources.proposals_found:
        print("Aviso: no se encontro el archivo de conceptos de la IA; solo se alinea OntoPriv.")
    targets = build_dpv_targets(load_dpv(dpv_path))
    print("Calculando embeddings y candidatos (puede tardar un poco)...")
    table = build_candidate_table(sources, targets)
    json_path, csv_path = write_candidate_files(table, out_dir)
    print(render_console(table, json_path, csv_path))


if __name__ == "__main__":
    _run()