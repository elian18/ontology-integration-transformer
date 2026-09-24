"""Rank the DPV candidates of every concept: meaning (embeddings) + spelling (lexical).

This closes the automatic part of the Scrumban row "similitud léxica + embeddings":
- Semantic similarity: every source concept and every DPV term are embedded with the project
  encoder (EMBEDDING_MODEL; today the multilingual paraphrase-multilingual-MiniLM-L12-v2,
  L2-normalized), so a Spanish label ('titular') can still meet an English DPV term
  ('Data Subject'). Cosine = dot product, clipped to [0, 1].
- Lexical similarity: S4-T04 (identifiers only).
- Combined score = w_lex * lexical + w_sem * semantic (weights in config.yaml, block
  ``alignment``). No threshold is applied: the state of the art reports that results depend on
  the chosen threshold, so every concept keeps its top-k candidates and the human decides
  (Sprint 5). The weights are measured in S4-T09 with a hand-labelled sample.
- Kind filter: a candidate must be compatible with the concept (class/individual -> DPV concept,
  property -> DPV property, punned -> both), see ``COMPATIBLE_TARGETS`` in S4-T03.

The embedder is injected (``embed_fn``) so the fast tests use a deterministic fake and never
load the model; the real run lazily imports the Sprint 2 ``embed``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.alignment.sources import AlignmentSources, SourceConcept, ORIGIN_AI, ORIGIN_ONTOLOGY
from src.alignment.dpv_targets import DpvTargets, COMPATIBLE_TARGETS
from src.alignment.lexical import score_lexical, LexicalScores

DEFAULT_TOP_K = 3
DEFAULT_WEIGHT_LEXICAL = 0.4
DEFAULT_WEIGHT_SEMANTIC = 0.6


@dataclass(frozen=True)
class AlignmentSettings:
    top_k: int = DEFAULT_TOP_K
    weight_lexical: float = DEFAULT_WEIGHT_LEXICAL
    weight_semantic: float = DEFAULT_WEIGHT_SEMANTIC


def alignment_settings(cfg: dict | None = None) -> AlignmentSettings:
    """Read the ``alignment`` block of config.yaml (defaults if missing); weights sum to 1."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("alignment", {}) or {}
    top_k = int(block.get("top_k", DEFAULT_TOP_K))
    w_lex = float(block.get("weight_lexical", DEFAULT_WEIGHT_LEXICAL))
    w_sem = float(block.get("weight_semantic", DEFAULT_WEIGHT_SEMANTIC))
    if top_k < 1:
        raise ValueError("alignment.top_k debe ser 1 o mayor.")
    if w_lex < 0 or w_sem < 0 or (w_lex + w_sem) == 0:
        raise ValueError("Los pesos de alignment deben ser no negativos y no ambos cero.")
    total = w_lex + w_sem
    return AlignmentSettings(top_k=top_k, weight_lexical=w_lex / total,
                             weight_semantic=w_sem / total)


@dataclass
class Candidate:
    """One DPV candidate for one concept."""
    rank: int                  # 1 = best
    dpv_iri: str
    dpv_name: str
    dpv_label: str
    dpv_kind: str
    lexical: float
    semantic: float
    score: float

    def as_dict(self) -> dict:
        return {"rank": self.rank, "dpv_iri": self.dpv_iri, "dpv_name": self.dpv_name,
                "dpv_label": self.dpv_label, "dpv_kind": self.dpv_kind,
                "lexical": round(self.lexical, 4), "semantic": round(self.semantic, 4),
                "score": round(self.score, 4)}


@dataclass
class ConceptCandidates:
    """A source concept with its ranked DPV candidates."""
    concept: SourceConcept
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def as_dict(self) -> dict:
        return {"concept": self.concept.as_dict(),
                "candidates": [c.as_dict() for c in self.candidates]}


@dataclass
class CandidateRanking:
    """Top-k compatible DPV candidates for every source concept."""
    settings: AlignmentSettings
    items: list[ConceptCandidates] = field(default_factory=list)
    semantic_matrix: np.ndarray | None = field(default=None, repr=False)
    lexical: LexicalScores | None = field(default=None, repr=False)

    def of_origin(self, origin: str) -> list[ConceptCandidates]:
        return [it for it in self.items if it.concept.origin == origin]

    def get(self, key: str) -> ConceptCandidates | None:
        return next((it for it in self.items if it.concept.key == key), None)


def semantic_matrix(source_texts: list[str], target_texts: list[str], embed_fn=None) -> np.ndarray:
    """Cosine similarity matrix (sources x targets), clipped to [0, 1]."""
    if not source_texts or not target_texts:
        return np.zeros((len(source_texts), len(target_texts)), dtype=float)
    if embed_fn is None:                       # lazy import: fast tests never load the model
        from src.ai.rag.embeddings import embed as embed_fn
    s = np.asarray(embed_fn(source_texts), dtype=float)
    t = np.asarray(embed_fn(target_texts), dtype=float)
    return np.clip(s @ t.T, 0.0, 1.0)


def _kind_masks(targets: DpvTargets) -> dict[str, np.ndarray]:
    kinds = np.array([t.kind for t in targets.targets])
    return {src_kind: np.isin(kinds, list(allowed)) for src_kind, allowed in
            COMPATIBLE_TARGETS.items()}


def rank_candidates(sources: AlignmentSources, targets: DpvTargets, embed_fn=None,
                    settings: AlignmentSettings | None = None,
                    lexical: LexicalScores | None = None,
                    semantic: np.ndarray | None = None) -> CandidateRanking:
    """Combine lexical + semantic scores and keep the top-k compatible DPV terms per concept.

    ``lexical`` and ``semantic`` can be passed precomputed (e.g. to reuse embeddings); when
    missing they are computed here."""
    if settings is None:
        settings = alignment_settings()
    if lexical is None:
        lexical = score_lexical(sources, targets)
    if semantic is None:
        semantic = semantic_matrix([c.semantic_text() for c in sources.concepts],
                                   [t.semantic_text() for t in targets.targets], embed_fn)
    lex = lexical.matrix
    if lex.shape != semantic.shape:
        raise ValueError("Las matrices lexica y semantica no tienen la misma forma.")

    combined = settings.weight_lexical * lex + settings.weight_semantic * semantic
    masks = _kind_masks(targets)
    n_targets = len(targets.targets)
    target_order = np.arange(n_targets)

    items: list[ConceptCandidates] = []
    for i, concept in enumerate(sources.concepts):
        allowed = np.zeros(n_targets, dtype=bool)
        for k in concept.kinds:
            if k in masks:
                allowed |= masks[k]
        idx = target_order[allowed]
        if idx.size == 0:
            items.append(ConceptCandidates(concept=concept))
            continue
        # Sort by combined score, then lexical, then IRI order (deterministic ties).
        order = np.lexsort((idx, -lex[i, idx], -combined[i, idx]))
        chosen = idx[order][:settings.top_k]
        cands = []
        for rank, j in enumerate(chosen, start=1):
            t = targets.targets[int(j)]
            cands.append(Candidate(rank=rank, dpv_iri=t.iri, dpv_name=t.name,
                                   dpv_label=t.label, dpv_kind=t.kind,
                                   lexical=float(lex[i, j]), semantic=float(semantic[i, j]),
                                   score=float(combined[i, j])))
        items.append(ConceptCandidates(concept=concept, candidates=cands))
    return CandidateRanking(settings=settings, items=items, semantic_matrix=semantic,
                            lexical=lexical)


def ranking_summary(ranking: CandidateRanking) -> dict:
    """Serializable summary (safe for JSON, the CLI and the web)."""
    def stats(items):
        best = [it.best.score for it in items if it.best]
        return {"concepts": len(items),
                "with_candidates": len(best),
                "mean_best_score": round(float(np.mean(best)), 4) if best else None}
    return {
        "settings": {"top_k": ranking.settings.top_k,
                     "weight_lexical": round(ranking.settings.weight_lexical, 4),
                     "weight_semantic": round(ranking.settings.weight_semantic, 4)},
        "ontology": stats(ranking.of_origin(ORIGIN_ONTOLOGY)),
        "ai": stats(ranking.of_origin(ORIGIN_AI)),
        "items": [it.as_dict() for it in ranking.items],
    }


def render_console(ranking: CandidateRanking, examples: int = 10) -> str:
    s = ranking.settings
    summary = ranking_summary(ranking)
    lines = [
        "Candidatos DPV por concepto (lexico + significado; referencia, NO es decision):",
        f"  Configuracion: top_k={s.top_k}, peso lexico={s.weight_lexical:.2f}, "
        f"peso semantico={s.weight_semantic:.2f}",
    ]
    for origin, name in ((ORIGIN_ONTOLOGY, "OntoPriv"), (ORIGIN_AI, "Propuestos por la IA")):
        st = summary["ontology" if origin == ORIGIN_ONTOLOGY else "ai"]
        if st["concepts"]:
            lines.append(f"  {name}: {st['concepts']} conceptos | con candidatos: "
                         f"{st['with_candidates']} | puntaje medio del mejor: {st['mean_best_score']}")
    ai_items = ranking.of_origin(ORIGIN_AI)[:examples]
    if ai_items:
        lines.append("  Ejemplos (propuestos por la IA):")
        for it in ai_items:
            c = it.concept
            label = f" ({c.label})" if c.label else ""
            tops = "; ".join(f"{x.dpv_label} {x.score:.2f} [lex {x.lexical:.2f} | sem {x.semantic:.2f}]"
                             for x in it.candidates)
            lines.append(f"    - {c.name}{label} -> {tops}")
    return "\n".join(lines)


def _run() -> None:
    """Demo: rank DPV candidates for OntoPriv + AI proposals (loads the embedding model)."""
    from pathlib import Path
    from src.config import load_config
    from src.ingest.ontology_loader import load_ontology
    from src.ingest.dpv_loader import load_dpv
    from src.alignment.sources import build_alignment_sources
    from src.alignment.dpv_targets import build_dpv_targets

    cfg = load_config().get("inputs", {})
    onto_path = cfg.get("ontology", "data/input/ontopriv.rdf")
    dpv_path = cfg.get("dpv", "vocab/dpv.ttl")
    if not Path(onto_path).exists() or not Path(dpv_path).exists():
        print("Falta la ontologia base o el DPV (revisa 'inputs' en config.yaml).")
        return
    print("Calculando embeddings (puede tardar un poco la primera vez)...")
    sources = build_alignment_sources(load_ontology(onto_path))
    targets = build_dpv_targets(load_dpv(dpv_path))
    print(render_console(rank_candidates(sources, targets)))


if __name__ == "__main__":
    _run()