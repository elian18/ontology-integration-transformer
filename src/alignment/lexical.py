"""Lexical similarity between the concepts to align and the DPV terms (how names are WRITTEN).

This is the first of the two automatic signals of the Scrumban row (similitud léxica +
embeddings). It only compares spelling; meaning is S4-T05's job. It makes no decision and
applies no threshold: it returns a score in [0, 1] for every (source, DPV target) pair.

What is compared (agreed design):
- Source side: the IDENTIFIER only (``SourceConcept.name_text()``: 'DataSubject' ->
  'Data Subject'). For the AI concepts that identifier is English CamelCase; their Spanish
  label never enters here, because comparing letters across languages is meaningless
  ('titular' vs 'Data Subject'). Spanish labels go to the embeddings (S4-T05).
- Target side: the DPV ``skos:prefLabel`` (English), or its humanized local name.

How the score is computed (kept simple so it can be explained in the thesis):
1. Normalize both texts: split CamelCase/underscores, lowercase, strip accents and
   punctuation, drop stop words ('of', 'in', 'the'...), naive singular ('categories' ->
   'category', 'rights' -> 'right').
2. RapidFuzz ``token_sort_ratio`` on the normalized tokens, divided by 100. Word order does not
   matter, extra words lower the score: 'Consent in verification' vs 'Consent' is partial,
   not a perfect match.
The whole source x target matrix is computed at once with ``rapidfuzz.process.cdist``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz, process

from src.alignment.sources import AlignmentSources, SourceConcept
from src.alignment.dpv_targets import DpvTargets, DpvTarget

# Function words that carry no meaning for matching (English + Spanish, since some OntoPriv
# local names are Spanish).
STOP_WORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or",
    "the", "to", "with",
    "de", "del", "el", "en", "la", "las", "los", "para", "por", "un", "una", "y",
})

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch))


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize_for_lexical(text: str | None) -> str:
    """'SpecialCategoriesOfPersonalData' -> 'special category personal data'."""
    if not text:
        return ""
    text = _CAMEL.sub(" ", text.replace("_", " ").replace("-", " "))
    text = _strip_accents(text).lower()
    tokens = [t for t in _NON_WORD.split(text) if t and t not in STOP_WORDS]
    return " ".join(_singular(t) for t in tokens)


def lexical_similarity(a: str | None, b: str | None) -> float:
    """Similarity in [0, 1] between two names, after normalization."""
    na, nb = normalize_for_lexical(a), normalize_for_lexical(b)
    if not na or not nb:
        return 0.0
    return fuzz.token_sort_ratio(na, nb) / 100.0


def source_lexical_text(concept: SourceConcept) -> str:
    """What a source contributes to the lexical match: its identifier, never its label."""
    return concept.name_text()


def target_lexical_text(target: DpvTarget) -> str:
    return target.name_text()


def lexical_matrix(source_texts: list[str], target_texts: list[str]) -> np.ndarray:
    """Score matrix (len(sources) x len(targets)) in [0, 1]."""
    if not source_texts or not target_texts:
        return np.zeros((len(source_texts), len(target_texts)), dtype=float)
    ns = [normalize_for_lexical(t) for t in source_texts]
    nt = [normalize_for_lexical(t) for t in target_texts]
    m = process.cdist(ns, nt, scorer=fuzz.token_sort_ratio, dtype=np.float32, workers=-1)
    m = np.asarray(m, dtype=float) / 100.0
    empty_s = np.array([not s for s in ns])
    empty_t = np.array([not t for t in nt])
    m[empty_s, :] = 0.0
    m[:, empty_t] = 0.0
    return m


@dataclass
class LexicalScores:
    """Lexical score of every source against every DPV target (same order as the inputs)."""
    source_keys: list[str]
    target_iris: list[str]
    matrix: np.ndarray = field(repr=False)

    def score(self, source_key: str, target_iri: str) -> float:
        i = self.source_keys.index(source_key)
        j = self.target_iris.index(target_iri)
        return float(self.matrix[i, j])

    def top(self, i: int, k: int = 3, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """Best k target indices for source row ``i`` (optionally only among ``allowed``)."""
        row = self.matrix[i]
        idx = np.argsort(-row, kind="stable")
        out = []
        for j in idx:
            if allowed is not None and int(j) not in allowed:
                continue
            out.append((int(j), float(row[j])))
            if len(out) == k:
                break
        return out


def score_lexical(sources: AlignmentSources, targets: DpvTargets) -> LexicalScores:
    """Lexical matrix between every source concept and every DPV target."""
    m = lexical_matrix([source_lexical_text(c) for c in sources.concepts],
                       [target_lexical_text(t) for t in targets.targets])
    return LexicalScores(source_keys=[c.key for c in sources.concepts],
                         target_iris=[t.iri for t in targets.targets], matrix=m)


def best_compatible(sources: AlignmentSources, targets: DpvTargets,
                    scores: LexicalScores) -> list[dict]:
    """Best lexical DPV match per source, respecting kind compatibility (for reports/demo)."""
    index_of = {t.iri: j for j, t in enumerate(targets.targets)}
    rows = []
    for i, concept in enumerate(sources.concepts):
        allowed = {index_of[t.iri] for t in targets.compatible_with(concept.kinds)}
        best = scores.top(i, k=1, allowed=allowed)
        if not best:
            continue
        j, s = best[0]
        t = targets.targets[j]
        rows.append({"key": concept.key, "origin": concept.origin, "name": concept.name,
                     "dpv_iri": t.iri, "dpv_label": t.label, "lexical": round(s, 4)})
    return rows


def render_console(rows: list[dict], high: float = 0.90, examples: int = 12) -> str:
    """Spanish summary: how many sources have a near-identical DPV name, with examples."""
    by_origin: dict = {}
    for r in rows:
        by_origin.setdefault(r["origin"], []).append(r)
    lines = ["Parecido lexico con el DPV (solo nombres; referencia, NO es decision):"]
    names = {"ontology": "OntoPriv", "ai": "Propuestos por la IA"}
    for origin, items in by_origin.items():
        n_high = sum(1 for r in items if r["lexical"] >= high)
        mean = sum(r["lexical"] for r in items) / len(items)
        lines.append(f"  {names.get(origin, origin)}: {len(items)} conceptos | "
                     f"nombre casi igual (>= {high:.2f}): {n_high} | promedio del mejor: {mean:.3f}")
    top = sorted(rows, key=lambda r: -r["lexical"])[:examples]
    lines.append("  Coincidencias mas fuertes:")
    for r in top:
        lines.append(f"    - [{names.get(r['origin'], r['origin'])}] {r['name']} -> "
                     f"{r['dpv_label']} ({r['lexical']:.3f})")
    return "\n".join(lines)


def _run() -> None:
    """Demo: lexical scores of OntoPriv + AI proposals against the DPV in config.yaml."""
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
    sources = build_alignment_sources(load_ontology(onto_path))
    targets = build_dpv_targets(load_dpv(dpv_path))
    scores = score_lexical(sources, targets)
    print(render_console(best_compatible(sources, targets, scores)))


if __name__ == "__main__":
    _run()