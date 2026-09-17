"""Score every OntoPriv concept by how close it is to the DPV.

This is the second signal the core/profile split (S3-T03) consumes. The intuition, agreed
with the design: a concept close to the DPV is general / GDPR-level and belongs in the
reusable core; a concept far from the DPV is particular to the jurisdiction (LOPDP) and
belongs in the Ecuador profile. This module only produces the score. It fixes NO threshold
and makes NO core/profile decision — that is S3-T03's job.

How the score is computed:
- Each OntoPriv entity is turned into a short text (humanized local name + its label/comment
  when present).
- Each DPV concept (skos:Concept) is turned into a short text (skos:prefLabel + a trimmed
  skos:definition when present).
- Both sets are embedded with the Sprint 2 encoder (all-MiniLM-L6-v2, already L2-normalized,
  384 dims), so cosine similarity is just the dot product.
- For each OntoPriv entity we keep its best-matching DPV concept and that similarity (0-1).

The embedder is injected (``embed_fn``) so the fast tests can pass a deterministic fake and
never touch the network; ``_run`` uses the real Sprint 2 ``embed``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import numpy as np
import rdflib
from rdflib.namespace import RDFS, SKOS

from src.ingest.ontology_loader import OntologyReport
from src.ingest.dpv_loader import DpvReport
from src.core.inventory import build_inventory, OntologyInventory, KIND_CLASS

_CONTEXT_CHARS = 240
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass
class ProximityEntry:
    """One OntoPriv entity with its closest DPV concept and the similarity."""
    iri: str
    local_name: str
    namespace: str
    kinds: tuple[str, ...]
    family: str
    best_dpv_iri: str | None
    best_dpv_label: str | None
    score: float

    def as_dict(self) -> dict:
        return {
            "iri": self.iri,
            "local_name": self.local_name,
            "namespace": self.namespace,
            "kinds": list(self.kinds),
            "family": self.family,
            "best_dpv_iri": self.best_dpv_iri,
            "best_dpv_label": self.best_dpv_label,
            "score": round(self.score, 4),
        }


@dataclass
class DpvProximityReport:
    """All OntoPriv entities scored against the DPV, newest signal for the split."""
    onto_path: str
    dpv_path: str
    n_dpv_concepts: int
    entries: list[ProximityEntry] = field(default_factory=list)

    def bands(self, hi: float = 0.60, lo: float = 0.40) -> dict:
        """Count entities in three reference bands (for choosing the cut in S3-T03)."""
        c = Counter()
        for e in self.entries:
            if e.score >= hi:
                c["cerca (>= %.2f)" % hi] += 1
            elif e.score >= lo:
                c["media (%.2f-%.2f)" % (lo, hi)] += 1
            else:
                c["lejos (< %.2f)" % lo] += 1
        return dict(c)

    def family_means(self) -> dict:
        """Average DPV proximity per class family — a quick read on which families lean core."""
        acc: dict = defaultdict(list)
        for e in self.entries:
            acc[e.family].append(e.score)
        return {fam: round(sum(v) / len(v), 4) for fam, v in
                sorted(acc.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))}


def _humanize(local_name: str) -> str:
    text = local_name.replace("_", " ").replace("-", " ")
    text = _CAMEL.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _trim(text: str, limit: int = _CONTEXT_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def _literal(graph: rdflib.Graph, subject, predicate) -> str | None:
    for obj in graph.objects(subject, predicate):
        if isinstance(obj, rdflib.Literal):
            return str(obj)
    return None


def _entity_text(graph: rdflib.Graph, iri, local_name: str, use_context: bool) -> str:
    parts = [_humanize(local_name)]
    if use_context:
        label = _literal(graph, iri, RDFS.label) or _literal(graph, iri, SKOS.prefLabel)
        comment = _literal(graph, iri, RDFS.comment) or _literal(graph, iri, SKOS.definition)
        if label and label.lower() != parts[0].lower():
            parts.append(label)
        if comment:
            parts.append(_trim(comment))
    return ". ".join(parts)


def _dpv_text(graph: rdflib.Graph, iri, use_context: bool) -> str:
    label = (_literal(graph, iri, SKOS.prefLabel) or _literal(graph, iri, RDFS.label)
             or _humanize(str(iri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]))
    if use_context:
        definition = _literal(graph, iri, SKOS.definition) or _literal(graph, iri, RDFS.comment)
        if definition:
            return f"{label}. {_trim(definition)}"
    return label


def collect_dpv_concepts(dpv: DpvReport, use_context: bool = True) -> tuple[list[str], list[str]]:
    """Return (iris, texts) for every named DPV skos:Concept."""
    if dpv.graph is None:
        raise ValueError("El reporte del DPV no trae grafo cargado.")
    iris, texts = [], []
    for concept in sorted(dpv.graph.subjects(SKOS.prefLabel, None), key=str):
        if not isinstance(concept, rdflib.URIRef):
            continue
        iris.append(str(concept))
        texts.append(_dpv_text(dpv.graph, concept, use_context))
    return iris, texts


def score_against_dpv(report: OntologyReport, dpv: DpvReport, embed_fn=None,
                      inventory: OntologyInventory | None = None,
                      use_context: bool = True) -> DpvProximityReport:
    """Score every OntoPriv entity by its best cosine similarity to a DPV concept."""
    if embed_fn is None:                       # lazy import so tests need no encoder/network
        from src.ai.rag.embeddings import embed as embed_fn
    if report.graph is None:
        raise ValueError("El reporte de la ontologia no trae grafo cargado.")
    if inventory is None:
        inventory = build_inventory(report)

    onto_iris = [rdflib.URIRef(e.iri) for e in inventory.entries]
    onto_texts = [_entity_text(report.graph, iri, e.local_name, use_context)
                  for iri, e in zip(onto_iris, inventory.entries)]
    dpv_iris, dpv_texts = collect_dpv_concepts(dpv, use_context)

    entries: list[ProximityEntry] = []
    if not onto_texts or not dpv_texts:
        return DpvProximityReport(report.path, dpv.path, len(dpv_iris), entries)

    onto_vecs = np.asarray(embed_fn(onto_texts), dtype=float)
    dpv_vecs = np.asarray(embed_fn(dpv_texts), dtype=float)
    sims = onto_vecs @ dpv_vecs.T               # cosine (vectors are L2-normalized)
    best_idx = sims.argmax(axis=1)
    best_score = sims.max(axis=1)

    for e, bi, sc in zip(inventory.entries, best_idx, best_score):
        entries.append(ProximityEntry(
            iri=e.iri,
            local_name=e.local_name,
            namespace=e.namespace,
            kinds=e.kinds,
            family=e.family,
            best_dpv_iri=dpv_iris[int(bi)],
            best_dpv_label=dpv_texts[int(bi)].split(". ")[0],
            score=float(sc),
        ))
    return DpvProximityReport(report.path, dpv.path, len(dpv_iris), entries)


def proximity_summary(report: DpvProximityReport, hi: float = 0.60, lo: float = 0.40) -> dict:
    """Serializable summary (safe for JSON, the CLI and the web)."""
    return {
        "onto_path": report.onto_path,
        "dpv_path": report.dpv_path,
        "n_dpv_concepts": report.n_dpv_concepts,
        "n_scored": len(report.entries),
        "bands": report.bands(hi, lo),
        "family_means": report.family_means(),
        "entries": [e.as_dict() for e in report.entries],
    }


def render_console(report: DpvProximityReport, hi: float = 0.60, lo: float = 0.40) -> str:
    lines = [
        f"Cercania al DPV: {report.onto_path}",
        f"  Conceptos del DPV comparados: {report.n_dpv_concepts}",
        f"  Entidades de OntoPriv puntuadas: {len(report.entries)}",
        "  Distribucion (umbrales de referencia, NO es el corte final):",
    ]
    for band, n in report.bands(hi, lo).items():
        lines.append(f"    - {band}: {n}")
    lines.append("  Cercania media por familia (mayor = mas 'nucleo'):")
    for fam, mean in report.family_means().items():
        lines.append(f"    - {fam}: {mean}")
    top = sorted(report.entries, key=lambda e: -e.score)[:5]
    lines.append("  Ejemplos mas cercanos al DPV:")
    for e in top:
        lines.append(f"    - {e.local_name} -> {e.best_dpv_label} ({e.score:.3f})")
    return "\n".join(lines)


def _run() -> None:
    """Demo: score the base ontology named in config.yaml against the DPV."""
    from pathlib import Path
    from src.config import load_config
    from src.ingest.ontology_loader import load_ontology
    from src.ingest.dpv_loader import load_dpv

    cfg = load_config().get("inputs", {})
    onto_path = cfg.get("ontology", "data/input/ontopriv.rdf")
    dpv_path = cfg.get("dpv", "vocab/dpv.ttl")
    if not Path(onto_path).exists() or not Path(dpv_path).exists():
        print("Falta la ontologia base o el DPV (revisa 'inputs' en config.yaml).")
        return
    report = score_against_dpv(load_ontology(onto_path), load_dpv(dpv_path))
    print(render_console(report))


if __name__ == "__main__":
    _run()