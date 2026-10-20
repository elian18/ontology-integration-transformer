"""Measure the current design and choose where the decider's DPV candidates come from (S7-T03).

No LLM call is made here. Three measurements, all against the automatic reference of
``src/alignment/reference.py`` (your decisions, no hand labelling):

1. BASELINE: the design of Sprint 6 (best candidate only, threshold ``auto.threshold`` = 0.75,
   the SKOS type the AI already gave in Sprint 4) on the 552 base concepts and on the blind
   sample. It must repeat the calibration of S6-T06 (34 proposals, 28 correct) and adds what the
   calibration did not report: coverage over the 173 aligned concepts (not only over the 92
   whose term was rank 1), F1 and SKOS type accuracy. These are the numbers the decider of
   S7-T04 has to beat in S7-T05, measured with the same ``score``.
2. WHERE THE 10 CANDIDATES COME FROM: the decider will choose among N DPV candidates, so the
   right term must be among them. Three sources are compared on the base (173), GDPRtEXT (21)
   and the blind sample (19), counting how often the right term is in the top 1 / 3 / 10:
   - ``hybrid``: the Sprint 4 ranking (0.4 lexical + 0.6 embeddings) with 10 candidates;
   - ``dpv``: a search by meaning in the ChromaDB collection "dpv" (S7-T01);
   - ``dpv_rerank``: the 50 nearest terms of "dpv", re-ranked with the hybrid formula.
   The three use the same stored DPV vectors. The criterion is fixed BEFORE measuring: the most
   hits in the top 10 of the base wins; on a tie, the hybrid ranking (already validated).
3. CONSISTENCY OF THE REFERENCE: how often the blind sample (Sprint 4) and the Sprint 5
   decisions agree on the concepts they share. It says how noisy the reference itself is.

Results: ``<outputs>/alignment-benchmark.json`` and the reference with its leak control in
``<outputs>/decision-reference.json`` (both used again by S7-T05).

CLI: ``py -m src.alignment.benchmark`` (needs the "dpv" collection: py -m src.alignment.knowledge).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from src.ai.rag import vector_store
from src.alignment.candidates import alignment_settings, rank_candidates
from src.alignment.dpv_targets import COMPATIBLE_TARGETS
from src.alignment.inherit import METHOD_LABELS, inherit
from src.alignment.knowledge import DPV_COLLECTION
from src.alignment.lexical import score_lexical
from src.alignment.reference import (REF_BASE, REF_BLIND, REF_GDPRTEXT, REF_LABELS,
                                     REFERENCE_FILE, Reference, agreement, base_reference,
                                     blind_reference, external_reference, score)
from src.alignment.sources import AlignmentSources

BENCHMARK_FILE = "alignment-benchmark.json"
SOURCE_HYBRID, SOURCE_DPV, SOURCE_RERANK = "hybrid", "dpv", "dpv_rerank"
SOURCES = (SOURCE_HYBRID, SOURCE_DPV, SOURCE_RERANK)
SOURCE_LABELS = {SOURCE_HYBRID: "ranking híbrido (actual)",
                 SOURCE_DPV: "colección dpv (por significado)",
                 SOURCE_RERANK: "dpv top 50 + híbrido"}
HITS_AT = (1, 3, 10)
RERANK_POOL = 50
DEFAULT_CANDIDATES = 10


def decider_settings(cfg: dict | None = None) -> dict:
    """The ``decider`` block of config.yaml: where the candidates come from and how many."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("decider", {}) or {}
    source = str(block.get("candidate_source", SOURCE_HYBRID))
    k = int(block.get("candidates", DEFAULT_CANDIDATES))
    if source not in SOURCES or not 3 <= k <= RERANK_POOL:
        raise ValueError("Valores inválidos en el bloque decider de config.yaml "
                         f"(candidate_source: {' | '.join(SOURCES)}; candidates entre 3 y "
                         f"{RERANK_POOL}).")
    return {"candidate_source": source, "candidates": k}


# --------------------------------------------------------------------------- 1. baseline

def baseline_outputs(data: dict, threshold: float, ai_filter: bool = True) -> dict[str, dict]:
    """What the Sprint 6 design proposes for each concept of a Sprint 4 candidates file."""
    out = {}
    for row in data.get("rows", []):
        if row.get("rank") != 1 or float(row.get("score") or 0) < threshold:
            continue
        relation = row.get("proposed_relation")
        if ai_filter and relation == "none":
            continue                                  # "descartada por la IA": no claim at all
        typed = relation if str(relation or "").startswith("skos:") else None
        out[row["concept_key"]] = {"term": row["dpv_iri"], "relation": typed, "none": False}
    return out


# --------------------------------------------------------------------------- 2. candidates

class MemoEmbed:
    """Embeds each distinct text once (the inheritance and the ranking share many texts)."""

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.cache: dict[str, list[float]] = {}

    def __call__(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            for t, v in zip(missing, self.embed_fn(missing)):
                self.cache[t] = [float(x) for x in v]
        return [self.cache[t] for t in texts]


def stored_dpv_vectors(targets, model_name: str, path=None, embed_fn=None) -> np.ndarray:
    """The DPV vectors of the "dpv" collection, in the order of ``targets``.

    Reading them only saves computing the embeddings again. When ChromaDB still cannot read a
    collection it has just built ("Nothing found on disk" after the retries of vector_store,
    seen on Windows) and ``embed_fn`` is given, the same texts the collection was built from
    are embedded again, rounded to float32 as ChromaDB stores them, so the candidates, the
    scores and the decider's cache keys stay the same."""
    meta = vector_store.metadata(DPV_COLLECTION, path)
    count = vector_store.count(name=DPV_COLLECTION, path=path)
    if count != len(targets.targets) or meta.get("embedding_model") != model_name:
        raise ValueError("La colección dpv no está al día con este modelo de embeddings: "
                         "corre primero py -m src.alignment.knowledge")
    ids = [t.iri for t in targets.targets]
    try:
        got = vector_store.get(ids, name=DPV_COLLECTION, path=path, include=("embeddings",))
    except Exception as exc:                          # noqa: BLE001 - only one is handled
        if embed_fn is None or not vector_store.not_on_disk(exc):
            raise
        texts = [t.semantic_text() for t in targets.targets]     # = knowledge.dpv_records
        return np.asarray(embed_fn(texts), dtype=np.float32).astype(float)
    by_id = dict(zip(got["ids"], got["embeddings"]))
    return np.asarray([by_id[i] for i in ids], dtype=float)


def _allowed_kinds(concept) -> tuple[str, ...]:
    allowed: set[str] = set()
    for k in concept.kinds:
        allowed |= COMPATIBLE_TARGETS.get(k, set())
    return tuple(sorted(allowed))


TIE_MARGIN = 10


def stable_order(ids, distances) -> list[tuple]:
    """Nearest first; exact ties (identical texts give identical vectors) are broken by id, so
    every run sees the same list and the decider's cache is hit (ChromaDB does not promise an
    order among equal distances). Ask ChromaDB for ``TIE_MARGIN`` extra results first."""
    return sorted(zip(ids, (float(d) for d in distances)), key=lambda r: (round(r[1], 6), r[0]))


def _query_dpv(concepts, vectors: np.ndarray, n: int, path=None) -> dict[str, list[tuple]]:
    """Nearest DPV terms per concept in the collection: [(iri, similarity), ...]."""
    groups: dict[tuple, list[int]] = {}
    for i, c in enumerate(concepts):
        groups.setdefault(_allowed_kinds(c), []).append(i)
    out: dict[str, list[tuple]] = {}
    for kinds, rows in groups.items():
        if not kinds:
            for i in rows:
                out[concepts[i].key] = []
            continue
        where = {"kind": kinds[0]} if len(kinds) == 1 else {"kind": {"$in": list(kinds)}}
        res = vector_store.query([vectors[i].tolist() for i in rows], n + TIE_MARGIN,
                                 name=DPV_COLLECTION, path=path, where=where,
                                 include=("distances",))
        for i, ids, dists in zip(rows, res["ids"], res["distances"]):
            out[concepts[i].key] = [(iri, 1.0 - d) for iri, d in stable_order(ids, dists)[:n]]
    return out


def candidate_lists(source: str, concepts, targets, concept_vectors: np.ndarray,
                    target_vectors: np.ndarray, *, k: int = DEFAULT_CANDIDATES, path=None,
                    cfg: dict | None = None) -> dict[str, list[str]]:
    """The top-k DPV IRIs of each concept with one of the three sources."""
    if not concepts:
        return {}
    settings = replace(alignment_settings(cfg), top_k=k)
    sources = AlignmentSources(ontology_path="", proposals_path=None, concepts=list(concepts))
    lexical = score_lexical(sources, targets)
    if source == SOURCE_HYBRID:
        semantic = np.clip(concept_vectors @ target_vectors.T, 0.0, 1.0)
        ranking = rank_candidates(sources, targets, settings=settings, lexical=lexical,
                                  semantic=semantic)
        return {it.concept.key: [c.dpv_iri for c in it.candidates] for it in ranking.items}
    pool = _query_dpv(concepts, concept_vectors, k if source == SOURCE_DPV else RERANK_POOL, path)
    if source == SOURCE_DPV:
        return {key: [iri for iri, _ in hits] for key, hits in pool.items()}
    if source != SOURCE_RERANK:
        raise ValueError(f"Origen de candidatos desconocido: {source}")
    index = {t.iri: j for j, t in enumerate(targets.targets)}
    out = {}
    for i, c in enumerate(concepts):
        scored = []
        for iri, sim in pool[c.key]:
            j = index[iri]
            lex = float(lexical.matrix[i, j])
            combined = settings.weight_lexical * lex + settings.weight_semantic * min(max(sim, 0), 1)
            scored.append((-combined, -lex, j, iri))
        out[c.key] = [iri for *_, iri in sorted(scored)[:k]]
    return out


def hits(lists: dict[str, list[str]], reference: Reference, at=HITS_AT) -> dict:
    """How many aligned concepts have an approved term within the first n candidates."""
    aligned = [i for i in reference.items if i.terms]
    ranks = []
    for item in aligned:
        found = [n for n, iri in enumerate(lists.get(item.key, []), start=1) if iri in item.terms]
        ranks.append(min(found) if found else None)
    return {"aligned": len(aligned),
            **{str(n): sum(1 for r in ranks if r is not None and r <= n) for n in at},
            "missing": sum(r is None for r in ranks)}


def choose_source(table: dict) -> str:
    """Pre-declared criterion: most top-10 hits on the base; tie -> top-3; then the hybrid."""
    def key(source):
        h = table[source][REF_BASE]
        return (h[str(HITS_AT[-1])], h[str(HITS_AT[1])], source == SOURCE_HYBRID)
    return max(SOURCES, key=key)


# --------------------------------------------------------------------------- inputs and run

@dataclass
class BenchmarkInputs:
    data: dict                 # Sprint 4 candidates of the base (rank, score and AI type)
    base: object               # BaseIndex: the 552 decided concepts
    targets: object            # DpvTargets: the 1116 alignable DPV terms
    s4_concepts: list          # the 620 SourceConcepts exactly as Sprint 4 ranked them
    external_concepts: list    # SourceConcepts of the external ontology (GDPRtEXT)
    sample: list[dict]         # rows of the blind sample of Sprint 4
    threshold: float           # auto.threshold of the Sprint 6 design


def load_inputs(cfg: dict | None = None) -> BenchmarkInputs:
    """The real files of the base (data/output, data/review, data/input, data/reference)."""
    from src.config import ROOT, load_config
    from src.alignment.calibrate import auto_settings
    from src.alignment.evaluate import read_reference_sample, reference_settings
    from src.alignment.export import load_candidate_file
    from src.alignment.inherit import base_paths, load_base, user_concepts
    from src.alignment.propose import load_dpv_targets
    from src.alignment.sources import build_alignment_sources
    from src.ingest.ontology_loader import load_ontology
    cfg = cfg if cfg is not None else load_config()
    paths = base_paths(cfg)
    if not paths["candidates"].exists():
        raise FileNotFoundError(f"Falta {paths['candidates']}")
    inputs = cfg.get("inputs", {}) or {}
    onto = ROOT / inputs.get("ontology", "data/input/ontopriv.rdf")
    external = ROOT / "data/input/gdprtext-ontology.ttl"
    ref = Path(reference_settings(cfg)["path"])
    ref = ref if ref.is_absolute() else ROOT / ref
    for f in (onto, external, ref):
        if not f.exists():
            raise FileNotFoundError(f"Falta {f}")
    return BenchmarkInputs(
        data=load_candidate_file(paths["candidates"]),
        base=load_base(paths["candidates"], paths["log"], cfg=cfg),
        targets=load_dpv_targets(cfg),
        s4_concepts=build_alignment_sources(load_ontology(onto)).concepts,
        external_concepts=user_concepts(external),
        sample=read_reference_sample(ref),
        threshold=auto_settings(cfg)["threshold"])


def run_benchmark(inputs: BenchmarkInputs, *, embed_fn=None, model_name: str | None = None,
                  path=None, cfg: dict | None = None) -> dict:
    """The three measurements of S7-T03 (no LLM call)."""
    from src.alignment.knowledge import embedding_model_name
    if embed_fn is None:
        from src.ai.rag.embeddings import embed as embed_fn
    model_name = model_name or embedding_model_name()
    embed = MemoEmbed(embed_fn)
    k = decider_settings(cfg)["candidates"]
    base = inputs.base

    base_vectors = np.asarray(embed([c.semantic_text() for c in base.concepts]), float)
    base_ref = base_reference(base, base_vectors, cfg)
    ext = inputs.external_concepts
    ext_vectors = np.asarray(embed([c.semantic_text() for c in ext]), float) if ext else None
    inherited = inherit(ext, base, embed_fn=embed, cfg=cfg).to_dict()
    ext_ref = external_reference(ext, inherited, base, ext_vectors, base_vectors, cfg)
    s4_by_key = {c.key: c for c in inputs.s4_concepts}
    s4_vectors = dict(zip(s4_by_key, embed([c.semantic_text() for c in inputs.s4_concepts])))
    blind_ref = blind_reference(inputs.sample, base_ref, base, s4_by_key, s4_vectors,
                                base_vectors, cfg)
    references = {REF_BASE: base_ref, REF_GDPRTEXT: ext_ref, REF_BLIND: blind_ref}

    baseline = baseline_outputs(inputs.data, inputs.threshold)
    target_vectors = stored_dpv_vectors(inputs.targets, model_name, path)
    ext_by_key = {c.key: (c, v) for c, v in zip(ext, ext_vectors if ext else [])}
    sets = {REF_BASE: [(s4_by_key[i.key], s4_vectors[i.key]) for i in base_ref.items
                       if i.terms and i.key in s4_by_key],
            REF_GDPRTEXT: [ext_by_key[i.key] for i in ext_ref.items if i.terms],
            REF_BLIND: [(s4_by_key[i.key], s4_vectors[i.key]) for i in blind_ref.items
                        if i.terms and i.key in s4_by_key]}
    table = {}
    for source in SOURCES:
        table[source] = {}
        for name, pairs in sets.items():
            concepts = [c for c, _ in pairs]
            vecs = np.asarray([v for _, v in pairs], float) if pairs else np.zeros((0, 1))
            lists = candidate_lists(source, concepts, inputs.targets, vecs, target_vectors, k=k,
                                    path=path, cfg=cfg)
            table[source][name] = hits(lists, references[name])
    winner = choose_source(table)
    configured = decider_settings(cfg)["candidate_source"]
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": model_name, "llm_calls": 0, "candidates": k,
        "references": {name: ref.summary() for name, ref in references.items()},
        "baseline": {"design": "puesto 1 + umbral + tipo de la IA del Sprint 4 (Sprint 6)",
                     "threshold": inputs.threshold,
                     REF_BASE: score(baseline, base_ref), REF_BLIND: score(baseline, blind_ref),
                     "reachable_rank1": table[SOURCE_HYBRID][REF_BASE]["1"]},
        "reference_consistency": agreement(blind_ref, base_ref),
        "candidate_sources": table,
        "winner": winner, "configured": configured, "config_matches": winner == configured,
        "_references": references,
    }


def write_results(result: dict, out_dir: Path) -> dict[str, Path]:
    """alignment-benchmark.json (numbers) and decision-reference.json (reference + leak control)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = result["_references"]
    bench = {k: v for k, v in result.items() if k != "_references"}
    paths = {"benchmark": out_dir / BENCHMARK_FILE, "reference": out_dir / REFERENCE_FILE}
    paths["benchmark"].write_text(json.dumps(bench, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    paths["reference"].write_text(json.dumps({"created": bench["created"],
                                              "references": {n: r.to_dict()
                                                             for n, r in refs.items()}},
                                             indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


# --------------------------------------------------------------------------- console

def _pct(v) -> str:
    """0.8235 -> '82,4 %' (half up, as a person rounds; Spanish decimal comma)."""
    if v is None:
        return "-"
    value = Decimal(str(v)) * 100
    return f"{value.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)} %".replace(".", ",")


def render_console(result: dict) -> str:
    r = result["references"]
    b = r[REF_BASE]
    lines = ["Medidor automático del Sprint 7 (S7-T03) - 0 llamadas al LLM",
             f"  Referencia ({REF_LABELS[REF_BASE]}): {b['aligned']} con término "
             f"({b['correspondences']} correspondencias), {b['no_match']} sin correspondencia",
             f"  Control de fuga: {b['with_forbidden']} conceptos tienen vecinos que no pueden ser "
             f"precedente ({b['forbidden_pairs']} pares: "
             + ", ".join(f"{METHOD_LABELS.get(k, k)} {v}"
                         for k, v in sorted(b["forbidden_by_rule"].items())) + ")",
             f"  GDPRtEXT: {r[REF_GDPRTEXT]['concepts']} conceptos heredados "
             f"({r[REF_GDPRTEXT]['aligned']} con término) | muestra ciega: "
             f"{r[REF_BLIND]['concepts']} ({r[REF_BLIND]['aligned']} con término)"]
    base = result["baseline"][REF_BASE]
    blind = result["baseline"][REF_BLIND]
    lines += [f"Línea base: diseño del Sprint 6 (puesto 1 + umbral "
              f"{result['baseline']['threshold']:.2f} + tipo de la IA guardado)",
              f"  Base: {base['proposals']} propuestas, {base['correct']} correctas | precisión "
              f"{_pct(base['precision'])} | cobertura {_pct(base['coverage'])} de "
              f"{base['aligned']} | F1 {_pct(base['f1'])}",
              f"        propuestas sobre conceptos sin correspondencia: {base['on_no_match']} | "
              f"'sin correspondencia' afirmada: {base['none_claimed']} | tipo SKOS igual al "
              f"tuyo: {base['relation_ok']} de {base['relation_typed']}",
              f"  Muestra ciega: propuestas {blind['proposals']}, correctas {blind['correct']}"]
    c = result["reference_consistency"]
    lines.append(f"Consistencia de la referencia: la muestra ciega (Sprint 4) y tus decisiones "
                 f"del Sprint 5 coinciden en {c['agree']} de {c['shared']} conceptos "
                 f"({_pct(c['rate'])})")
    k = result["candidates"]
    lines.append(f"Origen de los {k} candidatos (término correcto en el puesto 1 / top 3 / "
                 f"top {k}):")
    lines.append(f"  {'':34}{'base':>16}{'GDPRtEXT':>16}{'muestra ciega':>18}")
    for source, row in result["candidate_sources"].items():
        cells = [f"{h['1']}/{h['3']}/{h[str(HITS_AT[-1])]} de {h['aligned']}"
                 for h in (row[REF_BASE], row[REF_GDPRTEXT], row[REF_BLIND])]
        mark = "  <- gana" if source == result["winner"] else ""
        lines.append(f"  {SOURCE_LABELS[source]:<34}{cells[0]:>16}{cells[1]:>16}"
                     f"{cells[2]:>18}{mark}")
    lines.append("  Criterio fijado antes de medir: más aciertos en el top 10 de la base; "
                 "empate -> top 3; después, el híbrido.")
    if result["config_matches"]:
        lines.append(f"  config.yaml ya dice decider.candidate_source: {result['configured']}")
    else:
        lines.append(f"  CAMBIA config.yaml: decider.candidate_source: {result['winner']} "
                     f"(hoy dice {result['configured']})")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import base_paths
    parser = argparse.ArgumentParser(prog="py -m src.alignment.benchmark",
                                     description="Línea base y origen de los candidatos "
                                                 "del decisor (S7-T03), sin llamadas al LLM.")
    parser.parse_args(argv)
    try:
        inputs = load_inputs()
        print("Calculando embeddings de los conceptos (puede tardar un minuto)...")
        result = run_benchmark(inputs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    paths = write_results(result, base_paths()["candidates"].parent)
    print(render_console(result))
    print(f"  Guardado en: {paths['benchmark']}")
    print(f"               {paths['reference']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())