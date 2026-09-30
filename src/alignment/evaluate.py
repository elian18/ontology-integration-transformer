"""Measure how well the alignment works against a small hand-labelled sample (S4-T09).

The state of the art reports the lack of a gold standard as a limitation, and that results
depend on the chosen thresholds/weights. This module builds a small reference ourselves and
measures, so the thesis can report numbers instead of impressions ("measured, then corrected").

Three steps:
1. ``sample``: draw a stratified random sample of the concepts the AI justified (default 30:
   20 OntoPriv + 10 AI proposals; fixed seed) into ``alignment-reference-sample.csv``. The file
   is BLIND: it shows each concept and its top-k DPV candidates (name, label, definition) but
   NOT the scores nor the AI's relation, so the human label is not anchored to the AI.
2. The human fills two columns per concept: ``gold_dpv_name`` (the best DPV term for the
   concept: one of the candidates, another DPV term found with ``find``, or ``none``) and
   ``gold_relation`` (exactMatch / closeMatch / broadMatch / narrowMatch / relatedMatch / none,
   always FROM the concept TO the DPV term).
3. ``evaluate``: compares the labels with ``alignment-candidates.json`` and reports
   - ranking: Hit@1 and Hit@k (is the gold DPV term the first candidate / among the top-k?);
   - AI type: accuracy of the proposed relation on the gold term, the broad/narrow direction
     swaps (suspected in S4-T08), and whether the AI avoided exact/close matches when the gold
     says there is no DPV counterpart;
   - optionally (``--sweep``) Hit@1/Hit@k for several lexical/semantic weights, recomputing the
     ranking for the sampled concepts only (the embeddings are computed once).
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.alignment.justify import normalize_relation, RELATION_UNTYPED

REFERENCE_FILE = "alignment-reference-sample.csv"
EVALUATION_FILE = "alignment-evaluation.json"
DEFAULT_REFERENCE_DIR = "data/reference"          # tracked in git (data/output is not)
DEFAULT_SIZE = 30
DEFAULT_AI_SHARE = 1 / 3
DEFAULT_SEED = 42
DEFAULT_SWEEP = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

GOLD_NONE = "none"
_BASE_COLUMNS = ["sample_id", "concept_key", "origin", "concept_name", "concept_label",
                 "concept_definition"]
_GOLD_COLUMNS = ["gold_dpv_name", "gold_relation", "notes"]
_SWAPS = {frozenset({"skos:broadMatch", "skos:narrowMatch"})}


def reference_columns(k: int) -> list[str]:
    cand = []
    for i in range(1, k + 1):
        cand += [f"cand{i}_dpv", f"cand{i}_label", f"cand{i}_definition"]
    return _BASE_COLUMNS + cand + _GOLD_COLUMNS


def reference_settings(cfg: dict | None = None) -> dict:
    """``reference_*`` keys of the ``alignment`` block of config.yaml (defaults if missing)."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("alignment", {}) or {}
    out = {
        "path": Path(block.get("reference_sample",
                               f"{DEFAULT_REFERENCE_DIR}/{REFERENCE_FILE}")),
        "size": int(block.get("reference_size", DEFAULT_SIZE)),
        "ai_share": float(block.get("reference_ai_share", DEFAULT_AI_SHARE)),
        "seed": int(block.get("reference_seed", DEFAULT_SEED)),
    }
    if out["size"] < 1 or not 0.0 <= out["ai_share"] <= 1.0:
        raise ValueError("reference_size debe ser >= 1 y reference_ai_share estar entre 0 y 1.")
    return out


# --- 1. sample ------------------------------------------------------------------------------

def _concepts(rows: list[dict]) -> dict[str, list[dict]]:
    """Rows grouped by concept (file order), only concepts the AI was asked to justify."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("needs_justification"):
            out.setdefault(r["concept_key"], []).append(r)
    return out


def make_reference_sample(data: dict, size: int = DEFAULT_SIZE,
                          ai_share: float = DEFAULT_AI_SHARE,
                          seed: int = DEFAULT_SEED) -> list[dict]:
    """Stratified random sample (OntoPriv / AI) of justified concepts, blind to the AI output."""
    groups = _concepts(data["rows"])
    k = max((len(v) for v in groups.values()), default=0)
    by_origin: dict[str, list[str]] = {"ontology": [], "ai": []}
    for key, rows in groups.items():
        by_origin.setdefault(rows[0]["origin"], []).append(key)

    rng = random.Random(seed)
    n_ai = min(len(by_origin["ai"]), round(size * ai_share))
    n_onto = min(len(by_origin["ontology"]), size - n_ai)
    n_ai = min(len(by_origin["ai"]), size - n_onto)          # refill if ontology is short
    chosen = rng.sample(sorted(by_origin["ontology"]), n_onto) + \
        rng.sample(sorted(by_origin["ai"]), n_ai)
    chosen.sort(key=lambda key: (groups[key][0]["origin"], groups[key][0]["concept_name"].lower()))

    sample = []
    for n, key in enumerate(chosen, start=1):
        rows = sorted(groups[key], key=lambda r: r["rank"])
        first = rows[0]
        rec = {"sample_id": n, "concept_key": key, "origin": first["origin"],
               "concept_name": first["concept_name"], "concept_label": first.get("concept_label"),
               "concept_definition": first.get("concept_definition")}
        for i in range(1, k + 1):
            r = rows[i - 1] if i <= len(rows) else {}
            rec[f"cand{i}_dpv"] = r.get("dpv_name")
            rec[f"cand{i}_label"] = r.get("dpv_label")
            definition = r.get("dpv_definition") or ""
            rec[f"cand{i}_definition"] = definition[:220] or None
        rec.update({"gold_dpv_name": None, "gold_relation": None, "notes": None})
        sample.append(rec)
    return sample


def write_reference_sample(sample: list[dict], path: str | Path, force: bool = False) -> Path:
    """Write the blind sample; never overwrite an existing one (human labels) unless forced."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"Ya existe {path}; no se sobrescribe (puede tener etiquetas).")
    path.parent.mkdir(parents=True, exist_ok=True)
    k = sum(1 for c in sample[0] if c.endswith("_dpv")) if sample else 3
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reference_columns(k))
        writer.writeheader()
        for rec in sample:
            writer.writerow({c: ("" if rec.get(c) is None else rec.get(c))
                             for c in reference_columns(k)})
    return path


def read_reference_sample(path: str | Path) -> list[dict]:
    """Read the labelled sample; tolerates Excel re-saving it with ';' or in cp1252."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0] if text else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    return [dict(r) for r in csv.DictReader(text.splitlines(), delimiter=delimiter)]


def count_labelled(reference: list[dict]) -> int:
    """Rows of the sample that already have a ``gold_dpv_name`` (a term or 'none')."""
    return sum(1 for r in reference if str(r.get("gold_dpv_name") or "").strip())


# --- 2. metrics -----------------------------------------------------------------------------

def _clean_name(value) -> str:
    v = str(value or "").strip()
    for prefix in ("dpv:", "https://w3id.org/dpv#"):
        if v.lower().startswith(prefix):
            v = v[len(prefix):]
    return v


@dataclass
class EvaluationReport:
    k: int
    sample_size: int
    labelled: int = 0
    unlabelled: int = 0
    invalid_labels: list[str] = field(default_factory=list)
    with_gold_term: int = 0                 # labelled concepts whose gold is a DPV term
    hit_at_1: int = 0
    hit_at_k: int = 0
    gold_none: int = 0                      # labelled "no DPV counterpart"
    none_respected: int = 0                 # ... and the AI proposed no exact/close match
    type_evaluated: int = 0                 # gold term within top-k and gold relation given
    type_correct: int = 0
    direction_swaps: int = 0
    confusion: dict = field(default_factory=dict)        # "gold -> ai": count
    misses: list[dict] = field(default_factory=list)     # gold term outside the top-k
    by_origin: dict = field(default_factory=dict)        # origin -> {with_gold_term, hit_at_1, hit_at_k}

    @staticmethod
    def rate(num: int, den: int) -> float | None:
        return round(num / den, 3) if den else None

    def summary(self) -> dict:
        d = asdict(self)
        d["hit_at_1_rate"] = self.rate(self.hit_at_1, self.with_gold_term)
        d["hit_at_k_rate"] = self.rate(self.hit_at_k, self.with_gold_term)
        d["type_accuracy"] = self.rate(self.type_correct, self.type_evaluated)
        d["none_respected_rate"] = self.rate(self.none_respected, self.gold_none)
        return d


def evaluate(data: dict, reference: list[dict]) -> EvaluationReport:
    """Compare the hand labels with the candidates file (ranking + AI relation)."""
    groups = {}
    for r in data["rows"]:
        groups.setdefault(r["concept_key"], []).append(r)
    k = max((len(v) for v in groups.values()), default=0)
    rep = EvaluationReport(k=k, sample_size=len(reference))
    confusion: Counter = Counter()
    origins: dict = {}

    for ref in reference:
        gold_name = _clean_name(ref.get("gold_dpv_name"))
        if not gold_name:
            rep.unlabelled += 1
            continue
        rows = sorted(groups.get(ref.get("concept_key"), []), key=lambda r: r["rank"])
        if not rows:
            rep.invalid_labels.append(f"{ref.get('concept_name')}: concepto no esta en el archivo")
            continue
        gold_rel_raw = str(ref.get("gold_relation") or "").strip()
        gold_rel = normalize_relation(gold_rel_raw) if gold_rel_raw else None
        if gold_rel == RELATION_UNTYPED:
            rep.invalid_labels.append(f"{ref.get('concept_name')}: gold_relation '{gold_rel_raw}'")
            gold_rel = None
        rep.labelled += 1

        if gold_name.lower() == GOLD_NONE:
            rep.gold_none += 1
            if not any(r.get("proposed_relation") in ("skos:exactMatch", "skos:closeMatch")
                       for r in rows):
                rep.none_respected += 1
            continue

        o = origins.setdefault(rows[0]["origin"], {"with_gold_term": 0, "hit_at_1": 0,
                                                   "hit_at_k": 0})
        rep.with_gold_term += 1
        o["with_gold_term"] += 1
        names = [r["dpv_name"].lower() for r in rows]
        if names and names[0] == gold_name.lower():
            rep.hit_at_1 += 1
            o["hit_at_1"] += 1
        if gold_name.lower() in names:
            rep.hit_at_k += 1
            o["hit_at_k"] += 1
            row = rows[names.index(gold_name.lower())]
            if gold_rel is not None and row.get("proposed_relation"):
                ai_rel = row["proposed_relation"]
                rep.type_evaluated += 1
                confusion[f"{gold_rel} -> {ai_rel}"] += 1
                if ai_rel == gold_rel:
                    rep.type_correct += 1
                elif frozenset({ai_rel, gold_rel}) in _SWAPS:
                    rep.direction_swaps += 1
        else:
            rep.misses.append({"concept": rows[0]["concept_name"], "gold": gold_name,
                               "candidates": [r["dpv_name"] for r in rows]})

    rep.confusion = dict(confusion.most_common())
    rep.by_origin = origins
    return rep


# --- 3. weight sweep (optional, loads the embedding model) ----------------------------------

def weight_sweep(sources, targets, reference: list[dict], embed_fn=None,
                 weights=DEFAULT_SWEEP, k: int = 3) -> list[dict]:
    """Hit@1 / Hit@k of the labelled concepts for several lexical weights (semantic = 1 - w)."""
    from src.alignment.sources import AlignmentSources
    from src.alignment.lexical import score_lexical
    from src.alignment.candidates import semantic_matrix, rank_candidates, AlignmentSettings

    gold = {}
    for ref in reference:
        name = _clean_name(ref.get("gold_dpv_name"))
        if name and name.lower() != GOLD_NONE:
            gold[ref["concept_key"]] = name.lower()
    concepts = [c for c in sources.concepts if c.key in gold]
    if not concepts:
        return []
    sub = AlignmentSources(sources.ontology_path, sources.proposals_path, concepts=concepts,
                           proposals_found=sources.proposals_found)
    lexical = score_lexical(sub, targets)
    semantic = semantic_matrix([c.semantic_text() for c in concepts],
                               [t.semantic_text() for t in targets.targets], embed_fn)
    out = []
    for w in weights:
        settings = AlignmentSettings(top_k=k, weight_lexical=w, weight_semantic=1.0 - w)
        ranking = rank_candidates(sub, targets, settings=settings, lexical=lexical,
                                  semantic=semantic)
        h1 = hk = 0
        for item in ranking.items:
            names = [c.dpv_name.lower() for c in item.candidates]
            g = gold[item.concept.key]
            h1 += bool(names) and names[0] == g
            hk += g in names
        n = len(ranking.items)
        out.append({"weight_lexical": round(w, 2), "weight_semantic": round(1.0 - w, 2),
                    "concepts": n, "hit_at_1": round(h1 / n, 3), "hit_at_k": round(hk / n, 3)})
    return out


# --- console --------------------------------------------------------------------------------

def _pct(value) -> str:
    return "-" if value is None else f"{value * 100:.1f} %"


def render_console(report: EvaluationReport, sweep: list[dict] | None = None,
                   current_weight: float | None = None) -> str:
    s = report.summary()
    lines = [
        "Medicion de la alineacion contra la muestra etiquetada a mano:",
        (f"  Muestra: {report.sample_size} conceptos | etiquetados: {report.labelled} | "
         f"sin etiquetar: {report.unlabelled}"),
        f"  Con termino DPV correcto: {report.with_gold_term} | sin correspondencia en el DPV: "
        f"{report.gold_none}",
        "  Ranking (lexico + embeddings):",
        f"    - Hit@1 (el correcto sale primero): {report.hit_at_1}/{report.with_gold_term} "
        f"= {_pct(s['hit_at_1_rate'])}",
        f"    - Hit@{report.k} (el correcto esta entre los {report.k}): "
        f"{report.hit_at_k}/{report.with_gold_term} = {_pct(s['hit_at_k_rate'])}",
    ]
    for origin, o in report.by_origin.items():
        name = "OntoPriv" if origin == "ontology" else "IA"
        lines.append(f"      {name}: Hit@1 {o['hit_at_1']}/{o['with_gold_term']}, "
                     f"Hit@{report.k} {o['hit_at_k']}/{o['with_gold_term']}")
    lines += [
        "  Tipo propuesto por la IA (sobre el termino correcto):",
        f"    - Acierto: {report.type_correct}/{report.type_evaluated} = "
        f"{_pct(s['type_accuracy'])}",
        f"    - Direccion invertida (broad <-> narrow): {report.direction_swaps}",
        f"    - Sin correspondencia respetada (ni exact ni close): {report.none_respected}/"
        f"{report.gold_none} = {_pct(s['none_respected_rate'])}",
    ]
    if report.confusion:
        lines.append("  Confusiones (etiqueta humana -> IA):")
        for pair, n in list(report.confusion.items())[:8]:
            lines.append(f"    - {pair}: {n}")
    if report.misses:
        lines.append(f"  Correctos fuera del top-{report.k}:")
        for m in report.misses[:8]:
            lines.append(f"    - {m['concept']}: correcto {m['gold']}, candidatos "
                         f"{', '.join(m['candidates'])}")
    if report.invalid_labels:
        lines.append("  Etiquetas a revisar:")
        for msg in report.invalid_labels[:8]:
            lines.append(f"    - {msg}")
    if sweep:
        lines.append("  Prueba de pesos (mismos embeddings, solo conceptos con termino correcto):")
        for row in sweep:
            mark = "  <- actual" if current_weight is not None and \
                abs(row["weight_lexical"] - current_weight) < 1e-6 else ""
            lines.append(f"    - lexico {row['weight_lexical']:.1f} / semantico "
                         f"{row['weight_semantic']:.1f}: Hit@1 {_pct(row['hit_at_1'])}, "
                         f"Hit@{report.k} {_pct(row['hit_at_k'])}{mark}")
    return "\n".join(lines)


def find_dpv_terms(targets, text: str, limit: int = 15) -> list:
    """DPV terms whose name, label or definition contains ``text`` (helper for labelling)."""
    t = text.strip().lower()
    hits = [x for x in targets.targets
            if t in x.name.lower() or t in (x.label or "").lower()]
    if len(hits) < limit:
        hits += [x for x in targets.targets
                 if x not in hits and t in (x.definition or "").lower()]
    return hits[:limit]


# --- CLI ------------------------------------------------------------------------------------

def _run(argv: list[str] | None = None) -> None:
    """py -m src.alignment.evaluate [sample [--force] | find <texto> | --sweep]"""
    import sys
    from src.config import load_config
    from src.alignment.export import load_candidate_file, CANDIDATES_JSON

    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config()
    ref = reference_settings(cfg)
    out_dir = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    inputs = cfg.get("inputs", {})

    if argv and argv[0] == "find":
        from src.ingest.dpv_loader import load_dpv
        from src.alignment.dpv_targets import build_dpv_targets
        text = " ".join(argv[1:])
        if not text:
            print("Uso: py -m src.alignment.evaluate find <texto>")
            return
        targets = build_dpv_targets(load_dpv(inputs.get("dpv", "vocab/dpv.ttl")))
        hits = find_dpv_terms(targets, text)
        print(f"Terminos del DPV que contienen '{text}': {len(hits)}")
        for x in hits:
            print(f"  - {x.name} | {x.label} | {x.kind} | {(x.definition or '')[:110]}")
        return

    candidates = out_dir / CANDIDATES_JSON
    if not candidates.exists():
        print(f"No existe {candidates}. Corre primero: py -m src.alignment.export")
        return
    data = load_candidate_file(candidates)

    if argv and argv[0] == "sample":
        sample = make_reference_sample(data, ref["size"], ref["ai_share"], ref["seed"])
        try:
            path = write_reference_sample(sample, ref["path"], force="--force" in argv)
        except FileExistsError as exc:
            print(f"{exc}\nSi de verdad quieres otra muestra: "
                  f"py -m src.alignment.evaluate sample --force")
            return
        origins = Counter(s["origin"] for s in sample)
        print(f"Muestra creada: {path}")
        print(f"  {len(sample)} conceptos (OntoPriv {origins.get('ontology', 0)}, "
              f"IA {origins.get('ai', 0)}), semilla {ref['seed']}")
        print("  Llena 'gold_dpv_name' (el mejor termino DPV, o 'none') y 'gold_relation' "
              "(exactMatch, closeMatch, broadMatch, narrowMatch, relatedMatch o none).")
        print("  Para buscar un termino del DPV: py -m src.alignment.evaluate find <texto>")
        return

    if not ref["path"].exists():
        print(f"No existe {ref['path']}. Crea la muestra: py -m src.alignment.evaluate sample")
        return
    reference = read_reference_sample(ref["path"])
    if count_labelled(reference) == 0:
        print(f"La muestra {ref['path']} aun no tiene etiquetas ({len(reference)} conceptos).\n"
              f"  Llena 'gold_dpv_name' y 'gold_relation' en cada fila y vuelve a correr "
              f"este comando.\n"
              f"  Para buscar un termino del DPV: py -m src.alignment.evaluate find <texto>")
        return
    report = evaluate(data, reference)

    sweep = None
    current = data.get("metadata", {}).get("settings", {}).get("weight_lexical")
    if "--sweep" in argv:
        from src.ingest.ontology_loader import load_ontology
        from src.ingest.dpv_loader import load_dpv
        from src.alignment.sources import build_alignment_sources
        from src.alignment.dpv_targets import build_dpv_targets
        print("Recalculando el ranking con varios pesos (carga el modelo de embeddings)...")
        sources = build_alignment_sources(load_ontology(inputs.get("ontology",
                                                                   "data/input/ontopriv.rdf")))
        targets = build_dpv_targets(load_dpv(inputs.get("dpv", "vocab/dpv.ttl")))
        sweep = weight_sweep(sources, targets, reference, k=report.k)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"reference": str(ref["path"]), "metrics": report.summary(), "weight_sweep": sweep}
    (out_dir / EVALUATION_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(render_console(report, sweep, current))
    print(f"  Guardado en: {out_dir / EVALUATION_FILE}")


if __name__ == "__main__":
    _run()