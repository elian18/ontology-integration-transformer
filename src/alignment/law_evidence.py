"""The "ley + ontología" mode: the uploaded law gives each concept of the uploaded ontology
the articles closest to it, for a person to review them (S7-T11).

A person uploads their ontology AND a law. The ontology is aligned with the DPV exactly as in
"Alinear mi ontología" (S7-T06, S7-T08). The law does not change those decisions: measured in
S7-T05, giving the decider the article of the law lowered its precision on the base from
75,0 % to 64,1 % with the chosen rule, so the decider keeps the setting it was calibrated with.
What the law adds is EVIDENCE (step "evidence"): for each class and property, the articles of
the law closest in meaning to it (name, label and definition, the text the alignment embeds),
searched in the index of the law that S7-T07 built in the workspace. They go to the column
``law_evidence`` of ``alignment.csv`` ("art. 17 (0,32); art. 4 (0,41)", with the cosine
distance), for the person to confirm them in the comments of Sprint 10. They are NOT written to
the ontology, because no rule reached the precision the project asks of what enters it
automatically (the same treatment as the suggestions of the decider).

The calibration (``py -m src.alignment.law_evidence --calibrar``, no LLM) uses an external
ontology that already says where each concept is defined: GDPRtEXT links 80 of its 235
concepts to 38 articles of the GDPR (``rdfs:isDefinedBy`` -> ``...#article4-1``: 127 links to
paragraphs, 113 concept-article pairs). With the GDPR uploaded as the law, a link is correct
when its article is one of those (recitals do not count: the law is indexed by article). The
grid and the criterion were fixed BEFORE measuring (as the decider rule, S7-T05): ``top_k`` in
1, 2, 3 and a maximum distance in 0.30 .. 0.60 or none; among the rules with at least
``MIN_LINKS`` links and precision >= ``auto.min_precision`` (75 %), the one with the highest
coverage (concepts with a correct article); ties go to the rule with fewer links. The optimism
of choosing on the same 80 concepts is measured with cross-validation (chosen on one random
half, measured on the other, 100 times, seed 7). Precision is a LOWER bound: GDPRtEXT names
the article that DEFINES a concept (mostly article 4, the definitions), and the article about
its topic is also related to it.

Result (S7-T11): no rule reaches 75 % (the best, 1 article up to 0.30: 73,9 % on 23 links;
cross-validation 64,5 %), so the articles stay in the CSV. For the review the rule in
``law_evidence`` of config.yaml is the one with the highest coverage and the fewest links
(``review_rule``): 3 articles up to 0.55, the correct one among them for 53 of the 80 concepts.

CLI: ``py -m src.alignment.law_evidence --calibrar`` and
``py -m src.alignment.law_evidence ONTOLOGIA LEY [--formato turtle] [--sin-ia]`` (the whole
mode).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

METHOD_VERSION = 1
RESULT_FILE = "law-evidence.json"
CALIBRATION_FILE = "law-evidence-calibration.json"
GRID_K = (1, 2, 3)
GRID_DISTANCE = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, None)
MIN_LINKS = 10
CV_REPEATS, CV_SEED = 100, 7
CALIBRATION_ONTOLOGY = "data/input/gdprtext-ontology.ttl"
CALIBRATION_LAW = "data/input/laws/gdpr-en.pdf"
_ARTICLE = re.compile(r"#article(\d+)\b")


@dataclass(frozen=True)
class EvidenceSettings:
    top_k: int = 3
    max_distance: float | None = None

    def label(self) -> str:
        limit = ("sin distancia máxima" if self.max_distance is None
                 else f"distancia <= {self.max_distance:.2f}".replace(".", ","))
        noun = "artículo" if self.top_k == 1 else "artículos"
        return f"hasta {self.top_k} {noun} por concepto, {limit}"


def evidence_settings(cfg: dict | None = None) -> EvidenceSettings:
    """The ``law_evidence`` block of config.yaml (defaults for the keys that are missing)."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("law_evidence", {}) or {}
    d = EvidenceSettings()
    top_k = int(block.get("top_k", d.top_k))
    raw = block.get("max_distance", d.max_distance)
    max_distance = None if raw is None else float(raw)
    if top_k < 1 or (max_distance is not None and not 0 < max_distance <= 2):
        raise ValueError("Valores inválidos en el bloque law_evidence de config.yaml "
                         "(top_k >= 1; max_distance entre 0 y 2, o vacío).")
    return EvidenceSettings(top_k, max_distance)


# --------------------------------------------------------------------------- the search

def search(concepts, *, path, embed_fn=None, n: int = 3) -> dict[str, list[dict]]:
    """The ``n`` articles of the law closest to each concept, nearest first."""
    from src.ai.rag import vector_store
    from src.ingest.law_upload import LAW_COLLECTION
    if not concepts:
        return {}
    if embed_fn is None:                       # lazy: the fast tests never load the model
        from src.ai.rag.embeddings import embed as embed_fn
    if not vector_store.count(name=LAW_COLLECTION, path=path):
        raise ValueError("Primero hay que indexar la ley en el espacio de trabajo (S7-T07).")
    vectors = [[float(x) for x in v] for v in embed_fn([c.semantic_text() for c in concepts])]
    res = vector_store.query(vectors, n, name=LAW_COLLECTION, path=path,
                             include=("metadatas", "distances"))
    out = {}
    for c, metas, dists in zip(concepts, res["metadatas"], res["distances"]):
        hits = [{"number": int(m["number"]), "title": m.get("title") or "",
                 "distance": round(float(d), 4)} for m, d in zip(metas, dists)]
        out[c.key] = sorted(hits, key=lambda h: (h["distance"], h["number"]))
    return out


def select(hits: list[dict], top_k: int, max_distance: float | None) -> list[dict]:
    """The evidence a rule keeps: the first ``top_k`` hits within ``max_distance``."""
    return [h for h in hits[:top_k] if max_distance is None or h["distance"] <= max_distance]


# --------------------------------------------------------------------------- calibration

def ground_truth(graph: Graph, concepts) -> dict[str, set[int]]:
    """Article numbers each concept is defined in, from ``rdfs:isDefinedBy`` (GDPRtEXT)."""
    out = {}
    for c in concepts:
        numbers = {int(m.group(1)) for o in graph.objects(URIRef(c.key), RDFS.isDefinedBy)
                   for m in [_ARTICLE.search(str(o))] if m}
        if numbers:
            out[c.key] = numbers
    return out


def score(rows: list[dict], top_k: int, max_distance: float | None) -> dict:
    """Precision (correct links / links) and coverage (concepts with a correct link)."""
    links = correct = covered = 0
    for r in rows:
        kept = select(r["hits"], top_k, max_distance)
        good = sum(h["number"] in r["truth"] for h in kept)
        links, correct, covered = links + len(kept), correct + good, covered + (good > 0)
    return {"top_k": top_k, "max_distance": max_distance, "links": links, "correct": correct,
            "covered": covered, "concepts": len(rows),
            "precision": round(correct / links, 4) if links else None,
            "coverage": round(covered / len(rows), 4) if rows else None}


def grid_scores(rows: list[dict]) -> list[dict]:
    return [score(rows, k, d) for k in GRID_K for d in GRID_DISTANCE]


def choose(table: list[dict], min_precision: float, min_links: int = MIN_LINKS) -> dict | None:
    """Highest coverage among the rules that reach the precision; ties: fewer links, then the
    smaller ``top_k`` and the shorter distance."""
    ok = [r for r in table if r["links"] >= min_links and r["precision"] is not None
          and r["precision"] >= min_precision]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["coverage"], -r["links"], -r["top_k"],
                                  -(r["max_distance"] if r["max_distance"] is not None else 9)))


def review_rule(table: list[dict], min_links: int = MIN_LINKS) -> dict | None:
    """The rule for the review in the CSV: the highest coverage, whatever its precision (the
    person decides); ties: fewer links."""
    return choose(table, 0.0, min_links)


def cross_validate(rows: list[dict], min_precision: float, *, repeats: int = CV_REPEATS,
                   seed: int = CV_SEED) -> dict:
    """Choose on one random half of the concepts and measure on the other (both ways)."""
    from src.alignment.decider_eval import _nearest_rank
    rng = random.Random(seed)
    half, measured, without = len(rows) // 2, [], 0
    for _ in range(repeats):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        for train, test in ((shuffled[:half], shuffled[half:]),
                            (shuffled[half:], shuffled[:half])):
            best = choose(grid_scores(train), min_precision, max(1, MIN_LINKS // 2))
            if best is None:
                without += 1
                continue
            measured.append(score(test, best["top_k"], best["max_distance"]))
    precisions = [m["precision"] for m in measured if m["precision"] is not None]
    coverages = [m["coverage"] for m in measured]
    mean = (lambda v: round(sum(v) / len(v), 4) if v else None)          # noqa: E731
    return {"repeats": repeats, "seed": seed, "folds": 2 * repeats, "folds_without_rule": without,
            "precision_mean": mean(precisions), "coverage_mean": mean(coverages),
            "precision_p10": _nearest_rank(precisions, 0.1) if precisions else None,
            "precision_p90": _nearest_rank(precisions, 0.9) if precisions else None}


def calibrate(ontology_path=None, law_path=None, *, embed_fn=None, cfg: dict | None = None) -> dict:
    """The grid on GDPRtEXT + GDPR (see the module docstring); no LLM call."""
    from src.alignment.calibrate import auto_settings
    from src.alignment.inherit import user_concepts
    from src.config import ROOT
    from src.ingest.law_upload import chroma_path, prepare_law
    from src.transform.convert import load_graph
    from src.workspace.store import open_law
    ontology_path = Path(ontology_path or ROOT / CALIBRATION_ONTOLOGY)
    law_path = Path(law_path or ROOT / CALIBRATION_LAW)
    for p in (ontology_path, law_path):
        if not p.exists():
            raise FileNotFoundError(f"Falta el archivo de la calibración: {p}")
    ws = open_law(law_path.name, law_path.read_bytes(), cfg=cfg)
    prepare_law(ws, embed_fn=embed_fn)
    concepts = user_concepts(ontology_path)
    truth = ground_truth(load_graph(ontology_path), concepts)
    judged = [c for c in concepts if c.key in truth]
    hits = search(judged, path=chroma_path(ws), embed_fn=embed_fn, n=max(GRID_K))
    rows = [{"key": c.key, "name": c.name, "truth": sorted(truth[c.key]), "hits": hits[c.key]}
            for c in judged]
    min_precision = auto_settings(cfg)["min_precision"]
    table = grid_scores(rows)
    hit_at = {str(k): sum(any(h["number"] in r["truth"] for h in r["hits"][:k]) for r in rows)
              for k in GRID_K}
    return {"method_version": METHOD_VERSION, "ontology": ontology_path.name,
            "law": law_path.name, "concepts": len(concepts), "judged": len(rows),
            "truth_links": sum(len(r["truth"]) for r in rows), "hit_at": hit_at,
            "min_precision": min_precision, "min_links": MIN_LINKS, "grid": table,
            "chosen": choose(table, min_precision, MIN_LINKS),
            "review": review_rule(table, MIN_LINKS),
            "cross_validation": cross_validate(rows, min_precision), "rows": rows}


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}".replace(".", ",") + " %"


def _rule(r: dict) -> str:
    return EvidenceSettings(r["top_k"], r["max_distance"]).label()


def render_calibration(result: dict) -> str:
    n = result["judged"]
    lines = ["Evidencia de la ley: calibración (S7-T11)",
             f"  {result['ontology']} + {result['law']}: {n} conceptos con artículo conocido "
             f"(de {result['concepts']}), {result['truth_links']} enlaces de referencia",
             "  El artículo correcto está entre los primeros: "
             + ", ".join(f"{k}: {v} de {n}" for k, v in result["hit_at"].items()),
             "  Reglas (top_k · distancia máxima · enlaces · precisión · cobertura):"]
    for r in result["grid"]:
        limit = "—" if r["max_distance"] is None else f"{r['max_distance']:.2f}"
        mark = ("  <- para revisar" if r == result["review"] else
                "  <- precisión mínima" if r == result["chosen"] else "")
        lines.append(f"    {r['top_k']} · {limit:>4} · {r['links']:>3} · {_pct(r['precision']):>7}"
                     f" · {_pct(r['coverage']):>7}{mark}")
    chosen, review, cv = result["chosen"], result["review"], result["cross_validation"]
    if chosen is None:
        lines.append(f"  Ninguna regla llega a la precisión mínima ({_pct(result['min_precision'])}"
                     f") con {result['min_links']} enlaces o más: los artículos no entran a la "
                     "ontología.")
    else:
        lines.append(f"  Llega a la precisión mínima: {_rule(chosen)} ({_pct(chosen['precision'])}"
                     f", {chosen['links']} enlaces).")
    if review is not None:
        lines.append(f"  Para revisar en alignment.csv (la mayor cobertura con menos enlaces): "
                     f"{_rule(review)}, {review['links']} enlaces, el correcto para "
                     f"{review['covered']} de {n} conceptos")
        limit = "null" if review["max_distance"] is None else f"{review['max_distance']:.2f}"
        lines.append(f"  Para config.yaml: law_evidence: top_k: {review['top_k']}, "
                     f"max_distance: {limit}")
    else:
        lines.append(f"  Ninguna regla da {result['min_links']} enlaces o más para revisar.")
    lines.append(f"  Validación cruzada ({cv['folds']} mitades): precisión media "
                 f"{_pct(cv['precision_mean'])} (p10 {_pct(cv['precision_p10'])}, p90 "
                 f"{_pct(cv['precision_p90'])}), cobertura media {_pct(cv['coverage_mean'])}, "
                 f"sin regla en {cv['folds_without_rule']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- the workspace

def load_result(ws) -> dict | None:
    path = ws.work_dir / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def evidence_for_workspace(ws, *, embed_fn=None, cfg: dict | None = None) -> dict:
    """The evidence of every class and property of the ontology; save it; mark the step."""
    from src.alignment.inherit import user_concepts
    from src.ingest.law_upload import chroma_path, current_articles
    from src.workspace.store import DONE, ERROR, RUNNING
    settings = evidence_settings(cfg)
    ws.mark_step("evidence", RUNNING)
    try:
        articles = current_articles(ws)
        concepts = user_concepts(ws.input_path())
        hits = search(concepts, path=chroma_path(ws), embed_fn=embed_fn, n=settings.top_k)
        entries = [{"key": c.key, "name": c.name,
                    "articles": select(hits[c.key], settings.top_k, settings.max_distance)}
                   for c in concepts]
        cited = sorted({a["number"] for e in entries for a in e["articles"]})
        result = {"method_version": METHOD_VERSION, "settings": settings.__dict__,
                  "law": articles["file"], "law_articles": articles["n_articles"],
                  "summary": {"concepts": len(entries),
                              "with_evidence": sum(bool(e["articles"]) for e in entries),
                              "links": sum(len(e["articles"]) for e in entries),
                              "articles_cited": len(cited)},
                  "entries": entries}
        (ws.work_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    except Exception as exc:
        ws.mark_step("evidence", ERROR, str(exc)[:300])
        raise
    s = result["summary"]
    ws.mark_step("evidence", DONE, f"{s['with_evidence']} de {s['concepts']} conceptos con "
                                   f"artículos de la ley para revisar ({s['links']} enlaces)",
                 with_evidence=s["with_evidence"], links=s["links"])
    return result


def is_current(result: dict | None, cfg: dict | None = None) -> bool:
    return (bool(result) and result.get("method_version") == METHOD_VERSION
            and result["settings"] == evidence_settings(cfg).__dict__)


def by_key(result: dict | None) -> dict[str, list[dict]]:
    return {e["key"]: e["articles"] for e in (result or {}).get("entries", [])}


def _decimal(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def evidence_text(articles: list[dict]) -> str:
    """For the CSV: "art. 17 (0,32); art. 4 (0,41)"."""
    return "; ".join(f"art. {a['number']} ({_decimal(a['distance'])})" for a in articles)


# --------------------------------------------------------------------------- whole mode

def align_with_law(onto_name: str, onto_data: bytes, law_name: str, law_data: bytes, *,
                   fmt: str = "turtle", use_llm: bool | None = None, embed_fn=None, llm=None,
                   cfg: dict | None = None, base=None, targets=None, progress=None,
                   chroma_path=None, sleep_fn=None):
    """Upload an ontology and a law -> segment and index the law -> the alignment flow of
    S7-T06/T08 with the evidence step before writing (into alignment.csv) -> zip."""
    from src.alignment.user_alignment import align_workspace
    from src.ingest.law_upload import prepare_law
    from src.workspace.store import KIND_LAW, KIND_ONTOLOGY, open_upload
    ws = open_upload({KIND_ONTOLOGY: (onto_name, onto_data), KIND_LAW: (law_name, law_data)},
                     cfg=cfg)
    prepare_law(ws, embed_fn=embed_fn, progress=progress)
    return align_workspace(ws, fmt=fmt, use_llm=use_llm, embed_fn=embed_fn, llm=llm, cfg=cfg,
                           base=base, targets=targets, progress=progress,
                           chroma_path=chroma_path, sleep_fn=sleep_fn)


# --------------------------------------------------------------------------- CLI

def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import base_paths
    from src.alignment.user_alignment import render_console
    from src.config import load_config
    from src.transform.convert import ConversionError, output_format
    from src.workspace.store import WorkspaceError
    parser = argparse.ArgumentParser(prog="py -m src.alignment.law_evidence",
                                     description="Modo ley + ontología: evidencia de la ley "
                                                 "para cada concepto (S7-T11).")
    parser.add_argument("files", nargs="*", help="ONTOLOGIA LEY")
    parser.add_argument("--calibrar", action="store_true",
                        help="calibrar la evidencia con GDPRtEXT + GDPR (sin LLM)")
    parser.add_argument("--formato", default="turtle", help="rdfxml, owl, turtle o jsonld")
    parser.add_argument("--sin-ia", action="store_true", help="no llamar al LLM")
    args = parser.parse_args(argv)
    try:
        if args.calibrar:
            cfg = load_config()
            result = calibrate(cfg=cfg)
            out = base_paths(cfg)["candidates"].parent / CALIBRATION_FILE
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(render_calibration(result))
            print(f"  Guardado en: {out}")
            return 0
        if len(args.files) != 2:
            print("Error: indica la ontología y la ley (o --calibrar).")
            return 1
        onto, law = (Path(f) for f in args.files)
        for p in (onto, law):
            if not p.exists():
                print(f"Error: no existe el archivo {p}")
                return 1
        output_format(args.formato)
        ws, manifest, zip_path = align_with_law(onto.name, onto.read_bytes(), law.name,
                                                law.read_bytes(), fmt=args.formato,
                                                use_llm=False if args.sin_ia else None)
    except (WorkspaceError, ConversionError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(ws, manifest, zip_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())