"""Calibrate the threshold of the automatic proposals with the validated base (S6-T06).

In the "solo ontología" mode nobody reviews the concepts that did not inherit a decision
(S6-T05): the system proposes the best DPV candidate on its own, but ONLY when its score is
high enough. This module measures, on the 552 base concepts that a person decided in Sprint 5,
what would have happened with each threshold if the system had proposed alone:

- proposals: concepts whose best candidate (rank 1) reaches the threshold;
- correct: proposals whose term is one the person approved for that concept;
- precision = correct / proposals (how many automatic proposals are right);
- coverage = correct / reachable, where reachable = concepts whose rank-1 candidate is one the
  person approved (what a perfect threshold could recover);
- the same with the AI filter: a proposal is dropped when the AI said "none" for it.

The chosen threshold is the LOWEST one whose precision reaches ``auto.min_precision`` with at
least ``MIN_PROPOSALS`` proposals (a threshold that keeps 2 proposals proves nothing).

Honest limits, written in the report: 532 of the 552 decisions confirmed an assistant
proposal without changes (not an independent gold standard), so the 30-concept blind sample of
Sprint 4 (``data/reference/alignment-reference-sample.csv``) is checked too at the chosen
threshold. Both use the Sprint 4 candidates (3 candidates and a law article per concept).

CLI: ``py -m src.alignment.calibrate`` prints the table and writes
``<outputs>/threshold-calibration.json``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.alignment.evaluate import GOLD_NONE, read_reference_sample
from src.alignment.inherit import BaseIndex, base_paths, load_base
from src.alignment.mapping_review import STATUS_ALIGNED

CALIBRATION_FILE = "threshold-calibration.json"
THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(9))       # 0.50 .. 0.90
DEFAULT_MIN_PRECISION = 0.75
MIN_PROPOSALS = 10


def auto_settings(cfg: dict | None = None) -> dict:
    """The ``auto`` block of config.yaml with defaults."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("auto", {}) or {}
    s = {"threshold": float(block.get("threshold", 0.75)),
         "min_precision": float(block.get("min_precision", DEFAULT_MIN_PRECISION)),
         "use_llm": bool(block.get("use_llm", True)),
         "llm_max_calls": int(block.get("llm_max_calls", 20))}
    if not 0 < s["threshold"] <= 1 or not 0 < s["min_precision"] <= 1 or s["llm_max_calls"] < 1:
        raise ValueError("Valores del bloque auto de config.yaml fuera de rango.")
    return s


def top1_points(data: dict, base: BaseIndex) -> list[dict]:
    """One point per decided base concept: its rank-1 candidate and the human verdict."""
    first: dict[str, dict] = {}
    for r in data.get("rows", []):
        if r.get("rank") == 1:
            first[r["concept_key"]] = r
    points = []
    for c in base.concepts:
        row = first.get(c.key)
        if row is None:
            continue
        approved = {m["dpv_iri"] for m in c.mappings}
        points.append({"key": c.key, "name": c.name, "status": c.status,
                       "score": float(row.get("score") or 0.0), "dpv_iri": row["dpv_iri"],
                       "dpv_name": row.get("dpv_name"), "ai_relation": row.get("proposed_relation"),
                       "correct": row["dpv_iri"] in approved})
    return points


def sweep(points: list[dict], thresholds=THRESHOLDS, ai_filter: bool = False) -> list[dict]:
    reachable = sum(p["correct"] for p in points)
    out = []
    for t in thresholds:
        chosen = [p for p in points if p["score"] >= t
                  and not (ai_filter and p["ai_relation"] == "none")]
        correct = sum(p["correct"] for p in chosen)
        out.append({"threshold": t, "proposals": len(chosen), "correct": correct,
                    "on_no_match": sum(p["status"] != STATUS_ALIGNED for p in chosen),
                    "precision": round(correct / len(chosen), 4) if chosen else None,
                    "coverage": round(correct / reachable, 4) if reachable else None})
    return out


def choose_threshold(table: list[dict], min_precision: float,
                     min_proposals: int = MIN_PROPOSALS) -> float | None:
    for row in table:                                     # thresholds in increasing order
        if row["proposals"] >= min_proposals and (row["precision"] or 0) >= min_precision:
            return row["threshold"]
    return None


def blind_check(data: dict, reference: list[dict], threshold: float) -> dict:
    """The Sprint 4 blind sample at one threshold (an independent check)."""
    first = {r["concept_key"]: r for r in data.get("rows", []) if r.get("rank") == 1}
    proposals = correct = gold_none_proposed = labelled = 0
    for ref in reference:
        gold = (ref.get("gold_dpv_name") or "").strip()
        row = first.get(ref.get("concept_key"))
        if not gold or row is None:
            continue
        labelled += 1
        if float(row.get("score") or 0) < threshold:
            continue
        proposals += 1
        if gold.lower() == GOLD_NONE:
            gold_none_proposed += 1
        elif gold.lower() == str(row.get("dpv_name") or "").lower():
            correct += 1
    return {"threshold": threshold, "labelled": labelled, "proposals": proposals,
            "correct": correct, "gold_none_proposed": gold_none_proposed,
            "precision": round(correct / proposals, 4) if proposals else None}


def calibrate(data: dict, base: BaseIndex, reference: list[dict] | None = None,
              min_precision: float = DEFAULT_MIN_PRECISION) -> dict:
    points = top1_points(data, base)
    plain, with_ai = sweep(points), sweep(points, ai_filter=True)
    chosen = choose_threshold(with_ai, min_precision)
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "concepts": len(points),
        "aligned": sum(p["status"] == STATUS_ALIGNED for p in points),
        "reachable": sum(p["correct"] for p in points),
        "min_precision": min_precision,
        "min_proposals": MIN_PROPOSALS,
        "chosen_threshold": chosen,
        "sweep": plain,
        "sweep_ai_filter": with_ai,
        "blind_sample": (blind_check(data, reference, chosen)
                         if reference and chosen is not None else None),
        "notes": [
            "Referencia: decisiones del Sprint 5 (552 conceptos); 532 confirman propuestas del "
            "asistente sin cambios, no es un gold standard independiente.",
            "Control independiente: la muestra ciega de 30 conceptos del Sprint 4.",
            "Candidatos del Sprint 4: 3 por concepto y un artículo de la LOPDP como evidencia.",
        ],
    }


def _pct(v) -> str:
    return "-" if v is None else f"{v * 100:.0f} %"


def render_console(result: dict) -> str:
    lines = ["Calibración del umbral de propuestas automáticas (S6-T06)",
             f"  Base: {result['concepts']} conceptos decididos ({result['aligned']} alineados); "
             f"el mejor candidato acierta en {result['reachable']}.",
             "  umbral  propuestas  correctas  sobre 'sin corresp.'  precisión  cobertura"
             "   | con filtro IA: propuestas  precisión"]
    for a, b in zip(result["sweep"], result["sweep_ai_filter"]):
        mark = "  <- elegido" if b["threshold"] == result["chosen_threshold"] else ""
        lines.append(f"  {a['threshold']:.2f}    {a['proposals']:>6}    {a['correct']:>6}"
                     f"    {a['on_no_match']:>10}          {_pct(a['precision']):>6}"
                     f"     {_pct(a['coverage']):>6}   |   {b['proposals']:>6}"
                     f"       {_pct(b['precision']):>6}{mark}")
    chosen = result["chosen_threshold"]
    if chosen is None:
        lines.append(f"  Ningún umbral llega a {_pct(result['min_precision'])} de precisión.")
    else:
        lines.append(f"  Umbral elegido: {chosen:.2f} (el menor con precisión >= "
                     f"{_pct(result['min_precision'])} y al menos {result['min_proposals']} "
                     f"propuestas, con filtro IA).")
    blind = result.get("blind_sample")
    if blind:
        lines.append(f"  Muestra ciega del Sprint 4 con ese umbral: {blind['proposals']} "
                     f"propuestas de {blind['labelled']}, {blind['correct']} correctas "
                     f"({_pct(blind['precision'])}); {blind['gold_none_proposed']} sobre "
                     f"conceptos sin término en el DPV.")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.config import ROOT, load_config
    from src.alignment.export import load_candidate_file
    from src.alignment.evaluate import reference_settings

    parser = argparse.ArgumentParser(prog="py -m src.alignment.calibrate",
                                     description="Calibra el umbral de propuestas (S6-T06).")
    parser.parse_args(argv)
    cfg = load_config()
    paths = base_paths(cfg)
    if not paths["candidates"].exists():
        print(f"Error: falta {paths['candidates']}")
        return 1
    data = load_candidate_file(paths["candidates"])
    base = load_base(paths["candidates"], paths["log"], cfg=cfg)
    ref_path = Path(reference_settings(cfg)["path"])
    ref_path = ref_path if ref_path.is_absolute() else ROOT / ref_path
    reference = read_reference_sample(ref_path) if ref_path.exists() else None
    result = calibrate(data, base, reference, auto_settings(cfg)["min_precision"])
    out = paths["candidates"].parent / CALIBRATION_FILE
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(render_console(result))
    print(f"  Guardado en: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())