"""Before/after evaluation of the decider and calibration of its fixed rule (S7-T05).

Everything is measured against your decisions (``decision-reference.json`` of S7-T03), with no
hand labelling, and compared with the Sprint 6 design measured the same way
(``alignment-benchmark.json``). Three configurations of the decider (S7-T04):

- ``sin_precedentes``: 10 candidates with definitions and parents, no precedents;
- ``con_precedentes``: the same plus the 4 closest decisions of the base (production setting
  of the "solo ontología" mode);
- ``con_ley``: the same plus the LOPDP article that supports the concept (the AI concepts
  carry their source article; the OntoPriv entities get the closest article of the indexed
  law, "normativa"; the same evidence Sprint 4 used).

Leave one out: each of the 552 base concepts is decided as if it were new, never seeing itself
or its twins as precedents (leak control). The blind sample of Sprint 4 comes almost free
(23 of its 30 concepts are in the base and their answers are cached); GDPRtEXT (22 known
concepts) is decided with the first two configurations.

The fixed rule is chosen with a grid declared BEFORE running (``RULE_GRID``): which
confidences become an automatic answer, and a minimum hybrid score. Criterion of Sprint 6
(precision over coverage, nobody reviews the automatic answers): among the rules with at least
``MIN_PROPOSALS`` proposals and precision >= ``auto.min_precision`` (75 %) on the base with
``con_precedentes``, the one with the highest coverage; ties go to the more conservative rule.
The chosen rule is then only CONFIRMED (not re-tuned) on the blind sample and GDPRtEXT.

Second grid (``RULE_GRID_2``). The first run (October 2026) found no rule of the first grid
with 75 % precision, and its error analysis found two causes: the weak SKOS relations
(closeMatch and relatedMatch) and the concepts that are only properties (you aligned 44 of 352).
So a second grid was declared AFTER that analysis and BEFORE measuring it: high confidence, a
minimum score (0.0, 0.5, 0.6), the relations accepted for classes (all, all but relatedMatch,
only exactMatch) and for properties (the same, only exactMatch, never automatic). It is only
searched when the first grid has no rule, with the same criterion. Because it was declared
after looking at the data, its optimism is measured with cross-validation
(``cross_validate``): the rule is chosen on one random half of the base and measured on the
other half, 100 times with a fixed seed, so the expected precision on concepts not used to
choose it is reported next to the precision on the whole base. The thesis must say that this
grid came from the error analysis.

Suggestions (option D): with the chosen rule, the terms the LLM chose with high confidence that
the rule does not accept are counted as suggestions (never written into the ontology), with how
many of them match your decisions.

Quota: about 56 calls per configuration on the base + 1 for the blind sample + 3 for GDPRtEXT
(≈ 177 in total, gemini-3.1-flash-lite free tier: 500 a day). Every answer is cached, so the
run can be split: ``--max-llamadas`` limits a run, a 429 stops it cleanly, and the next run
continues where it stopped and prints the table with what is complete.

CLI: ``py -m src.alignment.decider_eval [--max-llamadas 200]``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from src.alignment.decider import (CACHE_FILE, PROMPT, SKOS_EXACT, SKOS_STRONG, STATUS_DECIDED,
                                   STATUS_INVALID, STATUS_PENDING, apply_rule, decide,
                                   settings_from_config, suggested)
from src.alignment.reference import (REF_BASE, REF_BLIND, REF_GDPRTEXT, REFERENCE_FILE, RefItem,
                                     Reference, score)

EVALUATION_FILE = "decider-evaluation.json"
CONFIG_PLAIN, CONFIG_PRECEDENTS, CONFIG_LAW = "sin_precedentes", "con_precedentes", "con_ley"
CONFIGS = (CONFIG_PLAIN, CONFIG_PRECEDENTS, CONFIG_LAW)
CONFIG_LABELS = {CONFIG_PLAIN: "Decisor sin precedentes",
                 CONFIG_PRECEDENTS: "Decisor con precedentes",
                 CONFIG_LAW: "Decisor con precedentes + ley"}
RULE_GRID = tuple({"accept": accept, "min_score": min_score}
                  for accept in (("alta",), ("alta", "media"))
                  for min_score in (0.0, 0.4, 0.5))
GRID_FIRST, GRID_SECOND = "primera", "segunda"
CLASS_RELATIONS = (("todas", None), ("sin related", SKOS_STRONG), ("solo exact", SKOS_EXACT))
PROPERTY_OPTIONS = ("igual", "solo exact", "no responde")


def _second_grid() -> tuple[dict, ...]:
    rules = []
    for min_score in (0.0, 0.5, 0.6):
        for classes, relations in CLASS_RELATIONS:
            for properties in PROPERTY_OPTIONS:
                if properties == "solo exact" and relations == SKOS_EXACT:
                    continue                          # the same rule as "igual"
                prop = {"igual": relations, "solo exact": SKOS_EXACT, "no responde": ()}
                rules.append({"accept": ("alta",), "min_score": min_score,
                              "relations": relations, "property_relations": prop[properties],
                              "classes": classes, "properties": properties})
    return tuple(rules)


RULE_GRID_2 = _second_grid()
MIN_PROPOSALS = 10
CV_REPEATS = 100
CV_SEED = 7
DEFAULT_BUDGET = 200
EXAMPLES = 15


# --------------------------------------------------------------------------- inputs

def reference_from_dict(data: dict) -> Reference:
    return Reference(data["name"], [RefItem(**i) for i in data["items"]])


@dataclass
class EvalInputs:
    references: dict            # name -> Reference (base, gdprtext, blind)
    concepts: dict              # concept key -> SourceConcept (Sprint 4 sources + GDPRtEXT)
    targets: object             # DpvTargets
    baseline: dict | None       # alignment-benchmark.json["baseline"] (S7-T03)
    min_precision: float = 0.75


def load_eval_inputs(cfg: dict | None = None) -> EvalInputs:
    from src.config import load_config
    from src.alignment.benchmark import BENCHMARK_FILE, load_inputs
    from src.alignment.calibrate import auto_settings
    from src.alignment.inherit import base_paths
    cfg = cfg if cfg is not None else load_config()
    out_dir = base_paths(cfg)["candidates"].parent
    ref_path = out_dir / REFERENCE_FILE
    if not ref_path.exists():
        raise FileNotFoundError(f"Falta {ref_path}: corre primero py -m src.alignment.benchmark")
    refs = json.loads(ref_path.read_text(encoding="utf-8"))["references"]
    bench_path = out_dir / BENCHMARK_FILE
    baseline = (json.loads(bench_path.read_text(encoding="utf-8")).get("baseline")
                if bench_path.exists() else None)
    inputs = load_inputs(cfg)
    concepts = {c.key: c for c in inputs.s4_concepts}
    concepts.update({c.key: c for c in inputs.external_concepts})
    return EvalInputs(references={n: reference_from_dict(d) for n, d in refs.items()},
                      concepts=concepts, targets=inputs.targets, baseline=baseline,
                      min_precision=auto_settings(cfg)["min_precision"])


def lopdp_evidence_fn(cfg: dict | None = None, articles: dict | None = None, retrieve_fn=None):
    """The article that supports a concept, as in Sprint 4 (deterministic, so it is cached).

    AI concepts: their own source article. OntoPriv entities: the closest article of the law
    indexed in "normativa" (Sprint 2)."""
    from src.alignment.justify import _evidence
    if articles is None:
        from src.config import ROOT, load_config
        from src.ingest.legal_segmenter import segment_articles
        from src.ingest.text_loader import load_legal_text
        cfg = cfg if cfg is not None else load_config()
        law = Path((cfg.get("inputs", {}) or {}).get("legal_text", "data/input/lopdp.pdf"))
        law = law if law.is_absolute() else ROOT / law
        segmented = segment_articles(load_legal_text(law).text, source=law.stem)["articles"]
        articles = {a["number"]: a for a in segmented}
    if retrieve_fn is None:
        from src.ai.rag import vector_store
        from src.ai.rag.retriever import retrieve_articles
        if vector_store.count() == 0:
            raise ValueError("La colección 'normativa' está vacía: corre "
                             "py -m src.ingest.cli --index")
        def retrieve_fn(query):
            return retrieve_articles(query, top_k=1)

    def evidence(concept):
        row = {"concept_articles": list(getattr(concept, "articles", []) or []),
               "concept_name": concept.name, "concept_label": concept.label,
               "concept_definition": concept.definition}
        return _evidence(row, articles.get, retrieve_fn)
    return evidence


# --------------------------------------------------------------------------- the rule

def rule_label(rule: dict) -> str:
    label = f"{' + '.join(rule['accept'])}, puntaje >= {rule['min_score']:.1f}"
    if "classes" in rule:
        label += f" | clases: {rule['classes']} | propiedades: {rule['properties']}"
    return label


def rule_kwargs(rule: dict) -> dict:
    """The arguments of ``apply_rule`` for one rule of either grid."""
    kwargs = {"accept": tuple(rule["accept"]), "min_score": rule["min_score"]}
    for key in ("relations", "property_relations"):
        if key in rule:
            kwargs[key] = None if rule[key] is None else tuple(rule[key])
    return kwargs


def rule_json(rule: dict, grid: str) -> dict:
    """The chosen rule as S7-T06 writes it to config.yaml (None = any relation)."""
    kw = rule_kwargs(rule)
    relations = kw.get("relations")
    props = kw.get("property_relations", relations)
    return {"grid": grid, "label": rule_label(rule), "accept": list(kw["accept"]),
            "min_score": kw["min_score"],
            "relations": None if relations is None else list(relations),
            "property_relations": None if props is None else list(props)}


def rule_table(decisions, reference: Reference, grid=None) -> list[dict]:
    """The score of every rule of the grid (default: the first one) on one set of decisions."""
    grid = RULE_GRID if grid is None else grid
    return [{**rule, "label": rule_label(rule),
             "score": score(apply_rule(decisions, **rule_kwargs(rule)), reference)}
            for rule in grid]


def choose_rule(table: list[dict], min_precision: float,
                min_proposals: int | None = None) -> dict | None:
    """Highest coverage among the rules that reach the precision; ties -> more conservative
    (fewer confidence levels, then higher minimum score), then higher precision, then the
    first one declared."""
    min_proposals = MIN_PROPOSALS if min_proposals is None else min_proposals
    ok = [r for r in table if r["score"]["proposals"] >= min_proposals
          and (r["score"]["precision"] or 0) >= min_precision]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["score"]["coverage"] or 0, -len(r["accept"]), r["min_score"],
                                  r["score"]["precision"] or 0))


def choose_final(first: list[dict], second: list[dict] | None,
                 min_precision: float) -> tuple[dict | None, str | None]:
    """The first grid decides; the second one is searched only when the first has no rule."""
    chosen = choose_rule(first, min_precision)
    if chosen is not None:
        return chosen, GRID_FIRST
    chosen = choose_rule(second or [], min_precision)
    return (chosen, GRID_SECOND) if chosen is not None else (None, None)


def _nearest_rank(values: list[float], q: float) -> float:
    """Percentile by nearest rank: the smallest value with at least q of the values <= it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def cross_validate(decisions, reference: Reference, grid=None, min_precision: float = 0.75, *,
                   repeats: int | None = None, seed: int | None = None,
                   min_proposals: int | None = None) -> dict:
    """Choose the rule on one random half of the reference and measure it on the other half.

    ``repeats`` random splits (fixed ``seed``: the same numbers on every machine), both halves
    used once for choosing, so 2 x ``repeats`` measurements on concepts that did not take part
    in the choice. A half needs half the minimum proposals. Precision percentiles are by
    nearest rank (p10: 10 % of the measurements are at or below it)."""
    grid = RULE_GRID_2 if grid is None else grid
    repeats = CV_REPEATS if repeats is None else repeats
    seed = CV_SEED if seed is None else seed
    min_proposals = (math.ceil(MIN_PROPOSALS / 2) if min_proposals is None else min_proposals)
    outputs = {rule_label(r): (r, apply_rule(decisions, **rule_kwargs(r))) for r in grid}
    by_key = reference.by_key()
    keys = sorted(by_key)
    half = len(keys) // 2
    rng = random.Random(seed)
    measured, chosen, without = [], Counter(), 0
    for _ in range(repeats):
        shuffled = keys[:]
        rng.shuffle(shuffled)
        a = Reference(reference.name, [by_key[k] for k in shuffled[:half]])
        b = Reference(reference.name, [by_key[k] for k in shuffled[half:]])
        for train, test in ((a, b), (b, a)):
            table = [{**rule, "label": label, "score": score(out, train)}
                     for label, (rule, out) in outputs.items()]
            best = choose_rule(table, min_precision, min_proposals)
            if best is None:
                without += 1
                continue
            chosen[best["label"]] += 1
            measured.append(score(outputs[best["label"]][1], test))
    precisions = [m["precision"] for m in measured if m["precision"] is not None]
    coverages = [m["coverage"] for m in measured if m["coverage"] is not None]
    mean = (lambda v: round(sum(v) / len(v), 4) if v else None)          # noqa: E731
    return {"repeats": repeats, "seed": seed, "folds": 2 * repeats,
            "folds_without_rule": without, "min_proposals_per_half": min_proposals,
            "precision_mean": mean(precisions),
            "precision_p10": _nearest_rank(precisions, 0.1) if precisions else None,
            "precision_p90": _nearest_rank(precisions, 0.9) if precisions else None,
            "coverage_mean": mean(coverages),
            "proposals_mean": mean([m["proposals"] for m in measured]),
            "most_chosen": [[label, n] for label, n in chosen.most_common(3)]}


# --------------------------------------------------------------------------- running

def _counts(decisions) -> dict:
    out = {"concepts": len(decisions)}
    for status in (STATUS_DECIDED, STATUS_PENDING, STATUS_INVALID):
        out[status] = sum(d.status == status for d in decisions)
    decided = [d for d in decisions if d.status == STATUS_DECIDED]
    out["by_confidence"] = {c: sum(d.confidence == c for d in decided)
                            for c in ("alta", "media", "baja")}
    out["said_none"] = sum(d.none for d in decided)
    return out


def _sets_for(config: str, inputs: EvalInputs) -> list[str]:
    # The LOPDP is the law of the base: GDPRtEXT has no LOPDP evidence (its law comes in S7-T11).
    names = [REF_BASE, REF_BLIND]
    return names if config == CONFIG_LAW else names + [REF_GDPRTEXT]


def evaluate(inputs: EvalInputs, *, llm=None, embed_fn=None, budget: int = DEFAULT_BUDGET,
             configs=CONFIGS, evidence_fn=None, cache_path=None, chroma_path=None,
             sleep_fn=time.sleep, embedding_model: str | None = None,
             cfg: dict | None = None) -> dict:
    """Decide every reference concept with each configuration (within the call budget)."""
    settings = settings_from_config(cfg)
    used = 0
    results: dict[str, dict] = {}
    stopped_by_quota = False
    for config in configs:
        results[config] = {"label": CONFIG_LABELS[config], "sets": {}}
        for name in _sets_for(config, inputs):
            ref = inputs.references.get(name)
            if ref is None or not ref.items:
                continue
            concepts = [inputs.concepts[i.key] for i in ref.items if i.key in inputs.concepts]
            forbidden = {i.key: set(i.forbidden) for i in ref.items}
            remaining = 0 if stopped_by_quota else max(budget - used, 0)
            report = decide(concepts, inputs.targets, llm=llm, embed_fn=embed_fn,
                            settings=replace(settings, max_calls=remaining),
                            forbidden=forbidden, use_precedents=config != CONFIG_PLAIN,
                            evidence_fn=evidence_fn if config == CONFIG_LAW else None,
                            cache_path=cache_path, chroma_path=chroma_path,
                            embedding_model=embedding_model, sleep_fn=sleep_fn, cfg=cfg)
            used += report.calls
            stopped_by_quota = stopped_by_quota or report.quota_exhausted
            counts = _counts(report.decisions)
            results[config]["sets"][name] = {
                "counts": counts, "complete": counts[STATUS_PENDING] == 0,
                "calls": report.calls, "cached": report.cached,
                "errors": report.errors[:3], "rules": rule_table(report.decisions, ref),
                "rules_2": rule_table(report.decisions, ref, RULE_GRID_2),
                "_decisions": report.decisions}
    return _summarize(results, inputs, used, stopped_by_quota)


def _summarize(results: dict, inputs: EvalInputs, used: int, quota: bool) -> dict:
    main = results.get(CONFIG_PRECEDENTS, {}).get("sets", {}).get(REF_BASE)
    chosen, grid, cv = None, None, None
    if main and main["complete"]:
        chosen, grid = choose_final(main["rules"], main.get("rules_2"), inputs.min_precision)
        cv = cross_validate(main["_decisions"], inputs.references[REF_BASE], RULE_GRID_2,
                            inputs.min_precision)
    examples, suggestions = [], None
    if chosen is not None:
        ref = inputs.references[REF_BASE].by_key()
        out = apply_rule(main["_decisions"], **rule_kwargs(chosen))
        by_key = {d.key: d for d in main["_decisions"]}
        for key, o in out.items():
            if o["term"] and not ref[key].terms:
                d = by_key[key]
                examples.append({"key": key, "name": d.name, "dpv_name": d.dpv_name,
                                 "relation": d.relation, "justification": d.justification})
        sug = suggested(main["_decisions"], out)
        suggestions = {"total": len(sug),
                       "correct": sum(s["term"] in ref[k].terms for k, s in sug.items()),
                       "on_no_match": sum(not ref[k].terms for k in sug)}
    pending = sum(s["counts"][STATUS_PENDING] for r in results.values()
                  for s in r["sets"].values())
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": PROMPT.manifest(), "llm_calls_this_run": used, "quota_exhausted": quota,
        "pending": pending, "min_precision": inputs.min_precision,
        "min_proposals": MIN_PROPOSALS, "grid": [rule_label(r) for r in RULE_GRID],
        "grid_2": [rule_label(r) for r in RULE_GRID_2],
        "baseline": inputs.baseline, "configs": results,
        "chosen_rule": rule_json(chosen, grid) if chosen else None,
        "cross_validation": cv, "suggestions": suggestions,
        "proposals_on_no_match": examples[:EXAMPLES],
        "proposals_on_no_match_total": len(examples)}


def to_json(result: dict) -> dict:
    """The result without the Decision objects (they are in the decider cache)."""
    clean = json.loads(json.dumps({k: v for k, v in result.items() if k != "configs"}))
    clean["configs"] = {
        name: {"label": r["label"],
               "sets": {s: {k: v for k, v in data.items() if not k.startswith("_")}
                        for s, data in r["sets"].items()}}
        for name, r in result["configs"].items()}
    return clean


# --------------------------------------------------------------------------- console

def _pct(v) -> str:
    from src.alignment.benchmark import _pct as pct
    return pct(v)


def _row(label: str, s: dict) -> str:
    none = f"{s['none_ok']}/{s['none_wrong']}" if s.get("none_claimed") is not None else "-"
    typed = f"{s['relation_ok']}/{s['relation_typed']}"
    return (f"  {label:<44}{s['proposals']:>6}{s['correct']:>7}{_pct(s['precision']):>10}"
            f"{_pct(s['coverage']):>10}{_pct(s['f1']):>9}{none:>16}{typed:>9}")


def _rule_of(data: dict, rule: dict) -> dict:
    """The row of a rule (by its label) in the tables of one set."""
    return next(r for r in data["rules"] + data.get("rules_2", [])
                if r["label"] == rule["label"])


def _rule_lines(rows: list[dict], result: dict, width: int) -> list[str]:
    chosen = result["chosen_rule"]                    # labels are unique across both grids
    out = []
    for r in rows:
        s = r["score"]
        mark = "  <- elegida" if chosen and r["label"] == chosen["label"] else ""
        out.append(f"    {r['label']:<{width}} propuestas {s['proposals']:>4} | precisión "
                   f"{_pct(s['precision']):>8} | cobertura {_pct(s['coverage']):>8}{mark}")
    return out


def render_console(result: dict) -> str:
    chosen = result["chosen_rule"] or {"label": rule_label(RULE_GRID[0]), "grid": None}
    head = (f"  {'':<44}{'prop.':>6}{'corr.':>7}{'precisión':>10}{'cobertura':>10}"
            f"{'F1':>9}{'ninguno ok/mal':>16}{'tipo':>9}")
    lines = ["Evaluación del decisor (S7-T05): tus 552 decisiones, dejando uno fuera",
             f"  Regla mostrada: {chosen['label']}"
             + ("" if result["chosen_rule"] else " (provisional: aún no hay regla elegida)"),
             head]
    if result.get("baseline"):
        lines.append(_row("Línea base (Sprint 6)", result["baseline"][REF_BASE]))
    for config, r in result["configs"].items():
        base = r["sets"].get(REF_BASE)
        if not base:
            continue
        mark = "" if base["complete"] else f"  (incompleta: {base['counts']['pending']} pendientes)"
        lines.append(_row(r["label"], _rule_of(base, chosen)["score"]) + mark)
    main = result["configs"].get(CONFIG_PRECEDENTS, {}).get("sets", {}).get(REF_BASE)
    floor = (f"{_pct(result['min_precision'])} de precisión con al menos "
             f"{result['min_proposals']} propuestas")
    grid = result["chosen_rule"]["grid"] if result["chosen_rule"] else None
    if main and main["complete"]:
        lines.append(f"  Grilla de reglas en '{CONFIG_LABELS[CONFIG_PRECEDENTS]}' "
                     f"(fijada antes de correr):")
        lines += _rule_lines(main["rules"], result, 30)
        if grid != GRID_FIRST:
            lines.append(f"  Ninguna regla de esta grilla llega a {floor}: se busca en la segunda.")
            lines.append("  Segunda grilla (declarada tras el análisis de errores de la primera: "
                         "relación SKOS y clase/propiedad):")
            width = max([len(r["label"]) for r in main.get("rules_2", [])] or [30])
            lines += _rule_lines(main.get("rules_2", []), result, width)
        if grid is None:
            lines.append(f"  Ninguna regla llega a {floor}, en ninguna de las dos grillas.")
        cv = result.get("cross_validation")
        if cv:
            lines.append(f"  Validación cruzada de la segunda grilla (elegir en una mitad, "
                         f"medir en la otra; {cv['repeats']} repeticiones):")
            if cv["precision_mean"] is None:
                lines.append(f"    sin regla en {cv['folds_without_rule']} de {cv['folds']} "
                             f"mitades")
            else:
                lines.append(f"    precisión {_pct(cv['precision_mean'])} (p10 "
                             f"{_pct(cv['precision_p10'])}, p90 {_pct(cv['precision_p90'])}) | "
                             f"cobertura {_pct(cv['coverage_mean'])} | sin regla en "
                             f"{cv['folds_without_rule']} de {cv['folds']} mitades")
    lines.append("  Confirmación con la regla elegida, sin reajustar:" if result["chosen_rule"]
                 else "  Confirmación con la regla mostrada (provisional), sin reajustar:")
    for config, r in result["configs"].items():
        for name in (REF_BLIND, REF_GDPRTEXT):
            data = r["sets"].get(name)
            if not data:
                continue
            s = _rule_of(data, chosen)["score"]
            where = "muestra ciega" if name == REF_BLIND else "GDPRtEXT"
            lines.append(f"    {r['label']:<36} {where:<14} propuestas {s['proposals']:>3}, "
                         f"correctas {s['correct']:>3} | precisión {_pct(s['precision']):>8} | "
                         f"cobertura {_pct(s['coverage']):>8} de {s['aligned']}")
    sug = result.get("suggestions")
    if sug:
        lines.append(f"  Sugeridas (confianza alta que la regla no acepta): {sug['total']} | "
                     f"coinciden con tu decisión {sug['correct']} | sobre conceptos sin "
                     f"correspondencia {sug['on_no_match']}")
    if result["proposals_on_no_match_total"]:
        lines.append(f"  Propuestas sobre conceptos que dejaste sin correspondencia: "
                     f"{result['proposals_on_no_match_total']} (ejemplos):")
        for e in result["proposals_on_no_match"][:8]:
            rel = str(e["relation"] or "sin tipo").replace("skos:", "")
            lines.append(f"    {e['name'][:40]:<40} -> {e['dpv_name']} {rel}")
    lines.append(f"  Llamadas al LLM en esta corrida: {result['llm_calls_this_run']} | "
                 f"decisiones pendientes: {result['pending']}")
    if result["quota_exhausted"]:
        lines.append("  CUOTA AGOTADA: vuelve a correr mañana; lo ya decidido sale de la caché.")
    elif result["pending"]:
        lines.append("  Quedan pendientes: vuelve a correr (sube --max-llamadas si hace falta).")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import base_paths
    parser = argparse.ArgumentParser(prog="py -m src.alignment.decider_eval",
                                     description="Evaluación antes/después del decisor y "
                                                 "calibración de su regla fija (S7-T05).")
    parser.add_argument("--max-llamadas", type=int, default=DEFAULT_BUDGET,
                        help="llamadas al LLM como máximo en esta corrida (caché: no se repiten)")
    args = parser.parse_args(argv)
    out_dir = base_paths()["candidates"].parent
    try:
        inputs = load_eval_inputs()
        evidence = lopdp_evidence_fn()
        print("Decidiendo (las respuestas ya guardadas salen de la caché)...")
        result = evaluate(inputs, budget=args.max_llamadas, evidence_fn=evidence,
                          cache_path=out_dir / "decider" / CACHE_FILE)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    path = out_dir / EVALUATION_FILE
    path.write_text(json.dumps(to_json(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print(render_console(result))
    print(f"  Guardado en: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())