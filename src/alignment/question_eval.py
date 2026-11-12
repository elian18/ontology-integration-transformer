"""Measure the questions of the AI with simulated people, without labelling (S10-T04).

Result (November 2026, 12 calls): ask.v1 was not adopted. It asked 24 questions and 8 served;
following its recommendation lowered the precision from 75.0 % to 66.3 %; with the person it
reached 80.3 % and 35.3 % against 84.6 % and 44.5 % of the fixed questions. The uploads ask the
fixed questions (``user_questions.fixed_questions``). A new run reads the cache (0 calls).

Simulated uploads. The 552 decisions of the base decided as if they were new (leave one out:
the decisions that calibrated the rule, 75.0 % precision and 29.5 % coverage) and the review of
the AI reviewer of what the rule accepted, both from their caches (0 calls). Each family of
OntoPriv-Core is one upload, and the concepts the AI took from the LOPDP another one: from 3 to
104 concepts, like the real uploads. In each one the AI asks (prompt ask.v1, at most 10
questions) and so do the fixed questions without AI: one question per doubt, in the same order,
with the same maximum (take the suggestion; for a reviewed one, the relation the AI reviewer
proposes or remove it; added after measuring ask.v1, it can only help the person).

Two simulated people answer every question:
- "siempre la recomendada" takes the recommended option;
- "la persona" takes the option that leaves its concepts closest to your decisions of Sprint 5,
  scored per concept: your term with your relation 3, your term with another relation 2,
  without correspondence 1 and another term 0; for a concept you left without correspondence,
  without correspondence 3 and any term 0. Ties go to "Dejar como está". A question "sirve"
  when "la persona" takes another option.

The rule to adopt ask.v1, fixed before measuring (accepted with the backlog of Sprint 10):
1. "siempre la recomendada" lowers neither the precision nor the coverage of the rule alone;
2. at least half of the questions serve;
3. it asks fewer questions than the fixed ones, and with "la persona" neither its precision
   nor its coverage is lower than theirs.
Precision and coverage are those of the decider (``reference.score``): the term against your
decisions, over the 552 concepts.

Quota: one call per upload with doubts (about 13). Every answer is cached
(``data/output/decider/ask-cache.jsonl``): ``--max-llamadas`` limits a run, a 429 stops it, and
the next run goes on from there.

CLI: ``py -m src.alignment.question_eval [--max-llamadas 20]``."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from src.alignment import user_questions as uq
from src.alignment.reference import STATUS_ALIGNED, Reference, score
from src.console import counted

EVALUATION_FILE = "user-questions-evaluation.json"
AI_GROUP = "(conceptos que la IA sacó de la LOPDP)"
DEFAULT_BUDGET = 20
EXAMPLES = 6
RIGHT_RELATION, RIGHT_TERM, NOTHING, WRONG = 3, 2, 1, 0
CONDITIONS = (
    "Siempre la recomendada no baja la precisión ni la cobertura de la regla sola",
    "Sirven al menos la mitad de las preguntas",
    "Menos preguntas que las fijas, sin peor precisión ni cobertura con la persona")


@dataclass
class SimUpload:
    """One family of the base, as an upload: the decider's entries and what it has now."""
    name: str
    entries: list[dict]
    info: dict = field(default_factory=dict)          # key -> label, definition
    audits: dict = field(default_factory=dict)        # key -> answer of the AI reviewer
    index: dict = field(default_factory=dict)         # key -> name, kinds, alignable
    state: dict = field(default_factory=dict)         # key -> DPV IRI -> correspondence


# --------------------------------------------------------------------------- inputs

def entries_of(decisions, accepted: dict, suggestions: dict) -> list[dict]:
    """The decider's decisions as the entries of an upload (accepted / suggested / other)."""
    from src.alignment import propose as dec
    out = []
    for d in decisions:
        a = accepted.get(d.key) or {}
        status = (dec.ACCEPTED if a.get("term") else dec.SUGGESTED if d.key in suggestions
                  else "other")
        out.append({"key": d.key, "name": d.name, "kinds": list(d.kinds), "status": status,
                    "decided_by": "rule", "dpv_iri": d.dpv_iri, "dpv_name": d.dpv_name,
                    "relation": a.get("relation", d.relation), "score": d.score,
                    "confidence": d.confidence, "justification": d.justification,
                    "candidates": list(d.candidates)})
    return out


def uploads_of(entries: list[dict], concepts: dict, audits: dict) -> list[SimUpload]:
    """One simulated upload per family (``concepts``: key -> SourceConcept)."""
    from src.alignment import propose as dec
    groups: dict[str, list[dict]] = {}
    for e in entries:
        family = getattr(concepts.get(e["key"]), "family", None) or AI_GROUP
        groups.setdefault(family, []).append(e)
    out = []
    for name, rows in sorted(groups.items(), key=lambda g: (-len(g[1]), g[0])):
        up = SimUpload(name, rows)
        for e in rows:
            c = concepts.get(e["key"])
            up.info[e["key"]] = {"label": getattr(c, "label", None),
                                 "definition": getattr(c, "definition", None)}
            up.index[e["key"]] = {"key": e["key"], "name": e["name"], "kinds": e["kinds"],
                                  "alignable": True}
            if e["key"] in audits:
                up.audits[e["key"]] = audits[e["key"]]
            if e["status"] == dec.ACCEPTED:
                up.state[e["key"]] = {e["dpv_iri"]: {"relation": e["relation"],
                                                     "dpv_name": e["dpv_name"],
                                                     "decided_by": "rule",
                                                     "version": dec.ALIGNMENT_VERSION}}
        out.append(up)
    return out


def load_inputs(cfg: dict | None = None, *, embed_fn=None, chroma_path=None,
                sleep_fn=time.sleep):
    """(uploads, targets, reference, rule, rule outputs) from the caches of the decider and
    of the AI reviewer, as their measurements left them (0 calls)."""
    from src.alignment import prompt_lab as lab
    from src.alignment.decider import (CACHE_FILE, STATUS_PENDING, apply_rule, decide_prompt,
                                       rule_settings, suggested)
    from src.alignment.decider_eval import load_eval_inputs
    from src.alignment.inherit import base_paths
    from src.alignment.reference import REF_BASE
    from src.config import load_config
    from src.validation import ai_auditor as au
    cfg = cfg if cfg is not None else load_config()
    inputs = load_eval_inputs(cfg)
    out = base_paths(cfg)["candidates"].parent / "decider"
    prompt, rule = decide_prompt(cfg), rule_settings(cfg)
    decisions, concepts, base = [], {}, None
    for half in lab.HALVES:
        _, seen, report, _, base = lab.decide_half(
            inputs, prompt, half, embed_fn=embed_fn, budget=0, cache_path=out / CACHE_FILE,
            chroma_path=chroma_path, sleep_fn=sleep_fn, cfg=cfg, base=base)
        decisions += report.decisions
        concepts.update({c.key: c for c in seen})
    pending = sum(d.status == STATUS_PENDING for d in decisions)
    if pending:
        raise ValueError(f"Faltan {counted(pending, 'decisión', 'decisiones')} del decisor en "
                         "su caché: corre primero py -m src.alignment.decider_eval")
    outputs = apply_rule(decisions, **rule.kwargs())
    accepted = {k: o for k, o in outputs.items() if o.get("term")}
    items = [au.item_for(concepts[k], o["term"], o["relation"], inputs.targets)
             for k, o in accepted.items()]
    review = au.ask_audit(items, settings=au.audit_settings(cfg), cache_path=out / au.CACHE_FILE)
    missing = len(items) - len(review.answers)
    if missing:
        raise ValueError(f"Faltan {counted(missing, 'revisión', 'revisiones')} de la IA "
                         "revisora en su caché: corre primero py -m src.validation.ai_auditor "
                         "--medir")
    entries = entries_of(decisions, accepted, suggested(decisions, outputs, rule.suggest))
    return (uploads_of(entries, concepts, review.answers), inputs.targets,
            inputs.references[REF_BASE], rule, outputs)


# --------------------------------------------------------------------------- the people

def concept_value(item, maps: dict) -> int:
    """How close a concept is to your decision (see the module docstring)."""
    if item is None:
        return NOTHING if not maps else WRONG
    if item.status != STATUS_ALIGNED:
        return RIGHT_RELATION if not maps else WRONG
    right = [iri for iri in maps if iri in item.terms]
    if right:
        return (RIGHT_RELATION if any(maps[i]["relation"] == item.terms[i] for i in right)
                else RIGHT_TERM)
    return NOTHING if not maps else WRONG


def _value(q: uq.Question, state: dict, ref: dict) -> int:
    return sum(concept_value(ref.get(d.key), state.get(d.key, {})) for d in q.doubts)


def person_choice(q: uq.Question, state: dict, ref: dict) -> int:
    """The option "la persona" takes (``len(q.options)`` = "Dejar como está")."""
    best, best_value = len(q.options), _value(q, state, ref)
    order = sorted(range(len(q.options)), key=lambda n: n != q.recommended)  # it first
    for n in order:
        value = _value(q, uq.answered(state, [q.options[n]]), ref)
        if value > best_value:
            best, best_value = n, value
    return best


def _outputs(rule_outputs: dict, state: dict, changed: set[str]) -> dict:
    """What the decider's measurement scores: the rule's answers with the changed concepts."""
    out = dict(rule_outputs)
    for key in changed:
        maps = state.get(key, {})
        if maps:
            iri, m = sorted(maps.items())[0]
            out[key] = {"term": iri, "relation": m["relation"]}
        else:
            out[key] = {"none": True}
    return out


def _answer(qs: uq.QuestionSet, up: SimUpload, ref: dict, who: str) -> tuple[dict, set, list]:
    chosen = [(len(q.options) if q.recommended is None else q.recommended)
              if who == "recommended" else person_choice(q, up.state, ref)
              for q in qs.questions]
    options = [q.options[n] for q, n in zip(qs.questions, chosen) if n < len(q.options)]
    changed = {key for o in options for key in o.after()}
    return uq.answered(up.state, options), changed, chosen


# --------------------------------------------------------------------------- measuring

def _metrics(s: dict) -> dict:
    return {k: s[k] for k in ("proposals", "correct", "precision", "coverage",
                              "relation_typed", "relation_ok", "relation_accuracy")}


def _not_lower(a, b) -> bool:
    return (a or 0.0) >= (b or 0.0)


def conditions(results: dict, questions: dict) -> list[bool]:
    """The three conditions to adopt the prompt, fixed before measuring."""
    r, rec, per, bas = (results[k] for k in ("rule", "recommended", "person", "baseline_person"))
    asked = questions["ai"]
    return [_not_lower(rec["precision"], r["precision"])
            and _not_lower(rec["coverage"], r["coverage"]),
            asked > 0 and 2 * questions["ai_served"] >= asked,
            asked < questions["baseline"] and _not_lower(per["precision"], bas["precision"])
            and _not_lower(per["coverage"], bas["coverage"])]


def evaluate(uploads: list[SimUpload], targets, reference: Reference, rule, rule_outputs: dict,
             *, llm=None, settings: uq.AskSettings | None = None, cache_path=None,
             budget: int = DEFAULT_BUDGET, sleep_fn=time.sleep) -> dict:
    settings = settings or uq.AskSettings()
    ref = reference.by_key()
    used, rows = 0, []
    states = {u: {} for u in ("recommended", "person", "baseline")}
    changed = {u: set() for u in states}
    served = {"ai": 0, "baseline": 0}
    for up in uploads:
        doubts = uq.find_doubts(up.entries, up.state, info=up.info, audits=up.audits, rule=rule,
                                candidates=settings.candidates)
        ai = uq.propose_questions(doubts, up.index, up.state, targets, llm=llm,
                                  settings=replace(settings, max_calls=max(budget - used, 0)),
                                  cache_path=cache_path, sleep_fn=sleep_fn)
        used += ai.ai["calls"]
        if ai.ai["quota_exhausted"]:
            budget = used                              # stop asking: the next run goes on
        base = uq.fixed_questions(doubts, up.index, up.state, targets,
                                  max_questions=settings.max_questions)
        picks = {}
        for who, qs in (("recommended", ai), ("person", ai), ("baseline", base)):
            state, keys, chosen = _answer(qs, up, ref, "recommended" if who == "recommended"
                                          else "person")
            states[who].update({k: state.get(k, {}) for k in keys})
            changed[who] |= keys
            picks[who] = chosen
        served["ai"] += sum(n < len(q.options) for q, n in zip(ai.questions, picks["person"]))
        served["baseline"] += sum(n < len(q.options)
                                  for q, n in zip(base.questions, picks["baseline"]))
        rows.append({"upload": up, "doubts": doubts, "ai": ai, "baseline": base,
                     "person": picks["person"], "baseline_person": picks["baseline"]})
    asked = sum(len(r["ai"].questions) for r in rows)
    asked_base = sum(len(r["baseline"].questions) for r in rows)
    results = {"rule": _metrics(score(rule_outputs, reference))}
    for who, name in (("recommended", "recommended"), ("person", "person"),
                      ("baseline", "baseline_person")):
        results[name] = _metrics(score(_outputs(rule_outputs, states[who], changed[who]),
                                       reference))
    questions = {"ai": asked, "ai_served": served["ai"],
                 "ai_joined": sum(len(q.doubts) > 1 for row in rows for q in row["ai"].questions),
                 "baseline": asked_base, "baseline_served": served["baseline"],
                 "refused_options": sum(len(row["ai"].refused) for row in rows),
                 "not_asked": {by: sum(s.by == by for row in rows for s in row["ai"].skipped)
                               for by in (uq.BY_AI, uq.BY_RULE, uq.BY_LIMIT, uq.NO_ANSWER)}}
    ok = conditions(results, questions)
    complete = all(row["ai"].complete for row in rows)
    first = next((row["ai"] for row in rows if row["ai"].prompt), None)
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": first.prompt if first else uq.ask_prompt(settings).manifest(),
        "model": first.model if first else None,
        "settings": {"max_questions": settings.max_questions, "batch_size": settings.batch_size,
                     "candidates": settings.candidates},
        "ai": {"calls": used, "cached": sum(row["ai"].ai["cached"] for row in rows),
               "pending": sum(row["ai"].ai["pending"] for row in rows),
               "quota_exhausted": any(row["ai"].ai["quota_exhausted"] for row in rows),
               "errors": [e for row in rows for e in row["ai"].ai["errors"]][:3]},
        "complete": complete,
        "doubts": sum(len(row["doubts"]) for row in rows),
        "questions": questions,
        "results": results,
        "conditions": [{"n": n, "text": t, "ok": v}
                       for n, (t, v) in enumerate(zip(CONDITIONS, ok), start=1)],
        "adopted": (all(ok) if complete else None),
        "_rows": rows}


# --------------------------------------------------------------------------- report

def _skipped_json(s: uq.Skipped) -> dict:
    return {"doubt": s.doubt.key, "name": s.doubt.name, "kind": s.doubt.kind, "by": s.by,
            "reason": s.reason}


def to_json(result: dict) -> dict:
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    out["uploads"] = []
    for row in result["_rows"]:
        ai, base, up = row["ai"], row["baseline"], row["upload"]
        out["uploads"].append({
            "name": up.name, "concepts": len(up.entries), "doubts": len(row["doubts"]),
            "ai": {"summary": uq.summary_text(ai), "complete": ai.complete, **ai.ai,
                   "questions": [uq.question_json(q, {"person": n}) for q, n in
                                 zip(ai.questions, row["person"])],
                   "skipped": [_skipped_json(s) for s in ai.skipped],
                   "refused": ai.refused},
            "baseline": {"questions": len(base.questions),
                         "served": sum(n < len(q.options) for q, n in
                                       zip(base.questions, row["baseline_person"]))}})
    return json.loads(json.dumps(out, ensure_ascii=False))


def _pct(v) -> str:
    from src.alignment.benchmark import _pct as pct
    return pct(v)


def render_console(result: dict, examples: int = EXAMPLES) -> str:
    q, res = result["questions"], result["results"]
    p = result["prompt"]
    lines = [f"Preguntas de la IA con opciones (S10-T04) · {p['name']}.v{p['version']}"
             + (f" · {result['model']}" if result.get("model") else ""),
             f"  Subidas simuladas: {counted(len(result['_rows']), 'familia', 'familias')} de "
             f"la base ({sum(len(r['upload'].entries) for r in result['_rows'])} conceptos), "
             f"{counted(result['doubts'], 'duda', 'dudas')}",
             f"  IA: {counted(result['ai']['calls'], 'llamada', 'llamadas')}, "
             f"{result['ai']['cached']} desde la caché"
             + (f", {result['ai']['pending']} dudas sin respuesta" if result['ai']['pending']
                else ""),
             "",
             f"  {'':<34}{'preguntas':>10}{'sirven':>9}{'precisión':>11}{'cobertura':>11}"
             f"{'relación':>10}"]

    def row(label, m, asked=None, useful=None):
        rel = f"{m['relation_ok']}/{m['relation_typed']}"
        lines.append(f"  {label:<34}{'-' if asked is None else asked:>10}"
                     f"{'-' if useful is None else useful:>9}{_pct(m['precision']):>11}"
                     f"{_pct(m['coverage']):>11}{rel:>10}")
    row("Regla sola", res["rule"])
    row("Con IA, siempre la recomendada", res["recommended"], q["ai"])
    row("Con IA, la persona", res["person"], q["ai"], q["ai_served"])
    row("Preguntas fijas, la persona", res["baseline_person"], q["baseline"],
        q["baseline_served"])
    lines += ["", f"  Condiciones para adoptar {p['name']}.v{p['version']} (fijadas antes de "
                  "medir):"]
    detail = {2: f" ({q['ai_served']} de {q['ai']})",
              3: f" ({q['ai']} preguntas con IA, {q['baseline']} fijas)"}
    lines += [f"    {c['n']}. {c['text']}: {'sí' if c['ok'] else 'no'}{detail.get(c['n'], '')}"
              for c in result["conditions"]]
    if result["adopted"] is None:
        lines.append("  Resultado: incompleto (faltan respuestas de la IA); vuelve a correr "
                     "para seguir desde la caché.")
    else:
        failed = [str(c["n"]) for c in result["conditions"] if not c["ok"]]
        if not failed:
            lines.append(f"  Resultado: se adopta {p['name']}.v{p['version']} (para usarla: "
                         "user_questions.ai: true)")
        else:
            which = (f"falla la condición {failed[0]}" if len(failed) == 1 else
                     f"fallan las condiciones {', '.join(failed[:-1])} y {failed[-1]}")
            lines.append(f"  Resultado: no se adopta {p['name']}.v{p['version']} ({which}); las "
                         "subidas hacen las preguntas fijas")
    no = q["not_asked"]
    lines.append(f"  Juntan varias dudas: {q['ai_joined']} de {q['ai']} preguntas. Sin pregunta: "
                 f"{counted(no[uq.BY_AI], 'duda', 'dudas')} por la IA, {no[uq.BY_RULE]} por la "
                 f"regla, "
                 f"{no[uq.BY_LIMIT]} por el máximo"
                 + (f", {no[uq.NO_ANSWER]} sin respuesta" if no[uq.NO_ANSWER] else "")
                 + f"; la regla dejó fuera {counted(q['refused_options'], 'opción', 'opciones')}")
    shown = 0
    for r in result["_rows"]:
        for qq, n in zip(r["ai"].questions, r["person"]):
            if shown == examples:
                break
            if shown == 0:
                lines += ["", "  Ejemplos (la persona elige como en tus decisiones):"]
            lines.append(f"  [{r['upload'].name}]")
            lines += uq.render_question(qq, chosen=n)
            shown += 1
    lines.append(f"  Detalle: data/output/{EVALUATION_FILE}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.question_eval",
        description="Mide las preguntas de la IA con personas simuladas (S10-T04).")
    parser.add_argument("--max-llamadas", type=int, default=DEFAULT_BUDGET,
                        help=f"llamadas a la IA en esta corrida (por defecto {DEFAULT_BUDGET})")
    args = parser.parse_args(argv)
    from src.alignment.inherit import base_paths
    from src.config import load_config
    cfg = load_config()
    try:
        uploads, targets, reference, rule, outputs = load_inputs(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    try:
        from src.ai.llm_client import LLMClient
        llm = LLMClient()
    except KeyError:
        print("Error: falta LLM_API_KEY en el .env (la medición pregunta a la IA).")
        return 1
    out = base_paths(cfg)["candidates"].parent
    settings = uq.ask_settings(cfg)
    result = evaluate(uploads, targets, reference, rule, outputs, llm=llm, settings=settings,
                      cache_path=out / "decider" / uq.CACHE_FILE, budget=args.max_llamadas)
    (out / EVALUATION_FILE).write_text(json.dumps(to_json(result), indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(render_console(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
