from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.ai.prompts import load_prompt

PROMPT = load_prompt("audit", 1)
RESULT_FILE = "ai-audit.json"
EVALUATION_FILE = "ai-audit-evaluation.json"
CACHE_FILE = "audit-cache.jsonl"
COHERENT, DOUBTFUL, INCOHERENT, INVALID = "coherente", "dudosa", "incoherente", "inválida"
VERDICTS = (COHERENT, DOUBTFUL, INCOHERENT)
RELATIONS = ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch")
# What counts as flagged: the two options measured on the 552 decisions (fixed beforehand).
FLAGS = {"incoherente": (INCOHERENT,), "incoherente o dudosa": (INCOHERENT, DOUBTFUL)}
MIN_GAIN = 0.05             # precision points the flagged ones must add when left out ...
MAX_LOST = 0.10             # ... taking away at most this share of the right ones
EXAMPLES = 5


@dataclass(frozen=True)
class AuditSettings:
    enabled: bool = True
    batch_size: int = 10
    max_calls: int = 10
    pause_seconds: float = 5.0
    max_tokens: int = 4096
    definition_chars: int = 400


def audit_settings(cfg: dict | None = None) -> AuditSettings:
    from src.config import load_config
    block = (cfg if cfg is not None else load_config()).get("audit", {}) or {}
    s = AuditSettings(**{k: v for k, v in block.items() if k in AuditSettings.__dataclass_fields__})
    if s.batch_size < 1 or s.max_calls < 0 or s.pause_seconds < 0:
        raise ValueError("Valores inválidos en el bloque audit de config.yaml.")
    return s


@dataclass
class AuditItem:
    key: str
    name: str
    kinds: tuple
    label: str | None
    definition: str | None
    relation: str | None              # "skos:exactMatch"…; None: accepted without a SKOS type
    dpv_iri: str
    dpv_name: str
    dpv_definition: str | None
    dpv_parents: list = field(default_factory=list)


def item_for(concept, dpv_iri: str, relation: str | None, targets) -> AuditItem:
    term = next((t for t in targets.targets if t.iri == dpv_iri), None)
    return AuditItem(key=concept.key, name=concept.name, kinds=tuple(concept.kinds),
                     label=concept.label, definition=concept.definition, relation=relation,
                     dpv_iri=dpv_iri, dpv_name=term.name if term else dpv_iri.rsplit("#", 1)[-1],
                     dpv_definition=term.definition if term else None,
                     dpv_parents=list(term.parents) if term else [])


# --------------------------------------------------------------------------- asking the AI

def _one_line(text, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit].rstrip() if limit else text


def relation_name(relation: str | None) -> str | None:
    return relation.split(":")[-1] if relation else None


def body(item: AuditItem, chars: int = 400) -> str:
    kind = "propiedad" if set(item.kinds) == {"property"} else "clase"
    lines = [f"Concepto: {item.name} ({kind})",
             f"Etiqueta: {_one_line(item.label) or '-'}",
             f"Definicion: {_one_line(item.definition, chars) or '-'}",
             f"Relacion propuesta: {relation_name(item.relation) or 'sin tipo (rdfs:seeAlso)'}",
             f"Termino DPV: {item.dpv_name}",
             f"Definicion DPV: {_one_line(item.dpv_definition, chars) or '-'}",
             f"Padres en el DPV: {', '.join(item.dpv_parents) or '-'}"]
    return "\n".join(lines)


def cache_key(text: str, model: str | None, generation: dict | None = None) -> str:
    content = {"prompt": PROMPT.sha256, "model": model, "body": text}
    if generation:
        content["generation"] = generation
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_reviews(raw: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    data = None
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if isinstance(data, dict):
        data = data.get("reviews")
    if not isinstance(data, list):
        raise ValueError("La respuesta del LLM no trae la lista de revisiones en JSON.")
    return [d for d in data if isinstance(d, dict)]


def validate_review(answer: dict) -> dict:
    verdict = _one_line(answer.get("verdict")).lower()
    relation = _one_line(answer.get("relation")).split(":")[-1]
    return {"verdict": verdict if verdict in VERDICTS else INVALID,
            "relation": relation if relation in RELATIONS else None,
            "explanation": _one_line(answer.get("explanation"), 500) or None}


@dataclass
class AuditReport:
    answers: dict = field(default_factory=dict)
    calls: int = 0
    cached: int = 0
    quota_exhausted: bool = False
    call_limit_reached: bool = False
    errors: list[dict] = field(default_factory=list)

    def ai(self, available: bool = True) -> dict:
        return {"available": available, "calls": self.calls, "cached": self.cached,
                "invalid": sum(a["verdict"] == INVALID for a in self.answers.values()),
                "quota_exhausted": self.quota_exhausted,
                "call_limit_reached": self.call_limit_reached,
                "errors": [e["error"] for e in self.errors][:3]}


def ask_audit(items: list[AuditItem], *, llm=None, settings: AuditSettings | None = None,
              cache_path=None, model_name: str | None = None,
              sleep_fn=time.sleep) -> AuditReport:
    from src.alignment.decider import generation_settings
    from src.alignment.justify import (QUOTA_WAIT_SECONDS, TRANSIENT_WAIT_SECONDS,
                                       _is_transient_error)
    from src.core.extract import _is_quota_error
    from src.transform.labels import _append_cache, _load_cache
    settings = settings or AuditSettings()
    report = AuditReport()
    model_name = model_name or getattr(llm, "model", None) or os.environ.get("LLM_MODEL")
    generation = generation_settings(llm)
    bodies = {i.key: body(i, settings.definition_chars) for i in items}
    keys = {i.key: cache_key(bodies[i.key], model_name, generation) for i in items}
    cache = _load_cache(cache_path)
    pending = []
    for item in items:
        hit = cache.get(keys[item.key])
        if hit is not None:
            report.answers[item.key] = hit["answer"]
            report.cached += 1
        else:
            pending.append(item)
    for start in range(0, len(pending), settings.batch_size):
        if llm is None:
            break
        if report.calls >= settings.max_calls:
            report.call_limit_reached = True
            break
        if report.calls and settings.pause_seconds:
            sleep_fn(settings.pause_seconds)
        batch = pending[start:start + settings.batch_size]
        ids = {f"C{n}": item for n, item in enumerate(batch, start=1)}
        context = "\n\n".join(f"=== {cid} ===\n{bodies[it.key]}" for cid, it in ids.items())
        raw, err = None, None
        for attempt in range(2):                       # one retry after a 429 or a 503
            try:
                raw = llm.ask(prompt=PROMPT.text, context=context,
                              max_tokens=settings.max_tokens, json_mode=True) or ""
                err = None
                break
            except Exception as exc:                   # noqa: BLE001 - report, don't crash
                err = str(exc)
                if attempt == 0 and _is_quota_error(err):
                    sleep_fn(QUOTA_WAIT_SECONDS)
                    continue
                if attempt == 0 and _is_transient_error(err):
                    sleep_fn(TRANSIENT_WAIT_SECONDS)
                    continue
                break
        report.calls += 1
        if err is None:
            try:
                parsed = parse_reviews(raw)
            except ValueError as exc:
                err = str(exc)
        if err is not None:
            report.errors.append({"concepts": [it.name for it in batch], "error": err[:300]})
            if _is_quota_error(err):
                report.quota_exhausted = True
                break
            continue
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        new = []
        for ans in parsed:
            item = ids.get(str(ans.get("concept") or "").strip().upper())
            if item is None or item.key in report.answers:
                continue
            answer = validate_review(ans)
            report.answers[item.key] = answer
            new.append({"cache_key": keys[item.key], "concept_key": item.key, "answer": answer,
                        "prompt": PROMPT.manifest(), "model": model_name, "created": now})
        _append_cache(cache_path, new)
    return report


def flagged(answer: dict | None, flag: str) -> bool:
    return bool(answer) and answer["verdict"] in FLAGS[flag]


# --------------------------------------------------------------------------- measuring

def reference_rows(cfg: dict, *, llm=None, embed_fn=None, decider_cache=None,
                   chroma_path=None, sleep_fn=time.sleep) -> list[dict]:
    """The correspondences the rule accepted on the 552 decisions (both halves, as the prompt
    lab decided them: from the cache), each with whether it is the person's term."""
    from src.alignment import prompt_lab as lab
    from src.alignment.decider import decide_prompt
    from src.alignment.decider_eval import load_eval_inputs
    inputs = load_eval_inputs(cfg)
    prompt = decide_prompt(cfg)
    rows, base = [], None
    for half in lab.HALVES:
        part, concepts, report, accepted, base = lab.decide_half(
            inputs, prompt, half, llm=llm, embed_fn=embed_fn, budget=lab.DEFAULT_BUDGET,
            cache_path=decider_cache, chroma_path=chroma_path, sleep_fn=sleep_fn, cfg=cfg,
            base=base)
        seen = {c.key: c for c in concepts}
        for ref in part.items:
            a = accepted.get(ref.key)
            if not a or not a["term"] or ref.key not in seen:
                continue
            rows.append({"half": half, "correct": a["term"] in ref.terms,
                         "person": dict(ref.terms),
                         "item": item_for(seen[ref.key], a["term"], a["relation"],
                                          inputs.targets)})
    return rows


def _rate(n: int, total: int) -> float | None:
    return round(n / total, 4) if total else None


def measure(rows: list[dict], answers: dict) -> dict:
    right = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    before = _rate(len(right), len(rows))
    options = {}
    for flag in FLAGS:
        caught = sum(flagged(answers.get(r["item"].key), flag) for r in wrong)
        lost = sum(flagged(answers.get(r["item"].key), flag) for r in right)
        kept = len(rows) - caught - lost
        after = _rate(len(right) - lost, kept)
        ok_gain = before is not None and after is not None and round(after - before, 4) >= MIN_GAIN
        ok_lost = len(right) == 0 or round(lost / len(right), 4) <= MAX_LOST
        options[flag] = {"caught": caught, "wrong": len(wrong), "lost": lost,
                         "right": len(right), "precision_before": before,
                         "precision_after": after, "adopted": ok_gain and ok_lost}
    chosen = [f for f, o in options.items() if o["adopted"]]
    best = max(chosen, key=lambda f: options[f]["precision_after"]) if chosen else None
    relations = [r for r in right if relation_name(r["item"].relation)]
    agree_decider = sum(relation_name(r["item"].relation) in
                        {relation_name(v) for v in r["person"].values()} for r in relations)
    agree_ai = sum((answers.get(r["item"].key) or {}).get("relation") in
                   {relation_name(v) for v in r["person"].values()} for r in relations)
    verdicts = {v: {"right": sum((answers.get(r["item"].key) or {}).get("verdict") == v
                                 for r in right),
                    "wrong": sum((answers.get(r["item"].key) or {}).get("verdict") == v
                                 for r in wrong)} for v in (*VERDICTS, INVALID)}
    return {"accepted": len(rows), "right": len(right), "wrong": len(wrong),
            "verdicts": verdicts, "options": options, "adopted": best,
            "relations": {"typed": len(relations), "decider": agree_decider, "ai": agree_ai}}


def _example(row: dict, answer: dict | None) -> dict:
    item = row["item"]
    return {"name": item.name, "dpv": item.dpv_name, "relation": relation_name(item.relation),
            "person": sorted(k.rsplit("#", 1)[-1] for k in row["person"]),
            "verdict": (answer or {}).get("verdict"),
            "explanation": (answer or {}).get("explanation")}


def evaluate(cfg: dict | None = None, *, llm=None, decider_llm=None, embed_fn=None,
             chroma_path=None, sleep_fn=time.sleep) -> dict:
    from src.alignment.decider import CACHE_FILE as DECIDER_CACHE
    from src.alignment.inherit import base_paths
    from src.config import load_config
    cfg = cfg if cfg is not None else load_config()
    out = base_paths(cfg)["candidates"].parent
    rows = reference_rows(cfg, llm=decider_llm, embed_fn=embed_fn,
                          decider_cache=out / "decider" / DECIDER_CACHE,
                          chroma_path=chroma_path, sleep_fn=sleep_fn)
    settings = audit_settings(cfg)
    settings = AuditSettings(**{**settings.__dict__,
                                "max_calls": max(settings.max_calls, len(rows))})
    if llm is None and rows:
        from src.ai.llm_client import LLMClient
        llm = LLMClient()
    report = ask_audit([r["item"] for r in rows], llm=llm, settings=settings,
                       cache_path=out / "decider" / CACHE_FILE, sleep_fn=sleep_fn)
    result = measure(rows, report.answers)
    best = result["adopted"] or "incoherente"
    groups = {"caught": [r for r in rows if not r["correct"]
                         and flagged(report.answers.get(r["item"].key), best)],
              "missed": [r for r in rows if not r["correct"]
                         and not flagged(report.answers.get(r["item"].key), best)],
              "false_alarms": [r for r in rows if r["correct"]
                               and flagged(report.answers.get(r["item"].key), best)]}
    result.update({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": PROMPT.manifest(), "model": getattr(llm, "model", None),
        "ai": report.ai(), "pending": sum(r["item"].key not in report.answers for r in rows),
        "criterion": {"min_gain": MIN_GAIN, "max_lost": MAX_LOST},
        "examples": {k: [_example(r, report.answers.get(r["item"].key)) for r in v]
                     for k, v in groups.items()}})
    (out / EVALUATION_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    return result


# --------------------------------------------------------------------------- one upload

def fingerprint(entries: list[dict]) -> str:
    rows = sorted((e["key"], e["dpv_iri"], e.get("relation") or "") for e in entries)
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()


def _accepted(decisions: dict) -> list[dict]:
    return [e for e in decisions["entries"] if e["status"] == "accepted"]


def _use_llm(cfg: dict | None, use_llm: bool | None) -> bool:
    from src.alignment.calibrate import auto_settings
    return auto_settings(cfg)["use_llm"] if use_llm is None else use_llm


def is_current(result: dict | None, ws, cfg: dict | None = None,
               use_llm: bool | None = None) -> bool:
    """A saved review can be reused: same prompt, same accepted correspondences and, when the
    AI is to be used, none left without review."""
    from src.alignment.propose import load_result as decisions_result
    decisions = decisions_result(ws)
    if not result or decisions is None or result["prompt"]["sha256"] != PROMPT.sha256:
        return False
    if result.get("decisions") != fingerprint(_accepted(decisions)):
        return False
    return not (_use_llm(cfg, use_llm) and audit_settings(cfg).enabled
                and result["summary"]["pending"])


def _audit(ws, cfg, llm, use_llm, targets, sleep_fn) -> dict:
    from src.alignment import law_definitions as ld
    from src.alignment.inherit import load_result as inherited_result
    from src.alignment.propose import load_dpv_targets, load_result, pending_concepts
    result = load_result(ws)
    if result is None:
        raise ValueError("Primero hay que decidir la alineación de esta subida.")
    settings = audit_settings(cfg)
    accepted = _accepted(result)
    sources = pending_concepts(ws, inherited_result(ws) or {}, ld.load_result(ws))
    concepts = {c.key: c for c in sources.concepts}
    targets = targets if targets is not None else load_dpv_targets(cfg)
    items = [item_for(concepts[e["key"]], e["dpv_iri"], e.get("relation"), targets)
             for e in accepted if e["key"] in concepts]
    available = _use_llm(cfg, use_llm) and settings.enabled
    error = None if available else ("IA desactivada (--sin-ia)" if settings.enabled
                                    else "desactivada en config.yaml (audit.enabled)")
    if available and llm is None and items:
        try:
            from src.ai.llm_client import LLMClient
            llm = LLMClient()
        except KeyError as exc:                        # LLM_API_KEY / LLM_MODEL missing
            available, error = False, f"Falta la variable {exc} en .env"
    report = ask_audit(items, llm=llm if available else None, settings=settings,
                       cache_path=ws.work_dir / CACHE_FILE, sleep_fn=sleep_fn)
    entries = []
    for i in items:
        answer = report.answers.get(i.key) or {}
        entries.append({"key": i.key, "name": i.name, "dpv_iri": i.dpv_iri,
                        "dpv_name": i.dpv_name, "relation": relation_name(i.relation),
                        "verdict": answer.get("verdict"), "ai_relation": answer.get("relation"),
                        "explanation": answer.get("explanation")})
    count = {v: sum(e["verdict"] == v for e in entries) for v in (*VERDICTS, INVALID)}
    out = {"prompt": PROMPT.manifest(),
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "decisions": fingerprint(accepted),
           "summary": {"accepted": len(items), **count,
                       "pending": sum(e["verdict"] is None for e in entries)},
           "ai": {**report.ai(available), **({"error": error} if error else {})},
           "entries": entries}
    ws.work_dir.mkdir(parents=True, exist_ok=True)
    (ws.work_dir / RESULT_FILE).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return out


def audit_workspace(ws, cfg: dict | None = None, *, llm=None, use_llm: bool | None = None,
                    targets=None, sleep_fn=time.sleep) -> dict:
    """Review the correspondences the rule accepted in an upload (never the inherited ones,
    which a person decided); it explains, it never changes what the rule accepted."""
    from src.workspace.store import DONE, ERROR, RUNNING
    ws.mark_step("audit", RUNNING)
    try:
        out = _audit(ws, cfg, llm, use_llm, targets, sleep_fn)
    except Exception as exc:
        ws.mark_step("audit", ERROR, str(exc)[:300])
        raise
    s = out["summary"]
    ws.mark_step("audit", DONE, summary_text(s), incoherent=s[INCOHERENT], doubtful=s[DOUBTFUL],
                 pending=s["pending"])
    return out


def current_result(ws) -> dict | None:
    """The saved review if it is of the correspondences accepted now (else None)."""
    from src.alignment.propose import load_result as decisions_result
    result, decisions = load_result(ws), decisions_result(ws)
    if result is None or decisions is None:
        return None
    return result if result.get("decisions") == fingerprint(_accepted(decisions)) else None


def by_key(result: dict | None) -> dict[str, dict]:
    return {e["key"]: e for e in (result or {}).get("entries", [])}


def load_result(ws) -> dict | None:
    path = ws.work_dir / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- console

def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}".replace(".", ",") + " %"


def ai_text(ai: dict) -> str:
    if not ai.get("available"):
        return "no usada" + (f" ({ai['error']})" if ai.get("error") else "")
    text = f"{ai['calls']} llamadas, {ai['cached']} desde la caché, {ai['invalid']} inválidas"
    if ai.get("quota_exhausted"):
        text += " (cuota agotada: vuelve a correr mañana)"
    elif ai.get("call_limit_reached"):
        text += " (límite de llamadas: vuelve a correr para seguir)"
    if ai.get("errors"):
        text += f"; error: {ai['errors'][0]}"
    return text


def render_evaluation(result: dict) -> str:
    prompt = result["prompt"]
    lines = [f"Revisión de la IA (prompt {prompt['name']}.v{prompt['version']}) de las "
             f"{result['accepted']} correspondencias que la regla aceptó en tus 552 decisiones",
             f"  IA: {ai_text(result['ai'])}" + (f"; faltan {result['pending']}"
                                                 if result["pending"] else ""),
             f"  Iguales a tu decisión: {result['right']} · distintas: {result['wrong']}",
             "  Veredicto de la IA      en las iguales   en las distintas"]
    for v, n in result["verdicts"].items():
        if n["right"] or n["wrong"] or v != INVALID:
            lines.append(f"    {v:<20}  {n['right']:>12}   {n['wrong']:>14}")
    lines.append(f"  Si se dejaran fuera las marcadas (criterio fijado antes: +"
                 f"{round(MIN_GAIN * 100)} puntos de precisión, perdiendo como máximo el "
                 f"{round(MAX_LOST * 100)} % de las iguales):")
    for flag, o in result["options"].items():
        lines.append(f"    {flag}: atrapa {o['caught']} de {o['wrong']} distintas, quita "
                     f"{o['lost']} de {o['right']} iguales; precisión "
                     f"{_pct(o['precision_before'])} -> {_pct(o['precision_after'])}"
                     f" ({'cumple' if o['adopted'] else 'no cumple'})")
    adopted = result["adopted"]
    lines.append("  Veredicto: " + (f"dejar fuera las que la IA marca «{adopted}»" if adopted
                                    else "la IA solo explica: no cambia lo que acepta la regla"))
    rel = result["relations"]
    if rel["typed"]:
        lines.append(f"  Relación SKOS en las {rel['typed']} iguales con tipo: coincide con la "
                     f"tuya la del decisor en {rel['decider']} y la de la IA revisora en "
                     f"{rel['ai']}")
    names = {"caught": "Atrapadas (distintas a tu decisión)", "missed": "No atrapadas",
             "false_alarms": "Falsas alarmas (iguales a tu decisión)"}
    for key, title in names.items():
        found = result["examples"][key]
        if not found:
            continue
        lines.append(f"  {title} ({len(found)}):")
        for e in found[:EXAMPLES]:
            lines.append(f"    - {e['name']} -> {e['dpv']} ({e['relation'] or 'sin tipo'}); "
                         f"tú: {', '.join(e['person']) or 'sin correspondencia'}; IA: "
                         f"{e['verdict']}. {e['explanation'] or ''}".rstrip())
    return "\n".join(lines)


def summary_text(s: dict) -> str:
    return (f"Coherentes {s[COHERENT]} · dudosas {s[DOUBTFUL]} · incoherentes {s[INCOHERENT]}"
            + (f" · inválidas {s[INVALID]}" if s[INVALID] else "")
            + (f" · sin revisar {s['pending']}" if s["pending"] else ""))


def render_workspace(result: dict, ws_id: str, examples: int | None = EXAMPLES) -> str:
    s = result["summary"]
    lines = [f"Revisión de la IA del espacio {ws_id}: {s['accepted']} correspondencias "
             "aceptadas por la regla",
             f"  IA: {ai_text(result['ai'])}",
             f"  {summary_text(s)}"]
    for v in (INCOHERENT, DOUBTFUL):
        found = [e for e in result["entries"] if e["verdict"] == v]
        for e in found[:examples] if examples is not None else found:
            lines.append(f"    [{v}] {e['name']} -> {e['dpv_name']} "
                         f"({e['relation'] or 'sin tipo'}): {e['explanation'] or ''}".rstrip())
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.config import load_config
    from src.validation.shape_inspector import open_workspace
    from src.workspace.store import WorkspaceError
    parser = argparse.ArgumentParser(prog="py -m src.validation.ai_auditor",
                                     description="La IA revisa las correspondencias que aceptó "
                                                 "la regla y explica las incoherentes.")
    parser.add_argument("workspace", nargs="?", help="id o carpeta del espacio de trabajo")
    parser.add_argument("--medir", action="store_true",
                        help="medir con las 552 decisiones de la base")
    parser.add_argument("--sin-ia", action="store_true", help="no llamar al LLM (solo la caché)")
    args = parser.parse_args(argv)
    if args.medir == bool(args.workspace):
        print("Error: indica un espacio de trabajo o --medir.")
        return 1
    cfg = load_config()
    try:
        if args.medir:
            from src.alignment.inherit import base_paths
            print("Midiendo con tus 552 decisiones (las del decisor salen de la caché)...")
            result = evaluate(cfg)
            print(render_evaluation(result))
            print(f"  Guardado en: {base_paths(cfg)['candidates'].parent / EVALUATION_FILE}")
            return 0
        ws = open_workspace(args.workspace, cfg)
        result = audit_workspace(ws, cfg, use_llm=not args.sin_ia)
    except (FileNotFoundError, ValueError, WorkspaceError) as exc:
        print(f"Error: {exc}")
        return 1
    except KeyError as exc:                            # LLM_API_KEY / LLM_MODEL missing
        print(f"Error: falta la variable {exc} en .env")
        return 1
    print(render_workspace(result, ws.id))
    print(f"  Guardado en: {ws.work_dir / RESULT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
