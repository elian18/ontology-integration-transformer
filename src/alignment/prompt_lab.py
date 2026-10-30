from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from src.ai.prompts import PromptNotFound, available, load_prompt
from src.alignment.decider import (CACHE_FILE, STATUS_DECIDED, STATUS_INVALID, STATUS_PENDING,
                                   apply_rule, decide, decide_prompt, rule_settings,
                                   settings_from_config)
from src.alignment.mapping_review import STATUS_ALIGNED
from src.alignment.reference import REF_BASE, Reference, score

PROMPT_NAME = "decide"
DEV, TEST = "desarrollo", "prueba"
HALVES = (DEV, TEST)
SPLIT_SEED = 2026
LAB_DIR = "prompt-lab"
SPLIT_FILE = "split.json"
MIN_GAIN = 0.03                       # F1 points a new version must gain on the test half
EXAMPLES = 8                          # examples per kind of error in the message for Gemini
DEFAULT_BUDGET = 40                   # half of the base: 276 concepts, 28 calls of 10
DEFINITIONS = "definiciones"          # the variant with the definitions the law gives (mode 3)
DEFINITIONS_BUDGET = 60               # the LOPDP defines ~450 OntoPriv concepts: ~45 calls
REQUIRED = ('"decisions"', '"concept"', '"dpv"', '"relation"', '"confidence"',
            '"justification"', "ninguno", "alta", "media", "baja", "exactMatch", "closeMatch",
            "broadMatch", "narrowMatch", "relatedMatch")
MAX_PROMPT_CHARS = 12000
_OUTSIDE = re.compile(r"el término '(.+)' no está entre los candidatos")

ACCEPTED_WRONG, NONE_WRONG, OTHER_TERM = "accepted_wrong", "none_wrong", "other_term"
RIGHT_LOW, RIGHT_RULE, TERM_ON_NONE, INVALID = ("right_term_low", "right_term_rule",
                                                "term_on_no_match", "invalid")
ERROR_LABELS = {
    ACCEPTED_WRONG: "la regla la aceptó, pero no es lo que decidió la persona",
    NONE_WRONG: 'dijo "ninguno" y la persona sí lo alineó',
    OTHER_TERM: "eligió un término distinto del que eligió la persona",
    RIGHT_LOW: "eligió el término de la persona, pero sin confianza alta",
    TERM_ON_NONE: "eligió un término y la persona lo dejó sin correspondencia",
    INVALID: "respondió un término que no estaba entre sus candidatos",
    RIGHT_RULE: "eligió el término de la persona con confianza alta, pero la relación o el "
                "puntaje no pasan la regla",
}


# --------------------------------------------------------------------------- the halves

def _only_property(concept) -> bool:
    kinds = set(getattr(concept, "kinds", ()) or ())
    return "property" in kinds and "class" not in kinds


def split(reference: Reference, concepts: dict, seed: int = SPLIT_SEED) -> dict[str, Reference]:
    """Two halves with the same mix of aligned / sin correspondencia and classes / properties."""
    groups: dict[tuple, list] = {}
    for item in sorted(reference.items, key=lambda i: i.key):
        groups.setdefault((item.status, _only_property(concepts.get(item.key))), []).append(item)
    rng = random.Random(seed)
    halves = {DEV: [], TEST: []}
    for group in sorted(groups):
        items = groups[group][:]
        rng.shuffle(items)
        extra = len(items) % 2 and len(halves[DEV]) <= len(halves[TEST])
        cut = len(items) // 2 + int(extra)
        halves[DEV] += items[:cut]
        halves[TEST] += items[cut:]
    return {h: Reference(f"{reference.name}-{h}", sorted(items, key=lambda i: i.key))
            for h, items in halves.items()}


# --------------------------------------------------------------------------- the prompt

def check_prompt(text: str) -> tuple[list[str], list[str]]:
    """What a new version must keep (errors) and what is only a warning."""
    errors = [f"falta {token}" for token in REQUIRED if token not in text]
    warnings = []
    if not text.isascii():
        warnings.append("tiene tildes o eñes (los prompts del proyecto se escriben sin ellas)")
    if len(text) > MAX_PROMPT_CHARS:
        warnings.append(f"es muy largo ({len(text)} caracteres)")
    return errors, warnings


def prompt_version(version: int):
    try:
        return load_prompt(PROMPT_NAME, version)
    except PromptNotFound as exc:
        raise ValueError(f"{exc} Guárdalo como src/ai/prompts/{PROMPT_NAME}.v{version}.txt") \
            from exc


# --------------------------------------------------------------------------- errors

def _relation(value) -> str:
    return str(value or "").replace("skos:", "")


def _terms_text(terms: dict, names: dict) -> str:
    if not terms:
        return "sin correspondencia"
    return "; ".join(f"{names.get(iri, iri.rsplit('#', 1)[-1])} ({_relation(rel)})"
                     for iri, rel in sorted(terms.items()))


def _person(item, names: dict) -> str:
    return _terms_text(item.terms if item.status == STATUS_ALIGNED else {}, names)


def _precedent(concept, names: dict) -> str:
    terms = ({m["dpv_iri"]: m["relation"] for m in concept.mappings}
             if concept.status == STATUS_ALIGNED else {})
    return f"{concept.name}: {_terms_text(terms, names)}"


def classify(decision, item, accepted: dict | None) -> str | None:
    """The kind of error of one decision against the person's (None = right)."""
    if decision is None:
        return None
    if decision.status == STATUS_INVALID:
        return INVALID
    if decision.status != STATUS_DECIDED:
        return None
    aligned = item.status == STATUS_ALIGNED
    if accepted and accepted.get("term"):
        return None if accepted["term"] in item.terms else ACCEPTED_WRONG
    if decision.none:
        return NONE_WRONG if aligned else None
    if decision.is_property:              # never automatic: no prompt makes them a proposal
        return None
    if not aligned:
        return TERM_ON_NONE
    if decision.dpv_iri not in item.terms:
        return OTHER_TERM
    return RIGHT_LOW if decision.confidence != "alta" else RIGHT_RULE


def written_term(decision) -> str | None:
    """The term an invalid answer named (it was not one of the concept's candidates)."""
    found = (m for m in map(_OUTSIDE.search, decision.issues or []) if m)
    return next((m.group(1) for m in found), None)


def example(decision, item, concept, names: dict, decided: dict) -> dict:
    """One error with what the LLM saw and what the person decided (``decided``: base key ->
    BaseConcept, for the precedents)."""
    precedents = [_precedent(decided[key], names) for key in decision.precedents
                  if key in decided]
    outside = decision.status == STATUS_INVALID
    return {"key": item.key, "name": item.name,
            "kinds": list(getattr(concept, "kinds", ()) or ()),
            "label": getattr(concept, "label", None),
            "definition": getattr(concept, "definition", None),
            "candidates": [names.get(iri, iri.rsplit("#", 1)[-1]) for iri in decision.candidates],
            "precedents": precedents,
            "answer": {"dpv": decision.dpv_name or ("ninguno" if decision.none else None)
                       or (written_term(decision) if outside else None),
                       "outside_candidates": outside,
                       "relation": _relation(decision.relation),
                       "confidence": decision.confidence,
                       "justification": decision.justification},
            "person": _person(item, names)}


def llm_choice(decisions, part: Reference) -> dict:
    """What the LLM chose before the rule: the right term, and "ninguno" when it was right."""
    by_key = {d.key: d for d in decisions}
    right = none_right = 0
    for item in part.items:
        d = by_key.get(item.key)
        if d is None or d.status != STATUS_DECIDED:
            continue
        if item.status == STATUS_ALIGNED:
            right += d.dpv_iri in item.terms
        else:
            none_right += d.none
    return {"right_term": right, "aligned": part.aligned(), "none_right": none_right,
            "no_match": len(part.items) - part.aligned()}


# --------------------------------------------------------------------------- one run

def lab_dir(cfg: dict | None = None) -> Path:
    from src.alignment.inherit import base_paths
    return base_paths(cfg)["candidates"].parent / LAB_DIR


def run_file(version: int, half: str, variant: str = "") -> str:
    return f"{PROMPT_NAME}-v{version}{'-' + variant if variant else ''}-{half}.json"


def gemini_file(version: int) -> str:
    return f"{PROMPT_NAME}-v{version}-{DEV}-para-gemini.txt"


def decide_half(inputs, prompt, half: str = DEV, *, llm=None, embed_fn=None,
                budget: int = DEFAULT_BUDGET, cache_path=None, chroma_path=None,
                sleep_fn=time.sleep, cfg: dict | None = None, base=None,
                definitions: dict | None = None):
    """One half of the 552 decisions decided with ``prompt`` and the production rule:
    (part, concepts it saw, report, accepted, base). ``definitions``: those the law gave."""
    from src.alignment import law_definitions as ld
    from src.alignment.inherit import load_base
    from src.alignment.knowledge import build_knowledge
    if half not in HALVES:
        raise ValueError(f"La mitad debe ser {DEV} o {TEST}.")
    part = split(inputs.references[REF_BASE], inputs.concepts)[half]
    concepts = [inputs.concepts[i.key] for i in part.items if i.key in inputs.concepts]
    if definitions is not None:
        if not ld.added(definitions):
            raise ValueError("La ley no dio ninguna definición: corre primero "
                             "--preparar-definiciones.")
        concepts = ld.with_definitions(concepts, definitions)
    base = base if base is not None else load_base(cfg=cfg)
    build_knowledge(cfg, embed_fn=embed_fn, targets=inputs.targets, base=base, path=chroma_path)
    report = decide(concepts, inputs.targets, llm=llm, embed_fn=embed_fn,
                    settings=replace(settings_from_config(cfg), max_calls=budget),
                    forbidden={i.key: set(i.forbidden) for i in part.items},
                    cache_path=cache_path, chroma_path=chroma_path, sleep_fn=sleep_fn, cfg=cfg,
                    prompt=prompt)
    accepted = apply_rule(report.decisions, **rule_settings(cfg).kwargs())
    return part, concepts, report, accepted, base


def run_half(inputs, version: int, half: str = DEV, *, llm=None, embed_fn=None,
             budget: int = DEFAULT_BUDGET, cache_path=None, chroma_path=None,
             sleep_fn=time.sleep, cfg: dict | None = None, base=None,
             definitions: dict | None = None) -> dict:
    """Decide one half of the 552 decisions with ``decide.v<version>`` and the production rule;
    with ``definitions`` (a result of ``law_definitions``), as the "ontología + ley" mode does:
    the concepts without a definition take the one the law gave them."""
    from src.alignment import law_definitions as ld
    prompt = prompt_version(version)
    problems, _ = check_prompt(prompt.text)
    if problems:
        raise ValueError(f"El prompt {PROMPT_NAME}.v{version} no sirve: {'; '.join(problems)}.")
    part, concepts, report, accepted, base = decide_half(
        inputs, prompt, half, llm=llm, embed_fn=embed_fn, budget=budget, cache_path=cache_path,
        chroma_path=chroma_path, sleep_fn=sleep_fn, cfg=cfg, base=base, definitions=definitions)
    used = {c.key: c for c in concepts}
    rule = rule_settings(cfg)
    names = {t.iri: t.name for t in inputs.targets.targets}
    decided = {c.key: c for c in base.concepts}
    by_key = {d.key: d for d in report.decisions}
    examples: dict[str, list] = {kind: [] for kind in ERROR_LABELS}
    for item in part.items:
        d = by_key.get(item.key)
        kind = classify(d, item, accepted.get(item.key))
        if kind is not None:
            examples[kind].append(example(d, item, used.get(item.key), names, decided))
    pending = sum(d.status == STATUS_PENDING for d in report.decisions)
    given = None
    if definitions is not None:
        given = {"law": definitions["law"]["file"], "added": len(ld.added(definitions)),
                 "in_half": sum(c.definition != (inputs.concepts[c.key].definition)
                                for c in concepts),
                 "fingerprint": ld.fingerprint(definitions)}
    return {"prompt": prompt.manifest(), "half": half, "seed": SPLIT_SEED,
            "variant": DEFINITIONS if definitions is not None else "", "definitions": given,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": next((d.model for d in report.decisions if d.model), None),
            "concepts": len(part.items), "aligned": part.aligned(), "complete": not pending,
            "pending": pending, "calls": report.calls, "cached": report.cached,
            "quota_exhausted": report.quota_exhausted,
            "call_limit_reached": report.call_limit_reached,
            "llm_errors": [e["error"] for e in report.errors][:3],
            "rule": rule.label(), "score": score(accepted, part),
            "llm": llm_choice(report.decisions, part),
            "errors": {kind: len(found) for kind, found in examples.items()},
            "examples": examples}


def save_run(result: dict, folder: Path, reference: Reference | None = None,
             concepts: dict | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    version = result["prompt"]["version"]
    path = folder / run_file(version, result["half"], result.get("variant", ""))
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if reference is not None and not (folder / SPLIT_FILE).exists():
        halves = split(reference, concepts or {})
        (folder / SPLIT_FILE).write_text(json.dumps(
            {"seed": SPLIT_SEED, **{h: [i.key for i in r.items] for h, r in halves.items()}},
            indent=2, ensure_ascii=False), encoding="utf-8")
    if result["half"] == DEV and result["complete"] and not result.get("variant"):
        text = gemini_message(result, load_prompt(PROMPT_NAME, version).text, next_version())
        (folder / gemini_file(version)).write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- for Gemini

def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}".replace(".", ",") + " %"


def _points(gain: float) -> str:
    return f"{'+' if gain >= 0 else ''}{100 * gain:.1f}".replace(".", ",") + " puntos"


CONTEXT_EXAMPLE = """=== C1 ===
Concepto: Data_breach (tipo: class)
Etiqueta: Violacion de seguridad de los datos
Definicion: Incidente que expone, destruye o altera datos personales sin autorizacion.
Candidatos DPV:
1. DataBreach | Data Breach | definicion: ... | padres en el DPV: Incident
2. PersonalDataBreach | Personal Data Breach | definicion: ... | padres en el DPV: DataBreach
(hasta 10 candidatos)
Precedentes (decisiones de una persona en conceptos parecidos):
- Security_incident: sin correspondencia
- Breach_notification: DataBreachNotification (closeMatch)"""


def _example_lines(n: int, e: dict) -> list[str]:
    kinds = ", ".join(e["kinds"]) or "?"
    head = f"{n}. Concepto: {e['name']} (tipo: {kinds})"
    if e.get("label"):
        head += f" · etiqueta: {e['label']}"
    lines = [head]
    if e.get("definition"):
        lines.append(f"   Definición: {' '.join(str(e['definition']).split())[:300]}")
    lines.append("   Candidatos: " + ", ".join(e["candidates"]))
    if e["precedents"]:
        lines.append("   Precedentes: " + " | ".join(e["precedents"]))
    a = e["answer"]
    said = a["dpv"] or "—"
    if a.get("outside_candidates"):
        said += " (no está entre sus candidatos)"
    if a["relation"]:
        said += f" ({a['relation']})"
    lines.append(f"   Respondió: {said}, confianza {a['confidence'] or '—'}. "
                 f"«{a['justification'] or ''}»")
    lines.append(f"   La persona decidió: {e['person']}")
    return lines


def next_version() -> int:
    return max(available().get(PROMPT_NAME, [0])) + 1


def gemini_message(result: dict, prompt_text: str, new_version: int | None = None,
                   examples: int = EXAMPLES) -> str:
    """Everything to paste in a Gemini chat to get the next version (one change)."""
    version = result["prompt"]["version"]
    new_version = new_version or version + 1
    s, llm = result["score"], result["llm"]
    rng = random.Random(SPLIT_SEED)
    lines = [
        "Necesito que mejores un prompt. Lo uso con un modelo de lenguaje "
        f"({result.get('model') or 'Gemini'}, temperatura 0, salida JSON) para alinear los "
        "conceptos de ontologías de privacidad con el Data Privacy Vocabulary (DPV).",
        "",
        "CÓMO LO USO",
        "- El modelo recibe primero el contexto (un bloque por concepto, hasta 10 por llamada) y "
        "después el prompt.",
        "- Un bloque del contexto se ve así:",
        CONTEXT_EXAMPLE,
        f"- Después, una regla fija decide qué entra a la ontología: {result['rule']}. "
        "Lo demás no entra.",
        "- Lo mido contra 552 decisiones que tomó una persona, divididas en dos mitades. Iteramos "
        "solo con la mitad de desarrollo; la de prueba se guarda para la decisión final.",
        "",
        f"PROMPT ACTUAL ({PROMPT_NAME}.v{version})",
        "```",
        prompt_text,
        "```",
        "",
        f"RESULTADO EN LA MITAD DE DESARROLLO ({result['concepts']} conceptos: "
        f"{result['aligned']} que la persona alineó y {result['concepts'] - result['aligned']} "
        "sin correspondencia)",
        f"- Con la regla: {s['proposals']} propuestas, {s['correct']} correctas: precisión "
        f"{_pct(s['precision'])}, cobertura {_pct(s['coverage'])}, F1 {_pct(s['f1'])}.",
        f"- Antes de la regla: eligió el término de la persona en {llm['right_term']} de "
        f"{llm['aligned']} conceptos alineados y dijo \"ninguno\" bien en {llm['none_right']} de "
        f"{llm['no_match']}.",
        "- Errores: " + "; ".join(f"{ERROR_LABELS[k]}: {n}"
                                  for k, n in result["errors"].items() if n),
        "",
        f"EJEMPLOS DE ERRORES (hasta {examples} por tipo)",
    ]
    for kind, found in result["examples"].items():
        if not found:
            continue
        shown = sorted(found, key=lambda e: e["key"])
        if len(shown) > examples:
            shown = sorted(rng.sample(shown, examples), key=lambda e: e["key"])
        lines += ["", f"### {ERROR_LABELS[kind]} ({len(found)})"]
        for n, e in enumerate(shown, start=1):
            lines += _example_lines(n, e)
    lines += [
        "",
        "LO QUE TE PIDO",
        "1. Lee los ejemplos y busca el patrón que causa más errores.",
        f"2. Propón {PROMPT_NAME}.v{new_version} con UN SOLO cambio (una regla nueva, una regla "
        "reescrita o un ejemplo inventado), el que más errores corrija sin crear otros. La meta "
        "es subir el F1 sin bajar la precisión del 75 %.",
        "3. No cambies la salida: la clave \"decisions\" con las claves \"concept\", \"dpv\", "
        "\"relation\", \"confidence\" y \"justification\"; la respuesta \"ninguno\"; las "
        "confianzas alta, media y baja; y las relaciones exactMatch, closeMatch, broadMatch, "
        "narrowMatch y relatedMatch.",
        "4. No nombres conceptos ni términos de estos ejemplos: el prompt debe servir para "
        "cualquier ontología y cualquier ley.",
        "5. Escríbelo sin tildes ni eñes, como el actual.",
        "Responde con una línea «Cambio: ...» que lo explique en una oración y, después, el "
        "prompt completo dentro de un bloque de código.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- comparing

def load_runs(folder: Path) -> dict[tuple[int, str, str], dict]:
    """Every saved run by (version, variant, half); the variant is "" or DEFINITIONS."""
    runs = {}
    for path in sorted(folder.glob(f"{PROMPT_NAME}-v*-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs[(data["prompt"]["version"], data.get("variant", ""), data["half"])] = data
    return runs


def is_stale(run: dict) -> bool:
    """The prompt file changed after the run (a published version is never edited)."""
    try:
        current = load_prompt(PROMPT_NAME, run["prompt"]["version"])
    except PromptNotFound:
        return True
    return current.sha256 != run["prompt"]["sha256"]


def verdict(current: dict | None, candidate: dict | None, min_precision: float,
            min_gain: float = MIN_GAIN) -> dict:
    """Adopt a version only with its test half: precision >= the floor and F1 + ``min_gain``."""
    if current is None or candidate is None:
        missing = [str(v) for v, r in (("en uso", current), ("nueva", candidate)) if r is None]
        return {"adopt": None, "reason": f"falta correr la mitad de prueba ({', '.join(missing)})"}
    if not (current["complete"] and candidate["complete"]):
        return {"adopt": None, "reason": "una de las corridas de prueba quedó incompleta"}
    if is_stale(current) or is_stale(candidate):
        return {"adopt": None,
                "reason": "un prompt cambió después de su corrida: vuelve a correrla"}
    precision = candidate["score"]["precision"]
    gain = (candidate["score"]["f1"] or 0) - (current["score"]["f1"] or 0)
    ok_precision = precision is not None and precision >= min_precision
    ok_gain = round(gain, 4) >= min_gain
    reason = (f"precisión {_pct(precision)} ({'>=' if ok_precision else '<'} "
              f"{_pct(min_precision)}); F1 {_points(gain)} ({'>=' if ok_gain else '<'} "
              f"+{100 * min_gain:.0f})")
    return {"adopt": ok_precision and ok_gain, "reason": reason, "gain": round(gain, 4)}


def combined(runs: list[dict]) -> dict:
    """The score of both halves together (the 552 decisions)."""
    proposals = sum(r["score"]["proposals"] for r in runs)
    correct = sum(r["score"]["correct"] for r in runs)
    aligned = sum(r["score"]["aligned"] for r in runs)
    precision = round(correct / proposals, 4) if proposals else None
    coverage = round(correct / aligned, 4) if aligned else None
    f1 = (round(2 * precision * coverage / (precision + coverage), 4)
          if precision and coverage else None)
    return {"proposals": proposals, "correct": correct, "aligned": aligned,
            "precision": precision, "coverage": coverage, "f1": f1}


def definitions_verdict(plain: list, with_law: list, min_precision: float) -> dict:
    """The definitions of the law keep reaching the decider only if, on the 552 decisions, the
    precision stays >= the floor and the coverage goes up (nothing is tuned with them)."""
    if not all(plain) or not all(with_law):
        return {"adopt": None, "reason": "faltan mitades: corre las dos, con y sin definiciones"}
    if not all(r["complete"] for r in plain + with_law):
        return {"adopt": None, "reason": "una de las corridas quedó incompleta"}
    if any(is_stale(r) for r in plain + with_law):
        return {"adopt": None,
                "reason": "un prompt cambió después de su corrida: vuelve a correrla"}
    before, after = combined(plain), combined(with_law)
    ok_precision = after["precision"] is not None and after["precision"] >= min_precision
    ok_coverage = (after["coverage"] or 0) > (before["coverage"] or 0)
    reason = (f"precisión {_pct(after['precision'])} ({'>=' if ok_precision else '<'} "
              f"{_pct(min_precision)}); cobertura {_pct(before['coverage'])} -> "
              f"{_pct(after['coverage'])}")
    return {"adopt": ok_precision and ok_coverage, "reason": reason, "before": before,
            "after": after}


def _row(run: dict | None) -> str:
    if run is None:
        return f"{'—':>6}{'':>6}{'':>11}{'':>11}{'':>9}"
    s = run["score"]
    mark = "" if run["complete"] else " *"
    return (f"{s['proposals']:>6}{s['correct']:>6}{_pct(s['precision']):>11}"
            f"{_pct(s['coverage']):>11}{_pct(s['f1']):>9}{mark}")


def _name(version: int, variant: str, in_use: int) -> str:
    if variant:
        return f"v{version} + definiciones"
    return f"v{version}" + (" (en uso)" if version == in_use else "")


def render_table(runs: dict, in_use: int, min_precision: float) -> str:
    rows = sorted({(v, variant) for v, variant, _ in runs} | {(in_use, "")})
    head = f"{'prop.':>6}{'corr.':>6}{'precisión':>11}{'cobertura':>11}{'F1':>9}"
    lines = [f"Banco de prompts ({PROMPT_NAME}): tus 552 decisiones en dos mitades (semilla "
             f"{SPLIT_SEED}), con la regla de producción",
             f"  {'versión':<20}{'desarrollo':<45}prueba",
             f"  {'':<20}{head}  {head}"]
    for v, variant in rows:
        stale = any(is_stale(runs[(v, variant, h)]) for h in HALVES if (v, variant, h) in runs)
        lines.append(f"  {_name(v, variant, in_use):<20}{_row(runs.get((v, variant, DEV)))}  "
                     f"{_row(runs.get((v, variant, TEST)))}"
                     + ("  (el prompt cambió: vuelve a correrla)" if stale else ""))
    if any(not r["complete"] for r in runs.values()):
        lines.append("  * incompleta: vuelve a correrla para terminarla (lo ya decidido sale de "
                     "la caché)")
    for v, variant in rows:
        if variant or v == in_use or (v, "", TEST) not in runs:
            continue
        result = verdict(runs.get((in_use, "", TEST)), runs[(v, "", TEST)], min_precision)
        if result["adopt"] is None:
            lines.append(f"  v{v} frente a v{in_use}: {result['reason']}")
        elif result["adopt"]:
            lines.append(f"  v{v} frente a v{in_use}: SE ADOPTA ({result['reason']}). Cambia "
                         f"decider.prompt_version a {v} en config/config.yaml.")
        else:
            lines.append(f"  v{v} frente a v{in_use}: no se adopta ({result['reason']}).")
    with_law = [runs.get((in_use, DEFINITIONS, h)) for h in HALVES]
    if any(with_law):
        result = definitions_verdict([runs.get((in_use, "", h)) for h in HALVES], with_law,
                                     min_precision)
        head = f"  Definiciones de la ley con v{in_use} (las 552 decisiones):"
        if result["adopt"] is None:
            lines.append(f"{head} {result['reason']}")
        elif result["adopt"]:
            lines.append(f"{head} AYUDAN ({result['reason']}): el decisor sigue recibiéndolas.")
        else:
            lines.append(f"{head} no ayudan ({result['reason']}): quedan en el módulo de "
                         "definiciones, pero el decisor no debería recibirlas.")
    return "\n".join(lines)


def render_run(result: dict, folder: Path, runs: dict, in_use: int) -> str:
    s, llm = result["score"], result["llm"]
    v, half, variant = result["prompt"]["version"], result["half"], result.get("variant", "")
    given = result.get("definitions")
    lines = [f"Banco de prompts: {PROMPT_NAME}.v{v}"
             + (" con las definiciones de la ley" if variant else "")
             + f" en la mitad de {half} ({result['concepts']} de tus 552 decisiones; "
             f"{result['aligned']} alineadas)"]
    if given:
        lines.append(f"  Definiciones de {given['law']}: {given['in_half']} conceptos de esta "
                     "mitad las recibieron")
    lines += [
             f"  IA: {result['calls']} llamadas, {result['cached']} respuestas desde la caché"
             + (" · CUOTA AGOTADA: vuelve a correr mañana" if result["quota_exhausted"] else
                " · límite de llamadas: vuelve a correr para seguir"
                if result["call_limit_reached"] else "")
             + (f" · pendientes: {result['pending']}" if result["pending"] else "")]
    if result["llm_errors"]:
        lines.append(f"  Error de la IA: {result['llm_errors'][0]}")
    lines += [f"  Regla: {result['rule']}",
              f"  Propuestas {s['proposals']} · correctas {s['correct']} · precisión "
              f"{_pct(s['precision'])} · cobertura {_pct(s['coverage'])} · F1 {_pct(s['f1'])}",
              f"  Antes de la regla: el término de la persona en {llm['right_term']} de "
              f"{llm['aligned']} alineados; \"ninguno\" bien en {llm['none_right']} de "
              f"{llm['no_match']}",
              "  Errores: " + (" · ".join(f"{ERROR_LABELS[k]} {n}"
                                          for k, n in result["errors"].items() if n)
                               or "ninguno")]
    other = v if variant else in_use
    previous = runs.get((other, "", half))
    if (previous is not None and (variant or v != in_use) and previous["complete"]
            and result["complete"]):
        gain = (s["f1"] or 0) - (previous["score"]["f1"] or 0)
        label = f"v{other} sin definiciones" if variant else f"v{in_use} (en uso)"
        lines.append(f"  Frente a {label} en esta mitad: precisión "
                     f"{_pct(previous['score']['precision'])} -> {_pct(s['precision'])}, "
                     f"cobertura {_pct(previous['score']['coverage'])} -> "
                     f"{_pct(s['coverage'])}, F1 {_pct(previous['score']['f1'])} -> "
                     f"{_pct(s['f1'])} ({_points(gain)})")
    if variant:
        lines.append("  Corre las dos mitades, con y sin definiciones, y compara: "
                     "py -m src.alignment.prompt_lab --tabla")
    elif half == DEV and result["complete"]:
        lines.append(f"  Para Gemini: {folder / gemini_file(v)}")
    elif half == DEV:
        lines.append("  El mensaje para Gemini se arma cuando la corrida esté completa.")
    else:
        lines.append("  Compara las versiones y mira si se adopta: py -m src.alignment.prompt_lab "
                     "--tabla")
    return "\n".join(lines)


# --------------------------------------------------------------------------- the law (mode 3)

def definitions_workspace(cfg: dict | None = None):
    """The workspace "OntoPriv + LOPDP" (mode 3) where the law defines the base concepts."""
    from src.config import ROOT
    from src.workspace.store import KIND_LAW, KIND_ONTOLOGY, open_upload
    files = (cfg or {}).get("inputs", {}) or {}
    paths = {KIND_ONTOLOGY: ROOT / files.get("ontology", "data/input/ontopriv.rdf"),
             KIND_LAW: ROOT / files.get("legal_text", "data/input/lopdp.pdf")}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}")
    return open_upload({kind: (path.name, path.read_bytes()) for kind, path in paths.items()},
                       cfg=cfg)


def definitions_cfg(cfg: dict, budget: int) -> dict:
    block = (cfg or {}).get("law_definitions", {}) or {}
    return {**cfg, "law_definitions": {**block, "enabled": True, "max_calls": budget}}


def prepare_definitions(cfg: dict, *, budget: int = DEFINITIONS_BUDGET, llm=None,
                        embed_fn=None, sleep_fn=None, progress=None, ws=None):
    """The definitions the LOPDP gives to the base concepts, with the steps of mode 3."""
    from src.alignment import law_definitions as ld
    from src.ingest.law_upload import prepare_law
    ws = ws or definitions_workspace(cfg)
    prepare_law(ws, embed_fn=embed_fn, progress=progress)
    run_cfg = definitions_cfg(cfg, budget)
    result = ld.load_result(ws)
    if not ld.is_current(result, ws, run_cfg):
        result = ld.definitions_for_workspace(ws, embed_fn=embed_fn, llm=llm, cfg=run_cfg,
                                              sleep_fn=sleep_fn, progress=progress)
    return ws, result


def load_definitions(ws) -> dict:
    """The definitions ready for the comparison: computed and without concepts left waiting."""
    from src.alignment import law_definitions as ld
    result = ld.load_result(ws)
    if result is None or result["skipped"]:
        raise ValueError("Primero prepara las definiciones: py -m src.alignment.prompt_lab "
                         "--preparar-definiciones")
    if result["summary"]["pending"]:
        raise ValueError(f"Faltan {result['summary']['pending']} definiciones: vuelve a correr "
                         "py -m src.alignment.prompt_lab --preparar-definiciones")
    return result


def render_definitions(result: dict, ws) -> str:
    from src.alignment import law_definitions as ld
    block = {**result["summary"], "law": result["law"]["file"], "skipped": result["skipped"],
             "law_concepts": result.get("law_concepts")}
    lines = ["Definiciones de la LOPDP para los conceptos de la base (modo ontología + ley)",
             f"  {ld.summary_line(block)}",
             f"  IA: {ld.ai_line(result)}",
             f"  Espacio de trabajo: {ws.path}"]
    if result["summary"]["pending"]:
        lines.append("  Faltan definiciones: vuelve a correr --preparar-definiciones para "
                     "seguir (lo ya definido sale de la caché).")
    else:
        lines.append("  Listas. Ahora: py -m src.alignment.prompt_lab --con-definiciones (y con "
                     "--mitad prueba)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def _run(argv: list[str] | None = None) -> int:
    from src.alignment.decider_eval import load_eval_inputs
    from src.alignment.inherit import base_paths
    from src.config import load_config
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.prompt_lab",
        description="Banco de prompts del decisor: mide una versión de decide.vN en una mitad de "
                    "tus 552 decisiones y prepara el mensaje para pedirle a Gemini la siguiente.")
    parser.add_argument("--version", type=int, help="versión del prompt (por defecto, la en uso)")
    parser.add_argument("--mitad", choices=HALVES, default=DEV)
    parser.add_argument("--tabla", action="store_true", help="comparar las versiones medidas")
    parser.add_argument("--revisar", type=int, metavar="N",
                        help="solo revisar que decide.vN mantiene la salida (sin llamar a la IA)")
    parser.add_argument("--preparar-definiciones", action="store_true",
                        help="la LOPDP define los conceptos de OntoPriv (modo ontología + ley)")
    parser.add_argument("--con-definiciones", action="store_true",
                        help="decidir con las definiciones que dio la ley")
    parser.add_argument("--max-llamadas", type=int)
    args = parser.parse_args(argv)
    cfg = load_config()
    in_use = decide_prompt(cfg).version
    folder = lab_dir(cfg)
    min_precision = float((cfg.get("auto", {}) or {}).get("min_precision", 0.75))
    try:
        if args.revisar is not None:
            prompt = prompt_version(args.revisar)
            errors, warnings = check_prompt(prompt.text)
            print(f"{PROMPT_NAME}.v{args.revisar}: " + ("no sirve: " + "; ".join(errors)
                                                        if errors else "mantiene la salida"))
            for w in warnings:
                print(f"  Aviso: {w}")
            return 1 if errors else 0
        if args.tabla:
            print(render_table(load_runs(folder), in_use, min_precision))
            return 0
        if args.preparar_definiciones:
            budget = DEFINITIONS_BUDGET if args.max_llamadas is None else args.max_llamadas
            ws, defined = prepare_definitions(cfg, budget=budget,
                                              progress=lambda label: print(f"  {label}..."))
            print(render_definitions(defined, ws))
            return 0
        definitions = load_definitions(definitions_workspace(cfg)) if args.con_definiciones \
            else None
        version = args.version or in_use
        print(f"Decidiendo la mitad de {args.mitad} con {PROMPT_NAME}.v{version}"
              + (" y las definiciones de la ley" if definitions else "")
              + " (lo ya decidido sale de la caché)...")
        inputs = load_eval_inputs(cfg)
        budget = DEFAULT_BUDGET if args.max_llamadas is None else args.max_llamadas
        result = run_half(inputs, version, args.mitad, budget=budget, cfg=cfg,
                          cache_path=base_paths(cfg)["candidates"].parent / "decider" / CACHE_FILE,
                          definitions=definitions)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    except KeyError as exc:                            # LLM_API_KEY / LLM_MODEL missing
        print(f"Error: falta la variable {exc} en .env")
        return 1
    save_run(result, folder, inputs.references[REF_BASE], inputs.concepts)
    print(render_run(result, folder, load_runs(folder), in_use))
    print(f"  Guardado en: {folder / run_file(version, args.mitad, result['variant'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())