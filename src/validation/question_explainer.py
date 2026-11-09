# The AI says in one sentence why a competency question stays yellow (S9-T06). One call for all
# the yellow questions of a run, a cache per question and a fixed sentence when there is no AI.
# Prompt v2 also names the row that holds each missing element; the row must exist (--medir).
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.ai.prompts import load_prompt
from src.console import counted
from src.validation import question_catalog as qc

PROMPTS = {v: load_prompt("explain", v) for v in (1, 2)}
PROMPT = PROMPTS[1]
CACHE_FILE = "questions-ai-cache.jsonl"
EVALUATION_FILE = "questions-explain-evaluation.json"
REFERENCE_FILE = Path(__file__).resolve().parents[2] / "data" / "reference" / "explain-reference.json"
AI, FIXED = "ia", "fija"
NONE_ROW = "ninguna"
MAX_CHARS = 300


@dataclass(frozen=True)
class ExplainSettings:
    explain: bool = True
    prompt_version: int = 1
    max_tokens: int = 2048
    rows_shown: int = 60


def explain_settings(cfg: dict | None = None) -> ExplainSettings:
    from src.config import load_config
    block = (cfg if cfg is not None else load_config()).get("questions", {}) or {}
    s = ExplainSettings(**{k: v for k, v in block.items()
                           if k in ExplainSettings.__dataclass_fields__})
    if s.max_tokens < 1 or s.rows_shown < 1 or s.prompt_version not in PROMPTS:
        raise ValueError("Valores inválidos en el bloque questions de config.yaml.")
    return s


# --------------------------------------------------------------------------- what the AI sees

def _one_line(text, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit].rstrip() if limit else text


@dataclass
class Item:
    """One yellow question as the AI sees it, with the ids of its elements and rows (v2)."""
    body: str
    elements: dict = field(default_factory=dict)       # "E1" -> name of the missing element
    rows: dict = field(default_factory=dict)           # "F1" -> text of the row


def _missing(q: qc.Question, result: dict) -> list[tuple[str, str]]:
    """(name, what the rule looked for) of each missing element; a value counts as one."""
    if q.group == qc.GENERIC:
        return []
    if q.kind == qc.VALUE:
        value = f"{q.expected.value:g} {q.expected.unit}".strip()
        return [(value, value)]
    return [(e.name, ", ".join(e.terms) or e.name) for e in q.expected.elements
            if e.name in result.get("missing", [])]


def item(q: qc.Question, result: dict, rows: list[str], aligned: list[str], shown: int = 60,
         version: int = 1) -> Item:
    """What was asked, what the rule wanted and what the query brought."""
    lines = [f"Pregunta: {q.question}", f"Criterio: {result['criterion']}"]
    if q.group == qc.GENERIC:
        lines += [f"Busca: {qc.looks_for_text(q)}",
                  f"Términos del DPV con los que la ontología sí se alinea: "
                  f"{', '.join(aligned[:shown]) or 'ninguno'}"]
    else:
        lines.append(f"Respuesta esperada (de la ley): {q.expected.text}")
        if q.kind == qc.VALUE:
            lines.append(f"Valor esperado: {q.expected.value:g} {q.expected.unit}; valores que "
                         f"trajo: {', '.join(result.get('values') or []) or 'ninguno'}")
        else:
            found = ", ".join(f["name"] for f in result.get("found", [])) or "ninguno"
            lines.append(f"Encontrados: {found}")
    missing, shown_rows = _missing(q, result), [_one_line(r, 160) for r in rows[:shown]]
    if version == 1:
        if missing and q.kind != qc.VALUE:
            lines.append("Faltan: " + "; ".join(f"{name} (se buscaron: {terms})"
                                               if terms != name else name
                                               for name, terms in missing))
        lines.append(f"Filas que trajo la consulta ({result['rows']}): "
                     + (" | ".join(shown_rows) or "ninguna"))
        return Item("\n".join(lines))
    elements = {f"E{n}": name for n, (name, _) in enumerate(missing, 1)}
    numbered = {f"F{n}": text for n, text in enumerate(shown_rows, 1)}
    if missing:
        lines.append("Elementos que faltan:")
        lines += [f"{eid}: {name}" + (f" (se buscaron: {terms})" if terms != name else "")
                  for eid, (name, terms) in zip(elements, missing)]
    lines.append(f"Filas que trajo la consulta ({result['rows']}"
                 + (f", se muestran {len(shown_rows)}" if len(shown_rows) < result["rows"] else "")
                 + "):" + ("" if numbered else " ninguna"))
    lines += [f"{fid}: {text}" for fid, text in numbered.items()]
    return Item("\n".join(lines), elements, numbered)


def body(q: qc.Question, result: dict, rows: list[str], aligned: list[str], shown: int = 60,
         version: int = 1) -> str:
    return item(q, result, rows, aligned, shown, version).body


def fixed_sentence(q: qc.Question, result: dict) -> str:
    """What is said when there is no AI: the rule's own reason, in plain words."""
    if q.group == qc.GENERIC:
        return f"La consulta no trajo {qc.looks_for_text(q)}."
    if q.kind == qc.VALUE:
        values = ", ".join(result.get("values") or []) or "ningún valor"
        return (f"La respuesta no trae el valor {q.expected.value:g} {q.expected.unit} "
                f"(trajo: {values}).")
    return (f"Ninguna fila nombra a {', '.join(result.get('missing', []))} ni cita un artículo "
            f"que solo le corresponda.")


# --------------------------------------------------------------------------- asking the AI

def cache_key(text: str, model: str | None, generation: dict | None = None,
              prompt=PROMPT) -> str:
    content = {"prompt": prompt.sha256, "model": model, "body": text}
    if generation:
        content["generation"] = generation
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_explanations(raw: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    data = None
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if isinstance(data, dict):
        data = data.get("explanations")
    if not isinstance(data, list):
        raise ValueError("La respuesta del LLM no trae la lista de explicaciones en JSON.")
    return [d for d in data if isinstance(d, dict)]


_ONE_ROW = re.compile(r"\b(la|una|esta) fila [FE]\d+\b", re.IGNORECASE)
_ROWS = re.compile(r"\b(las) filas [FE]\d+(?:(?:,\s*| y | e )[FE]\d+)*\b", re.IGNORECASE)
_ID = re.compile(r"\s*\(?\b[FE]\d+\b\)?")


def _like(word: str, m: re.Match) -> str:
    """``word`` with the capital of the text it replaces: «La fila F4» -> «Una fila»."""
    return word.capitalize() if m.group(1)[0].isupper() else word


def readable(text: str) -> str:
    """The sentence without the ids F1/E1 that v2 sometimes leaves although its rule 2 says
    not to (measured on the base): «la fila F4 contiene» -> «una fila contiene»."""
    text = _ONE_ROW.sub(lambda m: _like("una fila", m), text)
    text = _ROWS.sub(lambda m: _like("algunas filas", m), text)
    text = _ID.sub("", text).strip()
    return text[:1].upper() + text[1:]


def check_answer(answer: dict, it: Item, version: int) -> dict | None:
    """The answer of one question, or None when it is not usable: no sentence, too long or
    (v2) a row or an element that is not in the block."""
    text = _one_line(answer.get("explanation"))
    if not text or len(text) > MAX_CHARS:
        return None
    if version == 1:
        return {"explanation": text}
    matches = answer.get("matches") or []
    if not isinstance(matches, list):
        return None
    rows = {}
    for m in matches:
        eid = _one_line((m or {}).get("element")).upper() if isinstance(m, dict) else ""
        row = _one_line(m.get("row")) if isinstance(m, dict) else ""
        row = NONE_ROW if row.lower() == NONE_ROW else row.upper()
        if eid not in it.elements or (row != NONE_ROW and row not in it.rows):
            return None
        rows[eid] = None if row == NONE_ROW else it.rows[row]
    return {"explanation": text,
            "matches": [{"element": name, "row": rows.get(eid)}
                        for eid, name in it.elements.items()]}


@dataclass
class ExplainReport:
    answers: dict = field(default_factory=dict)
    calls: int = 0
    cached: int = 0
    invalid: int = 0
    errors: list[str] = field(default_factory=list)

    def ai(self, available: bool, error: str | None = None) -> dict:
        out = {"available": available, "calls": self.calls, "cached": self.cached,
               "invalid": self.invalid, "errors": self.errors[:3]}
        return {**out, "error": error} if error else out


def ask(items: dict[str, Item | str], *, llm, settings: ExplainSettings, cache_path=None,
        sleep_fn=time.sleep) -> ExplainReport:
    """Cache first; the rest in ONE call (one retry after a 429 or a 503)."""
    from src.alignment.decider import generation_settings
    from src.alignment.justify import (QUOTA_WAIT_SECONDS, TRANSIENT_WAIT_SECONDS,
                                       _is_transient_error)
    from src.core.extract import _is_quota_error
    from src.transform.labels import _append_cache, _load_cache
    items = {k: v if isinstance(v, Item) else Item(v) for k, v in items.items()}
    version, prompt = settings.prompt_version, PROMPTS[settings.prompt_version]
    report = ExplainReport()
    model = getattr(llm, "model", None) or os.environ.get("LLM_MODEL")
    generation = generation_settings(llm)
    keys = {qid: cache_key(it.body, model, generation, prompt) for qid, it in items.items()}
    cache = _load_cache(cache_path)
    pending = []
    for qid in items:
        hit = cache.get(keys[qid])
        if hit is not None:
            report.answers[qid] = hit["answer"] if "answer" in hit else {
                "explanation": hit["explanation"]}
            report.cached += 1
        else:
            pending.append(qid)
    if not pending or llm is None:
        return report
    context = "\n\n".join(f"=== {qid} ===\n{items[qid].body}" for qid in pending)
    raw, err = None, None
    for attempt in range(2):
        try:
            raw = llm.ask(prompt=prompt.text, context=context, max_tokens=settings.max_tokens,
                          json_mode=True) or ""
            err = None
            break
        except Exception as exc:                       # noqa: BLE001 - reported, never raised
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
            parsed = parse_explanations(raw)
        except ValueError as exc:
            err = str(exc)
    if err is not None:
        report.errors.append(err[:300])
        return report
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new = []
    for answer in parsed:
        qid = _one_line(answer.get("id")).upper()
        if qid not in pending or qid in report.answers:
            continue
        checked = check_answer(answer, items[qid], version)
        if checked is None:
            report.invalid += 1
            continue
        report.answers[qid] = checked
        new.append({"cache_key": keys[qid], "question": qid, "answer": checked,
                    "explanation": checked["explanation"], "prompt": prompt.manifest(),
                    "model": model, "created": now})
    _append_cache(cache_path, new)
    return report


def explain(catalog: qc.Catalog, results: list[dict], answers: dict, *, cfg: dict | None = None,
            llm=None, use_llm: bool | None = False, cache_path=None, version: int | None = None,
            sleep_fn=time.sleep) -> dict:
    """Adds ``explanation`` and ``explained_by`` (and, with v2, ``matches``) to each yellow
    result; returns what the AI did. ``use_llm``: False never asks (the default, for direct
    calls); None asks if config.yaml lets."""
    from src.validation.questions import row_texts
    from src.validation.shape_inspector import YELLOW
    settings = explain_settings(cfg)
    if version is not None:
        settings = ExplainSettings(**{**settings.__dict__, "prompt_version": version})
    by_id = {q.id: q for q in catalog.questions}
    yellow = [r for r in results if r["color"] == YELLOW]
    aligned = sorted({row_texts([row[1]]) for row in answers.get("G01", {}).get("rows", [])
                      if len(row) > 1 and row[1]})
    items = {}
    for r in yellow:
        rows = sorted({row_texts(row) for row in answers.get(r["id"], {}).get("rows", [])} - {""})
        items[r["id"]] = item(by_id[r["id"]], r, rows, aligned, settings.rows_shown,
                              settings.prompt_version)
    if use_llm is None:
        from src.alignment.calibrate import auto_settings
        use_llm = auto_settings(cfg)["use_llm"]
    available = bool(use_llm) and settings.explain
    error = None if available else ("IA desactivada (--sin-ia)" if settings.explain
                                    else "desactivada en config.yaml (questions.explain)")
    if available and llm is None and items:
        try:
            from src.ai.llm_client import LLMClient
            llm = LLMClient()
        except KeyError as exc:                        # LLM_API_KEY / LLM_MODEL missing
            available, error = False, f"Falta la variable {exc} en .env"
    report = ask(items, llm=llm if available else None, settings=settings,
                 cache_path=cache_path, sleep_fn=sleep_fn)
    for r in yellow:
        answer = report.answers.get(r["id"])
        r["explanation"] = (readable(answer["explanation"]) if answer
                            else fixed_sentence(by_id[r["id"]], r))
        r["explained_by"] = AI if answer else FIXED
        if answer and "matches" in answer:
            r["matches"] = answer["matches"]
    return {"prompt": PROMPTS[settings.prompt_version].manifest(), "questions": len(yellow),
            **report.ai(available, error)}


def ai_text(ai: dict) -> str:
    """What the AI did, in one line."""
    if not ai["questions"]:
        return "nada que explicar"
    if not ai["available"]:                            # what the cache had still counts
        text = f"no usada; {ai['error']}"
        if ai["cached"]:
            text += f"; {ai['cached']} desde la caché"
        if ai["cached"] < ai["questions"]:
            text += "; frase fija" + (" para el resto" if ai["cached"] else "")
        return text
    text = f"{counted(ai['calls'], 'llamada', 'llamadas')}, {ai['cached']} desde la caché"
    if ai["invalid"]:
        text += f", {counted(ai['invalid'], 'inválida', 'inválidas')}"
    if ai["errors"]:
        text += f"; error: {ai['errors'][0]}"
    return text


# --------------------------------------------------------------------------- measuring (--medir)

def load_reference(path: Path = REFERENCE_FILE) -> dict:
    """For each yellow question of the base, the words of the row that holds each missing
    element (written by hand from the rows of the base, before asking prompt v2)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


def is_right(result: dict, reference: dict) -> bool:
    """Right when, for every missing element, the AI found the row that holds it: v1 by naming
    words of that row in its sentence, v2 by citing it (a row the system checked exists)."""
    if result.get("explained_by") != AI:
        return False
    if "matches" in result:
        cited = {m["element"]: qc.normalize(m["row"] or "") for m in result["matches"]}
        return all(any(qc.normalize(w) in cited.get(element, "") for w in words)
                   for element, words in reference.items())
    text = qc.normalize(result.get("explanation") or "")
    return all(any(qc.normalize(w) in text for w in words) for words in reference.values())


def adopted(right: dict[int, dict[str, bool]]) -> int:
    """The rule fixed before measuring: v2 when it is right in more questions than v1 and wrong
    in none that v1 got right; otherwise v1."""
    one, two = right[1], right[2]
    better = sum(two.values()) > sum(one.values())
    keeps = all(two.get(q, False) for q, ok in one.items() if ok)
    return 2 if better and keeps else 1


def measure(cfg: dict | None = None, *, llm=None, engine: str | None = None,
            out_dir: str | Path | None = None, sleep_fn=time.sleep) -> dict:
    """Both prompts on the yellow questions of the base, judged against the reference."""
    from datetime import datetime, timezone
    from src.alignment.inherit import base_paths
    from src.config import load_config
    from src.validation import questions as qs
    cfg = cfg if cfg is not None else load_config()
    catalog = qc.load_catalog(qc.CATALOG_FILE)
    questions = qc.for_mode(catalog, "base")
    sets = {qs.BASE_SET: qs.base_files(cfg)[qs.BASE_SET]}
    used, runs = qs.run(sets, [q.id for q in questions], cfg=cfg, engine=engine)
    answers = runs[qs.BASE_SET]["answers"]
    results = [qs.judge(q, answers.get(q.id)) for q in questions]
    reference = load_reference()
    folder = Path(out_dir) if out_dir else base_paths(cfg)["candidates"].parent
    folder.mkdir(parents=True, exist_ok=True)
    out = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "engine": used, "reference": str(REFERENCE_FILE.relative_to(REFERENCE_FILE.parents[2])
                                              .as_posix()),
           "rule": "v2 si acierta en más preguntas que v1 y no falla ninguna que v1 acertó",
           "versions": {}}
    right = {}
    for version in PROMPTS:
        mine = copy.deepcopy(results)
        ai = explain(catalog, mine, answers, cfg=cfg, llm=llm, use_llm=True, version=version,
                     cache_path=folder / CACHE_FILE, sleep_fn=sleep_fn)
        rows = [{"id": r["id"], "explanation": r.get("explanation"),
                 "explained_by": r.get("explained_by"), "matches": r.get("matches"),
                 "right": is_right(r, reference.get(r["id"], {}))}
                for r in mine if r["id"] in reference]
        right[version] = {r["id"]: r["right"] for r in rows}
        out["versions"][version] = {"ai": ai, "right": sum(right[version].values()),
                                    "total": len(rows), "questions": rows}
    out["adopted"] = adopted(right)
    out["configured"] = explain_settings(cfg).prompt_version
    (folder / EVALUATION_FILE).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return out


def render_measure(out: dict) -> str:
    lines = ["Explicaciones de la IA sobre las preguntas amarillas de la base (S9-T06)",
             f"  Referencia: {out['reference']}",
             f"  Regla fijada antes de medir: {out['rule']}"]
    for version, v in out["versions"].items():
        lines += ["", f"  explain.v{version}: {v['right']} de {v['total']} correctas · IA: "
                      f"{ai_text(v['ai'])}"]
        for r in v["questions"]:
            mark = "bien" if r["right"] else "mal"
            lines.append(f"    [{mark}] {r['id']} ({r['explained_by']}): {r['explanation']}")
            for m in r.get("matches") or []:
                lines.append(f"         {m['element']} ← {m['row'] or 'ninguna fila'}")
    lines += ["", f"  Se adopta: explain.v{out['adopted']} (config.yaml dice v{out['configured']})"]
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import base_paths
    from src.config import load_config
    parser = argparse.ArgumentParser(
        prog="py -m src.validation.question_explainer",
        description="Mide las explicaciones de la IA (prompts explain.v1 y v2) sobre las "
                    "preguntas amarillas de la base.")
    parser.add_argument("--medir", action="store_true", help="medir v1 contra v2")
    args = parser.parse_args(argv)
    if not args.medir:
        print("Error: indica --medir.")
        return 1
    cfg = load_config()
    try:
        out = measure(cfg)
    except (FileNotFoundError, ValueError, RuntimeError, qc.CatalogError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_measure(out))
    print(f"  Guardado en: {base_paths(cfg)['candidates'].parent / EVALUATION_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
