"""Concepts of an uploaded law: the AI extracts them and a fixed rule accepts them (S7-T09).

The first half of the "solo ley" mode (the second half, S7-T10, turns the accepted concepts into
a profile that imports OntoPriv-Core and aligns it with the DPV). For a law segmented in its
workspace (S7-T07, ``work/law-articles.json``):

1. EXTRACT (step "extract"): the LLM reads the articles in batches with the prompt
   ``extract.v2`` (any law, in Spanish or English) and proposes concepts, each with the article
   it comes from. The prompt carries the language the segmentation detected (``{idioma}``),
   because without it a quarter of the GDPR definitions came back in Spanish; labels and
   definitions follow the law, names are in English (UpperCamelCase for classes,
   lowerCamelCase starting with a verb for properties, whose label names the relation). It is
   the extraction of Sprint 3 (``src.core.extract``: JSONL that survives a truncated answer,
   resume after every batch, article numbers such as "3°" read as 3) with a pause between
   calls and a limit of calls per run (block ``extraction`` of config.yaml), saved in
   ``work/law-extraction.json``. A run that stops (quota, limit of
   calls, failed batch) leaves the rest pending; the next run continues where it stopped, with
   the same prompt, model and segmentation (another one starts again). Without AI nothing is
   extracted and the step says so.
2. ACCEPT (step "accept"): the rule of ``src.core.acceptance`` (decision D1) decides each
   concept: accepted, repeated (joined to the first one of its kind with the same label) or
   rejected, with its reasons. Measured on the 93 concepts of the LOPDP it agrees with the
   decisions of Sprint 5 in 91 (``py -m src.core.acceptance``). Result:
   ``work/law-concepts.json``.

Nothing is reviewed by the person here; S7-T10 writes the accepted concepts, and the person's
comments come in Sprint 10.

CLI: ``py -m src.core.law_concepts ARCHIVO [--sin-ia]``.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.ai.prompts import load_prompt
from src.core.acceptance import (ACCEPTED, REASONS, REJECTED, REPEATED, RULE_TEXT,
                                 RULE_VERSION, apply_rule, summarize)

PROMPT_VERSION = 2
PROMPT = load_prompt("extract", PROMPT_VERSION)
METHOD_VERSION = 1
EXTRACTION_FILE = "law-extraction.json"
RESULT_FILE = "law-concepts.json"
LANGUAGE_LABELS = {"es": "español", "en": "inglés"}
PROMPT_LANGUAGES = {"es": "espanol", "en": "ingles"}     # the prompts are written in ASCII


@dataclass(frozen=True)
class ExtractionSettings:
    batch_size: int = 8
    max_tokens: int = 8192
    pause_seconds: float = 5.0
    max_calls: int = 30


def extraction_settings(cfg: dict | None = None) -> ExtractionSettings:
    """The ``extraction`` block of config.yaml (defaults for the keys that are missing)."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("extraction", {}) or {}
    d = ExtractionSettings()
    s = ExtractionSettings(batch_size=int(block.get("batch_size", d.batch_size)),
                           max_tokens=int(block.get("max_tokens", d.max_tokens)),
                           pause_seconds=float(block.get("pause_seconds", d.pause_seconds)),
                           max_calls=int(block.get("max_calls", d.max_calls)))
    if s.batch_size < 1 or s.max_tokens < 256 or s.pause_seconds < 0 or s.max_calls < 1:
        raise ValueError("Valores inválidos en el bloque extraction de config.yaml "
                         "(batch_size >= 1, max_tokens >= 256, pause_seconds >= 0, "
                         "max_calls >= 1).")
    return s


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_result(ws) -> dict | None:
    return _load(ws.work_dir / RESULT_FILE)


# --------------------------------------------------------------------------- 1. extract

def prompt_text(language: str) -> str:
    """The prompt with the language of the law filled in."""
    return PROMPT.text.replace("{idioma}", PROMPT_LANGUAGES[language])


def extract_for_workspace(ws, *, llm=None, use_llm: bool | None = None,
                          cfg: dict | None = None, sleep_fn=None) -> dict:
    """Extract (or continue extracting) the concepts of the law; mark the step "extract"."""
    from src.alignment.calibrate import auto_settings
    from src.alignment.propose import _llm_for
    from src.core.extract import extraction_summary, propose_profile_concepts
    from src.ingest.law_upload import current_articles
    from src.workspace.store import DONE, ERROR, RUNNING
    settings = extraction_settings(cfg)
    use_llm = auto_settings(cfg)["use_llm"] if use_llm is None else use_llm
    articles = current_articles(ws)
    source = f"{articles['file']} (segmentador v{articles['segmenter_version']})"
    path = ws.work_dir / EXTRACTION_FILE
    saved = _load(path)
    reusable = (bool(saved) and saved["prompt"]["sha256"] == PROMPT.sha256
                and saved["source"] == source)          # same prompt and segmentation
    ws.mark_step("extract", RUNNING)
    try:
        client, ai = _llm_for(use_llm, llm)
        if client is None:
            summary = saved if reusable else None
        else:
            report = propose_profile_concepts(
                articles["articles"], llm=client, source=source,
                batch_size=settings.batch_size, max_tokens=settings.max_tokens, out_path=path,
                resume=reusable and saved.get("model") == getattr(client, "model", None),
                prompt=PROMPT, prompt_text=prompt_text(articles["language"]),
                pause_seconds=settings.pause_seconds,
                sleep_fn=sleep_fn or time.sleep, max_calls=settings.max_calls)
            summary = extraction_summary(report)
            ai.update(model=report.model, calls=report.calls,
                      quota_exhausted=report.quota_exhausted,
                      call_limit_reached=report.call_limit_reached,
                      errors=[str(e["error"])[:300] for e in report.errors][:3])
    except Exception as exc:
        ws.mark_step("extract", ERROR, str(exc)[:300])
        raise
    wanted = [a["number"] for a in articles["articles"]]
    done = set((summary or {}).get("processed_articles", []))
    pending = [n for n in wanted if n not in done]
    concepts = (summary or {}).get("concepts", [])
    extraction = {"articles_total": len(wanted), "articles_processed": len(wanted) - len(pending),
                  "pending_articles": pending, "complete": not pending,
                  "concepts": len(concepts), "model": (summary or {}).get("model"),
                  "batches_with_errors": len((summary or {}).get("errors", []))}
    ws.mark_step("extract", DONE, f"{len(concepts)} conceptos de "
                                  f"{extraction['articles_processed']} de {len(wanted)} artículos",
                 concepts=len(concepts), complete=extraction["complete"])
    return {"extraction": extraction, "ai": ai, "concepts": concepts}


# --------------------------------------------------------------------------- 2. accept

def accept_for_workspace(ws, extracted: dict) -> dict:
    """Apply the rule to the extracted concepts; save ``work/law-concepts.json``; mark "accept"."""
    from src.ingest.law_upload import current_articles
    from src.workspace.store import DONE, ERROR, RUNNING
    articles = current_articles(ws)
    ws.mark_step("accept", RUNNING)
    try:
        entries = apply_rule(extracted["concepts"], {a["number"] for a in articles["articles"]})
        result = {"method_version": METHOD_VERSION, "rule_version": RULE_VERSION,
                  "rule": RULE_TEXT, "prompt": PROMPT.manifest(),
                  "law": {"file": articles["file"], "language": articles["language"],
                          "articles": articles["n_articles"],
                          "segmenter_version": articles["segmenter_version"]},
                  "extraction": extracted["extraction"], "ai": extracted["ai"],
                  "summary": summarize(entries), "entries": entries}
        (ws.work_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    except Exception as exc:
        ws.mark_step("accept", ERROR, str(exc)[:300])
        raise
    s = result["summary"]
    ws.mark_step("accept", DONE, f"{s['accepted']} aceptados, {s['repeated']} repetidos y "
                                 f"{s['rejected']} rechazados por la regla",
                 accepted=s["accepted"], repeated=s["repeated"], rejected=s["rejected"])
    return result


def is_current(result: dict | None, cfg: dict | None = None,
               use_llm: bool | None = None) -> bool:
    """A saved result can be reused: same method, rule, prompt and segmenter and, when the AI
    is to be used, the whole law extracted."""
    from src.alignment.calibrate import auto_settings
    from src.ingest.legal_segmenter import SEGMENTER_VERSION
    if not result or result.get("method_version") != METHOD_VERSION:
        return False
    use_llm = auto_settings(cfg)["use_llm"] if use_llm is None else use_llm
    return (result["rule_version"] == RULE_VERSION
            and result["prompt"]["sha256"] == PROMPT.sha256
            and result["law"]["segmenter_version"] == SEGMENTER_VERSION
            and not (use_llm and not result["extraction"]["complete"]))


def prepare_law_concepts(ws, *, llm=None, use_llm: bool | None = None, cfg: dict | None = None,
                         sleep_fn=None, progress=None) -> dict:
    """Both steps, only when the saved result is not current; returns the result."""
    from src.workspace.store import STEP_LABELS
    result = load_result(ws)
    if is_current(result, cfg, use_llm):
        return result
    say = progress or (lambda label: None)
    say(STEP_LABELS["extract"])
    extracted = extract_for_workspace(ws, llm=llm, use_llm=use_llm, cfg=cfg, sleep_fn=sleep_fn)
    say(STEP_LABELS["accept"])
    return accept_for_workspace(ws, extracted)


# --------------------------------------------------------------------------- console / CLI

def _ai_text(result: dict) -> str:
    ai, x = result["ai"], result["extraction"]
    if not ai.get("available"):
        return f"no usada ({ai.get('error')})"
    text = (f"{ai.get('calls', 0)} llamadas en esta corrida, {x['batches_with_errors']} lotes "
            "con error")
    if ai.get("quota_exhausted"):
        text += " (cuota agotada: vuelve a correr mañana para seguir)"
    elif ai.get("call_limit_reached"):
        text += " (límite de llamadas: vuelve a correr para seguir)"
    if ai.get("errors"):
        text += f"; error: {ai['errors'][0]}"
    return text


def render_console(result: dict, examples: int = 8) -> str:
    law, x, s = result["law"], result["extraction"], result["summary"]
    lines = ["Conceptos de la ley (S7-T09)",
             f"  Ley: {law['file']} · {LANGUAGE_LABELS.get(law['language'], law['language'])} · "
             f"{law['articles']} artículos",
             f"  Extracción con la IA: {x['articles_processed']} de {x['articles_total']} "
             f"artículos · IA: {_ai_text(result)}"]
    if not x["complete"]:
        lines.append(f"    Faltan {len(x['pending_articles'])} artículos: vuelve a correr para "
                     "seguir (lo ya extraído no se repite).")
    lines += [f"  Conceptos extraídos: {s['concepts']}",
              f"  Regla: {result['rule']}",
              f"    aceptados: {s['accepted']} ({s['classes']} clases, {s['properties']} "
              f"propiedades) · repetidos: {s['repeated']} · rechazados: {s['rejected']}"]
    if s["reasons"]:
        lines.append("    rechazos: " + "; ".join(f"{REASONS[code]} ({n})"
                                                  for code, n in s["reasons"].items()))
    shown = [e for e in result["entries"] if e["status"] == ACCEPTED][:examples]
    if shown:
        lines.append("  Ejemplos de aceptados:")
    for e in shown:
        arts = ", ".join(map(str, e["articles"]))
        lines.append(f"    {e['name']} \"{e['label']}\" [{e['kind']}] <- art. {arts}")
    for e in [e for e in result["entries"] if e["status"] == REPEATED][:3]:
        lines.append(f"    repetido: {e['name']} se une a {e['repeated_of']} (\"{e['label']}\")")
    for e in [e for e in result["entries"] if e["status"] == REJECTED][:3]:
        lines.append(f"    rechazado: {e['name']} ({', '.join(REASONS[c] for c in e['reasons'])})")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.workspace.store import WorkspaceError, open_law
    parser = argparse.ArgumentParser(prog="py -m src.core.law_concepts",
                                     description="Extrae con la IA los conceptos de una ley "
                                                 "subida y los acepta por regla (S7-T09).")
    parser.add_argument("file", help="ley en .pdf o .txt")
    parser.add_argument("--sin-ia", action="store_true", help="no llamar al LLM")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: no existe el archivo {path}")
        return 1
    try:
        ws = open_law(path.name, path.read_bytes())
        result = prepare_law_concepts(ws, use_llm=False if args.sin_ia else None)
    except (WorkspaceError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(result))
    print(f"  Guardado en: {ws.work_dir / RESULT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())