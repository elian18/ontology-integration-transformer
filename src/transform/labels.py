"""Labels in Spanish and English for the classes and properties of an uploaded ontology (S7-T08).

The ontology a person downloads must carry labels in both languages (Sprint 7 deliverable).
Most uploads do not: GDPRtEXT has English labels without a language tag, OntoPriv has no label
at all. Following decision D2 (option B), the LLM writes the label that is MISSING, in a module
apart (``<name>-labels``) that imports the user's ontology and never changes an original label:

1. Inventory: the ``rdfs:label`` and ``skos:prefLabel`` of each class and property, by language
   (``en-GB`` counts as ``en``; a label without a language tag counts as neither). A concept
   that already has a label in ``es`` and one in ``en`` is not sent. Named individuals are left
   out, as in the alignment: they are example instances (S4-T02, ``src.alignment.sources``).
2. Glossary of the base: the concepts the AI found in the LOPDP (Sprint 3) that a person kept
   in Sprint 5 (approved, or confirmed as the same as an OntoPriv entity) give their Spanish
   label (the words of the law, such as "titular" or "responsable del tratamiento") next to
   their English name and, for the duplicates, the name of the OntoPriv entity. The whole
   glossary goes with every call, so a term is translated the same way everywhere.
3. LLM (``src/ai/prompts/labels.v1.txt``, temperature 0, JSON mode), ``labels.batch_size``
   concepts per call; for each concept it returns one label per language and only the missing
   ones are written. When the concept had a label without a tag and the LLM returns that same
   text for a language, the original text is kept letter by letter (only the tag is added).
   Answers are cached by content in the workspace (prompt, model, glossary and what the LLM saw),
   so uploading the file again makes no call; the free tier is handled as in the decider (pause
   between calls, a limit of calls, one retry after a 429 or 503, stop on an exhausted quota).
   Without AI nothing is invented: those concepts stay "pendientes" and the next upload with the
   AI available completes them.

CLI: ``py -m src.transform.labels ARCHIVO [--sin-ia]`` (only this step, on the workspace of the
file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from src.ai.prompts import load_prompt
from src.core.dpv_proximity import _humanize as humanize

PROMPT_VERSION = 1
PROMPT = load_prompt("labels", PROMPT_VERSION)
METHOD_VERSION = 1
LANGUAGES = ("es", "en")
RESULT_FILE = "labels.json"
CACHE_FILE = "labels-cache.jsonl"
MODULE_SUFFIX = "-labels"
MODULE_VERSION = 1
MAX_LABEL_CHARS = 150
COMPLETE, TRANSLATED, PENDING = "complete", "translated", "pending"


@dataclass(frozen=True)
class LabelSettings:
    batch_size: int = 40
    max_calls: int = 20
    pause_seconds: float = 5.0
    max_tokens: int = 4096
    definition_chars: int = 200


def labels_settings(cfg: dict | None = None) -> LabelSettings:
    """The ``labels`` block of config.yaml (defaults for the keys that are missing)."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("labels", {}) or {}
    d = LabelSettings()
    s = LabelSettings(batch_size=int(block.get("batch_size", d.batch_size)),
                      max_calls=int(block.get("max_calls", d.max_calls)),
                      pause_seconds=float(block.get("pause_seconds", d.pause_seconds)),
                      max_tokens=int(block.get("max_tokens", d.max_tokens)),
                      definition_chars=int(block.get("definition_chars", d.definition_chars)))
    if s.batch_size < 1 or s.max_calls < 0 or s.pause_seconds < 0 or s.max_tokens < 256:
        raise ValueError("Valores inválidos en el bloque labels de config.yaml (batch_size >= 1, "
                         "max_calls >= 0, pause_seconds >= 0, max_tokens >= 256).")
    return s


# --------------------------------------------------------------------------- inventory

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


@dataclass
class LabelItem:
    """One class or property and the labels it already has."""
    key: str
    name: str
    kinds: tuple[str, ...]
    tagged: dict[str, list[str]] = field(default_factory=dict)    # language -> texts
    untagged: list[str] = field(default_factory=list)
    definition: str | None = None

    @property
    def missing(self) -> list[str]:
        return [lang for lang in LANGUAGES if lang not in self.tagged]


def label_items(graph: Graph, concepts) -> list[LabelItem]:
    """The labels (``rdfs:label``, ``skos:prefLabel``) of each concept, by language."""
    out = []
    for c in concepts:
        item = LabelItem(key=c.key, name=c.name, kinds=tuple(c.kinds), definition=c.definition)
        for prop in (RDFS.label, SKOS.prefLabel):
            for o in graph.objects(URIRef(c.key), prop):
                if not isinstance(o, Literal) or not str(o).strip():
                    continue
                lang = (o.language or "").split("-")[0].lower()
                text = str(o).strip()
                bucket = item.tagged.setdefault(lang, []) if lang else item.untagged
                if text not in bucket:
                    bucket.append(text)
        out.append(item)
    return out


# --------------------------------------------------------------------------- glossary

def base_glossary(cfg: dict | None = None) -> list[dict]:
    """Spanish terms of the LOPDP kept by a person, with their English names (module doc).

    Empty when the base is not built yet (the labels are then translated without it)."""
    from src.alignment.ai_concepts import ai_concepts, concept_decisions
    from src.alignment.decisions import (DECISION_APPROVED, DECISION_DUPLICATE,
                                         DEFAULT_SOURCE_ID, read_log)
    from src.alignment.export import load_candidate_file
    from src.alignment.inherit import base_paths
    paths = base_paths(cfg)
    if not paths["candidates"].exists():
        return []
    current = concept_decisions(read_log(paths["log"]), DEFAULT_SOURCE_ID)
    entries: dict[str, dict] = {}
    for c in ai_concepts(load_candidate_file(paths["candidates"])):
        d = current.get(c.key)
        if not c.label or d is None or d.decision not in (DECISION_APPROVED, DECISION_DUPLICATE):
            continue
        english = [humanize(c.name)]
        if d.same_as:                                  # a duplicate: the OntoPriv entity too
            english.append(humanize(d.same_as.rsplit("#", 1)[-1]))
        es = c.label.strip()
        entry = entries.setdefault(_norm(es), {"es": es, "en": []})
        for name in english:
            if _norm(name) not in {_norm(x) for x in entry["en"]}:
                entry["en"].append(name)
    return sorted(entries.values(), key=lambda e: _norm(e["es"]))


def glossary_text(glossary: list[dict]) -> str:
    if not glossary:
        return "Glosario: (vacio)"
    return "Glosario (espanol = ingles):\n" + "\n".join(
        f"- {e['es']} = {' / '.join(e['en'])}" for e in glossary)


def glossary_sha(glossary: list[dict]) -> str:
    return hashlib.sha256(glossary_text(glossary).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- the LLM

def label_body(item: LabelItem, settings: LabelSettings) -> str:
    """What the LLM sees for one concept, without its id in the batch."""
    current = [f'"{t}" ({lang})' for lang in sorted(item.tagged) for t in item.tagged[lang]]
    current += [f'"{t}" (sin idioma)' for t in item.untagged]
    lines = [f"Concepto: {item.name} (tipo: {', '.join(item.kinds)})",
             f"Etiquetas actuales: {'; '.join(current) if current else 'ninguna'}"]
    if item.definition:
        definition = re.sub(r"\s+", " ", item.definition).strip()[:settings.definition_chars]
        lines.append(f"Definicion: {definition}")
    return "\n".join(lines)


def cache_key(body: str, model: str | None, glossary: str, generation: dict | None = None) -> str:
    content = {"prompt": PROMPT.sha256, "model": model, "glossary": glossary, "body": body}
    if generation:
        content["generation"] = generation
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_labels(raw: str) -> list[dict]:
    """The answers in the LLM output: ``{"labels": [...]}``, a bare list, or JSON in fences."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    data = None
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if isinstance(data, dict):
        data = data.get("labels")
    if not isinstance(data, list):
        raise ValueError("La respuesta del LLM no trae la lista de etiquetas en JSON.")
    return [d for d in data if isinstance(d, dict)]


def clean_label(value) -> str | None:
    """One line, without quotes or a final period; None when empty or too long."""
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip("\"'“”«»").strip()
    text = text.rstrip(".").rstrip()
    if not text or len(text) > MAX_LABEL_CHARS:
        return None
    return text


def validate_answer(answer: dict) -> dict | None:
    labels = {lang: clean_label(answer.get(lang)) for lang in LANGUAGES}
    return labels if all(labels.values()) else None


@dataclass
class TranslateReport:
    answers: dict = field(default_factory=dict)       # key -> {"es", "en"}
    calls: int = 0
    cached: int = 0
    invalid: list[str] = field(default_factory=list)
    quota_exhausted: bool = False
    call_limit_reached: bool = False
    errors: list[dict] = field(default_factory=list)  # {"concepts": [names], "error": text}


def _load_cache(path) -> dict:
    if path is None or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            out[entry["cache_key"]] = entry
        except (ValueError, KeyError, TypeError):
            continue
    return out


def _append_cache(path, entries: list[dict]) -> None:
    if path is None or not entries:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8", newline="\n") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def translate(items: list[LabelItem], glossary: list[dict], *, llm=None,
              settings: LabelSettings | None = None, cache_path=None,
              model_name: str | None = None, sleep_fn=time.sleep) -> TranslateReport:
    """Ask the LLM for the labels of the items (cache first, in batches, within the limit)."""
    from src.alignment.decider import generation_settings
    from src.alignment.justify import (QUOTA_WAIT_SECONDS, TRANSIENT_WAIT_SECONDS,
                                       _is_transient_error)
    from src.core.extract import _is_quota_error
    settings = settings or LabelSettings()
    report = TranslateReport()
    model_name = model_name or getattr(llm, "model", None) or os.environ.get("LLM_MODEL")
    gloss = glossary_text(glossary)
    gen = generation_settings(llm)
    keys = {i.key: cache_key(label_body(i, settings), model_name, gloss, gen) for i in items}
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
            sleep_fn(settings.pause_seconds)           # stay under the per-minute limits
        batch = pending[start:start + settings.batch_size]
        ids = {f"C{n}": it for n, it in enumerate(batch, start=1)}
        blocks = [f"=== {cid} ===\n{label_body(it, settings)}" for cid, it in ids.items()]
        context = gloss + "\n\n" + "\n\n".join(blocks)
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
                parsed = parse_labels(raw)
            except ValueError as exc:
                err = str(exc)
        if err is not None:
            report.errors.append({"concepts": [it.name for it in batch], "error": err[:300]})
            if _is_quota_error(err):
                report.quota_exhausted = True
                break
            continue
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        new = []
        for ans in parsed:
            item = ids.get(str(ans.get("concept") or "").strip().upper())
            if item is None or item.key in report.answers:
                continue
            labels = validate_answer(ans)
            if labels is None:
                report.invalid.append(item.key)
                continue
            report.answers[item.key] = labels
            new.append({"cache_key": keys[item.key], "concept_key": item.key, "answer": labels,
                        "prompt": PROMPT.manifest(), "model": model_name, "created": now})
        _append_cache(cache_path, new)
    return report


# --------------------------------------------------------------------------- the result

def _keep_original(item: LabelItem, text: str) -> str:
    """The untagged original when the LLM returned the same text (only the tag is added)."""
    return next((u for u in item.untagged if _norm(u) == _norm(text)), text)


def entries_from(items: list[LabelItem], report: TranslateReport) -> list[dict]:
    out = []
    for item in items:
        entry = {"key": item.key, "name": item.name, "kinds": list(item.kinds),
                 "existing": {**{lang: item.tagged.get(lang, []) for lang in LANGUAGES},
                              "none": item.untagged},
                 "added": {}, "status": COMPLETE}
        if item.missing:
            answer = report.answers.get(item.key)
            if answer is None:
                entry["status"] = PENDING
            else:
                entry["status"] = TRANSLATED
                entry["added"] = {lang: _keep_original(item, answer[lang])
                                  for lang in item.missing}
        out.append(entry)
    return out


def final_labels(entry: dict) -> dict[str, str | None]:
    """The label of each language after this step: the first one it had, or the added one."""
    return {lang: (entry["existing"][lang] or [None])[0] or entry["added"].get(lang)
            for lang in LANGUAGES}


def added_text(entry: dict) -> str:
    """What the AI added, for the CSV: "es", "en", "es, en", "pendiente" or ""."""
    if entry["status"] == PENDING:
        return "pendiente"
    return ", ".join(lang for lang in LANGUAGES if lang in entry["added"])


def by_key(result: dict | None) -> dict[str, dict]:
    return {e["key"]: e for e in (result or {}).get("entries", [])}


def summarize(entries: list[dict]) -> dict:
    return {"concepts": len(entries),
            "complete": sum(e["status"] == COMPLETE for e in entries),
            "translated": sum(e["status"] == TRANSLATED for e in entries),
            "pending": sum(e["status"] == PENDING for e in entries),
            "added": {lang: sum(lang in e["added"] for e in entries) for lang in LANGUAGES}}


def label_ontology(graph: Graph, concepts, *, llm=None, use_llm: bool = True,
                   cfg: dict | None = None, cache_path=None, sleep_fn=time.sleep,
                   glossary: list[dict] | None = None) -> dict:
    """The labels of one ontology: an upload (this task) or, in S7-T10, a new profile."""
    from src.alignment.propose import _llm_for
    settings = labels_settings(cfg)
    glossary = base_glossary(cfg) if glossary is None else glossary
    items = label_items(graph, concepts)
    todo = [i for i in items if i.missing]
    client, ai = _llm_for(use_llm, llm) if todo else (None, {"available": use_llm,
                                                             "needed": False})
    report = translate(todo, glossary, llm=client, settings=settings, cache_path=cache_path,
                       sleep_fn=sleep_fn)
    ai.update(model=getattr(client, "model", None), calls=report.calls, cached=report.cached,
              invalid=len(report.invalid), quota_exhausted=report.quota_exhausted,
              call_limit_reached=report.call_limit_reached,
              errors=[e["error"] for e in report.errors][:3])
    entries = entries_from(items, report)
    return {"method_version": METHOD_VERSION, "prompt": PROMPT.manifest(),
            "glossary": {"entries": len(glossary), "sha256": glossary_sha(glossary)},
            "summary": summarize(entries), "ai": ai, "entries": entries}


def labels_for_workspace(ws, *, llm=None, use_llm: bool | None = None, cfg: dict | None = None,
                         sleep_fn=None, glossary: list[dict] | None = None) -> dict:
    """Run S7-T08 on the ontology of a workspace; save ``work/labels.json``; mark the step."""
    from src.alignment.calibrate import auto_settings
    from src.alignment.sources import build_alignment_sources
    from src.ingest.ontology_loader import load_ontology
    from src.transform.convert import load_graph
    from src.workspace.store import DONE, ERROR, RUNNING
    use_llm = auto_settings(cfg)["use_llm"] if use_llm is None else use_llm
    ws.mark_step("labels", RUNNING)
    try:
        path = ws.input_path()
        concepts = build_alignment_sources(load_ontology(path), proposals={}).concepts
        result = label_ontology(load_graph(path), concepts, llm=llm, use_llm=use_llm, cfg=cfg,
                                cache_path=ws.work_dir / CACHE_FILE,
                                sleep_fn=sleep_fn or time.sleep, glossary=glossary)
        (ws.work_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    except Exception as exc:
        ws.mark_step("labels", ERROR, str(exc)[:300])
        raise
    s = result["summary"]
    ws.mark_step("labels", DONE, f"{s['added']['es']} en español y {s['added']['en']} en "
                                 f"inglés agregadas", translated=s["translated"],
                 pending=s["pending"])
    return result


def load_result(ws) -> dict | None:
    path = ws.work_dir / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def is_current(result: dict | None, cfg: dict | None = None,
               use_llm: bool | None = None) -> bool:
    """A saved result can be reused: same method, prompt and glossary and, when the AI is to be
    used, no concept left pending."""
    from src.alignment.calibrate import auto_settings
    if not result or result.get("method_version") != METHOD_VERSION:
        return False
    use_llm = auto_settings(cfg)["use_llm"] if use_llm is None else use_llm
    return (result["prompt"]["sha256"] == PROMPT.sha256
            and result["glossary"]["sha256"] == glossary_sha(base_glossary(cfg))
            and not (use_llm and result["summary"]["pending"]))


# --------------------------------------------------------------------------- the module

def labels_graph(result: dict, *, iri: str, onto_iri: str, name: str, created: str) -> Graph | None:
    """The module ``<name>-labels`` with the added labels (None when nothing was added)."""
    added = [(e["key"], lang, text) for e in result.get("entries", [])
             for lang, text in e["added"].items()]
    if not added:
        return None
    g = Graph()
    g.bind("dct", DCTERMS)
    node = URIRef(iri)
    g.add((node, RDF.type, OWL.Ontology))
    g.add((node, OWL.imports, URIRef(onto_iri)))
    g.add((node, RDFS.label, Literal(f"Etiquetas en español e inglés de {name}", lang="es")))
    g.add((node, RDFS.comment, Literal(
        "Etiquetas que le faltaban a la ontología, escritas por la IA con el glosario de la "
        "LOPDP de la base del sistema (S7-T08). Ninguna etiqueta original se cambia; cada "
        "etiqueta lleva su idioma.", lang="es")))
    g.add((node, DCTERMS.created, Literal(created, datatype=XSD.dateTime)))
    g.add((node, OWL.versionInfo, Literal(str(MODULE_VERSION))))
    for key, lang, text in added:
        g.add((URIRef(key), RDFS.label, Literal(text, lang=lang)))
    return g


# --------------------------------------------------------------------------- console / CLI

def ai_line(result: dict) -> str:
    """One line about the AI of this step (console of this module and of the alignment)."""
    ai = result.get("ai") or {}
    if ai.get("needed") is False:
        return "no hizo falta (cada concepto ya tenía es y en)"
    if not ai.get("available"):
        return f"no usada ({ai.get('error')})"
    text = (f"{ai.get('calls', 0)} llamadas, {ai.get('cached', 0)} desde la caché, "
            f"{ai.get('invalid', 0)} respuestas inválidas")
    if ai.get("quota_exhausted"):
        text += " (cuota agotada: vuelve a correr mañana)"
    elif ai.get("call_limit_reached"):
        text += " (límite de llamadas: vuelve a correr para seguir)"
    if ai.get("errors"):
        text += f"; error: {ai['errors'][0]}"
    return text


def render_console(result: dict, examples: int = 6) -> str:
    s = result["summary"]
    lines = ["Etiquetas es/en (S7-T08)",
             f"  Conceptos: {s['concepts']} · ya tenían es y en: {s['complete']}",
             f"  Completadas por la IA: {s['translated']} (agregadas: {s['added']['es']} en "
             f"español, {s['added']['en']} en inglés) · pendientes: {s['pending']}",
             f"  Glosario de la base: {result['glossary']['entries']} términos de la LOPDP",
             f"  IA: {ai_line(result)}"]
    shown = [e for e in result["entries"] if e["status"] == TRANSLATED][:examples]
    if shown:
        lines.append("  Ejemplos:")
    for e in shown:
        final = final_labels(e)
        lines.append(f"    {e['name']}: {final['es']} (es) | {final['en']} (en)")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.workspace.store import WorkspaceError, open_ontology
    parser = argparse.ArgumentParser(prog="py -m src.transform.labels",
                                     description="Etiquetas en español e inglés para una "
                                                 "ontología subida (S7-T08).")
    parser.add_argument("file", help="ontología .rdf, .owl, .ttl o .jsonld")
    parser.add_argument("--sin-ia", action="store_true", help="no llamar al LLM")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: no existe el archivo {path}")
        return 1
    try:
        ws = open_ontology(path.name, path.read_bytes())
        result = labels_for_workspace(ws, use_llm=False if args.sin_ia else None)
    except (WorkspaceError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(result))
    print(f"  Guardado en: {ws.work_dir / RESULT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())