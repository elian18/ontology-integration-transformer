from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from src.ai.prompts import load_prompt

PROMPT_VERSION = 1
PROMPT = load_prompt("define", PROMPT_VERSION)
METHOD_VERSION = 1
RESULT_FILE = "law-definitions.json"
CACHE_FILE = "definitions-cache.jsonl"
MODULE_SUFFIX = "-definitions"
MODULE_VERSION = 1
MAX_DEFINITION_CHARS = 600

LAW_CONCEPT, ARTICLES = "law_concept", "articles"
NOT_APPLICABLE, INVALID, PENDING, NO_SOURCE = "not_applicable", "invalid", "pending", "no_source"
ADDED = (LAW_CONCEPT, ARTICLES)
STATUS_LABELS = {LAW_CONCEPT: "de un concepto de la ley",
                 ARTICLES: "escrita por la IA con los artículos",
                 NOT_APPLICABLE: "no aplica: la ley no lo define",
                 INVALID: "respuesta inválida de la IA",
                 PENDING: "pendiente",
                 NO_SOURCE: "sin fuente"}
SKIP_DISABLED, SKIP_FEW = "disabled", "few_missing"
_ALIGNABLE = {"class", "property"}
_EMPTY = {"", "no aplica", "n/a", "ninguna", "not applicable", "none"}


@dataclass(frozen=True)
class DefinitionSettings:
    enabled: bool = False
    min_missing: float = 0.10
    from_law_concepts: bool = True
    from_articles: bool = True
    articles_per_concept: int = 3
    article_chars: int = 1200
    batch_size: int = 10
    max_calls: int = 30
    pause_seconds: float = 5.0
    max_tokens: int = 8192


def definition_settings(cfg: dict | None = None) -> DefinitionSettings:
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("law_definitions", {}) or {}
    d = DefinitionSettings()
    s = DefinitionSettings(
        enabled=bool(block.get("enabled", d.enabled)),
        min_missing=float(block.get("min_missing", d.min_missing)),
        from_law_concepts=bool(block.get("from_law_concepts", d.from_law_concepts)),
        from_articles=bool(block.get("from_articles", d.from_articles)),
        articles_per_concept=int(block.get("articles_per_concept", d.articles_per_concept)),
        article_chars=int(block.get("article_chars", d.article_chars)),
        batch_size=int(block.get("batch_size", d.batch_size)),
        max_calls=int(block.get("max_calls", d.max_calls)),
        pause_seconds=float(block.get("pause_seconds", d.pause_seconds)),
        max_tokens=int(block.get("max_tokens", d.max_tokens)))
    if (not 0 <= s.min_missing <= 1 or s.articles_per_concept < 1 or s.article_chars < 100
            or s.batch_size < 1 or s.max_calls < 0 or s.pause_seconds < 0 or s.max_tokens < 256):
        raise ValueError("Valores inválidos en el bloque law_definitions de config.yaml "
                         "(min_missing entre 0 y 1, articles_per_concept >= 1, article_chars >= "
                         "100, batch_size >= 1, max_calls >= 0, pause_seconds >= 0, max_tokens "
                         ">= 256).")
    return s


def counted(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# --------------------------------------------------------------------------- 1. law concepts

def alignable(concepts) -> list:
    return [c for c in concepts if set(c.kinds) & _ALIGNABLE]


def skip_reason(settings: DefinitionSettings, total: int, missing: int) -> str | None:
    """Why the law is not asked: turned off, or the ontology lacks less than ``min_missing``."""
    if not settings.enabled:
        return SKIP_DISABLED
    if not missing or round(missing / total, 4) < settings.min_missing:
        return SKIP_FEW
    return None


def law_index(entries: list[dict]):
    from src.alignment.inherit import BaseConcept, BaseIndex
    concepts = [BaseConcept(key=f"law:{e['name']}", iri=f"law:{e['name']}", origin="law",
                            name=e["name"], label=e.get("label") or None,
                            definition=e["definition"], kinds=(e["kind"],),
                            status=e["definition"])    # tied law concepts that disagree: no match
                for e in entries
                if e["status"] == "accepted" and (e.get("definition") or "").strip()]
    return BaseIndex(concepts=concepts, source_id="law")


def match_law_concepts(concepts, entries: list[dict], *, embed_fn=None,
                       cfg: dict | None = None) -> dict[str, dict]:
    """Concepts that are a concept of the law: same name or label, or very similar."""
    from src.alignment.inherit import inherit
    index = law_index(entries)
    if not concepts or not index.concepts:
        return {}
    by_key = {f"law:{e['name']}": e for e in entries}
    report = inherit(list(concepts), index, embed_fn=embed_fn, cfg=cfg)
    return {i.key: {"entry": by_key[i.base_key], "method": i.method, "score": i.score}
            for i in report.inherited}


def law_digest(law: dict | None) -> str | None:
    if not law:
        return None
    accepted = sorted((e["name"], e.get("definition") or "", tuple(e["articles"]))
                      for e in law["entries"] if e["status"] == "accepted")
    return hashlib.sha256(json.dumps(accepted, ensure_ascii=False).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- 2. the articles

def nearest_articles(ws, concepts, *, embed_fn=None, n: int = 3) -> dict[str, list[int]]:
    from src.alignment.law_evidence import search
    from src.ingest.law_upload import chroma_path
    hits = search(list(concepts), path=chroma_path(ws), embed_fn=embed_fn, n=n)
    return {key: [h["number"] for h in found] for key, found in hits.items()}


def _one_line(text, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit] if limit else text


def prompt_text(language: str) -> str:
    from src.core.law_concepts import PROMPT_LANGUAGES
    return PROMPT.text.replace("{idioma}", PROMPT_LANGUAGES.get(language, language))


def article_block(article: dict, chars: int) -> str:
    title = _one_line(article.get("title"))
    head = f"=== ARTICULO {article['number']}" + (f": {title}" if title else "") + " ==="
    return f"{head}\n{_one_line(article['text'], chars)}"


def concept_body(concept, numbers: list[int]) -> str:
    lines = [f"Concepto: {concept.name} (tipo: {', '.join(concept.kinds)})"]
    if concept.label:
        lines.append(f"Etiqueta: {concept.label}")
    lines.append("Articulos: " + ", ".join(map(str, numbers)))
    return "\n".join(lines)


def cache_key(body: str, articles: str, language: str, model: str | None,
              generation: dict | None = None) -> str:
    content = {"prompt": PROMPT.sha256, "model": model, "language": language, "body": body,
               "articles": articles}
    if generation:
        content["generation"] = generation
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_definitions(raw: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    data = None
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if isinstance(data, dict):
        data = data.get("definitions")
    if not isinstance(data, list):
        raise ValueError("La respuesta del LLM no trae la lista de definiciones en JSON.")
    return [d for d in data if isinstance(d, dict)]


def clean_definition(value) -> str | None:
    text = _one_line(value).strip("\"'“”«»").strip()
    return None if text.casefold().rstrip(".") in _EMPTY else text


def _number(value) -> int | None:
    found = re.search(r"\d+", str(value if value is not None else ""))
    return int(found.group()) if found else None


def validate_answer(answer: dict, numbers: list[int]) -> dict:
    text = clean_definition(answer.get("definition"))
    if text is None:
        return {"status": NOT_APPLICABLE, "definition": None, "article": None}
    article = _number(answer.get("article"))
    if article not in numbers or len(text) > MAX_DEFINITION_CHARS:
        return {"status": INVALID, "definition": None, "article": article}
    return {"status": ARTICLES, "definition": text, "article": article}


@dataclass
class DefineReport:
    answers: dict = field(default_factory=dict)
    calls: int = 0
    cached: int = 0
    quota_exhausted: bool = False
    call_limit_reached: bool = False
    errors: list[dict] = field(default_factory=list)


def ask_definitions(items, articles: dict[int, dict], language: str, *, llm=None,
                    settings: DefinitionSettings | None = None, cache_path=None,
                    model_name: str | None = None, sleep_fn=time.sleep) -> DefineReport:
    """``items``: (concept, numbers of its nearest articles). Cache first, then in batches."""
    from src.alignment.decider import generation_settings
    from src.alignment.justify import (QUOTA_WAIT_SECONDS, TRANSIENT_WAIT_SECONDS,
                                       _is_transient_error)
    from src.core.extract import _is_quota_error
    from src.transform.labels import _append_cache, _load_cache
    settings = settings or DefinitionSettings()
    report = DefineReport()
    model_name = model_name or getattr(llm, "model", None) or os.environ.get("LLM_MODEL")
    generation = generation_settings(llm)
    texts = {n: article_block(a, settings.article_chars) for n, a in articles.items()}
    keys = {c.key: cache_key(concept_body(c, nums), "\n\n".join(texts[n] for n in nums),
                             language, model_name, generation) for c, nums in items}
    cache = _load_cache(cache_path)
    pending = []
    for c, nums in items:
        hit = cache.get(keys[c.key])
        if hit is not None:
            report.answers[c.key] = hit["answer"]
            report.cached += 1
        else:
            pending.append((c, nums))
    prompt = prompt_text(language)
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
        used = sorted({n for _, nums in batch for n in nums})
        context = ("\n\n".join(texts[n] for n in used) + "\n\n"
                   + "\n\n".join(f"=== {cid} ===\n{concept_body(c, nums)}"
                                 for cid, (c, nums) in ids.items()))
        raw, err = None, None
        for attempt in range(2):                       # one retry after a 429 or a 503
            try:
                raw = llm.ask(prompt=prompt, context=context, max_tokens=settings.max_tokens,
                              json_mode=True) or ""
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
                parsed = parse_definitions(raw)
            except ValueError as exc:
                err = str(exc)
        if err is not None:
            report.errors.append({"concepts": [c.name for c, _ in batch], "error": err[:300]})
            if _is_quota_error(err):
                report.quota_exhausted = True
                break
            continue
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        new = []
        for ans in parsed:
            item = ids.get(str(ans.get("concept") or "").strip().upper())
            if item is None or item[0].key in report.answers:
                continue
            c, nums = item
            answer = validate_answer(ans, nums)
            report.answers[c.key] = answer
            new.append({"cache_key": keys[c.key], "concept_key": c.key, "answer": answer,
                        "prompt": PROMPT.manifest(), "model": model_name, "created": now})
        _append_cache(cache_path, new)
    return report


# --------------------------------------------------------------------------- 3. the result

def _entry(concept, for_decider: set[str]) -> dict:
    return {"key": concept.key, "name": concept.name, "kinds": list(concept.kinds),
            "for_decider": concept.key in for_decider, "status": NO_SOURCE, "definition": None,
            "article": None, "articles": [], "law_concept": None}


def entries_from(missing, matched: dict, report: DefineReport, nearest: dict,
                 for_decider: set[str], asked: bool) -> list[dict]:
    out = []
    for c in missing:
        e = _entry(c, for_decider)
        if c.key in matched:
            m = matched[c.key]
            law = m["entry"]
            e.update(status=LAW_CONCEPT, definition=_one_line(law["definition"]),
                     article=law["articles"][0], articles=list(law["articles"][:3]),
                     law_concept={"name": law["name"], "label": law.get("label"),
                                  "method": m["method"], "score": m["score"]})
        elif asked and c.key in nearest:
            answer = report.answers.get(c.key)
            e["articles"] = nearest[c.key]
            if answer is None:
                e["status"] = PENDING
            else:
                e.update(status=answer["status"], definition=answer["definition"],
                         article=answer["article"])
        out.append(e)
    return sorted(out, key=lambda e: (list(STATUS_LABELS).index(e["status"]), e["name"].lower()))


def summarize(concepts: int, missing: int, entries: list[dict]) -> dict:
    by = Counter(e["status"] for e in entries)
    return {"concepts": concepts, "with_own": concepts - missing, "missing": missing,
            "missing_share": round(missing / concepts, 4) if concepts else 0.0,
            "added": by.get(LAW_CONCEPT, 0) + by.get(ARTICLES, 0),
            "to_decider": sum(e["for_decider"] and e["status"] in ADDED for e in entries),
            **{status: by.get(status, 0) for status in STATUS_LABELS}}


def _save(ws, result: dict) -> None:
    (ws.work_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                           encoding="utf-8")


def _pct(share: float) -> str:
    return f"{100 * share:.0f} %"


def step_detail(result: dict) -> str:
    s = result["summary"]
    if result["skipped"] == SKIP_DISABLED:
        return "Apagado en config.yaml (law_definitions.enabled)"
    if result["skipped"] == SKIP_FEW:
        return (f"No hizo falta: a {s['missing']} de {s['concepts']} conceptos les falta "
                f"definición (menos del {_pct(result['settings']['min_missing'])})")
    return (f"{counted(s['added'], 'definición', 'definiciones')} de la ley ({s[LAW_CONCEPT]} de "
            f"conceptos de la ley y {counted(s[ARTICLES], 'escrita', 'escritas')} por la IA con "
            f"los artículos) para {counted(s['missing'], 'concepto', 'conceptos')} sin "
            "definición")


def definitions_for_workspace(ws, *, embed_fn=None, llm=None, use_llm: bool | None = None,
                              cfg: dict | None = None, sleep_fn=None, progress=None) -> dict:
    """Mode "ontología + ley": the law gives the definitions the ontology lacks."""
    from src.alignment.calibrate import auto_settings
    from src.alignment.inherit import load_result as inherited_result, user_concepts
    from src.alignment.propose import _llm_for
    from src.ingest.law_upload import current_articles
    from src.workspace.store import DONE, ERROR, RUNNING, STEP_LABELS
    settings = definition_settings(cfg)
    use_llm = auto_settings(cfg)["use_llm"] if use_llm is None else use_llm
    articles = current_articles(ws)
    concepts = alignable(user_concepts(ws.input_path()))
    missing = [c for c in concepts if not c.definition]
    skipped = skip_reason(settings, len(concepts), len(missing))
    result = {"method_version": METHOD_VERSION, "prompt": PROMPT.manifest(),
              "settings": asdict(settings), "skipped": skipped,
              "law": {"file": articles["file"], "language": articles["language"],
                      "segmenter_version": articles["segmenter_version"]},
              "law_concepts": None, "law_concepts_digest": None,
              "ai": {"needed": False}, "summary": summarize(len(concepts), len(missing), []),
              "entries": []}
    if skipped:
        _save(ws, result)
        ws.mark_step("definitions", DONE, step_detail(result), skipped=skipped)
        return result

    law_entries = []
    if settings.from_law_concepts:
        from src.core.law_concepts import prepare_law_concepts
        law = prepare_law_concepts(ws, llm=llm, use_llm=use_llm, cfg=cfg, sleep_fn=sleep_fn,
                                   progress=progress)
        law_entries = law["entries"]
        result.update(law_concepts={"summary": law["summary"], "extraction": law["extraction"],
                                    "ai": law["ai"], "prompt": law["prompt"]},
                      law_concepts_digest=law_digest(law))
    (progress or (lambda label: None))(STEP_LABELS["definitions"])
    ws.mark_step("definitions", RUNNING)
    try:
        matched = match_law_concepts(missing, law_entries, embed_fn=embed_fn, cfg=cfg)
        rest = [c for c in missing if c.key not in matched]
        for_decider = {n["key"] for n in (inherited_result(ws) or {}).get("not_inherited", [])}
        asked = bool(rest) and settings.from_articles
        report, nearest = DefineReport(), {}
        if asked:
            nearest = nearest_articles(ws, rest, embed_fn=embed_fn,
                                       n=settings.articles_per_concept)
            order = sorted((c for c in rest if nearest.get(c.key)),
                           key=lambda c: (c.key not in for_decider, nearest[c.key][0],
                                          c.name.lower()))
            client, ai = _llm_for(use_llm, llm)
            report = ask_definitions([(c, nearest[c.key]) for c in order],
                                     {a["number"]: a for a in articles["articles"]},
                                     articles["language"], llm=client, settings=settings,
                                     cache_path=ws.work_dir / CACHE_FILE,
                                     sleep_fn=sleep_fn or time.sleep)
            ai.update(needed=True, model=getattr(client, "model", None), calls=report.calls,
                      cached=report.cached, quota_exhausted=report.quota_exhausted,
                      call_limit_reached=report.call_limit_reached,
                      errors=[e["error"] for e in report.errors][:3])
            result["ai"] = ai
        entries = entries_from(missing, matched, report, nearest, for_decider, asked)
        result.update(summary=summarize(len(concepts), len(missing), entries), entries=entries)
        _save(ws, result)
    except Exception as exc:
        ws.mark_step("definitions", ERROR, str(exc)[:300])
        raise
    s = result["summary"]
    ws.mark_step("definitions", DONE, step_detail(result), added=s["added"],
                 law_concept=s[LAW_CONCEPT], articles=s[ARTICLES], pending=s[PENDING])
    return result


def load_result(ws) -> dict | None:
    path = ws.work_dir / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def is_current(result: dict | None, ws, cfg: dict | None = None,
               use_llm: bool | None = None) -> bool:
    from src.alignment.calibrate import auto_settings
    from src.core import law_concepts as lc
    from src.ingest.legal_segmenter import SEGMENTER_VERSION
    if not result or result.get("method_version") != METHOD_VERSION:
        return False
    settings = definition_settings(cfg)
    if (result["settings"] != asdict(settings) or result["prompt"]["sha256"] != PROMPT.sha256
            or result["law"]["segmenter_version"] != SEGMENTER_VERSION):
        return False
    if result["skipped"]:
        return True
    use_llm = auto_settings(cfg)["use_llm"] if use_llm is None else use_llm
    if settings.from_law_concepts:
        law = lc.load_result(ws)
        if not lc.is_current(law, cfg, use_llm) or result["law_concepts_digest"] != law_digest(law):
            return False
    return not (use_llm and result["summary"][PENDING])


def by_key(result: dict | None) -> dict[str, dict]:
    return {e["key"]: e for e in (result or {}).get("entries", [])}


def added(result: dict | None) -> list[dict]:
    return [e for e in (result or {}).get("entries", []) if e["status"] in ADDED]


def fingerprint(result: dict | None) -> str | None:
    """What the decider and the evidence depend on (None when the law added nothing)."""
    rows = sorted((e["key"], e["definition"], e["article"], tuple(e["articles"]), e["status"])
                  for e in added(result))
    if not rows:
        return None
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def with_definitions(concepts, result: dict | None) -> list:
    defined = {e["key"]: e["definition"] for e in added(result)}
    return [replace(c, definition=defined[c.key]) if not c.definition and c.key in defined
            else c for c in concepts]


def cited_articles(result: dict | None) -> dict[str, list[int]]:
    return {e["key"]: e["articles"] for e in added(result) if e["status"] == LAW_CONCEPT}


def source_text(entry: dict, law: str) -> str:
    """For the CSV: where the definition comes from."""
    from src.alignment.inherit import METHOD_LABELS
    from src.core.law_profile import article_source
    if entry["status"] == LAW_CONCEPT:
        lc = entry["law_concept"]
        return (f"{article_source(law, entry['article'])} · concepto de la ley {lc['name']} "
                f"({METHOD_LABELS[lc['method']]})")
    if entry["status"] == ARTICLES:
        return f"{article_source(law, entry['article'])} · escrita por la IA"
    return "" if entry["status"] == NO_SOURCE else STATUS_LABELS[entry["status"]]


# --------------------------------------------------------------------------- the module

def definitions_graph(result: dict, *, iri: str, onto_iri: str, name: str,
                      created: str) -> Graph | None:
    from src.alignment.inherit import METHOD_LABELS
    from src.core.law_profile import article_source
    entries = added(result)
    if not entries:
        return None
    law, language = result["law"]["file"], result["law"]["language"]
    g = Graph()
    g.bind("skos", SKOS)
    g.bind("dct", DCTERMS)
    node = URIRef(iri)
    g.add((node, RDF.type, OWL.Ontology))
    g.add((node, OWL.imports, URIRef(onto_iri)))
    g.add((node, RDFS.label, Literal(f"Definiciones de {name} tomadas de {law}", lang="es")))
    g.add((node, RDFS.comment, Literal(
        "Definiciones que le faltaban a la ontología, tomadas de la ley: las de los conceptos "
        "que la IA extrajo de ella o escritas por la IA a partir de sus artículos. Cada una "
        "lleva su artículo (dct:source) y su origen; ninguna definición original cambia.",
        lang="es")))
    g.add((node, DCTERMS.created, Literal(created, datatype=XSD.dateTime)))
    g.add((node, DCTERMS.source, Literal(law)))
    g.add((node, OWL.versionInfo, Literal(str(MODULE_VERSION))))
    for e in entries:
        subject, text = URIRef(e["key"]), Literal(e["definition"], lang=language)
        g.add((subject, SKOS.definition, text))
        if e["status"] == LAW_CONCEPT:
            lc = e["law_concept"]
            origin = (f"Definición del concepto {lc['name']} que la IA extrajo de {law} "
                      f"({METHOD_LABELS[lc['method']]}).")
        else:
            origin = (f"Escrita por la IA a partir del art. {e['article']} de {law} (prompt "
                      f"define.v{result['prompt']['version']}).")
        ax = BNode()
        g.add((ax, RDF.type, OWL.Axiom))
        g.add((ax, OWL.annotatedSource, subject))
        g.add((ax, OWL.annotatedProperty, SKOS.definition))
        g.add((ax, OWL.annotatedTarget, text))
        g.add((ax, DCTERMS.source, Literal(article_source(law, e["article"]))))
        g.add((ax, RDFS.comment, Literal(origin, lang="es")))
    return g


# --------------------------------------------------------------------------- console

def ai_line(result: dict) -> str:
    parts = []
    law = result.get("law_concepts")
    if law:
        ai = law.get("ai") or {}
        parts.append(f"extracción {ai.get('calls', 0)} llamadas" if ai.get("available")
                     else f"extracción no usada ({ai.get('error')})")
    ai = result.get("ai") or {}
    if ai.get("needed"):
        if ai.get("available"):
            text = (f"definiciones {ai.get('calls', 0)} llamadas, {ai.get('cached', 0)} desde "
                    "la caché")
            if ai.get("quota_exhausted"):
                text += " (cuota agotada: vuelve a correr mañana)"
            elif ai.get("call_limit_reached"):
                text += " (límite de llamadas: vuelve a correr para seguir)"
            if ai.get("errors"):
                text += f"; error: {ai['errors'][0]}"
        else:
            text = f"definiciones no usada ({ai.get('error')})"
        parts.append(text)
    return "; ".join(parts) or "no hizo falta"


def added_text(block: dict, whose: str = "la ontología") -> str:
    what = counted(block["added"], "definición que le faltaba", "definiciones que le faltaban")
    taken = "tomada" if block["added"] == 1 else "tomadas"
    return (f"{what} a {whose}, {taken} de {block['law']} ({block[LAW_CONCEPT]} de conceptos de "
            f"la ley y {counted(block[ARTICLES], 'escrita', 'escritas')} por la IA con sus "
            "artículos)")


def module_note(block: dict) -> str:
    text = (f"El módulo de definiciones agrega {added_text(block)}; cada una cita su artículo y "
            "ninguna definición original cambia.")
    used = block["to_decider"]
    if used:
        text += (f" El decisor usó {'la' if used == 1 else 'las'} de "
                 f"{counted(used, 'concepto que no heredó', 'conceptos que no heredaron')} su "
                 "alineación.")
    return text


def summary_line(block: dict) -> str:
    """One line for the console of the alignment (``block``: the manifest entry)."""
    law = block["law"]
    if block["skipped"] == SKIP_DISABLED:
        return f"Definiciones de {law}: apagadas en config.yaml"
    if block["skipped"] == SKIP_FEW:
        return (f"Definiciones de {law}: no hizo falta (a {block['missing']} de "
                f"{block['concepts']} conceptos les falta definición)")
    text = (f"Definiciones de {law}: a {block['missing']} de {block['concepts']} conceptos les "
            f"faltaba; {block[LAW_CONCEPT]} de conceptos de la ley, "
            f"{counted(block[ARTICLES], 'escrita', 'escritas')} por la IA con los artículos, "
            f"{counted(block[NOT_APPLICABLE], 'no aplica', 'no aplican')}")
    if block[INVALID]:
        text += f", {counted(block[INVALID], 'respuesta inválida', 'respuestas inválidas')}"
    if block[PENDING]:
        text += f" · pendientes: {block[PENDING]}"
    return text