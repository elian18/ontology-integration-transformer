# The AI drafts the SPARQL query of each competency question and every draft is measured on the
# base (S9-T08). The prompts only get the question in Spanish and a summary of the ontology made
# from its triples (v2 also says how the base is built): never the reference query nor the
# expected answer. Each draft runs with the reference, on the same engine and files, and the same
# rule judges both: as the AI wrote it ("a la primera") and with the standard prefixes it used
# without declaring them, added by the system. Nothing of this enters the ontology or the uploads.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.collection import Collection
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from src.ai.prompts import load_prompt
from src.validation import question_catalog as qc
from src.validation import question_explainer as qe
from src.validation import questions as qs

PROMPTS = {v: load_prompt("sparql", v) for v in (1, 2)}
CACHE_FILE = "sparql-ai-cache.jsonl"
DRAFTS_DIR = "sparql-drafts"
EVALUATION_FILE = "sparql-drafts-evaluation.json"
AS_WRITTEN, PREFIXED = "as_written", "prefixed"     # the two columns of the table
SAME, OTHER, BROKEN, NO_DRAFT = "misma respuesta", "otra respuesta", "no corrió", "sin borrador"
OUTCOMES = (SAME, OTHER, BROKEN, NO_DRAFT)
RULE = ("misma respuesta: la regla de la pregunta da el mismo color y lo mismo que lee la persona "
        "(los mismos elementos, valores o conceptos); otra respuesta: corre pero da otra cosa; "
        "no corrió: la consulta falla, tarda demasiado o no es un SELECT")
ADOPTION = ("v2 si da la misma respuesta en más preguntas que v1 y no pierde ninguna que v1 "
            "acertó, contando con los prefijos completados")
KNOWN = {"http://www.semanticweb.org/ley-organica-proteccion-datos-personales#": "onto",
         "http://www.semanticweb.org/franc/ontologies/2023/9/OntologiaLOPDP#": "op",
         "http://www.semanticweb.org/profiles/ecuador-lopdp#": "ec",
         "https://w3id.org/dpv#": "dpv", str(SKOS): "skos", str(RDFS): "rdfs", str(RDF): "rdf",
         str(OWL): "owl", "http://www.w3.org/2001/XMLSchema#": "xsd", str(DCTERMS): "dct"}
MAPPINGS = (SKOS.exactMatch, SKOS.closeMatch, SKOS.broadMatch, SKOS.narrowMatch,
            SKOS.relatedMatch)
_SERVICE = re.compile(r"\bSERVICE\s+(?:SILENT\s+)?[<?$]", re.IGNORECASE)
STANDARD = {prefix: ns for ns, prefix in KNOWN.items()}           # the prefixes the system adds


@dataclass(frozen=True)
class DraftSettings:
    prompt_version: int = 1
    batch_size: int = 8
    max_tokens: int = 16384
    query_timeout_seconds: float = 20.0


def draft_settings(cfg: dict | None = None) -> DraftSettings:
    from src.config import load_config
    block = ((cfg if cfg is not None else load_config()).get("questions") or {}).get("drafts") or {}
    s = DraftSettings(**{k: v for k, v in block.items() if k in DraftSettings.__dataclass_fields__})
    if (s.prompt_version not in PROMPTS or s.batch_size < 1 or s.max_tokens < 1
            or s.query_timeout_seconds <= 0):
        raise ValueError("Valores inválidos en el bloque questions.drafts de config.yaml.")
    return s


# --------------------------------------------------------------------------- what the AI sees

def _namespace(iri: str) -> str:
    cut = max(iri.rfind("#"), iri.rfind("/"))
    return iri[:cut + 1]


def prefixes(g: Graph) -> dict[str, str]:
    """namespace -> prefix: the known ones by name, the rest ns1, ns2... by how often they appear."""
    seen = Counter(_namespace(str(t)) for triple in g for t in triple if isinstance(t, URIRef))
    out = {ns: KNOWN[ns] for ns in seen if ns in KNOWN}
    others = sorted((ns for ns in seen if ns not in KNOWN), key=lambda ns: (-seen[ns], ns))
    out.update({ns: f"ns{n}" for n, ns in enumerate(others, 1)})
    return out


def _curie(term, names: dict[str, str]) -> str:
    iri = str(term)
    ns = _namespace(iri)
    local = iri[len(ns):]
    return f"{names[ns]}:{local}" if ns in names and re.fullmatch(r"[\w.-]*", local) else f"<{iri}>"


def _classes_of(g: Graph, node, names) -> list[str]:
    """A named class, or the members of an owl:unionOf (the repaired domains and ranges)."""
    if isinstance(node, URIRef):
        return [_curie(node, names)]
    union = g.value(node, OWL.unionOf)
    if union is None:
        return []
    return sorted(_curie(m, names) for m in Collection(g, union) if isinstance(m, URIRef))


def _named(g: Graph, kinds) -> list:
    return sorted({s for k in kinds for s in g.subjects(RDF.type, k) if isinstance(s, URIRef)},
                  key=str)


def _languages(g: Graph, predicate) -> str:
    langs = Counter(o.language or "sin idioma" for o in g.objects(None, predicate)
                    if isinstance(o, Literal))
    return ", ".join(f"{n} en {lang}" for lang, n in sorted(langs.items())) or "ninguno"


def summary_text(files: list[Path]) -> str:
    """The ontology as the AI sees it, from its triples alone (the DPV is not listed)."""
    from src.transform.convert import load_graph
    g = Graph()
    for f in files:
        g += load_graph(Path(f))
    names = prefixes(g)
    c = lambda t: _curie(t, names)                              # noqa: E731
    classes = _named(g, (OWL.Class, RDFS.Class))
    props = {kind: _named(g, (t,)) for kind, t in (("objeto", OWL.ObjectProperty),
                                                     ("datos", OWL.DatatypeProperty))}
    schema = set(classes) | {p for ps in props.values() for p in ps}
    ontology_types = set(classes)
    individuals = sorted({s for s, o in g.subject_objects(RDF.type)
                          if isinstance(s, URIRef) and (o == OWL.NamedIndividual
                                                        or o in ontology_types)}, key=str)
    uses = Counter(p for p in g.predicates())
    lines = ["PREFIJOS"]
    lines += [f"PREFIX {p}: <{ns}>" for ns, p in sorted(names.items(), key=lambda x: x[1])]
    lines += ["", f"CIFRAS: {len(g)} tripletas; {len(classes)} clases; "
                  f"{len(props['objeto'])} propiedades de objeto; {len(props['datos'])} de datos; "
                  f"{len(individuals)} individuos.",
              "", "CLASES (A < B: A es subclase de B)"]
    for k in classes:
        supers = sorted(c(o) for o in g.objects(k, RDFS.subClassOf) if isinstance(o, URIRef))
        lines.append(c(k) + (f" < {', '.join(supers)}" if supers else ""))
    for kind, ps in props.items():
        lines += ["", f"PROPIEDADES DE {kind.upper()} (dominio -> rango; usos; < superpropiedad)"]
        for p in ps:
            dom = sorted({x for d in g.objects(p, RDFS.domain) for x in _classes_of(g, d, names)})
            rng = sorted({x for r in g.objects(p, RDFS.range) for x in _classes_of(g, r, names)})
            supers = sorted(c(o) for o in g.objects(p, RDFS.subPropertyOf) if isinstance(o, URIRef))
            text = f"{c(p)}: {' | '.join(dom) or '?'} -> {' | '.join(rng) or '?'}; {uses[p]} usos"
            lines.append(text + (f" < {', '.join(supers)}" if supers else ""))
    lines += ["", "INDIVIDUOS (tipos; propiedades que usa)"]
    for i in individuals:
        types = sorted(c(o) for o in g.objects(i, RDF.type) if o != OWL.NamedIndividual)
        used = sorted({c(p) for p in g.predicates(i) if p in schema})
        lines.append(f"{c(i)}: {', '.join(types) or 'owl:NamedIndividual'}"
                     + (f"; usa {', '.join(used)}" if used else ""))
    lines += ["", "ENLACES CON EL DPV"]
    counts = [f"{c(m)} ({sum(1 for _ in g.subject_objects(m))})" for m in MAPPINGS]
    lines.append("Los conceptos se enlazan con términos del DPV con " + ", ".join(counts) + ".")
    terms = sorted({c(o) for m in MAPPINGS for o in g.objects(None, m) if isinstance(o, URIRef)})
    lines.append(f"Términos del DPV enlazados ({len(terms)}): {', '.join(terms)}.")
    lines.append("En el DPV, la jerarquía de términos se escribe con rdfs:subClassOf y skos:broader.")
    lines += ["", "ANOTACIONES"]
    sources = sorted({str(o) for o in g.objects(None, DCTERMS.source) if isinstance(o, Literal)})
    lines.append(f"dct:source guarda el artículo de la ley del que sale el concepto, en "
                 f"{sum(1 for _ in g.subject_objects(DCTERMS.source))} conceptos"
                 + (f" (p. ej. \"{sources[0]}\")." if sources else "."))
    lines.append(f"rdfs:label: {_languages(g, RDFS.label)}. "
                 f"skos:definition: {_languages(g, SKOS.definition)}.")
    return "\n".join(lines)


def context(summary: str, questions: list[qc.Question]) -> str:
    return "\n\n".join(["=== RESUMEN DE LA ONTOLOGIA ===\n" + summary,
                        *[f"=== {q.id} ===\n{q.question}" for q in questions]])


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- asking

def parse_queries(raw: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    data = None
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if isinstance(data, dict):
        data = data.get("queries")
    if not isinstance(data, list):
        raise ValueError("La respuesta del LLM no trae la lista de consultas en JSON.")
    return [d for d in data if isinstance(d, dict)]


@dataclass
class DraftReport:
    drafts: dict = field(default_factory=dict)
    calls: int = 0
    cached: int = 0
    invalid: int = 0
    errors: list[str] = field(default_factory=list)

    def ai(self, available: bool, error: str | None = None) -> dict:
        out = {"available": available, "calls": self.calls, "cached": self.cached,
               "invalid": self.invalid, "errors": self.errors[:3]}
        return {**out, "error": error} if error else out


def _ask_once(llm, prompt, text: str, max_tokens: int, sleep_fn) -> tuple[str | None, str | None]:
    """(answer, error) after one retry on a 429 or a 503."""
    from src.alignment.justify import (QUOTA_WAIT_SECONDS, TRANSIENT_WAIT_SECONDS,
                                       _is_transient_error)
    from src.core.extract import _is_quota_error
    err = None
    for attempt in range(2):
        try:
            return llm.ask(prompt=prompt.text, context=text, max_tokens=max_tokens,
                           json_mode=True) or "", None
        except Exception as exc:                       # noqa: BLE001 - reported, never raised
            err = str(exc)
            if attempt == 0 and _is_quota_error(err):
                sleep_fn(QUOTA_WAIT_SECONDS)
            elif attempt == 0 and _is_transient_error(err):
                sleep_fn(TRANSIENT_WAIT_SECONDS)
            else:
                break
    return None, err


def ask(questions: list[qc.Question], summary: str, *, llm, settings: DraftSettings,
        cache_path=None, sleep_fn=time.sleep) -> DraftReport:
    """Cache first; the rest in batches of ``batch_size`` questions, one call each."""
    from src.alignment.decider import generation_settings
    from src.transform.labels import _append_cache, _load_cache
    prompt = PROMPTS[settings.prompt_version]
    report = DraftReport()
    model = getattr(llm, "model", None) or os.environ.get("LLM_MODEL")
    generation = generation_settings(llm)
    seen = digest(summary)
    keys = {q.id: qe.cache_key(f"{seen}\n{q.id}\n{q.question}", model, generation, prompt)
            for q in questions}
    cache = _load_cache(cache_path)
    pending = []
    for q in questions:
        hit = cache.get(keys[q.id])
        if hit is not None:
            report.drafts[q.id] = hit["sparql"]
            report.cached += 1
        else:
            pending.append(q)
    if not pending or llm is None:
        return report
    for start in range(0, len(pending), settings.batch_size):
        batch = pending[start:start + settings.batch_size]
        raw, err = _ask_once(llm, prompt, context(summary, batch), settings.max_tokens, sleep_fn)
        report.calls += 1
        if err is None:
            try:
                parsed = parse_queries(raw)
            except ValueError as exc:
                err = str(exc)
        if err is not None:
            report.errors.append(err[:300])
            continue
        wanted = {q.id for q in batch}
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        new = []
        for answer in parsed:
            qid = str(answer.get("id") or "").strip().upper()
            sparql = answer.get("sparql")
            if qid not in wanted or qid in report.drafts:
                continue
            if not isinstance(sparql, str) or not sparql.strip():
                report.invalid += 1
                continue
            report.drafts[qid] = sparql.strip()
            new.append({"cache_key": keys[qid], "question": qid, "sparql": sparql.strip(),
                        "prompt": prompt.manifest(), "model": model, "created": now})
        _append_cache(cache_path, new)
    return report


# --------------------------------------------------------------------------- the prefixes

_DECLARED = re.compile(r"\bPREFIX\s+([A-Za-z][\w.-]*)?:\s*<[^>]*>", re.IGNORECASE)
_HIDDEN = re.compile(r'<[^<>\s]*>|"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|#[^\n]*')
_USED = re.compile(r"(?<![\w?$:.-])([A-Za-z][\w-]*):(?=[\w])")


def complete_prefixes(sparql: str) -> tuple[str, list[str]]:
    """The query with a PREFIX for each standard prefix it uses without declaring it, and those
    prefixes. IRIs, strings and comments are not read; nothing else changes."""
    declared = {m.group(1) or "" for m in _DECLARED.finditer(sparql)}   # case counts in SPARQL
    body = _HIDDEN.sub(" ", sparql)                 # a declaration leaves "onto: ", not used
    missing = sorted({p for p in _USED.findall(body) if p in STANDARD and p not in declared})
    if not missing:
        return sparql, []
    return "\n".join([*(f"PREFIX {p}: <{STANDARD[p]}>" for p in missing), sparql]), missing


# --------------------------------------------------------------------------- measuring

def answer_of(q: qc.Question, judged: dict, answer: dict) -> tuple:
    """What the person reads of an answer: the elements, the values or the concepts."""
    if q.kind == qc.VALUE:
        return ("valores", tuple(judged["values"]))
    if q.kind == qc.AT_LEAST:
        return ("conceptos", frozenset(json.dumps(x, sort_keys=True)
                                       for x in qs._column(answer, "concept") if x))
    return ("elementos", tuple(sorted(f["name"] for f in judged["found"])))


def outcome(q: qc.Question, reference: dict, draft: dict | None, *, missing: str | None = None,
            refused: str | None = None) -> dict:
    """How the draft did against the reference, judged by the same rule. ``missing``: why there is
    no draft; ``refused``: why a draft was not run."""
    ref = qs.judge(q, reference)
    out = {"id": q.id, "group": q.group, "question": q.question,
           "reference": qs.answer_text(ref), "reference_color": ref["color"]}
    if missing is not None:
        return {**out, "outcome": NO_DRAFT, "reason": missing}
    if refused is not None:
        return {**out, "outcome": BROKEN, "reason": refused}
    mine = qs.judge(q, draft)
    if mine["color"] == qs.GRAY:
        return {**out, "outcome": BROKEN, "reason": mine["error"]}
    same = answer_of(q, mine, draft) == answer_of(q, ref, reference)   # then the color too
    return {**out, "outcome": SAME if same else OTHER, "draft": qs.answer_text(mine),
            "draft_color": mine["color"], "same_color": mine["color"] == ref["color"],
            "same_rows": qs.rows_of(draft) == qs.rows_of(reference)}


_OWN = ("outcome", "reason", "draft", "draft_color", "same_color", "same_rows")


def _own(r: dict) -> dict:
    return {k: r[k] for k in _OWN if k in r}


def table(rows: list[dict]) -> dict:
    groups = {g: [r for r in rows if r["group"] == g] for g in qc.GROUPS}
    out = {}
    for name, part in (*groups.items(), ("total", rows)):
        counts = {o: sum(r["outcome"] == o for r in part) for o in OUTCOMES}
        counts["same_color"] = sum(r["outcome"] == OTHER and r["same_color"] for r in part)
        counts["same_rows"] = sum(bool(r.get("same_rows")) for r in part)
        counts["questions"] = len(part)
        out[name] = counts
    return out


def column(rows: list[dict], version: int, kind: str) -> list[dict]:
    """One column of the measurement: each question as one version left it."""
    return [{"group": r["group"], **r["versions"][str(version)][kind]} for r in rows]


def adopted(rows: list[dict]) -> int:
    """The rule fixed before measuring, on the column with the prefixes completed."""
    def right(version):
        return {r["id"] for r in rows if r["versions"][str(version)][PREFIXED]["outcome"] == SAME}
    one, two = right(1), right(2)
    return 2 if len(two) > len(one) and one <= two else 1


def _header(q: qc.Question, prompt, model: str | None, added: list[str] | None = None) -> str:
    text = (f"# Borrador de la IA ({prompt.name}.v{prompt.version}, {model or 'sin modelo'}) para "
            f"{q.id} · {q.question}\n# No es la consulta de referencia (src/validation/queries/).\n")
    if added:
        text += f"# El sistema declaró los prefijos que la IA usó sin declarar: {', '.join(added)}.\n"
    return text


def _draft_ids(qid: str, version: int) -> tuple[str, str]:
    """The ids the engines give to a draft and to its prefixed copy (P01.v1, P01.v1.prefixed)."""
    return f"{qid}.v{version}", f"{qid}.v{version}.{PREFIXED}"


def measure(cfg: dict | None = None, *, llm=None, use_llm: bool = True, engine: str | None = None,
            out_dir: str | Path | None = None, sleep_fn=time.sleep) -> dict:
    """Drafts of the 23 questions of the base by every prompt, run with the references and judged
    by their rule, as written and with the prefixes completed."""
    from datetime import datetime, timezone
    from src.alignment.inherit import base_paths
    from src.config import load_config
    cfg = cfg if cfg is not None else load_config()
    settings = draft_settings(cfg)
    catalog = qc.load_catalog(qc.CATALOG_FILE)
    questions = qc.for_mode(catalog, "base")
    files = qs.base_files(cfg)[qs.BASE_SET]
    summary = summary_text(files[:-1])                # the base without the DPV (the last file)
    folder = Path(out_dir) if out_dir else base_paths(cfg)["candidates"].parent
    available, error = bool(use_llm), None if use_llm else "se pidió --sin-ia"
    if available and llm is None:
        try:
            from src.ai.llm_client import LLMClient
            llm = LLMClient()
        except KeyError as exc:                        # LLM_API_KEY / LLM_MODEL missing
            available, error = False, f"Falta la variable {exc} en .env"
    model = getattr(llm, "model", None) or os.environ.get("LLM_MODEL")
    drafts_dir = folder / DRAFTS_DIR
    for old in drafts_dir.rglob("*.rq") if drafts_dir.exists() else []:
        old.unlink()
    paths, plans, reports = {}, {}, {}
    for version, prompt in PROMPTS.items():
        mine = DraftSettings(**{**settings.__dict__, "prompt_version": version})
        report = ask(questions, summary, llm=llm if available else None, settings=mine,
                     cache_path=folder / CACHE_FILE, sleep_fn=sleep_fn)
        reports[version] = report
        where = drafts_dir / f"v{version}"
        where.mkdir(parents=True, exist_ok=True)
        for q in questions:
            raw_id, fixed_id = _draft_ids(q.id, version)
            sparql = report.drafts.get(q.id)
            if sparql is None:
                plans[raw_id] = {"missing": error or ("la IA no dio una consulta" if not
                                                      report.errors else
                                                      f"la llamada falló: {report.errors[0]}")}
                continue
            fixed, added = complete_prefixes(sparql)
            (where / f"{raw_id}.rq").write_text(_header(q, prompt, model) + sparql + "\n",
                                                encoding="utf-8")
            if added:
                (where / f"{fixed_id}.rq").write_text(_header(q, prompt, model, added) + fixed
                                                      + "\n", encoding="utf-8")
            if _SERVICE.search(sparql):
                plans[raw_id] = {"refused": "usa SERVICE (sale a internet): no se corre"}
                continue
            plans[raw_id] = {"added": added}
            paths[raw_id] = where / f"{raw_id}.rq"
            if added:
                paths[fixed_id] = where / f"{fixed_id}.rq"
    ids = [q.id for q in questions] + list(paths)
    used, runs = qs.run({qs.BASE_SET: files}, ids, cfg=cfg, engine=engine, paths=paths,
                        query_timeout=settings.query_timeout_seconds)
    answers = runs[qs.BASE_SET]["answers"]
    rows = []
    for q in questions:
        reference = answers.get(q.id)
        ref = qs.judge(q, reference)
        row = {"id": q.id, "group": q.group, "question": q.question,
               "reference": qs.answer_text(ref), "reference_color": ref["color"], "versions": {}}
        for version in PROMPTS:
            raw_id, fixed_id = _draft_ids(q.id, version)
            plan = plans[raw_id]
            raw = outcome(q, reference, answers.get(raw_id), missing=plan.get("missing"),
                          refused=plan.get("refused"))
            fixed = (outcome(q, reference, answers.get(fixed_id)) if plan.get("added") else raw)
            row["versions"][str(version)] = {AS_WRITTEN: _own(raw), PREFIXED: _own(fixed),
                                             "added": plan.get("added") or []}
        rows.append(row)
    out = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "model": model, "engine": used, "seconds": runs[qs.BASE_SET]["seconds"],
           "summary": {"sha256": digest(summary), "chars": len(summary)}, "rule": RULE,
           "versions": {str(v): {"prompt": p.manifest(), "ai": reports[v].ai(available, error),
                                 AS_WRITTEN: table(column(rows, v, AS_WRITTEN)),
                                 PREFIXED: table(column(rows, v, PREFIXED))}
                        for v, p in PROMPTS.items()},
           "questions": rows, "adoption": ADOPTION, "adopted": adopted(rows),
           "configured": settings.prompt_version, "drafts": qs._shown(drafts_dir)}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / EVALUATION_FILE).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return out


def ai_text(ai: dict) -> str:
    """What the AI did, in one line."""
    calls = f"{ai['calls']} llamada{'' if ai['calls'] == 1 else 's'}"
    text = (f"no usada; {ai['error']}" if not ai["available"] else calls)
    text += f"; {ai['cached']} borrador{'' if ai['cached'] == 1 else 'es'} desde la caché"
    if ai["invalid"]:
        text += f"; {ai['invalid']} inválido{'' if ai['invalid'] == 1 else 's'}"
    if ai["errors"]:
        text += f"; error: {ai['errors'][0]}"
    return text


def _draft_line(v: dict) -> str:
    return f"[{v['outcome']}] {v.get('draft') or v['reason']}"


def render(out: dict) -> str:
    versions = out["versions"]
    lines = ["Borradores SPARQL de la IA sobre la base (S9-T08)",
             f"  Modelo {out['model'] or 'sin modelo'} · motor {qs.engine_text(out['engine'])} · "
             f"borradores y referencias en la misma corrida ({out['seconds']:g} s)",
             f"  La IA vio la pregunta y un resumen de la ontología de {out['summary']['chars']} "
             f"caracteres (sha256 {out['summary']['sha256'][:12]}); nunca la consulta de "
             f"referencia",
             f"  Regla: {out['rule']}",
             "  Con prefijos: el sistema declara los prefijos estándar que la consulta usa sin "
             f"declarar ({', '.join(STANDARD)}); nada más cambia", ""]
    for v, data in versions.items():
        p = data["prompt"]
        lines.append(f"  IA {p['name']}.v{p['version']}: {ai_text(data['ai'])}")
    heads = [f"v{v} {label}" for v in versions for label in ("tal cual", "con prefijos")]
    cells = [data[kind]["total"] for data in versions.values() for kind in (AS_WRITTEN, PREFIXED)]
    show = [(SAME, SAME), (OTHER, OTHER), ("same_color", "  con el mismo color"), (BROKEN, BROKEN)]
    if any(c[NO_DRAFT] for c in cells):
        show.append((NO_DRAFT, NO_DRAFT))
    width = max(len(label) for _, label in show) + 2
    total = cells[0]["questions"]
    lines += ["", f"  De {total} preguntas".ljust(width + 2) + "  ".join(heads)]
    for key, label in show:
        lines.append("  " + label.capitalize().ljust(width) + "  ".join(
            str(c[key]).rjust(len(h)) for c, h in zip(cells, heads)))
    lines += ["", "  Por pregunta:"]
    for r in out["questions"]:
        lines += [f"    {r['id']} · {r['question']}", f"         Referencia: {r['reference']}"]
        for v, mine in r["versions"].items():
            lines.append(f"         v{v}: {_draft_line(mine[AS_WRITTEN])}")
            if mine["added"]:
                lines.append(f"             con prefijos ({', '.join(mine['added'])}): "
                             f"{_draft_line(mine[PREFIXED])}")
    lines += ["", f"  Regla para adoptar, fijada antes de medir: {out['adoption']}",
              f"  Se adopta: sparql.v{out['adopted']} (config.yaml dice v{out['configured']})"]
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import base_paths
    from src.config import load_config
    parser = argparse.ArgumentParser(
        prog="py -m src.validation.query_drafter",
        description="La IA redacta la consulta SPARQL de cada pregunta de competencia de la base "
                    "y se compara con la consulta de referencia.")
    parser.add_argument("--medir", action="store_true", help="redactar, correr y comparar")
    parser.add_argument("--motor", choices=[qs.JENA, qs.RDFLIB], help="forzar un motor")
    parser.add_argument("--sin-ia", action="store_true", help="solo los borradores de la caché")
    args = parser.parse_args(argv)
    if not args.medir:
        print("Error: indica --medir.")
        return 1
    cfg = load_config()
    try:
        out = measure(cfg, use_llm=not args.sin_ia, engine=args.motor)
    except (FileNotFoundError, ValueError, RuntimeError, qc.CatalogError) as exc:
        print(f"Error: {exc}")
        return 1
    folder = base_paths(cfg)["candidates"].parent
    print(render(out))
    print(f"  Borradores en: {out['drafts']}/")
    print(f"  Guardado en: {folder / EVALUATION_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
