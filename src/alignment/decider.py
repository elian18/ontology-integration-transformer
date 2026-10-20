"""The alignment decider: retrieve -> decide with precedents -> fixed rule (S7-T04).

Until Sprint 6 the AI only saw the best DPV candidate of the concepts above the threshold and
only gave it a SKOS type: the threshold, not the AI, decided. The measurement of S7-T03 showed
the limit of that design: the right term is the best candidate in 92 of 173 concepts, but it
is among the first 10 in 154. The decider lets the AI CHOOSE among them.

For every concept that did not inherit a decision:

1. RETRIEVE (the "R" of RAG, no LLM):
   - its N DPV candidates (``decider.candidates``, 10) from the source chosen in S7-T03
     (``decider.candidate_source``: the hybrid ranking), each with its definition and parents;
   - its ``decider.precedents`` (4) closest decided base concepts from the ChromaDB collection
     "decisions" (S7-T01), with what a person decided: term and relation, or "sin
     correspondencia". The concept itself and every base concept that is "the same" (the
     leak control of S7-T03) are never shown, and only concepts of a compatible kind are;
   - optionally, the law article that supports it (``evidence_fn``; S7-T05 and S7-T11).
   The scores are NOT shown to the LLM (it would just follow the rank 1).
2. DECIDE (the "G"): the prompt ``src/ai/prompts/decide.v1.txt``, in batches of
   ``decider.batch_size`` concepts per call, temperature 0, JSON mode (Gemini accepts it,
   S7-T02). Per concept the LLM returns: the chosen candidate or "ninguno", the SKOS relation,
   a confidence (alta | media | baja) and one sentence in Spanish. Answers are checked: a term
   that is not one of the concept's candidates is rejected, never written.
3. FIXED RULE (``apply_rule``): deterministic and explainable. Its parameters (which
   confidences are accepted, a minimum score, which SKOS relations are accepted for classes and
   for properties) are calibrated in S7-T05, not invented here. What the rule leaves out but
   the LLM said with high confidence is a SUGGESTION (``suggested``): it is never written into
   the ontology; it is shown so that a person can comment on it (Sprint 10).

Free tier, as in Sprints 3-6: pause between calls, call limit per run, one retry after a 429
or a 503, stop cleanly when the quota is exhausted. Every answer is cached by CONTENT (prompt,
model and everything the LLM saw), so a re-run never spends a call twice and an interrupted
run resumes where it stopped; a change of prompt, model, candidates or precedents makes a new
call. Each decision keeps its provenance for PROV-O (Sprint 10): candidates, precedents,
article, prompt version and model.

The LLM, the embedder and the evidence are injected: the fast tests use fakes.

CLI: ``py -m src.alignment.decider --base 10`` decides 10 base concepts as if they were new
(1 call) and compares them with your decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.ai.prompts import load_prompt
from src.ai.rag import vector_store
from src.alignment.benchmark import (TIE_MARGIN, MemoEmbed, SOURCE_HYBRID, candidate_lists,
                                     decider_settings, stable_order, stored_dpv_vectors)
from src.alignment.candidates import alignment_settings
from src.alignment.justify import (QUOTA_WAIT_SECONDS, RELATION_UNTYPED, TRANSIENT_WAIT_SECONDS,
                                   _is_transient_error, clean_justification, normalize_relation)
from src.alignment.knowledge import DECISIONS_COLLECTION, parse_record_metadata
from src.alignment.lexical import lexical_matrix, source_lexical_text, target_lexical_text
from src.core.dpv_proximity import _trim
from src.core.extract import _is_quota_error

PROMPT_VERSION = 1
PROMPT = load_prompt("decide", PROMPT_VERSION)
CACHE_FILE = "decider-cache.jsonl"
NONE_ANSWER = "ninguno"
CONFIDENCE_LEVELS = ("alta", "media", "baja")
SKOS_EXACT = ("skos:exactMatch",)
SKOS_STRONG = ("skos:exactMatch", "skos:closeMatch", "skos:broadMatch", "skos:narrowMatch")
SAME_AS_CLASSES = "same"            # property_relations: the same relations as for classes
_CONFIDENCE_ALIASES = {"alta": "alta", "high": "alta", "media": "media", "medium": "media",
                       "moderada": "media", "baja": "baja", "low": "baja"}
_NONE_ALIASES = {"ninguno", "ninguna", "none", "no", "sin correspondencia", "nada", ""}

STATUS_DECIDED = "decided"          # the LLM answered and the answer is valid
STATUS_PENDING = "pending"          # not asked yet (call limit, quota, failed batch)
STATUS_INVALID = "invalid"          # the LLM answered with a term outside the candidates
STATUS_NO_CANDIDATES = "no_candidates"
STATUS_LABELS = {STATUS_DECIDED: "decidido", STATUS_PENDING: "pendiente",
                 STATUS_INVALID: "respuesta inválida", STATUS_NO_CANDIDATES: "sin candidatos"}


# --------------------------------------------------------------------------- settings

@dataclass(frozen=True)
class DeciderSettings:
    candidate_source: str = SOURCE_HYBRID
    candidates: int = 10
    precedents: int = 4
    batch_size: int = 10
    max_calls: int = 60
    pause_seconds: float = 5.0
    max_tokens: int = 8192
    article_chars: int = 1200
    definition_chars: int = 300
    candidate_definition_chars: int = 200


def settings_from_config(cfg: dict | None = None) -> DeciderSettings:
    """The ``decider`` block of config.yaml (defaults for the keys that are missing)."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("decider", {}) or {}
    base = decider_settings(cfg)
    s = DeciderSettings(
        candidate_source=base["candidate_source"], candidates=base["candidates"],
        precedents=int(block.get("precedents", 4)),
        batch_size=int(block.get("batch_size", 10)),
        max_calls=int(block.get("max_calls", 60)),
        pause_seconds=float(block.get("pause_seconds", 5.0)),
        max_tokens=int(block.get("max_tokens", 8192)),
        article_chars=int(block.get("article_chars", 1200)))
    if (not 0 <= s.precedents <= 10 or s.batch_size < 1 or s.max_calls < 1
            or s.pause_seconds < 0 or s.max_tokens < 256 or s.article_chars < 100):
        raise ValueError("Valores inválidos en el bloque decider de config.yaml "
                         "(precedents 0-10, batch_size y max_calls >= 1, max_tokens >= 256).")
    return s


# --------------------------------------------------------------------------- data

@dataclass
class Decision:
    """What the decider did with one concept, with its provenance."""
    key: str
    name: str
    status: str
    dpv_iri: str | None = None
    dpv_name: str | None = None
    none: bool = False                   # the LLM said "ninguno" (sin correspondencia)
    relation: str | None = None          # skos:... (None if "ninguno" or invalid relation)
    confidence: str | None = None        # alta | media | baja
    justification: str | None = None
    score: float | None = None           # hybrid score of the chosen candidate (for the rule)
    rank: int | None = None              # its position among the candidates
    candidates: list[str] = field(default_factory=list)       # DPV IRIs shown, in order
    precedents: list[str] = field(default_factory=list)       # base keys shown as precedents
    evidence_article: int | None = None
    from_cache: bool = False
    issues: list[str] = field(default_factory=list)
    prompt: dict = field(default_factory=dict)
    model: str | None = None
    kinds: tuple[str, ...] = ()          # the concept's kinds (class, property)

    @property
    def is_property(self) -> bool:
        """Only a property: a concept that is also a class (OWL Full punning) counts as class."""
        return "property" in self.kinds and "class" not in self.kinds

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecideReport:
    decisions: list[Decision] = field(default_factory=list)
    calls: int = 0
    sent: int = 0
    cached: int = 0
    quota_exhausted: bool = False
    call_limit_reached: bool = False
    errors: list[dict] = field(default_factory=list)          # {concepts, error}

    def counts(self) -> dict:
        by = {s: 0 for s in STATUS_LABELS}
        for d in self.decisions:
            by[d.status] += 1
        decided = [d for d in self.decisions if d.status == STATUS_DECIDED]
        return {"concepts": len(self.decisions), **by,
                "chosen_term": sum(1 for d in decided if d.dpv_iri),
                "said_none": sum(1 for d in decided if d.none),
                "by_confidence": {c: sum(1 for d in decided if d.confidence == c)
                                  for c in CONFIDENCE_LEVELS},
                "calls": self.calls, "sent": self.sent, "cached": self.cached,
                "quota_exhausted": self.quota_exhausted,
                "call_limit_reached": self.call_limit_reached}


@dataclass
class _Item:
    """Everything shown to the LLM for one concept (without its batch id)."""
    concept: object
    candidates: list            # [(DpvTarget, score)]
    precedents: list[dict]
    evidence: dict | None
    body: str = ""
    cache_key: str = ""


# --------------------------------------------------------------------------- retrieve

def _kind_where(kinds) -> dict | None:
    flags = [{"is_class": True}] if "class" in kinds else []
    flags += [{"is_property": True}] if "property" in kinds else []
    if not flags:
        return None
    return flags[0] if len(flags) == 1 else {"$or": flags}


def check_decisions_collection(model_name: str, path=None) -> None:
    meta = vector_store.metadata(DECISIONS_COLLECTION, path)
    if (not meta or meta.get("embedding_model") != model_name
            or vector_store.count(name=DECISIONS_COLLECTION, path=path) != meta.get("items")):
        raise ValueError("La colección decisions no está al día con este modelo de embeddings: "
                         "corre primero py -m src.alignment.knowledge")


def find_precedents(concept, vector, n: int, forbidden: set[str] | None = None,
                    path=None) -> list[dict]:
    """The ``n`` closest decided base concepts of a compatible kind, minus the forbidden ones."""
    if n <= 0:
        return []
    where = _kind_where(concept.kinds)
    if where is None:
        return []
    banned = set(forbidden or ()) | {concept.key}
    res = vector_store.query([list(map(float, vector))], n + len(banned) + TIE_MARGIN,
                             name=DECISIONS_COLLECTION, path=path, where=where,
                             include=("metadatas", "distances"))
    metas = dict(zip(res["ids"][0], res["metadatas"][0]))
    out = []
    for key, dist in stable_order(res["ids"][0], res["distances"][0]):
        if key in banned:
            continue
        m = parse_record_metadata(metas[key])
        out.append({"key": key, "name": m["name"], "label": m.get("label") or None,
                    "definition": m.get("definition") or None, "status": m["status"],
                    "mappings": m.get("mappings", []), "similarity": round(1 - float(dist), 4)})
        if len(out) == n:
            break
    return out


def _candidate_scores(concepts, targets, cvec, tvec, lists, cfg) -> dict[str, list[tuple]]:
    """[(DpvTarget, hybrid score)] per concept, in the order of the candidate lists."""
    settings = alignment_settings(cfg)
    index = {t.iri: j for j, t in enumerate(targets.targets)}
    out = {}
    for i, c in enumerate(concepts):
        iris = lists.get(c.key, [])
        if not iris:
            out[c.key] = []
            continue
        chosen = [targets.targets[index[iri]] for iri in iris]
        lex = lexical_matrix([source_lexical_text(c)], [target_lexical_text(t) for t in chosen])[0]
        sem = np.clip(np.asarray([tvec[index[iri]] for iri in iris]) @ cvec[i], 0.0, 1.0)
        combined = settings.weight_lexical * lex + settings.weight_semantic * sem
        out[c.key] = [(t, round(float(s), 4)) for t, s in zip(chosen, combined)]
    return out


# --------------------------------------------------------------------------- prompt

def _decision_text(p: dict) -> str:
    if p["status"] != "aligned" or not p["mappings"]:
        return "sin correspondencia"
    return "; ".join(f"{m.get('dpv_name') or m['dpv_iri'].rsplit('#', 1)[-1]} "
                     f"({str(m.get('relation') or '').replace('skos:', '')})"
                     for m in p["mappings"])


def concept_body(item: _Item, settings: DeciderSettings) -> str:
    """The text the LLM sees for one concept, without its batch id."""
    c = item.concept
    lines = [f"Concepto: {c.name} (tipo: {', '.join(c.kinds)})"]
    if c.label:
        lines.append(f"Etiqueta: {c.label}")
    if c.definition:
        lines.append(f"Definicion: {_trim(c.definition, settings.definition_chars)}")
    if item.evidence:
        ev = item.evidence
        lines.append(f"Articulo {ev.get('number')} de la ley ({ev.get('title') or ''}): "
                     f"{_trim(ev.get('text') or '', settings.article_chars)}")
    lines.append("Candidatos DPV:")
    for n, (t, _score) in enumerate(item.candidates, start=1):
        definition = _trim(t.definition or "", settings.candidate_definition_chars) or "-"
        parents = ", ".join(t.parents) or "-"
        lines.append(f"{n}. {t.name} | {t.label} | definicion: {definition} | "
                     f"padres en el DPV: {parents}")
    if item.precedents:
        lines.append("Precedentes (decisiones de una persona en conceptos parecidos):")
        for p in item.precedents:
            label = f" ({p['label']})" if p.get("label") else ""
            lines.append(f"- {p['name']}{label}: {_decision_text(p)}")
    return "\n".join(lines)


def cache_key(body: str, model: str | None) -> str:
    payload = json.dumps({"prompt": PROMPT.sha256, "model": model, "body": body},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- answers

def parse_answer(raw: str) -> list[dict]:
    """The list of decisions in the LLM answer (JSON object, list, or JSON in code fences)."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    data = None
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1],
                      text[text.find("["):text.rfind("]") + 1]):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            break
        except (json.JSONDecodeError, ValueError):
            continue
    if isinstance(data, dict):
        data = data.get("decisions", data.get("decisiones"))
    if not isinstance(data, list):
        raise ValueError("La respuesta del LLM no trae una lista de decisiones en JSON.")
    return [d for d in data if isinstance(d, dict)]


def _plain(text) -> str:
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    return t.strip().lower()


def validate_answer(answer: dict, item: _Item) -> dict:
    """Check one answer against the concept's candidates; returns the fields of a Decision."""
    choice = str(answer.get("dpv") or "").strip()
    for prefix in ("dpv:", "https://w3id.org/dpv#"):
        if choice.lower().startswith(prefix):
            choice = choice[len(prefix):]
    choice = choice.split("|")[0].strip()
    issues = []
    confidence = _CONFIDENCE_ALIASES.get(_plain(answer.get("confidence")))
    if confidence is None:
        issues.append("confianza no válida: se toma como baja")
        confidence = "baja"
    justification = clean_justification(answer.get("justification"))
    if _plain(choice) in _NONE_ALIASES:
        return {"status": STATUS_DECIDED, "none": True, "confidence": confidence,
                "justification": justification, "issues": issues}
    by_name = {t.name.lower(): (n, t, s) for n, (t, s) in enumerate(item.candidates, start=1)}
    found = by_name.get(choice.lower())
    if found is None and choice.isdigit() and 1 <= int(choice) <= len(item.candidates):
        n = int(choice)
        found = (n, *item.candidates[n - 1])
    if found is None:
        return {"status": STATUS_INVALID, "confidence": confidence,
                "justification": justification,
                "issues": issues + [f"el término '{choice}' no está entre los candidatos"]}
    rank, target, score = found
    relation = normalize_relation(answer.get("relation"))
    if relation in (RELATION_UNTYPED, "none"):
        issues.append("relación SKOS no válida")
        relation = None
    return {"status": STATUS_DECIDED, "dpv_iri": target.iri, "dpv_name": target.name,
            "relation": relation, "confidence": confidence, "justification": justification,
            "score": score, "rank": rank, "issues": issues}


# --------------------------------------------------------------------------- cache

def load_cache(path: Path | None) -> dict[str, dict]:
    if path is None or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entry = json.loads(line)
                out[entry["cache_key"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _append_cache(path: Path | None, entries: list[dict]) -> None:
    if path is None or not entries:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- the decider

def decide(concepts, targets, *, llm=None, embed_fn=None, settings: DeciderSettings | None = None,
           forbidden: dict[str, set] | None = None, evidence_fn=None, use_precedents: bool = True,
           cache_path: str | Path | None = None, chroma_path=None, target_vectors=None,
           model_name: str | None = None, embedding_model: str | None = None,
           sleep_fn=time.sleep, cfg: dict | None = None) -> DecideReport:
    """Retrieve, decide (LLM, cached, in batches) and validate, for every concept."""
    from src.alignment.knowledge import embedding_model_name
    settings = settings or settings_from_config(cfg)
    if llm is None:                                   # lazy: the fast tests need no key
        from src.ai.llm_client import LLMClient
        llm = LLMClient()
    model_name = model_name or getattr(llm, "model", None)
    embedding_model = embedding_model or embedding_model_name()
    if embed_fn is None:
        from src.ai.rag.embeddings import embed as embed_fn
    embed = MemoEmbed(embed_fn)
    concepts = list(concepts)
    report = DecideReport()
    if not concepts:
        return report

    cvec = np.asarray(embed([c.semantic_text() for c in concepts]), float)
    tvec = (np.asarray(target_vectors, float) if target_vectors is not None
            else stored_dpv_vectors(targets, embedding_model, chroma_path))
    lists = candidate_lists(settings.candidate_source, concepts, targets, cvec, tvec,
                            k=settings.candidates, path=chroma_path, cfg=cfg)
    scored = _candidate_scores(concepts, targets, cvec, tvec, lists, cfg)
    if use_precedents and settings.precedents:
        check_decisions_collection(embedding_model, chroma_path)
    forbidden = forbidden or {}

    items: list[_Item] = []
    for i, c in enumerate(concepts):
        precedents = (find_precedents(c, cvec[i], settings.precedents, forbidden.get(c.key),
                                      chroma_path) if use_precedents else [])
        item = _Item(concept=c, candidates=scored.get(c.key, []), precedents=precedents,
                     evidence=evidence_fn(c) if evidence_fn else None)
        item.body = concept_body(item, settings)
        item.cache_key = cache_key(item.body, model_name)
        items.append(item)

    cache = load_cache(cache_path)
    answers: dict[str, dict] = {}
    pending: list[_Item] = []
    for item in items:
        if not item.candidates:
            continue
        hit = cache.get(item.cache_key)
        if hit is not None:
            answers[item.cache_key] = {**hit["answer"], "_from_cache": True}
            report.cached += 1
        else:
            pending.append(item)

    for start in range(0, len(pending), settings.batch_size):
        if report.calls >= settings.max_calls:
            report.call_limit_reached = True
            break
        if report.calls > 0 and settings.pause_seconds > 0:
            sleep_fn(settings.pause_seconds)
        batch = pending[start:start + settings.batch_size]
        ids = {f"C{n}": it for n, it in enumerate(batch, start=1)}
        context = "\n\n".join(f"=== {cid} ===\n{it.body}" for cid, it in ids.items())
        raw, err = None, None
        for attempt in range(2):                      # one retry after a 429 or a 503
            try:
                raw = llm.ask(prompt=PROMPT.text, context=context,
                              max_tokens=settings.max_tokens, json_mode=True) or ""
                err = None
                break
            except Exception as exc:                  # noqa: BLE001 - report, don't crash
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
                parsed = parse_answer(raw)
            except ValueError as exc:
                err = str(exc)
        if err is not None:
            report.errors.append({"concepts": [it.concept.name for it in batch],
                                  "error": err[:300]})
            if _is_quota_error(err):
                report.quota_exhausted = True
                break
            continue
        report.sent += len(batch)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        new_entries = []
        for ans in parsed:
            item = ids.get(str(ans.get("concept") or "").strip().upper())
            if item is None or item.cache_key in answers:
                continue
            fields = validate_answer(ans, item)
            answers[item.cache_key] = fields
            new_entries.append({"cache_key": item.cache_key, "concept_key": item.concept.key,
                                "answer": fields, "raw": ans, "prompt": PROMPT.manifest(),
                                "model": model_name, "created": now})
        _append_cache(cache_path, new_entries)

    for item in items:
        c = item.concept
        base = {"key": c.key, "name": c.name, "kinds": tuple(c.kinds),
                "candidates": [t.iri for t, _ in item.candidates],
                "precedents": [p["key"] for p in item.precedents],
                "evidence_article": (item.evidence or {}).get("number"),
                "prompt": PROMPT.manifest(), "model": model_name}
        if not item.candidates:
            report.decisions.append(Decision(status=STATUS_NO_CANDIDATES, **base))
            continue
        ans = answers.get(item.cache_key)
        if ans is None:
            report.decisions.append(Decision(status=STATUS_PENDING, **base))
            continue
        fields = {k: v for k, v in ans.items() if not k.startswith("_")}
        report.decisions.append(Decision(**base, **fields,
                                         from_cache=bool(ans.get("_from_cache"))))
    return report


# --------------------------------------------------------------------------- the fixed rule

def apply_rule(decisions: list[Decision], accept=("alta",), min_score: float = 0.0,
               accept_none=None, *, relations=None,
               property_relations=SAME_AS_CLASSES) -> dict[str, dict]:
    """Deterministic rule: what becomes an automatic answer (S7-T05 calibrates it).

    - a chosen term whose confidence is accepted, whose score >= ``min_score`` and whose SKOS
      relation is allowed -> proposal. ``relations`` are the relations allowed for classes
      (None = any); ``property_relations`` those allowed for concepts that are only properties
      (``SAME_AS_CLASSES`` = the same as for classes, None = any, () = never automatic);
    - "ninguno" with an accepted confidence (``accept_none``, default = ``accept``) -> "sin
      correspondencia", for classes and properties alike;
    - anything else (other confidences or relations, pending, invalid) -> no proposal."""
    accept = set(accept)
    accept_none = accept if accept_none is None else set(accept_none)
    if property_relations == SAME_AS_CLASSES:
        property_relations = relations
    out = {}
    for d in decisions:
        if d.status != STATUS_DECIDED:
            continue
        if d.none:
            if d.confidence in accept_none:
                out[d.key] = {"term": None, "relation": None, "none": True}
            continue
        allowed = property_relations if d.is_property else relations
        if (d.dpv_iri and d.confidence in accept and (d.score or 0.0) >= min_score
                and (allowed is None or d.relation in allowed)):
            out[d.key] = {"term": d.dpv_iri, "relation": d.relation, "none": False}
    return out


def suggested(decisions: list[Decision], automatic: dict[str, dict],
              levels=("alta",)) -> dict[str, dict]:
    """The terms the LLM chose with one of ``levels`` that the rule did not accept (option D)."""
    levels = set(levels)
    return {d.key: {"term": d.dpv_iri, "relation": d.relation}
            for d in decisions
            if d.status == STATUS_DECIDED and d.dpv_iri            # "ninguno" has no term
            and d.confidence in levels and d.key not in automatic}


# --------------------------------------------------------------------------- console / CLI

def render_console(report: DecideReport, reference=None, limit: int = 20) -> str:
    c = report.counts()
    conf = c["by_confidence"]
    lines = ["Decisor de alineación (S7-T04)",
             f"  Conceptos: {c['concepts']} | decididos: {c['decided']} (término: "
             f"{c['chosen_term']}, 'ninguno': {c['said_none']}) | pendientes: {c['pending']} | "
             f"inválidos: {c['invalid']} | sin candidatos: {c['no_candidates']}",
             f"  Confianza: alta {conf['alta']}, media {conf['media']}, baja {conf['baja']}",
             f"  Llamadas al LLM: {c['calls']} | conceptos enviados: {c['sent']} | "
             f"desde la caché: {c['cached']}"]
    if report.quota_exhausted:
        lines.append("  CUOTA AGOTADA: vuelve a correr más tarde; lo ya decidido no se repite.")
    if report.call_limit_reached:
        lines.append("  Se alcanzó decider.max_calls: vuelve a correr para continuar.")
    for e in report.errors[:3]:
        lines.append(f"  Error en un lote ({', '.join(e['concepts'][:3])}...): {e['error'][:120]}")
    ref = reference.by_key() if reference is not None else {}
    shown = [d for d in report.decisions if d.status in (STATUS_DECIDED, STATUS_INVALID)][:limit]
    if shown:
        lines.append("  Concepto -> decisor (confianza) | tu decisión")
    for d in shown:
        mine = ""
        if d.key in ref:
            r = ref[d.key]
            mine = ("sin correspondencia" if not r.terms else
                    "; ".join(f"{iri.rsplit('#', 1)[-1]} {rel.replace('skos:', '')}"
                              for iri, rel in r.terms.items()))
            ok = ((d.none and not r.terms) or (d.dpv_iri in r.terms))
            mine = f"{mine}  {'OK' if ok else 'distinto'}"
        what = ("ninguno" if d.none else
                f"{d.dpv_name} {str(d.relation or 'sin tipo').replace('skos:', '')}"
                if d.status == STATUS_DECIDED else "inválido")
        lines.append(f"    {d.name[:38]:<38} -> {what} ({d.confidence or '-'}) | {mine}")
    return "\n".join(lines)


def load_forbidden(path: str | Path, name: str = "base") -> dict[str, set]:
    """The leak control saved by S7-T03 (decision-reference.json)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["references"][name]["items"]
    return {i["key"]: set(i.get("forbidden") or {}) for i in items}


def _sample(items, n: int) -> list:
    """A fixed, spread sample of n reference items (same every run)."""
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.benchmark import load_inputs
    from src.alignment.inherit import base_paths
    from src.alignment.reference import REFERENCE_FILE, Reference, RefItem
    parser = argparse.ArgumentParser(prog="py -m src.alignment.decider",
                                     description="Prueba el decisor con conceptos de la base "
                                                 "resueltos como nuevos (S7-T04).")
    parser.add_argument("--base", type=int, default=10,
                        help="cuántos conceptos de la base decidir (10 = 1 llamada)")
    parser.add_argument("--sin-precedentes", action="store_true",
                        help="no mostrar decisiones parecidas al LLM")
    args = parser.parse_args(argv)
    out_dir = base_paths()["candidates"].parent
    ref_path = out_dir / REFERENCE_FILE
    if not ref_path.exists():
        print(f"Error: falta {ref_path}. Corre primero py -m src.alignment.benchmark (S7-T03).")
        return 1
    inputs = load_inputs()
    data = json.loads(ref_path.read_text(encoding="utf-8"))["references"]["base"]["items"]
    reference = Reference("base", [RefItem(**i) for i in data])
    chosen = _sample(sorted(reference.items, key=lambda i: i.key), args.base)
    by_key = {c.key: c for c in inputs.s4_concepts}
    concepts = [by_key[i.key] for i in chosen if i.key in by_key]
    print(f"Decidiendo {len(concepts)} conceptos de la base como si fueran nuevos...")
    try:
        report = decide(concepts, inputs.targets, forbidden=load_forbidden(ref_path),
                        use_precedents=not args.sin_precedentes,
                        cache_path=out_dir / "decider" / CACHE_FILE)
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(report, reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())