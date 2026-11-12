"""Questions with options about what stays in doubt in an upload (S10-T04).

The doubts: the suggestions (the AI chose a term with high confidence and the rule left it
out; not the "solo relacionadas", the relation the rule already found unreliable) and the
correspondences the rule accepted that the AI reviewer marked incoherent or doubtful.

What the uploads ask (``fixed_questions``): one question per doubt, at most ``max_questions``,
in a fixed order (what is already in the ontology first, then the suggestions with the highest
score), without a recommended option. A suggestion: take it or leave it. A reviewed one: the
relation the AI reviewer proposes, remove it, or leave it. Each question shows what the AIs
said; the person decides.

The AI questions (prompt ask.v1, ``propose_questions``: the AI chooses what to ask, joins the
doubts with the same pattern and recommends an option) were measured on the base and not
adopted (py -m src.alignment.question_eval, November 2026): with names only it could not tell
the right suggestions from the wrong ones (it asked about 16 of the 33 right ones and 13 of the
37 you left without correspondence) and following its recommendation lowered the precision
from 75.0 % to 66.3 %. ``user_questions.ai`` keeps it off.

Every option goes through the fixed rule of S10-T03 (``user_decisions.check_state``); the AI
options also have to change only the concepts of their question, with terms shown for them."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field

from src.ai.prompts import load_prompt
from src.alignment import user_decisions as ud
from src.console import counted

PROMPT_NAME = "ask"
CACHE_FILE = "ask-cache.jsonl"
SUGGESTED, REVIEWED = "sugerida", "revisada"
LEAVE = "Dejar como está"
ALIGN, REMOVE = "alinear", "quitar"
RELATION_WORDS = {"skos:exactMatch": "es lo mismo que", "skos:closeMatch": "es casi lo mismo que",
                  "skos:broadMatch": "es un caso más específico de",
                  "skos:narrowMatch": "es más general que",
                  "skos:relatedMatch": "solo está relacionado con"}
DEFAULT_RELATION = "skos:closeMatch"          # a suggestion without a valid SKOS relation
BY_AI, BY_RULE, BY_LIMIT, NO_ANSWER = "ia", "regla", "máximo", "sin respuesta"
_ID = re.compile(r"^D\d+$")


@dataclass(frozen=True)
class AskSettings:
    ai: bool = False                # the AI chooses and recommends (ask.v1: not adopted)
    prompt_version: int = 1
    max_questions: int = 10
    batch_size: int = 30            # doubts per call
    max_calls: int = 10
    pause_seconds: float = 5.0
    max_tokens: int = 8192
    candidates: int = 5             # DPV candidates shown per doubt
    definition_chars: int = 300


def ask_settings(cfg: dict | None = None) -> AskSettings:
    from src.config import load_config
    block = (cfg if cfg is not None else load_config()).get("user_questions", {}) or {}
    s = AskSettings(**{k: v for k, v in block.items() if k in AskSettings.__dataclass_fields__})
    if (s.max_questions < 1 or s.batch_size < 1 or s.max_calls < 0 or s.pause_seconds < 0
            or s.candidates < 1):
        raise ValueError("Valores inválidos en el bloque user_questions de config.yaml.")
    return s


def ask_prompt(settings: AskSettings | None = None):
    return load_prompt(PROMPT_NAME, (settings or AskSettings()).prompt_version)


# --------------------------------------------------------------------------- the doubts

@dataclass
class Doubt:
    id: str
    key: str
    name: str
    kinds: tuple
    kind: str                                   # SUGGESTED | REVIEWED
    label: str | None = None
    definition: str | None = None
    current: list[dict] = field(default_factory=list)      # what it has now (user_decisions)
    proposal: dict = field(default_factory=dict)            # the decider's choice
    audit: dict | None = None                               # verdict, relation, explanation
    why: str = ""                                           # why the rule left it out
    candidates: list[str] = field(default_factory=list)     # DPV IRIs shown

    @property
    def is_property(self) -> bool:
        return "property" in self.kinds and "class" not in self.kinds

    def priority(self) -> tuple:
        verdict = (self.audit or {}).get("verdict")
        group = 0 if verdict == "incoherente" else 1 if self.kind == REVIEWED else 2
        return group, -(self.proposal.get("score") or 0.0), self.name.lower()

    def terms(self) -> set[str]:
        """The DPV terms its options may use: the ones shown for it."""
        return ({*self.candidates, *(m["dpv_iri"] for m in self.current)}
                | ({self.proposal["dpv_iri"]} if self.proposal.get("dpv_iri") else set()))


def left_out(entry: dict, rule) -> str:
    """Why the fixed rule did not accept a suggestion, in Spanish."""
    kinds = set(entry.get("kinds") or [])
    prop = "property" in kinds and "class" not in kinds
    if prop and not rule.property_relations:
        return "las propiedades nunca se alinean solas"
    score = entry.get("score")
    if score is not None and score < rule.min_score:
        return f"su puntaje ({score:.2f}) es menor que {rule.min_score:g}"
    allowed = rule.property_relations if prop else rule.relations
    if allowed is not None and entry.get("relation") not in allowed:
        return "la regla no acepta esa relación"
    return "no pasó la regla"


def find_doubts(entries: list[dict], state: dict[str, dict[str, dict]], *, info=None,
                audits=None, rule=None, skip=(), candidates: int = 5) -> list[Doubt]:
    """The doubts of an upload, in the order they are asked (ids D1, D2...).

    ``entries``: the decider's entries (status accepted / suggested...); ``state``: what the
    ontology has now (``user_decisions.replay``); ``info``: key -> {label, definition};
    ``audits``: key -> the AI reviewer's answer; ``skip``: what the person already decided."""
    from src.alignment import propose as dec
    from src.alignment.decider import rule_settings
    from src.validation.ai_auditor import DOUBTFUL, INCOHERENT
    rule = rule or rule_settings()
    info, audits, skip = info or {}, audits or {}, set(skip)
    out = []
    for e in entries:
        if e["key"] in skip:
            continue
        audit = audits.get(e["key"])
        if e["status"] == dec.SUGGESTED and e.get("relation") != "skos:relatedMatch":
            kind, why = SUGGESTED, left_out(e, rule)
        elif e["status"] == dec.ACCEPTED and (audit or {}).get("verdict") in (INCOHERENT,
                                                                              DOUBTFUL):
            kind, why = REVIEWED, ""
        else:
            continue
        shown = [*([e["dpv_iri"]] if e.get("dpv_iri") else []),
                 *(c for c in e.get("candidates") or [] if c != e.get("dpv_iri"))]
        about = info.get(e["key"], {})
        if audit is not None:                    # a workspace says ai_relation, the cache relation
            audit = {"verdict": audit.get("verdict"),
                     "relation": audit.get("relation") or audit.get("ai_relation"),
                     "explanation": audit.get("explanation")}
        out.append(Doubt(id="", key=e["key"], name=e["name"], kinds=tuple(e.get("kinds") or ()),
                         kind=kind, label=about.get("label"), definition=about.get("definition"),
                         current=ud._listing(state.get(e["key"], {})),
                         proposal={k: e.get(k) for k in ("dpv_iri", "dpv_name", "relation",
                                                         "score", "confidence",
                                                         "justification")},
                         audit=audit if kind == REVIEWED else None, why=why,
                         candidates=shown[:candidates]))
    out.sort(key=Doubt.priority)
    for n, d in enumerate(out, start=1):
        d.id = f"D{n}"
    return out


# --------------------------------------------------------------------------- asking the AI

def _one_line(text, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit].rstrip() if limit else text


def _short(iri: str) -> str:
    return re.split(r"[#/:]", iri)[-1]


def _relation_text(relation: str | None) -> str:
    return f"{_short(relation)} ({RELATION_WORDS[relation]})" if relation in RELATION_WORDS \
        else "sin relación SKOS"


def doubt_body(d: Doubt, terms: dict, chars: int = 300) -> str:
    """The text the AI sees for one doubt, without its id."""
    lines = [f"Concepto: {d.name} ({'propiedad' if d.is_property else 'clase'})",
             f"Etiqueta: {_one_line(d.label) or '-'}",
             f"Definicion: {_one_line(d.definition, chars) or '-'}"]
    now = "; ".join(f"{_relation_text(m['relation'])} {m['dpv_name']}, decidida por "
                    f"{ud.WHO.get(m['decided_by'], m['decided_by'])}" for m in d.current)
    lines.append(f"Estado actual: {now or 'sin correspondencia'}")
    p = d.proposal
    if d.kind == SUGGESTED:
        score = f", puntaje {p['score']:.2f}" if p.get("score") is not None else ""
        lines.append(f"Duda: sugerida. La IA del sistema eligio "
                     f"{_relation_text(p.get('relation'))} {p.get('dpv_name')} con confianza "
                     f"{p.get('confidence') or '-'}{score}, "
                     f"pero la regla no la acepto: {d.why}.")
    else:
        a = d.audit or {}
        use = f" (relacion que usaria: {a['relation']})" if a.get("relation") else ""
        lines.append(f"Duda: revisada. La IA revisora la marco como {a.get('verdict')}: "
                     f"{_one_line(a.get('explanation')) or '-'}{use}")
    if p.get("justification"):
        lines.append(f"Motivo de la IA del sistema: {_one_line(p['justification'], chars)}")
    lines.append("Candidatos DPV:")
    for n, iri in enumerate(d.candidates, start=1):
        t = terms.get(iri)
        if t is None:
            lines.append(f"{n}. {_short(iri)}")
            continue
        lines.append(f"{n}. {t.name} | {t.label or '-'} | definicion: "
                     f"{_one_line(t.definition, 200) or '-'} | padres en el DPV: "
                     f"{', '.join(t.parents) or '-'}")
    return "\n".join(lines)


def cache_key(context: str, model: str | None, generation: dict | None, prompt) -> str:
    content = {"prompt": prompt.sha256, "model": model, "body": context}
    if generation:
        content["generation"] = generation
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_answer(raw: str) -> dict:
    """{"preguntas": [...], "sin_pregunta": [...]} from the answer (JSON, maybe in fences)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    data = None
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if not isinstance(data, dict) or not isinstance(data.get("preguntas"), list):
        raise ValueError("La respuesta del LLM no trae la lista de preguntas en JSON.")
    return {"preguntas": [q for q in data["preguntas"] if isinstance(q, dict)],
            "sin_pregunta": [s for s in data.get("sin_pregunta") or [] if isinstance(s, dict)]}


@dataclass
class AskReport:
    answers: list[dict] = field(default_factory=list)       # one parsed answer per batch
    calls: int = 0
    cached: int = 0
    pending: int = 0                                         # doubts of batches not answered
    quota_exhausted: bool = False
    call_limit_reached: bool = False
    errors: list[str] = field(default_factory=list)

    def ai(self) -> dict:
        return {"calls": self.calls, "cached": self.cached, "pending": self.pending,
                "quota_exhausted": self.quota_exhausted,
                "call_limit_reached": self.call_limit_reached, "errors": self.errors[:3]}


def ask_ai(doubts: list[Doubt], targets, *, llm=None, settings: AskSettings | None = None,
           cache_path=None, model_name: str | None = None, sleep_fn=time.sleep,
           prompt=None) -> AskReport:
    """One call per batch of doubts (cache first, within ``max_calls``); without ``llm`` only
    the cache answers."""
    from src.alignment.decider import generation_settings
    from src.alignment.justify import (QUOTA_WAIT_SECONDS, TRANSIENT_WAIT_SECONDS,
                                       _is_transient_error)
    from src.core.extract import _is_quota_error
    from src.transform.labels import _append_cache, _load_cache
    settings = settings or AskSettings()
    prompt = prompt or ask_prompt(settings)
    model_name = model_name or getattr(llm, "model", None) or os.environ.get("LLM_MODEL")
    generation = generation_settings(llm)
    terms = {t.iri: t for t in targets.targets}
    cache = _load_cache(cache_path)
    report = AskReport()
    for start in range(0, len(doubts), settings.batch_size):
        batch = doubts[start:start + settings.batch_size]
        context = "\n\n".join(f"=== {d.id} ===\n{doubt_body(d, terms, settings.definition_chars)}"
                              for d in batch)
        key = cache_key(context, model_name, generation, prompt)
        if key in cache:
            report.answers.append(cache[key]["answer"])
            report.cached += 1
            continue
        if llm is None or report.quota_exhausted:
            report.pending += len(batch)
            continue
        if report.calls >= settings.max_calls:
            report.call_limit_reached = True
            report.pending += len(batch)
            continue
        if report.calls and settings.pause_seconds:
            sleep_fn(settings.pause_seconds)
        raw, err = None, None
        for attempt in range(2):                       # one retry after a 429 or a 503
            try:
                raw = llm.ask(prompt=prompt.text, context=context,
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
        answer = None
        if err is None:
            try:
                answer = parse_answer(raw)
            except ValueError as exc:
                err = str(exc)
        if err is not None:
            report.errors.append(err[:300])
            report.pending += len(batch)
            report.quota_exhausted = _is_quota_error(err)
            continue
        report.answers.append(answer)
        _append_cache(cache_path, [{"cache_key": key, "doubts": [d.key for d in batch],
                                    "answer": answer, "prompt": prompt.manifest(),
                                    "model": model_name,
                                    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime())}])
    return report


# --------------------------------------------------------------------------- the fixed rule

@dataclass
class Option:
    text: str
    checked: list[ud.Checked]

    def after(self) -> dict[str, list[dict]]:
        return {c.concept_key: c.after for c in self.checked}


@dataclass
class Question:
    id: str
    doubts: list[Doubt]
    text: str
    options: list[Option]                     # without "Dejar como está" (always the last)
    recommended: int | None = None            # index in options (only the AI recommends)
    reason: str = ""                          # why it is asked (what the AIs said)

    def priority(self) -> tuple:
        return min(d.priority() for d in self.doubts)


@dataclass
class Skipped:
    doubt: Doubt
    by: str                                   # BY_AI | BY_RULE | BY_LIMIT | NO_ANSWER
    reason: str


@dataclass
class QuestionSet:
    questions: list[Question] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    refused: list[dict] = field(default_factory=list)       # options the rule left out
    doubts: int = 0
    ai: dict = field(default_factory=dict)
    prompt: dict = field(default_factory=dict)
    model: str | None = None

    @property
    def complete(self) -> bool:
        return not self.ai.get("pending")


def _change(d: Doubt, raw: dict, targets) -> tuple[ud.Change | None, str]:
    action = _one_line(raw.get("accion")).lower()
    if action == REMOVE:
        return ud.Change(d.key, ud.REMOVE), ""
    if action != ALIGN:
        return None, f"«{action or '-'}» no es alinear ni quitar"
    name, relation = _one_line(raw.get("termino")), _one_line(raw.get("relacion"))
    if not name or not relation:
        what = "término" if not name else "relación"
        return None, f"no dice con qué {what} alinear «{d.name}»"
    term = ud._term(targets, name)
    if term is not None and term.iri not in d.terms():
        return None, (f"«{term.name}» no es uno de los términos que se mostraron para "
                      f"«{d.name}»")
    return ud.Change(d.key, ud.SET, relation=relation, dpv=term.iri if term else name), ""


def _option(raw: dict, ids: dict[str, Doubt], index, state, targets) -> tuple[Option | None,
                                                                              str]:
    changes, seen = [], set()
    for c in raw.get("cambios") or []:
        if not isinstance(c, dict):
            continue
        d = ids.get(_one_line(c.get("duda")).upper())
        if d is None:
            return None, "cambia un concepto que no es de la pregunta"
        if d.id in seen:
            return None, f"cambia dos veces «{d.name}»"
        change, why = _change(d, c, targets)
        if change is None:
            return None, why
        seen.add(d.id)
        changes.append(change)
    if not changes:
        return None, "no cambia nada"
    checked = ud.check_state(index, state, changes, targets)
    bad = next((c for c in checked if not c.ok), None)
    if bad is not None:
        return None, bad.reason
    return Option(_one_line(raw.get("texto"), 120) or "Aplicar el cambio", checked), ""


def validate(answers: list[dict], doubts: list[Doubt], index, state, targets, *,
             max_questions: int = 10) -> tuple[list[Question], list[Skipped], list[dict]]:
    """The questions of the AI after the fixed rule, in the fixed order, at most
    ``max_questions``; what was not asked, with who left it out and why."""
    by_id = {d.id: d for d in doubts}
    used: set[str] = set()
    questions, skipped, refused = [], [], []
    said: dict[str, str] = {}
    for answer in answers:
        for s in answer.get("sin_pregunta", []):
            did = _one_line(s.get("duda")).upper()
            if did in by_id:
                said.setdefault(did, _one_line(s.get("motivo"), 300) or "la IA no preguntó")
        for raw in answer.get("preguntas", []):
            names = raw.get("dudas") if isinstance(raw.get("dudas"), list) else []
            ids = list(dict.fromkeys(_one_line(i).upper() for i in names))
            own = {i: by_id[i] for i in ids if _ID.match(i) and i in by_id and i not in used}
            text = _one_line(raw.get("pregunta"), 400)
            if not own or not text:
                continue
            options, kept = [], {}
            raw_options = raw.get("opciones") if isinstance(raw.get("opciones"), list) else []
            for n, o in enumerate(raw_options[:3], start=1):          # the prompt asks 1 to 3
                if not isinstance(o, dict):
                    continue
                option, why = _option(o, own, index, state, targets)
                if option is not None and any(x.after() == option.after() for x in options):
                    option, why = None, "repite otra opción"
                if option is None:
                    refused.append({"question": text, "option": _one_line(o.get("texto"), 120),
                                    "reason": why})
                    continue
                kept[n] = len(options)
                options.append(option)
            try:
                rec = kept.get(int(raw.get("recomendada")))
            except (TypeError, ValueError):
                rec = None
            used.update(own)
            if rec is None:
                why = ("la opción recomendada no pasó la regla" if options
                       else "ninguna opción pasó la regla")
                skipped += [Skipped(d, BY_RULE, why) for d in own.values()]
                continue
            questions.append(Question("", list(own.values()), text, options, rec,
                                      _one_line(raw.get("motivo"), 300)))
    for d in doubts:
        if d.id not in used:
            skipped.append(Skipped(d, BY_AI, said[d.id]) if d.id in said
                           else Skipped(d, NO_ANSWER, "la IA no dijo nada de esta duda"))
    questions.sort(key=Question.priority)
    for q in questions[max_questions:]:
        skipped += [Skipped(d, BY_LIMIT, f"quedó fuera del máximo de {max_questions} preguntas")
                    for d in q.doubts]
    questions = questions[:max_questions]
    for n, q in enumerate(questions, start=1):
        q.id = f"P{n}"
    return questions, skipped, refused


def propose_questions(doubts: list[Doubt], index, state, targets, *, llm=None,
                      settings: AskSettings | None = None, cache_path=None,
                      model_name: str | None = None, sleep_fn=time.sleep) -> QuestionSet:
    """Ask the AI about ``doubts`` and keep what passes the fixed rule."""
    settings = settings or AskSettings()
    prompt = ask_prompt(settings)
    report = ask_ai(doubts, targets, llm=llm, settings=settings, cache_path=cache_path,
                    model_name=model_name, sleep_fn=sleep_fn, prompt=prompt)
    questions, skipped, refused = validate(report.answers, doubts, index, state, targets,
                                           max_questions=settings.max_questions)
    return QuestionSet(questions, skipped, refused, len(doubts), report.ai(), prompt.manifest(),
                       model_name or getattr(llm, "model", None) or os.environ.get("LLM_MODEL"))


# --------------------------------------------------------------------------- what is asked

def _fixed(d: Doubt) -> tuple[str, str, list[tuple[str, ud.Change]]]:
    """The text, the reasons and the options of the fixed question of a doubt."""
    p = d.proposal
    said = (f" «{_one_line(p['justification'], 300).rstrip('.')}»" if p.get("justification")
            else "")
    if d.kind == SUGGESTED:
        relation = p.get("relation") or DEFAULT_RELATION
        words = RELATION_WORDS[relation]
        return (f"¿«{d.name}» {words} «{p['dpv_name']}» del DPV?",
                f"La IA lo propuso:{said or ' sin motivo'}. La regla no lo aceptó porque "
                f"{d.why}.",
                [(f"Sí, {words} {p['dpv_name']}",
                  ud.Change(d.key, ud.SET, relation=relation, dpv=p["dpv_iri"]))])
    a, m = d.audit or {}, d.current[0]
    options = []
    proposed = f"skos:{a.get('relation')}"
    if proposed in RELATION_WORDS and proposed != m["relation"]:
        words = RELATION_WORDS[f"skos:{a['relation']}"]
        options.append((f"Cambiar a: {words} {m['dpv_name']} (lo que propone la IA revisora)",
                        ud.Change(d.key, ud.SET, relation=a["relation"], dpv=m["dpv_iri"])))
    options.append(("Quitar la correspondencia", ud.Change(d.key, ud.REMOVE)))
    return (f"¿Qué hacemos con «{d.name}»? La regla aceptó que "
            f"{RELATION_WORDS.get(m['relation'], 'tiene relación con')} «{m['dpv_name']}» y la "
            f"IA revisora la marcó como {a.get('verdict')}.",
            f"La IA revisora: «{_one_line(a.get('explanation'), 300).rstrip('.') or '-'}».",
            options)


def fixed_questions(doubts: list[Doubt], index, state, targets, *,
                    max_questions: int = 10) -> QuestionSet:
    """What the uploads ask: one question per doubt, in the fixed order, at most
    ``max_questions``, with fixed options and no recommended one."""
    out = QuestionSet(doubts=len(doubts), prompt={"name": "preguntas fijas"})
    for d in doubts:
        text, reason, raw = _fixed(d)
        options = []
        for label, change in raw:
            checked = ud.check_state(index, state, [change], targets)
            if checked[0].ok:
                options.append(Option(label, checked))
            else:
                out.refused.append({"question": text, "option": label,
                                    "reason": checked[0].reason})
        if not options:
            out.skipped.append(Skipped(d, BY_RULE, out.refused[-1]["reason"]))
            continue
        out.questions.append(Question("", [d], text, options, reason=reason))
    for q in out.questions[max_questions:]:
        out.skipped.append(Skipped(q.doubts[0], BY_LIMIT,
                                   f"quedó fuera del máximo de {max_questions} preguntas"))
    out.questions = out.questions[:max_questions]
    for n, q in enumerate(out.questions, start=1):
        q.id = f"P{n}"
    return out


# --------------------------------------------------------------------------- answering

def answered(state: dict[str, dict[str, dict]], options: list[Option]) -> dict:
    """``state`` with the chosen options applied (each question changes its own concepts)."""
    out = {k: dict(v) for k, v in state.items()}
    for option in options:
        for key, after in option.after().items():
            out[key] = {m["dpv_iri"]: {k: m[k] for k in ("relation", "dpv_name", "decided_by",
                                                          "version")} for m in after}
    return out


# --------------------------------------------------------------------------- console

def _change_line(c: ud.Checked) -> str:
    return f"{c.concept_name}: {ud.state_text(c.before)} -> {ud.state_text(c.after)}"


def render_question(q: Question, *, chosen: int | None = None) -> list[str]:
    """A question as the console shows it; ``chosen``: the option a simulated person took
    (len(options) = "Dejar como está")."""
    lines = [f"  {q.id}. {q.text}"]
    letters = "abcd"
    for n, o in enumerate([*q.options, None]):
        tags = [t for t, on in (("recomendada", q.recommended is not None
                                 and n == q.recommended), ("elegida", chosen == n)) if on]
        tag = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"     {letters[n]}) {o.text if o else LEAVE}{tag}")
        if o is not None:
            lines += [f"        {_change_line(c)}" for c in o.checked]
    if q.reason:
        label = "Por qué" if q.recommended is not None else "Por qué se pregunta"
        lines.append(f"     {label}: {q.reason}")
    return lines


def summary_text(qs: QuestionSet) -> str:
    joined = sum(len(q.doubts) > 1 for q in qs.questions)
    by = {b: sum(s.by == b for s in qs.skipped) for b in (BY_AI, BY_RULE, BY_LIMIT, NO_ANSWER)}
    return (f"{counted(qs.doubts, 'duda', 'dudas')}, "
            f"{counted(len(qs.questions), 'pregunta', 'preguntas')} "
            f"({joined} juntan varias dudas); sin pregunta: {by[BY_AI]} por la IA, "
            f"{by[BY_RULE]} por la regla, {by[BY_LIMIT]} por el máximo"
            + (f", {by[NO_ANSWER]} sin respuesta" if by[NO_ANSWER] else ""))


def question_json(q: Question, chosen: dict | None = None) -> dict:
    return {"id": q.id, "doubts": [d.key for d in q.doubts], "names": [d.name for d in q.doubts],
            "kinds": [d.kind for d in q.doubts], "text": q.text, "reason": q.reason,
            "recommended": q.recommended,
            "options": [{"text": o.text, "changes": [_change_line(c) for c in o.checked]}
                        for o in q.options],
            **(chosen or {})}
