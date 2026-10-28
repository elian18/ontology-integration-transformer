from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.alignment.calibrate import auto_settings
from src.alignment.export import (CANDIDATES_JSON, build_candidate_table, load_candidate_file,
                                  write_candidate_files)
from src.alignment.justify import RELATION_UNTYPED, justify_candidates
from src.ingest.ontology_loader import load_ontology
from src.alignment.sources import build_alignment_sources

# Sprint 6 proposer (kept for the S6-T09 check).
STATUS_PROPOSED = "proposed"            # best candidate above the threshold, AI agrees (or none)
STATUS_BELOW = "below_threshold"        # best candidate too weak: no proposal
STATUS_AI_NONE = "ai_none"              # the AI said the best candidate is something else
_ALIGNABLE = {"class", "property"}      # individuals: never aligned

# S7-T06: the decider in the user's flow.
RESULT_FILE = "alignment-decisions.json"
METHOD, METHOD_VERSION = "decider", 1
ALIGNMENT_VERSION = 1                   # first alignment of an upload (Sprint 10 adds versions)
ACCEPTED = "accepted"                   # written into the ontology
SUGGESTED = "suggested"                 # high confidence, rule left it out: never written
NO_MATCH = "no_match"                   # the LLM said "ninguno" with an accepted confidence
NO_PROPOSAL = "no_proposal"             # decided, but nothing the rule accepts or suggests
UNDECIDED = "undecided"                 # the AI could not decide and the rule without AI failed
STATUS_LABELS = {ACCEPTED: "aceptada automáticamente",
                 SUGGESTED: "sugerida (no entra a la ontología)",
                 NO_MATCH: "sin correspondencia (decidido por la regla)",
                 NO_PROPOSAL: "sin propuesta",
                 UNDECIDED: "sin decidir (la IA no estuvo disponible)"}
BY_RULE, BY_RULE_WITHOUT_AI = "rule", "rule_without_ai"
DECIDED_BY_LABELS = {"base": "persona (base)", BY_RULE: "regla",
                     BY_RULE_WITHOUT_AI: "regla sin IA"}


# --------------------------------------------------------------------------- inputs

def pending_concepts(ws, inherited: dict, definitions: dict | None = None) -> list:
    """User classes and properties that inherited nothing, as SourceConcepts (individuals are
    never aligned; a property also counts when the base had no concept of its kind, which the
    inheritance reports with the same reason as an individual). ``definitions``: those the law
    gave to the concepts that had none ("ontología + ley")."""
    from src.alignment.law_definitions import with_definitions
    keys = {n["key"] for n in inherited.get("not_inherited", [])}
    report = load_ontology(ws.input_path())
    sources = build_alignment_sources(report, proposals={})   # never the LOPDP proposals
    concepts = [c for c in sources.concepts if c.key in keys and set(c.kinds) & _ALIGNABLE]
    return replace(sources, concepts=with_definitions(concepts, definitions))


def load_dpv_targets(cfg: dict | None = None):
    from src.config import ROOT, load_config
    from src.ingest.dpv_loader import load_dpv
    from src.alignment.dpv_targets import build_dpv_targets
    cfg = cfg if cfg is not None else load_config()
    path = Path((cfg.get("inputs", {}) or {}).get("dpv", "vocab/dpv.ttl"))
    return build_dpv_targets(load_dpv(path if path.is_absolute() else ROOT / path))


# --------------------------------------------------------------------------- Sprint 6 proposer

def build_candidates(ws, sources, *, targets=None, embed_fn=None, cfg=None) -> dict:
    """Rank the DPV candidates of the pending concepts, or reuse the ones already saved."""
    path = ws.work_dir / CANDIDATES_JSON
    wanted = {c.key for c in sources.concepts}
    if path.exists():
        data = load_candidate_file(path)
        if {r["concept_key"] for r in data.get("rows", [])} == wanted:
            return data                                # same concepts: keep (and its AI types)
    targets = targets if targets is not None else load_dpv_targets(cfg)
    table = build_candidate_table(sources, targets, embed_fn=embed_fn)
    for row in table.rows:
        row["needs_justification"] = False             # set later, only for the proposals
    write_candidate_files(table, ws.work_dir)
    return load_candidate_file(path)


def best_rows(data: dict) -> dict[str, dict]:
    return {r["concept_key"]: r for r in data.get("rows", []) if r.get("rank") == 1}


def mark_for_ai(data: dict, threshold: float) -> int:
    """Only the rank-1 row of concepts above the threshold goes to the AI."""
    n = 0
    for row in data.get("rows", []):
        row["needs_justification"] = bool(row.get("rank") == 1
                                          and float(row.get("score") or 0) >= threshold)
        n += row["needs_justification"]
    return n


def ask_ai(ws, data: dict, *, llm=None, settings=None, sleep_fn=None) -> dict:
    """Type the marked rows with the AI (resumable). Returns what happened, never raises."""
    kwargs = {"settings": settings, "out_dir": ws.work_dir}
    if sleep_fn is not None:
        kwargs["sleep_fn"] = sleep_fn
    try:
        report = justify_candidates(data, llm=llm, **kwargs)
    except KeyError as exc:                            # LLM_API_KEY / LLM_MODEL missing
        return {"available": False, "error": f"Falta la variable {exc} en .env"}
    except Exception as exc:                           # noqa: BLE001 - report, don't crash
        return {"available": False, "error": str(exc)[:300]}
    return {"available": True, "calls": report.calls, "rows_typed": report.rows_filled,
            "quota_exhausted": report.quota_exhausted,
            "call_limit_reached": report.call_limit_reached,
            "remaining": report.remaining_concepts,
            "errors": [e["error"] for e in report.errors][:3]}


def decide(data: dict, threshold: float) -> list[dict]:
    """Sprint 6: one entry per pending concept (proposed, below threshold or AI none)."""
    out = []
    for key, row in best_rows(data).items():
        score = float(row.get("score") or 0)
        relation = row.get("proposed_relation")
        if score < threshold:
            status = STATUS_BELOW
        elif relation == "none":
            status = STATUS_AI_NONE
        else:
            status = STATUS_PROPOSED
        typed = status == STATUS_PROPOSED and relation not in (None, RELATION_UNTYPED)
        out.append({
            "key": key, "name": row.get("concept_name"), "kinds": row.get("concept_kinds"),
            "status": status, "dpv_iri": row["dpv_iri"], "dpv_name": row.get("dpv_name"),
            "dpv_label": row.get("dpv_label"), "score": round(score, 4),
            "relation": relation if typed else None,
            "justification": row.get("justification") if status != STATUS_BELOW else None,
            "validated": False,
        })
    return sorted(out, key=lambda e: (-e["score"], e["name"] or ""))


# --------------------------------------------------------------------------- S7-T06

def _llm_for(use_llm: bool, llm):
    """The LLM to use (None = none) and what the result says about it."""
    if not use_llm:
        return None, {"available": False, "error": "IA desactivada (--sin-ia)"}
    if llm is not None:
        return llm, {"available": True}
    try:
        from src.ai.llm_client import LLMClient
        return LLMClient(), {"available": True}
    except KeyError as exc:                            # LLM_API_KEY / LLM_MODEL missing
        return None, {"available": False, "error": f"Falta la variable {exc} en .env"}


def entries_from(decisions, rule, threshold: float, targets=None) -> list[dict]:
    """What the rule (or, without AI, the rule of Sprint 6) does with each decision."""
    from src.alignment.decider import STATUS_PENDING, apply_rule, suggested
    auto = apply_rule(decisions, **rule.kwargs())
    hints = suggested(decisions, auto, rule.suggest)
    names = {t.iri: t.name for t in getattr(targets, "targets", [])}
    out = []
    for d in decisions:
        entry = {"key": d.key, "name": d.name, "kinds": list(d.kinds), "status": NO_PROPOSAL,
                 "decided_by": BY_RULE, "dpv_iri": None, "dpv_name": None, "relation": None,
                 "score": d.score, "rank": d.rank, "confidence": d.confidence,
                 "justification": d.justification, "decision": d.status,
                 "candidates": list(d.candidates), "precedents": list(d.precedents),
                 "from_cache": d.from_cache, "issues": list(d.issues), "model": d.model,
                 "version": ALIGNMENT_VERSION}
        if d.key in auto:
            if auto[d.key]["none"]:
                entry["status"] = NO_MATCH
            else:
                entry.update(status=ACCEPTED, dpv_iri=d.dpv_iri, dpv_name=d.dpv_name,
                             relation=d.relation)
        elif d.key in hints:
            entry.update(status=SUGGESTED, dpv_iri=d.dpv_iri, dpv_name=d.dpv_name,
                         relation=d.relation)
        elif d.status == STATUS_PENDING:               # the AI could not decide it
            entry.update(decided_by=BY_RULE_WITHOUT_AI, score=d.best_score, rank=1,
                         confidence=None, justification=None)
            never = d.is_property and rule.property_relations == ()
            if not never and d.candidates and (d.best_score or 0.0) >= threshold:
                entry.update(status=ACCEPTED, dpv_iri=d.candidates[0],
                             dpv_name=names.get(d.candidates[0],
                                                d.candidates[0].rsplit("#", 1)[-1]))
            else:
                entry["status"] = UNDECIDED
        out.append(entry)
    order = list(STATUS_LABELS)
    return sorted(out, key=lambda e: (order.index(e["status"]), -(e["score"] or 0.0),
                                      e["name"] or ""))


def decision_summary(entries: list[dict], ai: dict, rule, threshold: float) -> dict:
    by = Counter(e["status"] for e in entries)
    accepted = [e for e in entries if e["status"] == ACCEPTED]
    return {"concepts": len(entries),
            "accepted": by.get(ACCEPTED, 0),
            "accepted_by_rule": sum(e["decided_by"] == BY_RULE for e in accepted),
            "accepted_without_ai": sum(e["decided_by"] == BY_RULE_WITHOUT_AI for e in accepted),
            "suggested": by.get(SUGGESTED, 0), "no_match": by.get(NO_MATCH, 0),
            "no_proposal": by.get(NO_PROPOSAL, 0), "undecided": by.get(UNDECIDED, 0),
            "pending_ai": sum(e["decision"] == "pending" for e in entries),
            "by_relation": dict(Counter(e["relation"] or "rdfs:seeAlso" for e in accepted)),
            "rule": rule.label(), "threshold_without_ai": threshold, "ai": ai}


def is_current(result: dict | None, cfg: dict | None = None, use_llm: bool | None = None,
               law_definitions: str | None = None) -> bool:
    """True if a saved result can be reused: same method, prompt, rule and definitions from the
    law (their fingerprint), and, when the AI is to be used, no concept left undecided."""
    if not result or result.get("method") != METHOD:
        return False
    settings = auto_settings(cfg)
    use_llm = settings["use_llm"] if use_llm is None else use_llm
    from src.alignment.decider import decide_prompt, rule_settings
    return (result.get("method_version") == METHOD_VERSION
            and (result.get("prompt") or {}).get("sha256") == decide_prompt(cfg).sha256
            and result.get("rule") == rule_settings(cfg).as_dict()
            and result.get("threshold_without_ai") == settings["threshold"]
            and result.get("law_definitions") == law_definitions
            and not (use_llm and result["summary"].get("pending_ai")))


def propose_for_workspace(ws, *, embed_fn=None, llm=None, use_llm: bool | None = None,
                          targets=None, cfg: dict | None = None, sleep_fn=None, base=None,
                          chroma_path=None) -> dict:
    """Run S7-T06 for a workspace whose inheritance (S6-T05) is done; save and mark steps."""
    from src.alignment.decider import CACHE_FILE, decide as decide_all, decide_prompt
    from src.alignment.decider import rule_settings, settings_from_config
    from src.alignment import law_definitions as ld
    from src.alignment.inherit import load_result
    from src.alignment.knowledge import build_knowledge
    from src.workspace.store import DONE, ERROR, RUNNING
    settings = auto_settings(cfg)
    rule = rule_settings(cfg)
    prompt = decide_prompt(cfg)
    use_llm = settings["use_llm"] if use_llm is None else use_llm
    inherited = load_result(ws)
    if inherited is None:
        raise ValueError("Primero hay que heredar lo validado (paso S6-T05).")

    definitions = ld.for_decider(ld.load_result(ws), cfg)
    ws.mark_step("candidates", RUNNING)
    try:
        sources = pending_concepts(ws, inherited, definitions)
        if sources.concepts:
            targets = targets if targets is not None else load_dpv_targets(cfg)
            build_knowledge(cfg, embed_fn=embed_fn, targets=targets, base=base,
                            path=chroma_path)
    except Exception as exc:
        ws.mark_step("candidates", ERROR, str(exc)[:300])
        raise
    ws.mark_step("candidates", DONE, f"{len(sources.concepts)} conceptos sin herencia",
                 concepts=len(sources.concepts))

    ws.mark_step("propose", RUNNING)
    try:
        llm, ai = _llm_for(use_llm, llm) if sources.concepts else (llm, {"available": use_llm})
        decisions = []
        if sources.concepts:
            run = settings_from_config(cfg)
            if llm is None:
                run = replace(run, max_calls=0)        # only the cache answers
            kwargs = {"sleep_fn": sleep_fn} if sleep_fn is not None else {}
            report = decide_all(sources.concepts, targets, llm=llm, embed_fn=embed_fn,
                                settings=run, cache_path=ws.work_dir / CACHE_FILE,
                                chroma_path=chroma_path, cfg=cfg, prompt=prompt, **kwargs)
            decisions = report.decisions
            ai.update(model=decisions[0].model if decisions else None, calls=report.calls,
                      cached=report.cached)
            if ai["available"]:
                ai.update(quota_exhausted=report.quota_exhausted,
                          call_limit_reached=report.call_limit_reached,
                          errors=[e["error"] for e in report.errors][:3])
        entries = entries_from(decisions, rule, settings["threshold"], targets)
    except Exception as exc:
        ws.mark_step("propose", ERROR, str(exc)[:300])
        raise
    result = {"method": METHOD, "method_version": METHOD_VERSION,
              "alignment_version": ALIGNMENT_VERSION, "rule": rule.as_dict(),
              "threshold_without_ai": settings["threshold"], "prompt": prompt.manifest(),
              "law_definitions": ld.fingerprint(definitions),
              "summary": decision_summary(entries, ai, rule, settings["threshold"]),
              "entries": entries}
    (ws.work_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    s = result["summary"]
    ws.mark_step("propose", DONE, f"{s['accepted']} aceptadas y {s['suggested']} sugeridas de "
                                  f"{s['concepts']}",
                 **{k: s[k] for k in ("accepted", "accepted_without_ai", "suggested",
                                      "no_match", "no_proposal", "undecided")})
    return result


def load_result(ws) -> dict | None:
    path = ws.work_dir / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- console

def render_console(result: dict, examples: int = 8) -> str:
    s = result["summary"]
    ai = s["ai"]
    lines = ["Decisor en tu ontología (S7-T06)",
             f"  Conceptos sin herencia: {s['concepts']}",
             f"  Regla: {s['rule']}",
             f"  Aceptadas automáticamente: {s['accepted']}"
             + (f" ({s['accepted_without_ai']} con la regla sin IA: puesto 1 y puntaje >= "
                f"{s['threshold_without_ai']:g}, sin tipo SKOS)" if s["accepted_without_ai"]
                else ""),
             f"  Sugeridas (no entran a la ontología): {s['suggested']}",
             f"  Sin correspondencia (la IA dijo 'ninguno' con confianza alta): "
             f"{s['no_match']}",
             f"  Sin propuesta: {s['no_proposal']}"]
    if s["undecided"]:
        lines.append(f"  Sin decidir (la IA no estuvo disponible): {s['undecided']}")
    if ai.get("available"):
        extra = (" (cuota agotada: vuelve a subir el archivo mañana para decidir el resto)"
                 if ai.get("quota_exhausted") else
                 " (se llegó al límite de llamadas: vuelve a subir el archivo para seguir)"
                 if ai.get("call_limit_reached") else "")
        lines.append(f"  IA: {ai.get('calls', 0)} llamadas, {ai.get('cached', 0)} desde la "
                     f"caché{extra}")
    else:
        lines.append(f"  IA: no usada ({ai.get('error')})")
    for status, title in ((ACCEPTED, "Aceptadas"), (SUGGESTED, "Sugeridas")):
        shown = [e for e in result["entries"] if e["status"] == status][:examples]
        if shown:
            lines.append(f"  {title}:")
        for e in shown:
            rel = e["relation"] or "sin tipo (rdfs:seeAlso)"
            score = f"puntaje {e['score']:.2f}" if e["score"] is not None else "sin puntaje"
            lines.append(f"    {e['name']} -> dpv:{e['dpv_name']}  {rel}  ({score}, "
                         f"{DECIDED_BY_LABELS[e['decided_by']]})")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import inherit_for_workspace, load_result as inherited_result
    from src.workspace.store import WorkspaceError, open_ontology
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.propose",
        description="Alinea lo que no se heredó con el decisor y su regla (S7-T06).")
    parser.add_argument("file", help="ontología .rdf, .owl, .ttl o .jsonld")
    parser.add_argument("--sin-ia", action="store_true",
                        help="no llamar al LLM (solo la caché y la regla sin IA)")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: no existe el archivo {path}")
        return 1
    try:
        ws = open_ontology(path.name, path.read_bytes())
        if inherited_result(ws) is None:
            inherit_for_workspace(ws)
        result = propose_for_workspace(ws, use_llm=False if args.sin_ia else None)
    except (WorkspaceError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(result))
    print(f"  Guardado en: {ws.work_dir / RESULT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())