"""Automatic proposals for the concepts that inherited nothing (S6-T06).

After S6-T05 every user concept that is the same as a base concept already carries a validated
decision. For the rest the system proposes ON ITS OWN, without a person, so it is strict:

1. Candidates: the Sprint 4 ranking (0.4 lexical + 0.6 embeddings, top-3 DPV terms of a
   compatible kind), computed only for these concepts and saved in the workspace
   (``work/alignment-candidates.json``; the base file in data/output is never touched).
2. Threshold: only the best candidate (rank 1) can become a proposal, and only when its score
   reaches ``auto.threshold`` (config.yaml), calibrated with the base in ``calibrate``: 0.75
   gave 82 % precision (28 of 34) with the AI filter. Below it: "sin propuesta".
3. The AI (one candidate per concept, in batches, ``auto.llm_max_calls`` calls at most):
   proposes the SKOS type and a one-line justification. If it says "none", the proposal is
   dropped (that filter raised precision from 78 % to 82 % in the calibration). If there is
   no AI (``--sin-ia``, no API key, quota exhausted), the proposal stays WITHOUT a type; the
   output module will write it as ``rdfs:seeAlso`` (S6-T07), which claims no SKOS relation.
   The run resumes: rows already typed are never sent again (S4-T08 code, reused).

Nothing here is validated: every proposal is labelled "automática, no validada" with its score
and, when there is one, the AI's justification. The result goes to ``work/proposals.json``.

CLI: ``py -m src.alignment.propose ARCHIVO [--sin-ia]`` (runs S6-T05 first if needed).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.alignment.calibrate import auto_settings
from src.alignment.export import (CANDIDATES_JSON, build_candidate_table, load_candidate_file,
                                  write_candidate_files)
from src.alignment.justify import RELATION_UNTYPED, justify_candidates, justify_settings
from src.ingest.ontology_loader import load_ontology
from src.alignment.sources import build_alignment_sources

RESULT_FILE = "proposals.json"
STATUS_PROPOSED = "proposed"            # best candidate above the threshold, AI agrees (or none)
STATUS_BELOW = "below_threshold"        # best candidate too weak: no proposal
STATUS_AI_NONE = "ai_none"              # the AI said the best candidate is something else
STATUS_LABELS = {STATUS_PROPOSED: "propuesta automática", STATUS_BELOW: "sin propuesta",
                 STATUS_AI_NONE: "descartada por la IA"}
_SKIP_REASONS = {"kind"}                # individuals: never aligned


# --------------------------------------------------------------------------- inputs

def pending_concepts(ws, inherited: dict) -> list:
    """User concepts that inherited nothing (except individuals), as SourceConcepts."""
    keys = {n["key"] for n in inherited.get("not_inherited", [])
            if n.get("reason") not in _SKIP_REASONS}
    report = load_ontology(ws.input_path())
    sources = build_alignment_sources(report, proposals={})   # never the LOPDP proposals
    return replace(sources, concepts=[c for c in sources.concepts if c.key in keys])


def load_dpv_targets(cfg: dict | None = None):
    from src.config import ROOT, load_config
    from src.ingest.dpv_loader import load_dpv
    from src.alignment.dpv_targets import build_dpv_targets
    cfg = cfg if cfg is not None else load_config()
    path = Path((cfg.get("inputs", {}) or {}).get("dpv", "vocab/dpv.ttl"))
    return build_dpv_targets(load_dpv(path if path.is_absolute() else ROOT / path))


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


# --------------------------------------------------------------------------- deciding

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
    """One entry per pending concept: proposed (typed or not), below threshold or AI none."""
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


# --------------------------------------------------------------------------- workspace

def summary(entries: list[dict], ai: dict | None, threshold: float) -> dict:
    by = Counter(e["status"] for e in entries)
    proposed = [e for e in entries if e["status"] == STATUS_PROPOSED]
    return {"concepts": len(entries), "threshold": threshold,
            "proposed": by.get(STATUS_PROPOSED, 0),
            "proposed_typed": sum(e["relation"] is not None for e in proposed),
            "proposed_untyped": sum(e["relation"] is None for e in proposed),
            "below_threshold": by.get(STATUS_BELOW, 0),
            "ai_none": by.get(STATUS_AI_NONE, 0),
            "by_relation": dict(Counter(e["relation"] for e in proposed if e["relation"])),
            "ai": ai or {"available": False, "error": "IA desactivada (--sin-ia)"}}


def propose_for_workspace(ws, *, embed_fn=None, llm=None, use_llm: bool | None = None,
                          targets=None, cfg: dict | None = None, sleep_fn=None) -> dict:
    """Run S6-T06 for a workspace whose inheritance (S6-T05) is done; save and mark steps."""
    from src.alignment.inherit import load_result
    from src.workspace.store import DONE, ERROR, RUNNING
    settings = auto_settings(cfg)
    use_llm = settings["use_llm"] if use_llm is None else use_llm
    inherited = load_result(ws)
    if inherited is None:
        raise ValueError("Primero hay que heredar lo validado (paso S6-T05).")

    ws.mark_step("candidates", RUNNING)
    try:
        sources = pending_concepts(ws, inherited)
        data = (build_candidates(ws, sources, targets=targets, embed_fn=embed_fn, cfg=cfg)
                if sources.concepts else {"rows": [], "metadata": {}})
    except Exception as exc:
        ws.mark_step("candidates", ERROR, str(exc)[:300])
        raise
    ws.mark_step("candidates", DONE, f"{len(sources.concepts)} conceptos sin herencia",
                 concepts=len(sources.concepts))

    ws.mark_step("propose", RUNNING)
    to_ai = mark_for_ai(data, settings["threshold"])
    ai = None
    if use_llm and to_ai:
        js = replace(justify_settings(cfg), max_calls=settings["llm_max_calls"])
        ai = ask_ai(ws, data, llm=llm, settings=js, sleep_fn=sleep_fn)
    elif use_llm:
        ai = {"available": True, "calls": 0, "rows_typed": 0}
    entries = decide(data, settings["threshold"])
    result = {"summary": summary(entries, ai, settings["threshold"]), "proposals": entries}
    (ws.work_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    s = result["summary"]
    ws.mark_step("propose", DONE, f"{s['proposed']} propuestas automáticas de {s['concepts']}",
                 **{k: s[k] for k in ("proposed", "proposed_typed", "proposed_untyped",
                                      "below_threshold", "ai_none")})
    return result


def load_result(ws) -> dict | None:
    path = ws.work_dir / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- console

def render_console(result: dict, examples: int = 8) -> str:
    s = result["summary"]
    ai = s["ai"]
    lines = ["Propuestas automáticas (S6-T06)",
             f"  Conceptos sin herencia: {s['concepts']}   umbral: {s['threshold']:.2f}",
             f"  Propuestas automáticas: {s['proposed']} ({s['proposed_typed']} con tipo SKOS "
             f"de la IA, {s['proposed_untyped']} sin tipo)",
             f"  Sin propuesta (mejor candidato bajo el umbral): {s['below_threshold']}",
             f"  Descartadas porque la IA dijo 'none': {s['ai_none']}"]
    if ai.get("available"):
        extra = " (cuota agotada: vuelve a correr mañana para tipar el resto)" \
            if ai.get("quota_exhausted") else ""
        lines.append(f"  IA: {ai.get('calls', 0)} llamadas, {ai.get('rows_typed', 0)} "
                     f"tipadas{extra}")
    else:
        lines.append(f"  IA: no usada ({ai.get('error')})")
    shown = [e for e in result["proposals"] if e["status"] == STATUS_PROPOSED][:examples]
    if shown:
        lines.append("  Propuestas (automáticas, no validadas):")
    for e in shown:
        rel = e["relation"] or "sin tipo (rdfs:seeAlso)"
        lines.append(f"    {e['name']} -> dpv:{e['dpv_name']}  {rel}  (puntaje {e['score']:.2f})")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import inherit_for_workspace, load_result as inherited_result
    from src.workspace.store import WorkspaceError, open_ontology
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.propose",
        description="Propuestas automáticas para lo que no se heredó (S6-T06).")
    parser.add_argument("file", help="ontología .rdf, .owl, .ttl o .jsonld")
    parser.add_argument("--sin-ia", action="store_true", help="no llamar al LLM (sin tipo SKOS)")
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