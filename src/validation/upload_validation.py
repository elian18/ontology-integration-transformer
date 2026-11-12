from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from src.validation import ai_auditor as au
from src.validation import logic_inspector as li
from src.validation import questions as qs
from src.validation import shape_inspector as si
from src.validation.shapes import counted
from src.validation.shape_inspector import BASE, GREEN, NEW, ONTOLOGY, RED, YELLOW

RESULT_FILE = "validation.json"
LOG_FILE = "validation-log.txt"
NOT_RUN, FAILED = "sin revisar", "error"
SHAPES, LOGIC, QUESTIONS, AUDIT = "shapes", "logic", "questions", "audit"
TITLES = {SHAPES: "Forma", LOGIC: "Lógica", QUESTIONS: "Preguntas", AUDIT: "IA revisora"}
WHAT = {SHAPES: ("violación nueva", "violaciones nuevas"),
        LOGIC: ("inconsistencia nueva", "inconsistencias nuevas")}
FLAGGED = (au.INCOHERENT, au.DOUBTFUL)


def overall(statuses) -> str:
    statuses = list(statuses)
    if RED in statuses:
        return RED
    if YELLOW in statuses or FAILED in statuses:
        return YELLOW
    return GREEN


def audit_status(audit: dict | None) -> str:
    if not audit:
        return NOT_RUN
    s = audit["summary"]
    if s[au.INCOHERENT] or s[au.DOUBTFUL]:
        return YELLOW
    if s["accepted"] and s["pending"] == s["accepted"]:
        return NOT_RUN
    return GREEN


def shapes_block(report: dict) -> dict:
    s, e = report["summary"], report["engine"]
    return {"status": report["status"], "engine": f"{e['name']} {e['version']}",
            "new": s["violations"][NEW],
            "inherited": s["violations"][ONTOLOGY] + s["violations"][BASE],
            "warnings": sum(s["warnings"].values()), "completeness": report["completeness"],
            "seconds": report["seconds"]}


def logic_block(report: dict) -> dict:
    s, e = report["summary"], report["engine"]
    engine = (f"Apache Jena {e['version']}, {li.REASONERS[e['reasoner']]}" if e["name"] == li.JENA
              else f"owlrl {e['version']}, OWL 2 RL en Python")
    return {"status": report["status"], "engine": engine, "consistent": report["consistent"],
            "new": s["errors"][NEW], "inherited": s["errors"][ONTOLOGY] + s["errors"][BASE],
            "warnings": sum(s["warnings"].values()), "seconds": report["seconds"]}


def questions_block(report: dict) -> dict:
    """Each generic question with its color and a short answer (the rows stay in work/)."""
    return {"status": report["status"], "engine": qs.engine_short(report["engine"]),
            "summary": report["summary"], "catalog": report["catalog"],
            "seconds": report["seconds"], "ai": report["explanations"],
            "results": [{"id": r["id"], "question": r["question"], "criterion": r["criterion"],
                         "color": r["color"], "applies": r.get("applies", True),
                         "answer": qs.answer_text(r),
                         **({k: r[k] for k in ("explanation", "explained_by", "matches")
                             if k in r})} for r in report["results"]]}


def audit_block(audit: dict | None) -> dict:
    if not audit:
        return {"status": NOT_RUN}
    return {"status": audit_status(audit), "summary": audit["summary"], "ai": audit["ai"],
            "prompt": audit["prompt"],
            "flagged": [{k: e[k] for k in ("name", "dpv_name", "relation", "verdict",
                                            "ai_relation", "explanation")}
                        for e in audit["entries"] if e["verdict"] in FLAGGED]}


def _inspect(ws, step: str, run, block, detail):
    """Run one inspector as a step of the workspace; a failure never blocks the download."""
    from src.workspace.store import DONE, ERROR, RUNNING
    ws.mark_step(step, RUNNING)
    try:
        report = run()
    except Exception as exc:                           # noqa: BLE001 - reported, not raised
        ws.mark_step(step, ERROR, str(exc)[:300])
        return None, {"status": FAILED, "error": str(exc)[:300]}
    out = block(report)
    ws.mark_step(step, DONE, detail(out), result=out["status"])
    return report, out


def counts_text(key: str, block: dict) -> str:
    return (f"{counted(block['new'], *WHAT[key])}, "
            f"{counted(block['inherited'], 'heredada', 'heredadas')}, "
            f"{counted(block['warnings'], 'advertencia', 'advertencias')}")


def validate_workspace(ws, cfg: dict | None = None, *, engine: str | None = None,
                       progress=None, llm=None, use_llm: bool | None = False,
                       sleep_fn=None) -> dict:
    """The form and logic inspectors and the competency questions on what was written, plus the
    AI review already done; saves ``validation.json`` and ``validation-log.txt`` in ``output/``
    (they go in the zip)."""
    from src.workspace.store import STEP_LABELS
    say = progress or (lambda label: None)
    say(STEP_LABELS[SHAPES])
    shape_report, shapes = _inspect(ws, SHAPES, lambda: si.inspect_workspace(ws, cfg),
                                    shapes_block,
                                    lambda o: f"{o['status']}: {counts_text(SHAPES, o)}")
    say(STEP_LABELS[LOGIC])
    logic_report, logic = _inspect(ws, LOGIC, lambda: li.inspect_workspace(ws, cfg, engine),
                                   logic_block,
                                   lambda o: f"{o['status']}: {counts_text(LOGIC, o)}")
    say(STEP_LABELS[QUESTIONS])
    question_report, questions = _inspect(ws, QUESTIONS, lambda: qs.inspect_workspace(
        ws, cfg, llm=llm, use_llm=use_llm, sleep_fn=sleep_fn or time.sleep),
                                          questions_block,
                                          lambda o: f"{o['status']}: "
                                                    f"{qs.counts_text(o['summary'])}")
    audit_result = au.current_result(ws)
    blocks = {SHAPES: shapes, LOGIC: logic, QUESTIONS: questions,
              AUDIT: audit_block(audit_result)}
    result = {"workspace": ws.id,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "status": overall(b["status"] for b in blocks.values()), "inspectors": blocks}
    out = ws.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    logs = [si.render_log(shape_report) if shape_report else _failed(SHAPES, shapes),
            li.render_log(logic_report) if logic_report else _failed(LOGIC, logic),
            qs.render_workspace(question_report) if question_report
            else _failed(QUESTIONS, questions),
            au.render_workspace(audit_result, ws.id, examples=None) if audit_result
            else "Revisión de la IA: no se hizo"]
    (out / LOG_FILE).write_text(f"Validación del espacio {ws.id}: {result['status']}\n\n"
                                + "\n\n".join(logs) + "\n", encoding="utf-8")
    return result


def _failed(key: str, block: dict) -> str:
    return f"{TITLES[key]}: no se pudo correr ({block['error']})"


def load_result(ws) -> dict | None:
    path = ws.output_dir / RESULT_FILE
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    result["status"] = si.current_color(result["status"])          # an S8 result: "ámbar"
    for block in result.get("inspectors", {}).values():
        block["status"] = si.current_color(block["status"])
    return result


# --------------------------------------------------------------------------- console / README

def inspector_text(key: str, block: dict) -> str:
    title = TITLES[key]
    if block["status"] == FAILED:
        return f"{title}: no se pudo correr ({block['error']})"
    if key == QUESTIONS:
        return (f"{title} ({block['engine']}): {block['status']} · "
                f"{qs.counts_text(block['summary'], len(block['results']))}")
    if key != AUDIT:
        return f"{title} ({block['engine']}): {block['status']} · {counts_text(key, block)}"
    if "summary" not in block:
        return f"{title}: {NOT_RUN} · no se hizo la revisión"
    s, status = block["summary"], block["status"]
    title += f" ({block['prompt']['name']}.v{block['prompt']['version']})"
    if not s["accepted"]:
        return (f"{title}: {status} · la regla no aceptó ninguna correspondencia que revisar"
                + by_user_text(s))
    if status == NOT_RUN:
        ai = block["ai"]
        why = ai.get("error") or next(iter(ai.get("errors") or []), "sin respuesta")
        return (f"{title}: {status} · la IA no revisó las {s['accepted']} aceptadas por la "
                f"regla: {why}")
    return (f"{title}: {status} · {counted(s[au.INCOHERENT], 'incoherente', 'incoherentes')} y "
            f"{counted(s[au.DOUBTFUL], 'dudosa', 'dudosas')} de "
            f"{counted(s['accepted'], 'aceptada', 'aceptadas')} por la regla"
            + (f", {s['pending']} sin revisar" if s["pending"] else "")
            + "; solo explica, no cambia lo aceptado" + by_user_text(s))


def by_user_text(summary: dict) -> str:
    """S10-T03: the concepts the person decided on are not reviewed again."""
    n = summary.get("by_user") or 0
    return (f" ({counted(n, 'concepto decidido', 'conceptos decididos')} por el usuario: no "
            + ("se revisa" if n == 1 else "se revisan") + " de nuevo)" if n else "")


def notes(key: str, block: dict) -> list[str]:
    """The lines under an inspector: why each yellow question is yellow (S9-T06)."""
    if key != QUESTIONS or block["status"] == FAILED:
        return []
    return [f"{r['id']} {r['question']} {qs.why_text(r)}" for r in block["results"]
            if r.get("explanation")]


def summary_lines(result: dict) -> list[str]:
    lines = [f"Validación: {result['status']} (no bloquea la descarga)"]
    for k, b in result["inspectors"].items():
        lines += [f"  {inspector_text(k, b)}", *(f"    - {n}" for n in notes(k, b))]
    return lines
