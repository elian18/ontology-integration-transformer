"""Bridge for the page "Preguntas de la base" (S9-T10): what the base answers, read only.

It reads what the commands already saved in data/output (the questions of the base and the bridge
with the GDPR) and never runs them: when a result is missing, it says which command makes it."""
from __future__ import annotations

import json
from pathlib import Path

from src.alignment import gdpr_bridge as gb
from src.validation import question_catalog as qc
from src.validation import question_explainer as qe
from src.validation import questions as qs

from . import align_ontology as ao

RUN_QUESTIONS = "py -m src.validation.questions --base"
RUN_BRIDGE = "py -m src.alignment.gdpr_bridge"
SOURCES = {"paper": "Artículo de OntoPriv", "lopdp": "LOPDP"}
FORMAT_LABELS = {"owl": "OWL (RDF/XML)", "turtle": "Turtle", "jsonld": "JSON-LD"}
COLOR_LABELS = {qs.GREEN: "Verde", qs.YELLOW: "Amarillo", qs.GRAY: "Gris"}
VERDICT_ORDER = {"incoherente": 0, "dudosa": 1, "coherente": 2}
VERDICT_BADGES = {"incoherente": "red", "dudosa": "yellow"}


def base_cfg() -> dict | None:
    """The config of the base (None: config.yaml); the tests point it to a folder of theirs."""
    return None


def output_dir(cfg: dict | None = None) -> Path:
    from src.alignment.inherit import base_paths
    return base_paths(cfg)["candidates"].parent


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _n(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# --------------------------------------------------------------------------- the questions

def origin_text(q: qc.Question | None) -> str | None:
    if q is None:
        return None
    return f"{SOURCES.get(q.source, q.source)} · {q.reference}"


def question_rows(report: dict, catalog: qc.Catalog, group: str) -> list[dict]:
    """Each question of ``group`` as the page shows it."""
    by_id = {q.id: q for q in catalog.questions}
    rows = []
    for r in report["results"]:
        if r["group"] != group:
            continue
        q = by_id.get(r["id"])
        rows.append({"id": r["id"], "question": ao.plain(r["question"]),
                     "badge": ao.question_badge(r), "color": r["color"],
                     "answer": ao.plain(qs.answer_text(r)), "criterion": r["criterion"],
                     "expected": ao.plain(q.expected.text) if q and q.expected else None,
                     "origin": origin_text(q),
                     "why": ao.plain(r["explanation"]) if r.get("explanation") else None,
                     "why_label": ("Por qué (IA)" if r.get("explained_by") == qe.AI
                                   else "Por qué"),
                     "query": ao._query(r["id"])})
    return rows


def group_caption(report: dict, group: str) -> str:
    results = [r for r in report["results"] if r["group"] == group]
    return f"{qs.counts_text(qs.summary(results), len(results))}."


def header_lines(report: dict, catalog: qc.Catalog) -> list[str]:
    cat = report["catalog"]
    when = report["date"][:16].replace("T", " ") + " UTC"
    lines = [f"Corrida del {when} · {qs.engine_text(report['engine'])} · "
             f"{report['seconds']:g} s · {report['triples']} tripletas "
             f"({', '.join(Path(f).name for f in report['files'])})",
             f"Catálogo {cat['file']} v{cat['version']} (sha256 {cat['sha256'][:12]})"]
    ai = report.get("explanations")
    if ai and ai.get("questions"):
        p = ai["prompt"]
        lines.append(f"IA que explica las amarillas (prompt {p['name']}.v{p['version']}): "
                     f"{qe.ai_text(ai)}")
    return lines


def stale(report: dict, catalog: qc.Catalog) -> bool:
    """The catalog changed after the run: the page would show old questions."""
    return report["catalog"]["sha256"] != catalog.sha256


# --------------------------------------------------------------------------- original and formats

def original_view(report: dict) -> dict | None:
    original = report.get("original")
    if not original:
        return None
    base = {r["id"]: r["color"] for r in report["results"]}
    differ = {d["id"] for d in original["differences"]}
    rows = [{"Pregunta": qid, "OntoPriv original": COLOR_LABELS.get(color, color),
             "Base del sistema": COLOR_LABELS.get(base.get(qid), base.get(qid)),
             "Mismas filas": "no" if qid in differ else "sí"}
            for qid, color in original["colors"].items()]
    files = ", ".join(original["files"])
    if differ:
        message = (f"{_n(len(differ), 'pregunta da', 'preguntas dan')} otras filas en el "
                   f"OntoPriv original ({files}) que en la base: {', '.join(sorted(differ))}.")
    else:
        message = (f"Las {len(rows)} preguntas del artículo dan las mismas filas en el OntoPriv "
                   f"original ({files}) que en la base: al transformarlo no se perdió nada de "
                   "lo que OntoPriv sabía responder.")
    return {"message": message, "rows": rows}


def formats_view(report: dict) -> dict | None:
    formats = report.get("formats")
    if formats is None:
        return None
    total = len(report["results"])
    rows = [{"Formato": FORMAT_LABELS.get(key, key),
             "Preguntas con otras filas que en RDF/XML": ", ".join(d["id"] for d in diffs)
             or "ninguna"} for key, diffs in formats.items()]
    if any(formats.values()):
        message = "Algún formato no da las mismas filas que RDF/XML: revisa la tabla."
    else:
        message = (f"Los cuatro formatos (RDF/XML, {', '.join(FORMAT_LABELS[k] for k in formats)})"
                   f" dan las mismas filas en las {total} preguntas.")
    return {"message": message, "rows": rows}


# --------------------------------------------------------------------------- the bridge

def bridge_view(bridge: dict, folder: Path) -> dict:
    s = bridge["summary"]
    prompt = bridge["prompt"]
    lines = [f"{bridge['pairs']} pares OntoPriv ↔ GDPRtEXT que llegan al mismo término del DPV "
             f"con exactMatch o closeMatch en los dos lados, por "
             f"{_n(bridge['terms'], 'término', 'términos')} (con cualquier relación SKOS serían "
             f"{bridge['loose_pairs']}; no entran). Cada par va en el módulo como "
             "skos:closeMatch.",
             *[line.strip() for line in gb.validation_lines(bridge["validation"])],
             f"IA revisora ({prompt['name']}.v{prompt['version']}): "
             f"{_n(s['coherente'], 'coherente', 'coherentes')}, "
             f"{_n(s['dudosa'], 'dudosa', 'dudosas')}, "
             f"{_n(s['incoherente'], 'incoherente', 'incoherentes')}"
             + (f", {s['pending']} sin revisar" if s.get("pending") else "")
             + "; solo explica, no cambia los pares."]
    rows = sorted(bridge["rows"], key=lambda r: VERDICT_ORDER.get(r["verdict"], 3))
    table = [{"IA revisora": r["verdict"] or "sin revisar", "OntoPriv": r["ontopriv"],
              "GDPRtEXT": r["gdprtext"], "Explicación": r["explanation"] or "",
              "Término del DPV": r["dpv_terms"],
              "Relaciones": f"{r['ontopriv_relation']} / {r['gdprtext_relation']}",
              "Artículos del GDPR": r["gdpr_articles"] or ""} for r in rows]
    flagged = [{"title": f":{VERDICT_BADGES[r['verdict']]}-badge[{r['verdict'].capitalize()}] "
                         f"**{ao.plain(r['ontopriv'])}** ↔ **{ao.plain(r['gdprtext'])}** · por "
                         f"{r['dpv_terms']}",
                "explanation": ao.plain(r["explanation"] or "")}
               for r in rows if r["verdict"] in VERDICT_BADGES]
    downloads = []
    for name, label, mime in ((gb.CSV_FILE, "Descargar las explicaciones (CSV)", "text/csv"),
                              (f"{gb.MODULE}.ttl", "Descargar el módulo del puente (Turtle)",
                               "text/turtle")):
        path = folder / name
        if path.exists():
            downloads.append({"label": label, "data": path.read_bytes(), "file_name": name,
                              "mime": mime})
    return {"lines": lines, "flagged": flagged, "rows": table, "downloads": downloads,
            "created": bridge.get("created")}


# --------------------------------------------------------------------------- the page

def view(cfg: dict | None = None) -> dict:
    """Everything the page shows; None in a part means its command has not run yet."""
    folder = output_dir(cfg)
    report = _json(folder / qs.REPORT_FILE)
    bridge = _json(folder / gb.RESULT_FILE)
    out = {"questions": None, "original": None, "formats": None, "bridge": None,
           "commands": {"questions": RUN_QUESTIONS, "bridge": RUN_BRIDGE}}
    if report is not None:
        catalog = qc.load_catalog(qc.CATALOG_FILE)
        out["questions"] = {
            "header": header_lines(report, catalog), "stale": stale(report, catalog),
            "groups": [{"title": f"{qc.GROUPS[g]} ({len(rows)})", "caption":
                        group_caption(report, g), "rows": rows}
                       for g in qc.GROUPS if (rows := question_rows(report, catalog, g))]}
        out["original"] = original_view(report)
        out["formats"] = formats_view(report)
    if bridge is not None:
        out["bridge"] = bridge_view(bridge, folder)
    return out
