"""Bridge for the "Alinear mi ontología" page (S6-T08): the user's flow in the web.

The person only uploads a file; everything else is automatic (agreed in Sprint 6): the
workspace (S6-T04), inheriting the validated decisions (S6-T05), the automatic proposals
(S6-T06), writing the modules and the zip (S6-T07). This adapter calls
``src.alignment.user_alignment.align_upload`` and turns its results into what the page shows:
a few numbers, a read-only table per concept and the zip in the chosen format.

Nothing here asks the person to review: proposals are shown as "automática, no validada" and
the validated module only holds what a person decided while building the base.
"""
from __future__ import annotations

import csv
import io

from src.alignment.bundle import PROTEGE_JSONLD_NOTE
from src.alignment import user_alignment as ua
from src.transform.convert import OUTPUT_FORMATS, ConversionError, output_format
from src.workspace.store import WorkspaceError, load_workspace

ACCEPTED_TYPES = ["rdf", "owl", "xml", "ttl", "jsonld", "json"]
FORMATS = [(f.key, f.label) for f in OUTPUT_FORMATS.values()]
DEFAULT_FORMAT = "turtle"

# Order and wording of the states shown to the person (codes from S6-T07).
STATES = [ua.INHERITED_ALIGNED, ua.PROPOSED, ua.INHERITED_NO_MATCH, ua.NO_PROPOSAL,
          ua.AI_REJECTED, ua.NOT_ALIGNABLE]
STATE_LABELS = {ua.INHERITED_ALIGNED: "Validada",
                ua.PROPOSED: "Propuesta automática",
                ua.INHERITED_NO_MATCH: "Sin correspondencia",
                ua.NO_PROPOSAL: "Sin propuesta",
                ua.AI_REJECTED: "Sin propuesta",
                ua.NOT_ALIGNABLE: "No se alinea"}
STATE_HELP = {
    "Validada": "Tu concepto es el mismo que uno de la base (OntoPriv + LOPDP) y recibe la "
                "correspondencia que una persona ya validó.",
    "Propuesta automática": "El sistema encontró un término del DPV muy parecido (puntaje sobre "
                            "el umbral). No está validada: va en un módulo aparte.",
    "Sin correspondencia": "En la base se decidió que este concepto no tiene equivalente en el "
                           "DPV.",
    "Sin propuesta": "Ningún término del DPV se parece lo suficiente; el sistema prefiere no "
                     "proponer antes que equivocarse.",
    "No se alinea": "Individuos (ejemplos de datos): no se alinean, igual que en la base.",
}


# Help of the four numbers at the top of the page (their labels are plural).
METRIC_HELP = {"Validadas": STATE_HELP["Validada"],
               "Propuestas automáticas": STATE_HELP["Propuesta automática"],
               "Sin correspondencia": STATE_HELP["Sin correspondencia"],
               "Sin propuesta": STATE_HELP["Sin propuesta"]}


def workspace_cfg() -> dict | None:
    """Config used for the workspaces (None = config.yaml). Tests point it elsewhere."""
    return None


def align(name: str, data: bytes, *, progress=None, base=None, targets=None,
          embed_fn=None, llm=None) -> str:
    """Run the whole flow for an upload; returns the workspace id (reused if seen before)."""
    ws, _manifest, _zip = ua.align_upload(name, data, fmt=DEFAULT_FORMAT, progress=progress,
                                          base=base, targets=targets, embed_fn=embed_fn,
                                          llm=llm, cfg=workspace_cfg())
    return ws.id


def error_message(exc: Exception) -> str:
    """What the person reads when something fails (Spanish, no traceback)."""
    if isinstance(exc, (WorkspaceError, ConversionError)):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return ("El sistema todavía no tiene su base construida (faltan los archivos del "
                f"Sprint 4-5): {exc}")
    return f"No se pudo alinear la ontología: {type(exc).__name__}: {exc}"


def _rows(ws) -> list[dict]:
    path = ws.output_dir / ua.CSV_FILE
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def view(ws_id: str) -> dict:
    """Everything the page shows for one workspace."""
    ws = load_workspace(ws_id, cfg=workspace_cfg())
    manifest = ua.load_manifest(ws)
    if manifest is None:
        raise WorkspaceError("La alineación de este archivo no terminó; vuelve a subirlo.")
    by = manifest["summary"]["by_status"]
    ai = (manifest.get("proposals") or {}).get("ai") or {}
    return {
        "id": ws.id,
        "file": manifest["uploaded_file"],
        "created": manifest["created"],
        "concepts": manifest["summary"]["concepts"],
        "metrics": {"Validadas": by.get(ua.INHERITED_ALIGNED, 0),
                    "Propuestas automáticas": by.get(ua.PROPOSED, 0),
                    "Sin correspondencia": by.get(ua.INHERITED_NO_MATCH, 0),
                    "Sin propuesta": by.get(ua.NO_PROPOSAL, 0) + by.get(ua.AI_REJECTED, 0)},
        "validated_mappings": manifest["summary"]["validated_mappings"],
        "proposed_mappings": manifest["summary"]["proposed_mappings"],
        "proposed_untyped": manifest["summary"]["proposed_untyped"],
        "threshold": (manifest.get("proposals") or {}).get("threshold"),
        "ai_note": _ai_note(ai, manifest["summary"]["proposed_mappings"]),
        "iri_generated": manifest.get("ontology_iri_generated", False),
        "rows": table_rows(_rows(ws)),
    }


def _ai_note(ai: dict, proposed: int) -> str | None:
    if not proposed:
        return None
    if not ai.get("available"):
        return ("La IA no estuvo disponible: las propuestas automáticas van sin tipo SKOS "
                "(rdfs:seeAlso).")
    if ai.get("quota_exhausted"):
        return ("Se agotó la cuota diaria de la IA: algunas propuestas quedaron sin tipo SKOS. "
                "Vuelve a subir el archivo mañana para completarlas.")
    return None


def table_rows(rows: list[dict]) -> list[dict]:
    """Rows for the read-only table, in Spanish, best first."""
    out = []
    for r in rows:
        state = STATE_LABELS.get(r.get("status"), r.get("status"))
        score = r.get("score")
        out.append({
            "Concepto": r.get("concept_name"),
            "Estado": state,
            "Relación": r.get("relation") or "",
            "Término del DPV": r.get("dpv_name") or "",
            "Confianza": round(float(score), 2) if score not in (None, "") else None,
            "Origen": (f"{r['method']} · {r['base_concept'].rsplit('#', 1)[-1]}"
                       if r.get("base_concept") else ("IA + puntaje" if state ==
                                                      "Propuesta automática" else "")),
        })
    order = {label: i for i, label in enumerate(dict.fromkeys(STATE_LABELS.values()))}
    return sorted(out, key=lambda r: (order.get(r["Estado"], 99), (r["Concepto"] or "").lower()))


def filter_rows(rows: list[dict], state: str | None) -> list[dict]:
    return rows if not state or state == "Todos" else [r for r in rows if r["Estado"] == state]


def download(ws_id: str, fmt: str) -> dict:
    """The zip in ``fmt`` (built in memory; the alignment itself is not recomputed)."""
    ws = load_workspace(ws_id, cfg=workspace_cfg())
    package = ua.build_user_package(ws, fmt)
    spec = output_format(fmt)
    return {"data": package.data, "file_name": ua.package_name(ws, spec),
            "label": spec.label, "main_file": package.main_file,
            "note": PROTEGE_JSONLD_NOTE if spec.key == "jsonld" else None}


def zip_names(data: bytes) -> list[str]:
    import zipfile
    return zipfile.ZipFile(io.BytesIO(data)).namelist()