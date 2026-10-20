"""Bridge for the "Alinear mi ontología" page (S6-T08, S7-T06): the user's flow in the web.

The person only uploads a file; everything else is automatic: the workspace (S6-T04),
inheriting the validated decisions (S6-T05), the decider and its calibrated rule for the rest
(S7-T06), writing the module and the zip (S6-T07). This adapter calls
``src.alignment.user_alignment.align_upload`` and turns its results into what the page shows:
a few numbers, a read-only table per concept and the zip in the chosen format.

Nothing here asks the person to review. The ontology holds what a person decided while building
the base and what the rule accepted; the suggestions (high confidence, left out by the rule)
are only shown, so the person can comment on them later (Sprint 10).
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

# Wording of the states shown to the person (codes from user_alignment), in display order.
STATE_LABELS = {ua.INHERITED_ALIGNED: "Validada en la base",
                ua.ACCEPTED: "Aceptada automáticamente",
                ua.SUGGESTED: "Sugerida",
                ua.INHERITED_NO_MATCH: "Sin correspondencia",
                ua.RULE_NO_MATCH: "Sin correspondencia",
                ua.NO_PROPOSAL: "Sin propuesta",
                ua.UNDECIDED: "Sin decidir",
                ua.NOT_ALIGNABLE: "No se alinea"}
STATE_HELP = {
    "Validada en la base": "Tu concepto es el mismo que uno de la base (OntoPriv + LOPDP) y "
                           "recibe la correspondencia que una persona ya validó.",
    "Aceptada automáticamente": "La IA eligió un término del DPV entre los más parecidos, "
                                "guiada por decisiones parecidas de la base, y la regla "
                                "calibrada la aceptó (confianza alta, puntaje suficiente y una "
                                "relación fuerte). Entra a la ontología.",
    "Sugerida": "La IA la propuso con confianza alta, pero la regla no la acepta (relación "
                "débil, puntaje bajo o una propiedad). No entra a la ontología: queda para que "
                "la comentes.",
    "Sin correspondencia": "El DPV no tiene un término equivalente: lo decidió una persona en "
                           "la base o la IA con confianza alta. No genera tripletas.",
    "Sin propuesta": "La IA no estuvo segura (confianza media o baja): el sistema prefiere no "
                     "proponer antes que equivocarse.",
    "Sin decidir": "La IA no estuvo disponible y el mejor candidato no llegó al umbral de la "
                   "regla sin IA. Vuelve a subir el archivo cuando la IA esté disponible.",
    "No se alinea": "Individuos (ejemplos de datos): no se alinean, igual que en la base.",
}
DECIDED_BY = {"base": "base", "rule": "regla", "rule_without_ai": "regla sin IA"}

# The five numbers at the top of the page (their labels are plural) and their states.
METRICS = {"Validadas en la base": (ua.INHERITED_ALIGNED,),
           "Aceptadas automáticamente": (ua.ACCEPTED,),
           "Sugeridas": (ua.SUGGESTED,),
           "Sin correspondencia": (ua.INHERITED_NO_MATCH, ua.RULE_NO_MATCH),
           "Sin propuesta": (ua.NO_PROPOSAL, ua.UNDECIDED)}
METRIC_HELP = {"Validadas en la base": STATE_HELP["Validada en la base"],
               "Aceptadas automáticamente": STATE_HELP["Aceptada automáticamente"],
               "Sugeridas": STATE_HELP["Sugerida"],
               "Sin correspondencia": STATE_HELP["Sin correspondencia"],
               "Sin propuesta": STATE_HELP["Sin propuesta"] + " También cuenta lo que quedó "
                                                              "sin decidir."}


def workspace_cfg() -> dict | None:
    """Config used for the workspaces (None = config.yaml). Tests point it elsewhere."""
    return None


def align(name: str, data: bytes, *, progress=None, base=None, targets=None,
          embed_fn=None, llm=None, chroma_path=None, sleep_fn=None) -> str:
    """Run the whole flow for an upload; returns the workspace id (reused if seen before)."""
    ws, _manifest, _zip = ua.align_upload(name, data, fmt=DEFAULT_FORMAT, progress=progress,
                                          base=base, targets=targets, embed_fn=embed_fn,
                                          llm=llm, cfg=workspace_cfg(), chroma_path=chroma_path,
                                          sleep_fn=sleep_fn)
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
    if manifest is None or "decider" not in manifest:
        raise WorkspaceError("La alineación de este archivo no terminó; vuelve a subirlo.")
    s = manifest["summary"]
    by = s["by_status"]
    decider = manifest.get("decider") or {}
    return {
        "id": ws.id,
        "file": manifest["uploaded_file"],
        "created": manifest["created"],
        "version": manifest.get("alignment_version"),
        "concepts": s["concepts"],
        "metrics": {label: sum(by.get(code, 0) for code in codes)
                    for label, codes in METRICS.items()},
        "mappings": s["mappings"],
        "validated_mappings": s["validated_mappings"],
        "accepted_mappings": s["accepted_mappings"],
        "suggested": s["suggested"],
        "rule": decider.get("rule"),
        "ai_note": _ai_note(decider),
        "iri_generated": manifest.get("ontology_iri_generated", False),
        "rows": table_rows(_rows(ws)),
    }


def _ai_note(decider: dict) -> str | None:
    """A warning only when the AI left concepts without their decision."""
    pending = decider.get("pending_ai") or 0
    if not pending:
        return None
    ai = decider.get("ai") or {}
    threshold = decider.get("threshold_without_ai")
    fallback = (f"se aplicó la regla sin IA (mejor candidato con puntaje >= {threshold:g}, "
                "sin tipo SKOS)" if threshold is not None else "se aplicó la regla sin IA")
    if not ai.get("available"):
        return (f"La IA no estuvo disponible: a {pending} conceptos {fallback}. Vuelve a subir "
                "el archivo cuando la IA esté disponible para que el decisor los complete.")
    if ai.get("quota_exhausted"):
        return (f"Se agotó la cuota diaria de la IA: a {pending} conceptos {fallback}. Vuelve "
                "a subir el archivo mañana para completarlos (lo ya decidido no se repite).")
    return (f"Se llegó al límite de llamadas a la IA de esta subida: a {pending} conceptos "
            f"{fallback}. Vuelve a subir el archivo para seguir (lo ya decidido no se repite).")


def _decided_by(r: dict) -> str:
    who = DECIDED_BY.get(r.get("decided_by") or "", "")
    if who == "base" and r.get("base_concept"):
        return f"base · {r['method']} · {r['base_concept'].rsplit('#', 1)[-1]}"
    return who


def table_rows(rows: list[dict]) -> list[dict]:
    """Rows for the read-only table, in Spanish, in the order of the states."""
    out = []
    for r in rows:
        score = r.get("score")
        out.append({
            "Concepto": r.get("concept_name"),
            "Estado": STATE_LABELS.get(r.get("status"), r.get("status")),
            "Relación": r.get("relation") or "",
            "Término del DPV": r.get("dpv_name") or "",
            "Puntaje": round(float(score), 2) if score not in (None, "") else None,
            "Confianza de la IA": r.get("confidence") or "",
            "Decidido por": _decided_by(r),
            "Justificación": r.get("justification") or "",
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