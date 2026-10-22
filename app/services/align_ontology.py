"""Bridge for the "Alinear con el DPV" page (S6-T08, S7-T06, S7-T08, S7-T12): the user's flow
in the web, in the three modes of the system.

The person picks what to upload (S7-T12) and only uploads; everything else is automatic:

- "Mi ontología" (``ontology``): the workspace (S6-T04), inheriting the validated decisions
  (S6-T05), the decider and its calibrated rule for the rest (S7-T06), the labels in Spanish
  and English that the ontology lacked (S7-T08), writing the modules and the zip (S6-T07):
  ``src.alignment.user_alignment.align_upload``.
- "Una ley" (``law``): the law is split into articles and indexed (S7-T07), the AI extracts its
  concepts and a rule accepts them (S7-T09), they become a profile that imports OntoPriv-Core
  (S7-T10) and the profile goes through the same flow: ``src.core.law_profile.align_law``.
- "Mi ontología y una ley" (``ontology+law``): the ontology is aligned as in the first mode and
  each concept gets the articles of the law closest to it, to review them; they do not enter
  the ontology (S7-T11): ``src.alignment.law_evidence.align_with_law``.

This adapter turns the results into what the page shows: a few numbers, what was done with
the law, a read-only table per concept and the zip in the chosen format.

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
from src.workspace.store import KIND_LAW, KIND_ONTOLOGY, WorkspaceError, load_workspace

ACCEPTED_TYPES = ["rdf", "owl", "xml", "ttl", "jsonld", "json"]
LAW_TYPES = ["txt", "pdf"]
FORMATS = [(f.key, f.label) for f in OUTPUT_FORMATS.values()]
DEFAULT_FORMAT = "turtle"

# The three modes (S7-T12): code, what the person reads, what is uploaded, in display order.
MODE_ONTOLOGY, MODE_LAW, MODE_BOTH = KIND_ONTOLOGY, KIND_LAW, "ontology+law"
MODES = {MODE_ONTOLOGY: "Mi ontología", MODE_LAW: "Una ley", MODE_BOTH: "Mi ontología y una ley"}
MODE_FILES = {MODE_ONTOLOGY: (KIND_ONTOLOGY,), MODE_LAW: (KIND_LAW,),
              MODE_BOTH: (KIND_ONTOLOGY, KIND_LAW)}
MODE_HELP = {
    MODE_ONTOLOGY: "Tu ontología se alinea con el DPV tal como está.",
    MODE_LAW: "La IA saca los conceptos de la ley y el sistema arma con ellos una ontología "
              "sobre OntoPriv-Core, ya alineada.",
    MODE_BOTH: "Tu ontología se alinea igual que en el primer modo y, además, cada concepto "
               "trae los artículos de la ley más cercanos, para que los revises.",
}
MODE_WAIT = {
    MODE_ONTOLOGY: "Sube un archivo para empezar. La alineación arranca sola y tarda desde unos "
                   "segundos hasta pocos minutos, según el tamaño de tu ontología.",
    MODE_LAW: "Sube tu ley (.txt o PDF) para empezar. Tarda más que con una ontología: la IA "
              "lee la ley por partes (unos minutos para una ley del tamaño del GDPR).",
    MODE_BOTH: "Sube tu ontología y tu ley para empezar; arranca sola cuando estén las dos. "
               "Tarda casi lo mismo que alinear solo tu ontología.",
}
MODE_RUNNING = {MODE_ONTOLOGY: "Alineando tu ontología…",
                MODE_LAW: "Convirtiendo tu ley en una ontología alineada…",
                MODE_BOTH: "Alineando tu ontología y buscando sus artículos en la ley…"}
FILE_NAMES = {KIND_ONTOLOGY: "tu ontología", KIND_LAW: "tu ley"}

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


def run(mode: str, files: dict[str, tuple[str, bytes]], **kwargs) -> str:
    """The flow of ``mode`` for the uploaded ``files`` ({kind: (name, bytes)}); returns the
    workspace id. ``kwargs``: progress, base, targets, embed_fn, llm, chroma_path, sleep_fn."""
    if mode not in MODES:
        raise WorkspaceError(f"Modo desconocido: {mode}")
    missing = [FILE_NAMES[k] for k in MODE_FILES[mode] if not files.get(k)]
    if missing:
        raise WorkspaceError(f"Falta subir {' y '.join(missing)}.")
    if mode == MODE_ONTOLOGY:
        return align(*files[KIND_ONTOLOGY], **kwargs)
    common = dict(fmt=DEFAULT_FORMAT, cfg=workspace_cfg(), **kwargs)
    if mode == MODE_LAW:
        from src.core.law_profile import align_law
        ws, _manifest, _zip = align_law(*files[KIND_LAW], **common)
    else:
        from src.alignment.law_evidence import align_with_law
        ws, _manifest, _zip = align_with_law(*files[KIND_ONTOLOGY], *files[KIND_LAW], **common)
    return ws.id


def ready(mode: str, files: dict[str, tuple[str, bytes] | None]) -> bool:
    """Every file the mode needs was uploaded."""
    return all(files.get(k) for k in MODE_FILES[mode])


def missing_text(mode: str, files: dict[str, tuple[str, bytes] | None]) -> str:
    """What the page says while it waits for the files."""
    given = [k for k in MODE_FILES[mode] if files.get(k)]
    if mode == MODE_BOTH and len(given) == 1:
        other = KIND_LAW if given[0] == KIND_ONTOLOGY else KIND_ONTOLOGY
        return f"Falta {FILE_NAMES[other]}: la alineación arranca cuando estén las dos."
    return MODE_WAIT[mode]


def error_message(exc: Exception) -> str:
    """What the person reads when something fails (Spanish, no traceback)."""
    if isinstance(exc, (WorkspaceError, ConversionError)):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return ("El sistema todavía no tiene su base construida (falta un archivo de los "
                f"Sprints 3-5): {exc}")
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
    mode = ws.manifest.get("mode") or MODE_ONTOLOGY
    evidence = manifest.get("law_evidence")
    return {
        "id": ws.id,
        "mode": mode,
        "mode_label": MODES.get(mode, mode),
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
        "labels": labels_caption(manifest.get("labels")),
        "labels_note": _labels_note(manifest.get("labels")),
        "profile": profile_caption(manifest.get("profile")),
        "extraction_note": extraction_note(_law_concepts(ws) if mode == MODE_LAW else None),
        "core": bool(manifest["files"].get("core")),
        "evidence": evidence_caption(evidence),
        "rows": table_rows(_rows(ws), with_law=evidence is not None),
    }


def _law_concepts(ws) -> dict | None:
    from src.core.law_concepts import load_result
    return load_result(ws)


def _n(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def profile_caption(profile: dict | None) -> str | None:
    """What the "Una ley" mode built from the law (S7-T10), in one line."""
    if not profile:
        return None
    accepted = profile.get("law_concepts") or {}
    extracted = _n(accepted.get("concepts", 0), "concepto", "conceptos")
    kinds = (f"{_n(profile['classes'], 'clase', 'clases')} y "
             f"{_n(profile['properties'], 'propiedad', 'propiedades')}")
    text = (f"De {profile['law']}: la IA extrajo {extracted} y la regla aceptó "
            f"{accepted.get('accepted', 0)} ({kinds})")
    repeated, rejected = accepted.get("repeated", 0), accepted.get("rejected", 0)
    left = [_n(n, one, many) for n, one, many in ((repeated, "repetido", "repetidos"),
                                                  (rejected, "rechazado", "rechazados")) if n]
    if left:
        verb = "entró" if repeated + rejected == 1 else "entraron"
        text += f"; {' y '.join(left)} no {verb}"
    built = (f"{_n(profile['reused'], 'reutiliza', 'reutilizan')} conceptos de OntoPriv-Core, "
             f"{_n(profile['linked'], 'se enlaza', 'se enlazan')} con él y "
             f"{_n(profile['new'], 'es nuevo', 'son nuevos')}")
    return f"{text}. Con ellos se armó tu ontología: {built}."


def extraction_note(concepts: dict | None) -> str | None:
    """A warning only when the AI did not read the whole law (S7-T09)."""
    x = (concepts or {}).get("extraction") or {}
    if not x or x.get("complete"):
        return None
    ai = concepts.get("ai") or {}
    what = (f"la IA leyó {x['articles_processed']} de {x['articles_total']} artículos de la ley "
            f"(faltan {len(x['pending_articles'])})")
    if not ai.get("available"):
        return (f"La IA no estuvo disponible: {what}. Vuelve a subir el archivo cuando la IA "
                "esté disponible para completar tu ontología.")
    if ai.get("quota_exhausted"):
        return (f"Se agotó la cuota diaria de la IA: {what}. Vuelve a subir el archivo mañana "
                "para seguir (lo ya leído no se repite).")
    if ai.get("call_limit_reached"):
        return (f"Se llegó al límite de llamadas a la IA de esta subida: {what}. Vuelve a subir "
                "el archivo para seguir (lo ya leído no se repite).")
    return (f"La IA falló en algunos lotes: {what}. Vuelve a subir el archivo para "
            "intentarlo de nuevo.")


def evidence_caption(evidence: dict | None) -> str | None:
    """What the "Mi ontología y una ley" mode found in the law (S7-T11), in one line."""
    if not evidence:
        return None
    if not evidence["links"]:
        return (f"Ningún concepto tiene artículos de {evidence['law']} lo bastante cercanos para "
                "revisarlos.")
    have = "concepto tiene" if evidence["concepts"] == 1 else "conceptos tienen"
    return (f"Artículos de {evidence['law']} para revisar: {evidence['with_evidence']} de "
            f"{evidence['concepts']} {have} artículos cercanos "
            f"({_n(evidence['links'], 'enlace', 'enlaces')} a "
            f"{_n(evidence['articles_cited'], 'artículo distinto', 'artículos distintos')}). "
            "Están en la columna "
            "«Artículos de la ley» de la tabla y en alignment.csv; no entran a la ontología, "
            "porque la búsqueda sola no es lo bastante precisa: revísalos.")


def labels_caption(labels: dict | None) -> str | None:
    """What S7-T08 did, in one line (None for a workspace aligned before it)."""
    if not labels:
        return None
    text = f"Etiquetas en español e inglés: {labels['complete']} conceptos ya las tenían"
    if labels["translated"]:
        added = labels["added"]
        text += (f"; la IA completó las de {labels['translated']} ({added['es']} en español y "
                 f"{added['en']} en inglés) en un módulo aparte, sin cambiar tus etiquetas")
    return text + "."


def _labels_note(labels: dict | None) -> str | None:
    """A warning only when concepts were left without their labels."""
    pending = (labels or {}).get("pending") or 0
    if not pending:
        return None
    ai = labels.get("ai") or {}
    what = f"{pending} conceptos quedaron sin sus etiquetas en español o inglés"
    if not ai.get("available"):
        return (f"La IA no estuvo disponible: {what}. Vuelve a subir el archivo cuando la IA "
                "esté disponible para completarlas.")
    if ai.get("quota_exhausted"):
        return (f"Se agotó la cuota diaria de la IA: {what}. Vuelve a subir el archivo mañana "
                "(lo ya traducido no se repite).")
    if ai.get("call_limit_reached"):
        return (f"Se llegó al límite de llamadas a la IA de esta subida: {what}. Vuelve a subir "
                "el archivo para seguir (lo ya traducido no se repite).")
    return (f"La IA no devolvió etiquetas válidas: {what}. Vuelve a subir el archivo para "
            "intentarlo de nuevo.")


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


def table_rows(rows: list[dict], with_law: bool = False) -> list[dict]:
    """Rows for the read-only table, in Spanish, in the order of the states; ``with_law`` adds
    the articles of the law to review ("Mi ontología y una ley", S7-T11)."""
    out = []
    for r in rows:
        score = r.get("score")
        row = {
            "Concepto": r.get("concept_name"),
            "Etiqueta (es)": r.get("label_es") or "",
            "Etiqueta (en)": r.get("label_en") or "",
            "Estado": STATE_LABELS.get(r.get("status"), r.get("status")),
            "Relación": r.get("relation") or "",
            "Término del DPV": r.get("dpv_name") or "",
            "Puntaje": round(float(score), 2) if score not in (None, "") else None,
            "Confianza de la IA": r.get("confidence") or "",
            "Decidido por": _decided_by(r),
            "Justificación": r.get("justification") or "",
        }
        if with_law:
            row["Artículos de la ley"] = r.get("law_evidence") or ""
        out.append(row)
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