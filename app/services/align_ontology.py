from __future__ import annotations

import csv
import re
from functools import lru_cache, partial
from pathlib import Path

from src.alignment import user_alignment as ua
from src.console import counted
from src.provenance import upload as up
from src.transform.convert import OUTPUT_FORMATS, ConversionError, input_format, serialize
from src.validation import ai_auditor as au
from src.validation import question_explainer as qe
from src.validation import questions as qs
from src.validation import upload_validation as uv
from src.validation.shapes import VIOLATION
from src.validation.shape_inspector import (BASE, GREEN, NEW, ONTOLOGY, RED, YELLOW,
                                            current_color)
from src.workspace.store import (KIND_LAW, KIND_ONTOLOGY, STEP_LABELS, WorkspaceError,
                                 load_workspace, steps_for)

ACCEPTED_TYPES = ["rdf", "owl", "xml", "ttl", "jsonld", "json"]
LAW_TYPES = ["txt", "pdf"]
DEFAULT_FORMAT = "turtle"

# The two paths of the start (S10-T09): align and transform, or only change the format.
PATH_ALIGN, PATH_TRANSFORM = "align", "transform"
PATHS = {PATH_ALIGN: "Alinear y transformar", PATH_TRANSFORM: "Solo transformar"}
PATH_HELP = {
    PATH_ALIGN: "Enlaza tu ontología o tu ley con el DPV, el vocabulario de privacidad de "
                "referencia, y descárgala en el formato que quieras.",
    PATH_TRANSFORM: "Cambia el formato de tu ontología (por ejemplo, de .owl a .ttl) sin tocar "
                    "su contenido.",
}
PATH_ICONS = {PATH_ALIGN: ":material/hub:", PATH_TRANSFORM: ":material/swap_horiz:"}
START_TITLE = "Transforma y alinea tu ontología de privacidad"
START_CAPTION = "¿Qué quieres hacer?"
MODES_CAPTION = ("¿Qué tienes? El resto es automático: alineación con el Data Privacy "
                 "Vocabulary (DPV) y etiquetas en español e inglés.")
TRANSFORM_NOTE = "Tu archivo no pasa por la IA: solo se convierte al formato que elijas."
TRANSFORM_READY = "Tu ontología está lista para descargar"
TRANSFORM_JSONLD = ("JSON-LD: para aplicaciones web y de datos enlazados. Protégé 5.6 no lo "
                    "abre.")
TRANSFORM_CAPTION = ("El contenido es el mismo en los cuatro formatos. Protégé 5.6 no abre "
                     "JSON-LD: para Protégé usa RDF/XML, OWL o Turtle.")

# The three modes (S7-T12): code, what the person reads, what is uploaded, in display order.
MODE_ONTOLOGY, MODE_LAW, MODE_BOTH = KIND_ONTOLOGY, KIND_LAW, "ontology+law"
MODES = {MODE_ONTOLOGY: "Mi ontología", MODE_LAW: "Una ley", MODE_BOTH: "Mi ontología y una ley"}
MODE_FILES = {MODE_ONTOLOGY: (KIND_ONTOLOGY,), MODE_LAW: (KIND_LAW,),
              MODE_BOTH: (KIND_ONTOLOGY, KIND_LAW)}
# Step 1 of the assistant (S8-T01): the question and the icon of each card.
MODE_HELP = {
    MODE_ONTOLOGY: "¿Ya tienes una ontología y quieres alinearla con el DPV?",
    MODE_LAW: "¿Quieres crear una ontología a partir de un texto legal?",
    MODE_BOTH: "¿Tienes las dos y quieres que la ley complete las definiciones de tu ontología?",
}
MODE_ICONS = {MODE_ONTOLOGY: ":material/schema:", MODE_LAW: ":material/gavel:",
              MODE_BOTH: ":material/join_inner:"}
# Step 2: the uploaders and the line under them while they wait.
UPLOAD_LABELS = {KIND_ONTOLOGY: "Tu ontología (.rdf, .owl, .ttl o .jsonld)",
                 KIND_LAW: "Tu ley (.txt o .pdf)"}
MODE_WAIT = {
    MODE_ONTOLOGY: "Arranca sola al subir tu ontología. Tarda de unos segundos a pocos minutos, "
                   "según su tamaño.",
    MODE_LAW: "Arranca sola al subir tu ley. Tarda más que con una ontología: la IA la lee por "
              "partes (unos minutos para una ley del tamaño del GDPR).",
    MODE_BOTH: "Arranca sola cuando estén los dos archivos. Si a tu ontología le faltan "
               "definiciones, la IA las busca en la ley: tarda unos minutos más.",
}
# What happens with the files: the help of the uploaders (S10-T09).
DATA_NOTE = (
    "- Lo que subes se guarda en un espacio de trabajo propio en el servidor. Si subes el mismo "
    "archivo otra vez, se reutiliza lo ya calculado.\n"
    "- Para decidir la alineación y completar las etiquetas, el sistema envía a la IA (Gemini, "
    "de Google) los nombres, las etiquetas y las definiciones de tus conceptos. Si subes una "
    "ley, también le envía su texto, por partes.\n"
    "- Tu archivo nunca se modifica: la alineación, las etiquetas y las definiciones van en "
    "módulos aparte que lo importan.\n"
    "- La base del sistema (OntoPriv y la LOPDP) no cambia con lo que subes."
)
MODE_RUNNING = {MODE_ONTOLOGY: "Alineando tu ontología…",
                MODE_LAW: "Convirtiendo tu ley en una ontología alineada…",
                MODE_BOTH: "Alineando tu ontología con ayuda de la ley…"}
FILE_NAMES = {KIND_ONTOLOGY: "tu ontología", KIND_LAW: "tu ley"}

# Each step of the run in plain words, for the progress bar (S10-T11).
STEP_TEXT = {"segment": "Leyendo tu ley", "index": "Ordenando los artículos de tu ley",
             "extract": "Buscando con la IA los conceptos de tu ley",
             "accept": "Revisando los conceptos encontrados", "profile": "Armando tu ontología",
             "definitions": "Completando las definiciones con tu ley",
             "inherit": "Reutilizando lo que ya está revisado",
             "candidates": "Buscando sus equivalentes en el vocabulario de privacidad",
             "propose": "Eligiendo con la IA el mejor equivalente",
             "labels": "Poniendo los nombres en español e inglés",
             "evidence": "Buscando en tu ley el artículo de cada concepto",
             "audit": "La IA revisa los enlaces", "write": "Escribiendo tu ontología",
             "shapes": "Revisando que esté bien formada",
             "logic": "Revisando que no tenga contradicciones",
             "questions": "Comprobando que responda preguntas básicas",
             "package": "Preparando la descarga"}
PROGRESS_START = "Empezando…"
NEXT_LABEL = "Ver tu ontología"

# Wording of the states shown to the person (codes from user_alignment), in display order.
STATE_LABELS = {ua.INHERITED_ALIGNED: "Validada en la base",
                ua.USER_ALIGNED: "Decidida por el usuario",
                ua.ACCEPTED: "Aceptada automáticamente",
                ua.SUGGESTED: "Sugerida",
                ua.INHERITED_NO_MATCH: "Sin correspondencia",
                ua.USER_NO_MATCH: "Sin correspondencia",
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
    "Decidida por el usuario": "La eligió quien subió el archivo, después de probarlo. Entra a "
                               "la ontología en una versión nueva y la IA revisora no la vuelve "
                               "a revisar.",
    "Sin correspondencia": "El DPV no tiene un término equivalente: lo decidió una persona en "
                           "la base, la IA con confianza alta o el usuario. No genera tripletas.",
    "Sin propuesta": "La IA no estuvo segura (confianza media o baja): el sistema prefiere no "
                     "proponer antes que equivocarse.",
    "Sin decidir": "La IA no estuvo disponible y el mejor candidato no llegó al umbral de la "
                   "regla sin IA. Vuelve a subir el archivo cuando la IA esté disponible.",
    "No se alinea": "Individuos (ejemplos de datos): no se alinean, igual que en la base.",
}
DECIDED_BY = {"base": "base", "rule": "regla", "rule_without_ai": "regla sin IA",
              "user": "usuario"}

# The validation panel of step 3 (S8-T11): a traffic light per inspector (S9-T01), plus gray
# when it did not run; no color blocks the download.
LIGHTS = {GREEN: ("green", "Verde"), YELLOW: ("yellow", "Amarillo"), RED: ("red", "Rojo"),
          uv.NOT_RUN: ("gray", "Sin revisar"), uv.FAILED: ("gray", "No corrió")}
CARD_TITLES = {uv.SHAPES: "Forma (SHACL)", uv.LOGIC: "Lógica (razonador)",
               uv.QUESTIONS: "Preguntas (SPARQL)", uv.AUDIT: "IA revisora"}
# The quality criterion each card answers (S9-T07).
CRITERIA = {uv.SHAPES: "Interoperabilidad", uv.LOGIC: "Coherencia",
            uv.QUESTIONS: "Completitud", uv.AUDIT: "Coherencia"}
VALIDATION_HELP = {
    "Forma (SHACL)": "12 reglas sobre lo que se entrega: que cada término del DPV exista, que "
                     "las relaciones SKOS no se contradigan, que cada correspondencia diga quién "
                     "la decidió y que cada etiqueta y definición tenga su idioma. Los avisos "
                     "son recomendaciones, como un concepto sin definición.",
    "Lógica (razonador)": "Apache Jena (o owlrl si no hay Java) busca contradicciones, como un "
                          "individuo en dos clases disjuntas o un valor de un tipo que no "
                          "corresponde.",
    "Preguntas (SPARQL)": "8 preguntas de competencia que cualquier ontología de privacidad "
                          "debería poder responder, como qué derechos del titular o qué bases "
                          "legales representa. Se responden con SPARQL sobre tu ontología, sus "
                          "módulos y el DPV. Si una no responde, la IA explica por qué.",
    "IA revisora": "Lee cada correspondencia que aceptó la regla y dice si el término del DPV "
                   "trata de lo mismo que tu concepto. Solo explica: no quita nada de lo "
                   "aceptado.",
    "Criterios": "coherencia: que no haya contradicciones (Lógica) y que cada correspondencia "
                 "aceptada trate de lo mismo que tu concepto (IA revisora); completitud: que tu "
                 "ontología responda las preguntas de competencia (Preguntas); "
                 "interoperabilidad: que lo que se entrega se pueda enlazar con otros "
                 "vocabularios, con términos del DPV que existen, relaciones SKOS que no se "
                 "contradicen y etiquetas con su idioma (Forma).",
    "Colores": "rojo: algo que escribió el sistema rompe una regla; amarillo: avisos, "
               "hallazgos heredados (ya estaban en tu ontología o en la base del sistema) o "
               "preguntas que tu ontología no responde; verde: nada que revisar; gris: el "
               "inspector no corrió, no se revisó o la pregunta no aplica. Ningún color "
               "bloquea la descarga.",
}
QUESTIONS_TAB = "Preguntas de competencia"
QUESTION_BADGES = {GREEN: ("green", "Responde"), YELLOW: ("yellow", "No responde")}
NO_QUESTIONS = ("Se validó antes de las preguntas  \n"
                "Vuelve a subir el archivo para responderlas")
NO_VALIDATION = ("Esta alineación es anterior a la validación: vuelve a subir el archivo para "
                 "validarla (lo ya calculado no se repite).")
FLAGGED_COLUMNS = ["Concepto", "Término del DPV", "Relación", "Veredicto", "Explicación"]
LAW_TAB = "Lo que dice la ley de cada concepto"

# The five numbers at the top of the page (their labels are plural) and their states; the
# person's one only when they decided something (S10-T03).
USER_METRIC = "Decididas por el usuario"
METRICS = {"Validadas en la base": (ua.INHERITED_ALIGNED,),
           "Aceptadas automáticamente": (ua.ACCEPTED,),
           USER_METRIC: (ua.USER_ALIGNED,),
           "Sugeridas": (ua.SUGGESTED,),
           "Sin correspondencia": (ua.INHERITED_NO_MATCH, ua.RULE_NO_MATCH, ua.USER_NO_MATCH),
           "Sin propuesta": (ua.NO_PROPOSAL, ua.UNDECIDED)}
METRIC_HELP = {"Validadas en la base": STATE_HELP["Validada en la base"],
               "Aceptadas automáticamente": STATE_HELP["Aceptada automáticamente"],
               USER_METRIC: STATE_HELP["Decidida por el usuario"],
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


_STEP_KEYS = {label: key for key, label in STEP_LABELS.items()}


def progress_of(mode: str, label: str) -> tuple[str, float]:
    """What the progress bar says when the run starts the step called ``label`` (the name the
    pipeline gives it) and how much of the run is already done (0 to 1)."""
    key = _STEP_KEYS.get(label)
    steps = steps_for(mode)
    done = steps.index(key) / len(steps) if key in steps else 0.0
    return f"{STEP_TEXT.get(key, label)}…", done


def file_size(n: int) -> str:
    """'378,4 KB', '1,2 MB' (in thousands)."""
    size = f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.1f} KB"
    return size.replace(".", ",")


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
    definitions = manifest.get("law_definitions")
    raw = _rows(ws)
    return {
        "id": ws.id,
        "mode": mode,
        "mode_label": MODES.get(mode, mode),
        "file": manifest["uploaded_file"],
        "source": source_name(ws),
        "created": manifest["created"],
        "version": manifest.get("alignment_version"),
        "concepts": s["concepts"],
        "metrics": {label: n for label, codes in METRICS.items()
                    if (n := sum(by.get(code, 0) for code in codes)) or label != USER_METRIC},
        "mappings": s["mappings"],
        "validated_mappings": s["validated_mappings"],
        "accepted_mappings": s["accepted_mappings"],
        "user_mappings": s.get("user_mappings", 0),
        "suggested": s["suggested"],
        "rule": decider.get("rule"),
        "ai_note": _ai_note(decider),
        "iri_generated": manifest.get("ontology_iri_generated", False),
        "labels": labels_caption(manifest.get("labels")),
        "labels_note": _labels_note(manifest.get("labels")),
        "profile": profile_caption(manifest.get("profile")),
        "extraction_note": extraction_note(_law_concepts(ws) if mode != MODE_ONTOLOGY else None),
        "core": bool(manifest["files"].get("core")),
        "evidence": evidence_caption(evidence),
        "definitions": definitions_caption(definitions),
        "definitions_note": definitions_note(definitions),
        "rows": table_rows(raw, with_law=evidence is not None,
                           with_definitions=bool((definitions or {}).get("added"))),
        "concept_options": concept_options(raw),
        "validation": validation_view(ws),
    }


def _law_concepts(ws) -> dict | None:
    from src.core.law_concepts import load_result
    return load_result(ws)


def profile_caption(profile: dict | None) -> str | None:
    """What the "Una ley" mode built from the law (S7-T10), in one line."""
    if not profile:
        return None
    accepted = profile.get("law_concepts") or {}
    extracted = counted(accepted.get("concepts", 0), "concepto", "conceptos")
    kinds = (f"{counted(profile['classes'], 'clase', 'clases')} y "
             f"{counted(profile['properties'], 'propiedad', 'propiedades')}")
    text = (f"De {profile['law']}: la IA extrajo {extracted} y la regla aceptó "
            f"{accepted.get('accepted', 0)} ({kinds})")
    repeated, rejected = accepted.get("repeated", 0), accepted.get("rejected", 0)
    left = [counted(n, one, many) for n, one, many in ((repeated, "repetido", "repetidos"),
                                                  (rejected, "rechazado", "rechazados")) if n]
    if left:
        verb = "entró" if repeated + rejected == 1 else "entraron"
        text += f"; {' y '.join(left)} no {verb}"
    built = (f"{counted(profile['reused'], 'reutiliza', 'reutilizan')} conceptos de OntoPriv-Core, "
             f"{counted(profile['linked'], 'se enlaza', 'se enlazan')} con él y "
             f"{counted(profile['new'], 'es nuevo', 'son nuevos')}")
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
            f"({counted(evidence['links'], 'enlace', 'enlaces')} a "
            f"{counted(evidence['articles_cited'], 'artículo distinto', 'artículos distintos')}"
            "). Están en la columna "
            "«Artículos de la ley» de la tabla y en alignment.csv; no entran a la ontología, "
            "porque la búsqueda sola no es lo bastante precisa: revísalos."
            + (f" En {counted(evidence['cited_by_ai'], 'concepto', 'conceptos')} el artículo es "
               "el que citó la IA al sacar su definición de la ley (citado)."
               if evidence.get("cited_by_ai") else ""))


def definitions_caption(defined: dict | None) -> str | None:
    """What the law did for the definitions ("Mi ontología y una ley"), in one line."""
    if not defined or defined["skipped"] == "disabled":
        return None
    if defined["skipped"]:
        return (f"Definiciones: tu ontología ya las tenía casi todas ({defined['with_own']} de "
                f"{defined['concepts']}), así que no hizo falta buscarlas en {defined['law']}.")
    text = (f"Definiciones de {defined['law']}: a {defined['missing']} de "
            f"{defined['concepts']} conceptos les faltaba")
    sources = [counted(n, one, many) + what for n, one, many, what in (
        (defined["law_concept"], "salió", "salieron", " de conceptos de la ley"),
        (defined["articles"], "la escribió", "las escribió", " la IA con sus artículos")) if n]
    text += "; " + " y ".join(sources) if sources else "; la ley no dio ninguna"
    if defined["not_applicable"]:
        one = defined["not_applicable"] == 1
        text += (f" ({counted(defined['not_applicable'], 'no aplica', 'no aplican')}: la ley no "
                 f"{'lo' if one else 'los'} define)")
    if defined["added"]:
        text += ". Van en un módulo aparte, cada una con su artículo"
    used = defined.get("to_decider") or 0
    if used:
        text += (f", y el decisor usó {'una' if used == 1 else used} para alinear lo que no "
                 "heredaste de la base")
    return text + "."


def definitions_note(defined: dict | None) -> str | None:
    """A warning only when concepts were left waiting for their definition."""
    pending = (defined or {}).get("pending") or 0
    if not pending:
        return None
    ai = defined.get("ai") or {}
    what = f"{pending} conceptos quedaron sin su definición de la ley"
    if not ai.get("available"):
        return (f"La IA no estuvo disponible: {what}. Vuelve a subir los archivos cuando la IA "
                "esté disponible para completarlas.")
    if ai.get("quota_exhausted"):
        return (f"Se agotó la cuota diaria de la IA: {what}. Vuelve a subir los archivos mañana "
                "(lo ya definido no se repite).")
    if ai.get("call_limit_reached"):
        return (f"Se llegó al límite de llamadas a la IA de esta subida: {what}. Vuelve a subir "
                "los archivos para seguir (lo ya definido no se repite).")
    return (f"La IA falló en algunos lotes: {what}. Vuelve a subir los archivos para "
            "intentarlo de nuevo.")


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


def table_rows(rows: list[dict], with_law: bool = False,
               with_definitions: bool = False) -> list[dict]:
    """Rows for the read-only table, in Spanish, in the order of the states; ``with_law`` adds
    the articles of the law to review and ``with_definitions`` the definitions it gave ("Mi
    ontología y una ley")."""
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
            "Revisión de la IA": r.get("ai_review") or "",
        }
        if with_definitions:
            row["Definición de la ley"] = r.get("law_definition") or ""
        if with_law:
            row["Artículos de la ley"] = r.get("law_evidence") or ""
        out.append(row)
    order = {label: i for i, label in enumerate(dict.fromkeys(STATE_LABELS.values()))}
    return sorted(out, key=lambda r: (order.get(r["Estado"], 99), (r["Concepto"] or "").lower()))


def filter_rows(rows: list[dict], state: str | None) -> list[dict]:
    return rows if not state or state == "Todos" else [r for r in rows if r["Estado"] == state]


# ------------------------------------------------------------------ step 3 of the assistant (S8-T01)

REVIEW_COLUMNS = ["Concepto", "Término del DPV", "Relación", "Confianza de la IA",
                  "Justificación"]


def notes(v: dict) -> list[str]:
    """The AI warnings, shown under the banner: they change what the person gets."""
    return [n for n in (v["extraction_note"], v.get("definitions_note"), v["ai_note"],
                        v["labels_note"]) if n]


def summary_captions(v: dict) -> list[str]:
    """The lines of the "Resumen" tab, under the five numbers."""
    parts = [f"{v['validated_mappings']} de la base",
             f"{v['accepted_mappings']} aceptadas por la regla"]
    if v.get("user_mappings"):
        parts.append(f"{v['user_mappings']} del usuario")
    out = [f"{v['mode_label']} · {counted(v['concepts'], 'concepto', 'conceptos')} · "
           f"{counted(v['mappings'], 'correspondencia', 'correspondencias')} en la ontología "
           f"({', '.join(parts[:-1])} y {parts[-1]}) · {v['suggested']} sugeridas fuera de la "
           f"ontología · versión {v['version']}"]
    out += [line for line in (v["profile"], f"Regla: {v['rule']}." if v["rule"] else None,
                              v["labels"], v.get("definitions")) if line]
    if v["iri_generated"]:
        out.append("Tu ontología no declaraba un IRI de ontología: se le asignó uno en la copia "
                   "del paquete (tu archivo original no cambia).")
    return out


def suggested_rows(rows: list[dict]) -> list[dict]:
    """What the AI proposed with high confidence and the rule left out ("Para revisar")."""
    return [{c: r.get(c, "") for c in REVIEW_COLUMNS}
            for r in filter_rows(rows, STATE_LABELS[ua.SUGGESTED])]


def evidence_rows(rows: list[dict]) -> list[dict]:
    """One row per concept with what the law says of it, to review ("Mi ontología y una ley"):
    the definition it gave (when there is that column) and its articles."""
    columns = ["Concepto", "Estado", "Definición de la ley", "Artículos de la ley"]
    out = {}
    for r in rows:
        if (r.get("Artículos de la ley") or r.get("Definición de la ley")) \
                and r["Concepto"] not in out:
            out[r["Concepto"]] = {c: r[c] for c in columns if c in r}
    return list(out.values())


# ------------------------------------------------------------------ validation panel (S8-T11)

def badge(status: str) -> str:
    color, label = LIGHTS[current_color(status)]
    return f":{color}-badge[{label}]"


def card_text(key: str, block: dict) -> str:
    """Two short lines per inspector (markdown)."""
    if block["status"] == uv.FAILED:
        return f"No pudo correr  \n{block['error']}"
    if key == uv.QUESTIONS:
        return questions_card_text(block)
    if key != uv.AUDIT:
        what = "Reglas rotas" if key == uv.SHAPES else "Contradicciones"
        return (f"{what}: {counted(block['new'], 'nueva', 'nuevas')} · "
                f"{counted(block['inherited'], 'heredada', 'heredadas')}  \n"
                f"Avisos: {block['warnings']}")
    if "summary" not in block:
        return "No se hizo la revisión"
    s = block["summary"]
    if not s["accepted"]:
        return ("Nada que revisar  \nLa regla no aceptó ninguna correspondencia"
                + (f" · del usuario: {s['by_user']}" if s.get("by_user") else ""))
    if block["status"] == uv.NOT_RUN:
        why = block["ai"].get("error") or next(iter(block["ai"].get("errors") or []), "")
        return f"No revisó las {s['accepted']} aceptadas  \n{why}".rstrip()
    text = (f"Incoherentes: {s['incoherente']} · dudosas: {s['dudosa']}  \n"
            f"De {counted(s['accepted'], 'aceptada', 'aceptadas')} por la regla")
    return (text + (f" · sin revisar: {s['pending']}" if s["pending"] else "")
            + (f" · del usuario: {s['by_user']}" if s.get("by_user") else ""))


def questions_card_text(block: dict) -> str:
    if "summary" not in block:                         # validated before the questions
        return NO_QUESTIONS
    s = block["summary"]
    text = f"Responden: {s[GREEN]} · sin responder: {s[YELLOW]}"
    if s[qs.GRAY]:
        text += f" · no corrieron: {s[qs.GRAY]}"
    applies = s[GREEN] + s[YELLOW] + s[qs.GRAY]
    text += f"  \nDe {counted(applies, 'pregunta que aplica', 'preguntas que aplican')}"
    na = s.get(qs.NOT_APPLICABLE, 0)
    return text + (f" · no {'aplica' if na == 1 else 'aplican'}: {na}" if na else "")


def card_caption(key: str, block: dict) -> str:
    if key == uv.QUESTIONS:
        ai = block.get("ai") or {}
        prompt = ai.get("prompt") if ai.get("questions") else None   # only if it explained
        return " · ".join(t for t in (block.get("engine"), prompt and
                                      f"prompt {prompt['name']}.v{prompt['version']}") if t)
    if key == uv.AUDIT:
        prompt = block.get("prompt")
        return (f"prompt {prompt['name']}.v{prompt['version']} · " if prompt else "") + \
            "solo explica"
    return block.get("engine") or ""


def validation_headline(result: dict) -> str:
    ins = result["inspectors"]
    if result["status"] == RED:
        return ("El sistema escribió algo que rompe una regla: mira el detalle. La descarga "
                "sigue disponible.")
    audit = ins[uv.AUDIT]
    if result["status"] == GREEN:
        text = "Ninguna regla se rompe y el razonador no encontró contradicciones."
        if audit["status"] == GREEN and (audit.get("summary") or {}).get("accepted"):
            text += " La IA revisora no marcó nada."
        elif audit["status"] == uv.NOT_RUN:
            text += " La IA no revisó las correspondencias aceptadas."
        answered = ((ins.get(uv.QUESTIONS) or {}).get("summary") or {}).get(GREEN)
        if answered:
            text += (" Tu ontología responde " + ("la pregunta de competencia que aplica."
                                                  if answered == 1 else
                                                  f"las {answered} preguntas de competencia "
                                                  "que aplican."))
        return text
    parts = []
    inherited = sum(ins[k].get("inherited", 0) for k in (uv.SHAPES, uv.LOGIC))
    warnings = sum(ins[k].get("warnings", 0) for k in (uv.SHAPES, uv.LOGIC))
    flagged = len(audit.get("flagged") or [])
    if inherited:
        parts.append(f"{counted(inherited, 'hallazgo heredado', 'hallazgos heredados')} (ya "
                     "estaban en tu ontología o en la base del sistema)")
    if warnings:
        parts.append(f"{counted(warnings, 'aviso', 'avisos')} (recomendaciones, como conceptos sin "
                     "definición)")
    if flagged:
        parts.append(f"{counted(flagged, 'correspondencia', 'correspondencias')} que la IA "
                     "revisora marcó para que las mires")
    questions = (ins.get(uv.QUESTIONS) or {}).get("summary") or {}
    word = ("pregunta de competencia", "preguntas de competencia")
    if questions.get(YELLOW):
        parts.append(f"{counted(questions[YELLOW], *word)} que tu ontología no responde")
    if questions.get(qs.GRAY):
        gray = questions[qs.GRAY]
        parts.append(f"{counted(gray, *word)} que no se {'pudo' if gray == 1 else 'pudieron'} "
                     "responder")
    failed = [CARD_TITLES.get(k, uv.TITLES[k]) for k, b in ins.items()
              if b["status"] == uv.FAILED]
    if failed:
        parts.append("no pudo correr: " + ", ".join(failed))
    return ("Nada de lo que escribió el sistema rompe una regla. Para mirar: " + "; ".join(parts)
            + ".")


def _rule_rows(ws, block: dict) -> list[dict]:
    from src.validation.shape_inspector import load_result
    report = load_result(ws) if block["status"] != uv.FAILED else None
    rows = []
    for r in ((report or {}).get("summary") or {}).get("rules", {}).values():
        if r[NEW] + r[ONTOLOGY] + r[BASE]:
            rows.append({"Regla": r["name"],
                         "Tipo": "regla" if r["severity"] == VIOLATION else "aviso",
                         "Nuevas": r[NEW], "Heredadas": r[ONTOLOGY] + r[BASE]})
    return rows


def _kind_rows(ws, block: dict) -> list[dict]:
    from src.validation.logic_inspector import load_result
    report = load_result(ws) if block["status"] != uv.FAILED else None
    rows = []
    for k in ((report or {}).get("summary") or {}).get("kinds", {}).values():
        if k["errors"] + k["warnings"]:
            rows.append({"Tipo": k["label"], "Contradicciones": k["errors"],
                         "Avisos": k["warnings"], "Nuevas": k[NEW],
                         "Heredadas": k[ONTOLOGY] + k[BASE]})
    return rows


def flagged_rows(result: dict | None) -> list[dict]:
    """What the AI reviewer marked, with why ("Para revisar")."""
    flagged = ((result or {}).get("inspectors", {}).get(uv.AUDIT) or {}).get("flagged") or []
    first = sorted(flagged, key=lambda f: (f["verdict"] != au.INCOHERENT, f["name"].lower()))
    return [dict(zip(FLAGGED_COLUMNS, (f["name"], f["dpv_name"], f["relation"] or "",
                                       f["verdict"], f["explanation"] or "")))
            for f in first]


_MARKDOWN = re.compile(r"([\\`*_\[\]<>#|~$])")


def plain(text: str) -> str:
    """``text`` shown as it is in markdown: no italics, links or formulas by accident."""
    return _MARKDOWN.sub(r"\\\1", text)


def question_badge(r: dict) -> str:
    color, label = QUESTION_BADGES.get(current_color(r["color"])) or (
        "gray", "No aplica" if not r.get("applies", True) else "No corrió")
    return f":{color}-badge[{label}]"


def _query(qid: str) -> str | None:
    try:
        return qs.query_text(qid)
    except OSError:
        return None


def questions_view(block: dict | None) -> dict | None:
    """The tab "Preguntas de competencia": each question with its answer, color, the sentence of
    the AI and its query (None for an upload validated before the questions)."""
    if not block or block["status"] == uv.NOT_RUN:
        return None
    if block["status"] == uv.FAILED:
        return {"caption": f"Las preguntas no pudieron correr: {block['error']}. Esto no "
                           "bloquea la descarga.", "rows": []}
    results = block["results"]
    rows = [{"id": r["id"], "question": plain(r["question"]), "badge": question_badge(r),
             "answer": plain(r["answer"]), "criterion": r["criterion"],
             "why": plain(r["explanation"]) if r.get("explanation") else None,
             "why_label": "Por qué (IA)" if r.get("explained_by") == qe.AI else "Por qué",
             "query": _query(r["id"])} for r in results]
    caption = (f"{counted(len(results), 'pregunta', 'preguntas')}, las mismas para cualquier "
               "ontología de privacidad, respondidas con SPARQL sobre tu ontología, sus módulos "
               f"y el DPV ({block['engine']}, {block['seconds']:g} s): "
               f"{qs.counts_text(block['summary'], len(results))}.")
    return {"caption": caption, "rows": rows}


def validation_view(ws) -> dict | None:
    """The panel: the overall color, one card per inspector, each with its criterion, the detail
    tables and the competency questions."""
    result = uv.load_result(ws)
    if result is None:
        return None
    ins = result["inspectors"]
    blocks = {k: ins.get(k) or {"status": uv.NOT_RUN} for k in CARD_TITLES}
    return {"status": result["status"], "badge": badge(result["status"]),
            "headline": validation_headline(result),
            "cards": [{"key": k, "title": CARD_TITLES[k], "criterion": CRITERIA[k],
                       "badge": badge(b["status"]), "text": card_text(k, b),
                       "caption": card_caption(k, b)} for k, b in blocks.items()],
            "rules": _rule_rows(ws, ins[uv.SHAPES]), "kinds": _kind_rows(ws, ins[uv.LOGIC]),
            "flagged": flagged_rows(result), "questions": questions_view(ins.get(uv.QUESTIONS)),
            "log_file": uv.LOG_FILE}


def validation_log(ws_id: str, cfg: dict | None = None) -> bytes:
    """validation-log.txt of ``ws_id`` (what its download button calls when clicked)."""
    return (load_workspace(ws_id, cfg=cfg).output_dir / uv.LOG_FILE).read_bytes()


# ------------------------------------------------------------------ provenance card (S10-T08)

PROVENANCE_TITLE = "Procedencia de un concepto"
PROVENANCE_HELP = ("De qué artículo sale, cómo se produjo, quién decidió su correspondencia y "
                   "sus versiones, leído del módulo de procedencia (PROV-O) que va en cada .zip.")
PROVENANCE_PLACEHOLDER = "Elige un concepto para ver su ficha"
NO_PROVENANCE = ("Esta alineación es anterior a la procedencia: vuelve a subir el archivo para "
                 "verla.")
VERSION_COLUMNS = ["Versión", "Correspondencia", "Decidida por", "Cuándo", "Estado"]
SYSTEM_AGENT = "ontology-integration-transformer"
# Records that say why a concept has no correspondence (src/provenance/upload.py).
WHY_RECORDS = ("decider-", "inherited-no-match-", "user-removal-")
LAW_RECORDS = ("law-concept-", "law-acceptance-")


def concept_options(raw: list[dict]) -> dict[str, str]:
    """{IRI: "Name · state"} of the concepts of alignment.csv, by name."""
    out = {}
    for r in raw:
        iri = r.get("concept_iri")
        if iri and iri not in out:
            out[iri] = (f"{r.get('concept_name') or up.local(iri)} · "
                        f"{STATE_LABELS.get(r.get('status'), r.get('status'))}")
    return dict(sorted(out.items(), key=lambda kv: (kv[1].lower(), kv[0])))


@lru_cache(maxsize=8)
def _graph_file(path: str, stamp: tuple):
    from src.transform.convert import load_graph
    return load_graph(path)


def _graph_at(path: Path):
    """A module of the workspace, read again only when its file changes."""
    stat = path.stat()
    return _graph_file(str(path), (stat.st_mtime_ns, stat.st_size))


def provenance_card(ws_id: str, iri: str) -> dict:
    """The card of one concept, read with the fixed query of its provenance module."""
    ws = load_workspace(ws_id, cfg=workspace_cfg())
    manifest = ua.load_manifest(ws) or {}
    entry = manifest.get("provenance")
    path = ws.path / entry["file"] if entry else None
    if path is None or not path.exists():
        return {"missing": NO_PROVENANCE}
    records = up.records_of(_graph_at(path), iri)
    return card(records, iri, [r for r in _rows(ws) if r.get("concept_iri") == iri],
                version=manifest.get("alignment_version") or 1,
                file=manifest.get("uploaded_file") or "", mode=ws.manifest.get("mode"))


def _kind(record: dict, prefixes: tuple) -> bool:
    return up.local(record["record"]).startswith(prefixes)


def _triple(r: dict) -> str:
    prop = r.get("property") or ""
    relation = f"skos:{up.local(prop)}" if "/skos/core#" in prop else "rdfs:seeAlso"
    return f"{relation} dpv:{up.local(r.get('target') or '')}"


def _by(r: dict) -> str:
    who = r.get("who")
    shown = who and who not in (r["decidedBy"], SYSTEM_AGENT)
    return f"{r['decidedBy']} ({who})" if shown else r["decidedBy"]


def _in_force(r: dict, records: list[dict]) -> bool:
    """A correspondence is in force when it is an owl:Axiom; a record of why there is none, when
    nothing replaced it and no correspondence is in force."""
    if r.get("decidedBy"):
        return r.get("current") == "true"
    return not r.get("replaced") and not any(x.get("current") == "true" for x in records)


def _state_of(r: dict, records: list[dict]) -> str:
    if _in_force(r, records):
        return "vigente"
    return f"reemplazada el {up._when(r['replaced'])}" if r.get("replaced") else "reemplazada"


_ARTICLE = re.compile(r"^(.*), art\. (\d+)$")


def _articles(sources: list[str]) -> str:
    """'ley.pdf, art. 21', 'ley.pdf, art. 25' -> 'ley.pdf, arts. 21 y 25'."""
    by_law: dict[str, list[int]] = {}
    other = []
    for s in sources:
        m = _ARTICLE.match(s)
        if m:
            by_law.setdefault(m.group(1), []).append(int(m.group(2)))
        else:
            other.append(s)
    out = []
    for law, nums in by_law.items():
        nums = [str(n) for n in sorted(set(nums))]
        out.append(f"{law}, art. {nums[0]}" if len(nums) == 1 else
                   f"{law}, arts. {', '.join(nums[:-1])} y {nums[-1]}")
    return "; ".join(out + list(dict.fromkeys(other)))


def _sources(records: list[dict], law: bool) -> list[str]:
    return [s for r in records if r.get("sources") and _kind(r, LAW_RECORDS) == law
            for s in r["sources"].split("; ")]


def _origin(records: list[dict], file: str, mode: str | None) -> str:
    law = _sources(records, True)
    if law:
        return f"**De qué artículo sale:** {plain(_articles(law))}"
    text = (f"de {plain(file)}, sin un artículo citado" if mode == MODE_LAW else
            f"de tu ontología ({plain(file)}), no de un artículo de una ley")
    defined = _sources(records, False)
    if defined:
        text += f"; su definición sale de {plain(_articles(defined))}"
    return f"**De qué artículo sale:** ninguno: viene {text}."


_SENTENCE = re.compile(r"(?<!\bart\.)(?<!\barts\.)(?<=\.)\s+(?=[A-ZÁÉÍÓÚÑ«¿])")


def _headline(records: list[dict], rows: list[dict], version) -> str:
    current = [r for r in records if r.get("decidedBy") and _in_force(r, records)]
    if current:
        lines = [f"{_triple(r)} · decidida por **{plain(_by(r))}** en la versión {r['version']}"
                 for r in current]
        head = "Correspondencia" if len(current) == 1 else "Correspondencias"
        return f"**{head} en la versión {version}:**  \n" + "  \n".join(lines)
    why = [r for r in records if _kind(r, WHY_RECORDS) and _in_force(r, records)]
    if why:                                       # the whole text is in its step, below
        text = _SENTENCE.split(why[-1].get("comment") or why[-1]["label"], maxsplit=1)[0]
    elif rows:
        state = STATE_LABELS.get(rows[0].get("status"), rows[0].get("status"))
        text = f"{state}. {STATE_HELP[state]}" if state in STATE_HELP else state
    else:
        text = "La procedencia no dice por qué."
    return f"**Sin correspondencia en la versión {version}.** {plain(text)}"


def _version_rows(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        if r.get("decidedBy"):
            what, by = _triple(r), _by(r)
        elif _kind(r, ("user-removal-",)):
            what, by = "ninguna (quitada)", "usuario"
        else:
            continue
        out.append(dict(zip(VERSION_COLUMNS, (int(r.get("version") or 1), what, by,
                                              up._when(r.get("at")), _state_of(r, records)))))
    return out


def _step(r: dict, records: list[dict]) -> dict:
    badge_ = None
    if r.get("decidedBy") or _kind(r, WHY_RECORDS):     # what was decided: in force or not
        badge_ = (":green-badge[Vigente]" if _in_force(r, records) else
                  ":gray-badge[Reemplazada]")
    who = r.get("who") if r.get("who") != r.get("decidedBy") else None
    extra = [f"Revisa: {r['revises']}" if r.get("revises") else None, r.get("notes")]
    return {"title": plain(r["label"]), "badge": badge_, "text": plain(r.get("comment") or ""),
            "meta": " · ".join(x for x in (r.get("activity"), who, up._when(r.get("at")))
                               if x),
            "extra": [plain(x) for x in extra if x]}


def card(records: list[dict], iri: str, rows: list[dict], *, version=1, file: str = "",
         mode: str | None = None) -> dict:
    """What the card shows: the correspondence in force (or why there is none), the article it
    comes from, the versions and every step of its provenance (``records`` from records.rq,
    ``rows`` the lines of alignment.csv of the concept)."""
    name = (rows[0].get("concept_name") if rows else None) or up.local(iri)
    state = STATE_LABELS.get(rows[0].get("status"), rows[0].get("status")) if rows else None
    return {"missing": None, "name": plain(name), "iri": iri, "state": state,
            "headline": _headline(records, rows, version),
            "origin": _origin(records, file, mode), "versions": _version_rows(records),
            "steps": [_step(r, records) for r in records]}


# ------------------------------------------------------------------ only transform (S10-T10)

@lru_cache(maxsize=4)
def _uploaded_graph(name: str, data: bytes):
    """The uploaded ontology read once per file (no workspace, no AI)."""
    from rdflib import Graph
    fmt = input_format(name)                  # a .owx or an unknown extension says why
    graph = Graph()
    try:
        graph.parse(data=data, format=fmt)
    except Exception as exc:                  # rdflib raises many parser-specific errors
        raise ConversionError(f"No se pudo leer {name}: {exc}") from exc
    if not len(graph):
        raise ConversionError(f"{name} está vacío: no tiene nada que convertir.")
    return graph


def transformed_bytes(name: str, data: bytes, key: str) -> bytes:
    """The uploaded ontology written in ``key`` (what a download button calls when clicked)."""
    return serialize(_uploaded_graph(name, data), key)


def transform_downloads(name: str, data: bytes) -> list[dict]:
    """The four buttons of "Solo transformar": the file is read now (an error shows at once);
    each format is written only when its button is clicked."""
    _uploaded_graph(name, data)
    stem = name.rsplit(".", 1)[0] or "ontologia"
    return [{"key": f.key, "label": f.label.split(" (")[0],
             "help": TRANSFORM_JSONLD if f.key == "jsonld" else f.label,
             "file_name": f"{stem}{f.extension}", "mime": f.media_type,
             "data": partial(transformed_bytes, name, data, f.key)}
            for f in OUTPUT_FORMATS.values()]


# ------------------------------------------------------------------ the ready screen (S10-T12)

READY_TITLE = "¡Tu ontología está lista!"
REVIEW_BUTTON = "Revisa tu ontología antes de confirmar tu descarga"
REVIEW_HINT = "Podemos afinar algunos detalles juntos."
PREVIEW_CONCEPTS = 16                     # concepts in the preview, with their DPV terms
PREVIEW_ALIGNED = 10                      # of them, at most this many with a DPV term
PREVIEW_ROOT = "root"                     # the node of the ontology itself, in the middle
SUBCLASS = "es un tipo de"
RELATION_WORDS = {"exactMatch": "igual a", "closeMatch": "parecido a",
                  "broadMatch": "es un tipo de", "narrowMatch": "más general que",
                  "relatedMatch": "relacionado con", "seeAlso": "ver también"}
PREVIEW_CAPTION = ("En azul, tus conceptos; en naranja, su equivalente en el vocabulario de "
                   "privacidad (DPV); en gris, los que no tienen uno. Puedes mover los nodos y "
                   "acercarte.")


def _subclasses(ws, manifest: dict, names: dict) -> list[tuple[str, str]]:
    """The subClassOf links between concepts of the table, read from the person's ontology."""
    from rdflib import RDFS, URIRef
    file = (manifest.get("files") or {}).get("ontology")
    path = ws.path / file if file else None
    if path is None or not path.exists():
        return []
    graph = _graph_at(path)
    return sorted({(str(s), str(o)) for s, o in graph.subject_objects(RDFS.subClassOf)
                   if isinstance(o, URIRef) and str(s) in names and str(o) in names
                   and s != o})


def preview(ws_id: str) -> dict:
    """A small piece of the ontology to draw: its aligned concepts first with their DPV terms,
    then the concepts linked to them by subClassOf, then the rest, up to PREVIEW_CONCEPTS."""
    ws = load_workspace(ws_id, cfg=workspace_cfg())
    rows = _rows(ws)
    names, states, links = {}, {}, {}
    for r in rows:
        iri = r.get("concept_iri")
        if not iri:
            continue
        names.setdefault(iri, r.get("concept_name") or up.local(iri))
        states.setdefault(iri, STATE_LABELS.get(r.get("status"), r.get("status")))
        if r.get("status") in ua.WRITTEN and r.get("dpv_iri"):
            links.setdefault(iri, []).append(r)
    manifest = ua.load_manifest(ws) or {}
    sub = _subclasses(ws, manifest, names)
    chosen = [iri for iri in names if iri in links][:PREVIEW_ALIGNED]
    for s, o in sub:
        for a, b in ((s, o), (o, s)):
            if a in chosen and b not in chosen and len(chosen) < PREVIEW_CONCEPTS:
                chosen.append(b)
    rest = sorted((iri for iri in names if iri not in chosen), key=lambda iri: iri in links)
    chosen += rest[:PREVIEW_CONCEPTS - len(chosen)]      # the ones without a term first
    file = source_name(ws)
    nodes = [{"id": PREVIEW_ROOT, "label": file, "kind": "root", "title": file}]
    nodes += [{"id": iri, "label": names[iri], "kind": "aligned" if iri in links else "none",
               "title": f"{names[iri]}: {states[iri]}"} for iri in chosen]
    edges, terms = [], {}
    for iri in chosen:
        for r in links.get(iri, []):
            terms.setdefault(r["dpv_iri"], r.get("dpv_name") or up.local(r["dpv_iri"]))
            word = RELATION_WORDS.get(up.local(r.get("relation") or ""), "")
            edges.append({"source": iri, "target": r["dpv_iri"], "label": word, "kind": "dpv"})
    nodes += [{"id": iri, "label": name, "kind": "dpv", "title": f"DPV: {name}"}
              for iri, name in terms.items()]
    shown = set(chosen)
    inner = [(s, o) for s, o in sub if s in shown and o in shown]
    edges += [{"source": s, "target": o, "label": SUBCLASS, "kind": "subclass"}
              for s, o in inner]
    tops = [iri for iri in chosen if iri not in {s for s, _ in inner}]
    edges += [{"source": iri, "target": PREVIEW_ROOT, "label": "", "kind": "root"}
              for iri in tops]                          # one graph, not scattered pieces
    return {"nodes": nodes, "edges": edges, "shown": len(chosen), "total": len(names)}


def preview_caption(p: dict) -> str:
    part = (f"Se ven {p['shown']} de tus {p['total']} conceptos. " if p["shown"] < p["total"]
            else "")
    return part + PREVIEW_CAPTION


# ------------------------------------------------------------- what we did, step by step (S10-T13)

WALK_TITLES = ("Qué encontramos", "Cómo los enlazamos con el vocabulario de privacidad",
               "Cómo la revisamos")
WALK_EXAMPLES = 5
DPV_LEAD = ("El DPV (Data Privacy Vocabulary, del W3C) es el vocabulario de referencia para "
            "hablar de privacidad. Enlazar tus conceptos con él permite que otros sistemas "
            "entiendan tu ontología.")
# Each state in plain words: (one, many), with {n} for the number.
WALK_STATES = (
    ((ua.INHERITED_ALIGNED,), ("{n} ya estaba revisado por una persona en la base del sistema.",
                               "{n} ya estaban revisados por una persona en la base del "
                               "sistema.")),
    ((ua.ACCEPTED,), ("La IA eligió el equivalente de {n} entre los candidatos del DPV y la "
                      "regla del sistema lo aceptó.", "La IA eligió el equivalente de {n} entre "
                      "los candidatos del DPV y la regla del sistema los aceptó.")),
    ((ua.USER_ALIGNED,), ("Tú decidiste {n}.", "Tú decidiste {n}.")),
    ((ua.SUGGESTED,), ("En {n}, la IA propuso un equivalente que la regla no aceptó: te lo "
                       "preguntaremos.", "En {n}, la IA propuso un equivalente que la regla no "
                       "aceptó: te los preguntaremos.")),
    ((ua.INHERITED_NO_MATCH, ua.RULE_NO_MATCH, ua.USER_NO_MATCH),
     ("{n} no tiene un equivalente en el DPV (no todo concepto lo tiene).",
      "{n} no tienen un equivalente en el DPV (no todo concepto lo tiene).")),
    ((ua.NO_PROPOSAL, ua.UNDECIDED), ("En {n} no hubo una propuesta lo bastante clara.",
                                      "En {n} no hubo una propuesta lo bastante clara.")),
    ((ua.NOT_ALIGNABLE,), ("{n} es un ejemplo de datos y no se enlaza.",
                           "{n} son ejemplos de datos y no se enlazan.")),
)
# A correspondence as a sentence: "Consentimiento es igual a Consent".
RELATION_VERBS = {"exactMatch": "es igual a", "closeMatch": "se parece a",
                  "broadMatch": "es un tipo de", "narrowMatch": "es más general que",
                  "relatedMatch": "está relacionado con", "seeAlso": "se relaciona con"}
REVIEW_LEADS = {GREEN: "Todo está en orden.",
                YELLOW: "Está lista para usarse; hay detalles que puedes mirar.",
                RED: "Encontramos un problema que conviene corregir antes de usarla."}
NO_REVIEW = ("Esta ontología se alineó antes de que existiera la revisión: vuelve a subir el "
             "archivo para revisarla.")


def _plural(n: int, one: str, many: str) -> str:
    return (one if n == 1 else many).format(n=n)


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} y {items[-1]}"


def _arts(numbers) -> str:
    """'del art. 4', 'de los arts. 4, 25 y 89'."""
    nums = [str(n) for n in sorted(set(numbers))]
    return f"del art. {nums[0]}" if len(nums) == 1 else f"de los arts. {_and(nums)}"


def _name(r: dict) -> str:
    """'**Datos personales** (PersonalData)', or the name alone."""
    label, name = r.get("label_es") or "", r.get("concept_name") or ""
    if label and label.lower() != name.lower():
        return f"**{plain(label)}** ({plain(name)})"
    return f"**{plain(name)}**"


def _found(ws, manifest: dict, rows: dict) -> dict:
    profile = manifest.get("profile")
    concepts = manifest["summary"]["concepts"]
    points, examples = [], []
    if profile:                                                # "Una ley"
        x = (profile.get("law_concepts") or {}).get("extraction") or profile.get(
            "extraction") or {}
        read = f" ({counted(x['articles_total'], 'artículo', 'artículos')})" if x else ""
        lead = (f"Leímos {plain(profile['law'])}{read} y la IA encontró "
                f"{counted(concepts, 'concepto', 'conceptos')} de protección de datos.")
        points.append(f"{_plural(profile['classes'], '{n} es una idea o un actor', '{n} son '
                                 'ideas o actores')} (como «consentimiento» o «responsable») y "
                      f"{_plural(profile['properties'], '{n} es una relación', '{n} son '
                                 'relaciones')} entre ellos.")
        if profile.get("reused"):
            points.append(_plural(profile["reused"], "{n} ya estaba en la base del sistema y se "
                                  "reutiliza tal cual.", "{n} ya estaban en la base del sistema "
                                  "y se reutilizan tal cual."))
        by_name = {r["concept_name"]: r for r in rows.values()}
        entries = sorted((e for e in (_law_concepts(ws) or {}).get("entries", [])
                          if e.get("status") == "accepted" and e.get("articles")),
                         key=lambda e: e["name"] not in by_name)    # with its Spanish name first
        examples = [f"{_name(by_name.get(e['name'], {'concept_name': e['name']}))} · sale "
                    f"{_arts(e['articles'])}" for e in entries[:WALK_EXAMPLES]]
    else:                                                      # an ontology (and a law)
        lead = (f"Tu ontología ({plain(manifest['uploaded_file'])}) tiene "
                f"{counted(concepts, 'concepto', 'conceptos')}.")
        defined = manifest.get("law_definitions") or {}
        if defined and not defined.get("skipped"):
            added = defined["law_concept"] + defined["articles"]
            points.append(f"A {defined['missing']} les faltaba la definición; tu ley "
                          f"({plain(defined['law'])}) dio {added}.")
        evidence = manifest.get("law_evidence") or {}
        if evidence.get("links"):
            points.append(f"Buscamos en {plain(evidence['law'])} los artículos más cercanos a "
                          f"cada concepto: {evidence['with_evidence']} de "
                          f"{evidence['concepts']} tienen alguno.")
        examples = [_name(r) + (f" · cerca {_arts(map(int, nums))} de tu ley" if (
            nums := re.findall(r"art\. (\d+)", r.get("law_evidence") or "")) else "")
            for r in list(rows.values())[:WALK_EXAMPLES]]
    labels = manifest.get("labels") or {}
    if labels.get("translated"):
        points.append(_plural(labels["translated"], "La IA completó el nombre en español e "
                              "inglés de 1 concepto.", "La IA completó los nombres en español e "
                              "inglés de {n} conceptos."))
    return {"title": WALK_TITLES[0], "lead": lead, "points": points,
            "examples_title": "Algunos de tus conceptos", "examples": examples}


def _linked(manifest: dict, rows: dict) -> dict:
    by = manifest["summary"]["by_status"]
    points = [_plural(n, one, many) for codes, (one, many) in WALK_STATES
              if (n := sum(by.get(c, 0) for c in codes))]
    examples = []
    for r in rows.values():
        if r.get("status") in ua.WRITTEN and r.get("dpv_name") and len(examples) < WALK_EXAMPLES:
            verb = RELATION_VERBS.get(up.local(r.get("relation") or ""), "se relaciona con")
            examples.append(f"{_name(r)} {verb} *{plain(r['dpv_name'])}*")
    return {"title": WALK_TITLES[1], "lead": DPV_LEAD, "points": points,
            "examples_title": "Algunos enlaces", "examples": examples}


def _checked(ws) -> dict:
    result = uv.load_result(ws)
    if result is None:
        return {"title": WALK_TITLES[2], "lead": NO_REVIEW, "points": [], "examples_title": "",
                "examples": []}
    ins, points = result["inspectors"], []
    for key, ok, bad, base in (
            (uv.SHAPES, "Está bien formada: cumple las reglas de lo que se entrega",
             ("{n} error de forma nuevo", "{n} errores de forma nuevos"),
             ("el único problema de forma que aparece", "los {n} problemas de forma que "
              "aparecen", "lo", "los")),
            (uv.LOGIC, "No tiene contradicciones propias",
             ("{n} contradicción nueva", "{n} contradicciones nuevas"),
             ("la única contradicción que aparece", "las {n} contradicciones que aparecen",
              "la", "las"))):
        b = ins.get(key) or {}
        if b.get("status") in (None, uv.NOT_RUN, uv.FAILED):
            continue
        text = ok if not b["new"] else "Tiene " + _plural(b["new"], *bad)
        if b.get("inherited"):
            n = b["inherited"]
            one, many, it, them = base
            text += (f"; {_plural(n, one, many)} ya {'estaba' if n == 1 else 'estaban'} en tu "
                     f"archivo o en la base del sistema: no {it if n == 1 else them} causó la "
                     "alineación")
        points.append(text + ".")
    q = (ins.get(uv.QUESTIONS) or {})
    if q.get("summary"):
        s = q["summary"]
        applies = s[GREEN] + s[YELLOW] + s[qs.GRAY]
        first = next((r["question"] for r in q.get("results", []) if r.get("applies", True)), "")
        points.append(f"Responde {s[GREEN]} de {applies} preguntas básicas, como «{first}»."
                      if first else f"Responde {s[GREEN]} de {applies} preguntas básicas.")
    a = ins.get(uv.AUDIT) or {}
    if a.get("summary", {}).get("accepted") and a.get("status") != uv.NOT_RUN:
        s = a["summary"]
        parts = [_plural(s["coherente"], "{n} está bien", "{n} están bien"),
                 _plural(s["dudosa"], "{n} es dudoso", "{n} son dudosos"),
                 _plural(s["incoherente"], "{n} parece errado", "{n} parecen errados")]
        points.append(_plural(s["accepted"], "La IA volvió a revisar el enlace que eligió: ",
                              "La IA volvió a revisar los {n} enlaces que eligió: ")
                      + _and(parts) + ".")
    elif a.get("status") == uv.NOT_RUN and a.get("summary", {}).get("accepted"):
        points.append("La IA no pudo volver a revisar los enlaces que eligió.")
    return {"title": WALK_TITLES[2], "lead": REVIEW_LEADS.get(result["status"], ""),
            "points": points, "examples_title": "", "examples": []}


def walkthrough(ws_id: str) -> list[dict]:
    """The three screens of "Lo que hicimos" in plain words: no SHACL, SPARQL, Jena or IRIs."""
    ws = load_workspace(ws_id, cfg=workspace_cfg())
    manifest = ua.load_manifest(ws)
    if manifest is None or "decider" not in manifest:
        raise WorkspaceError("La alineación de este archivo no terminó; vuelve a subirlo.")
    rows = {}
    for r in _rows(ws):
        rows.setdefault(r["concept_iri"], r)
    return [_found(ws, manifest, rows), _linked(manifest, rows), _checked(ws)]


# ------------------------------------------------------------------ you decide (S10-T14)

DECIDE_TITLE = "Tú decides"
DECIDE_LEAD = ("Encontramos {n} para afinar. Responder es opcional: lo que no toques se queda "
               "como está.")
NOTHING_TO_DECIDE = "No encontramos nada en duda: no hace falta que decidas nada."
APPLIED_TITLE = "¡Lista con los cambios que aprobaste!"
NOTHING_APPLIED = "No cambiaste nada: tu ontología se queda como estaba."
RECALCULATING = "Recalculando tu ontología…"
OLD_QUESTIONS = ("Las preguntas eran de la versión {old} y tu ontología ya va en la {new}: "
                 "vuelve a mirarlas.")
ADJUST_MAX = 3                            # DPV candidates offered for another concept


def decisions_view(ws_id: str) -> dict:
    """The questions of the current version (S10-T05) as the web asks them, one at a time."""
    from src.alignment import user_questions as uq
    ws = load_workspace(ws_id, cfg=workspace_cfg())
    data = uq.load_questions(ws) or {"version": None, "questions": []}
    questions = [{"n": q["n"], "text": plain(q["text"]),
                  "why": plain(_SENTENCE.split(q.get("reason") or "", maxsplit=1)[0]),
                  "concept": q["doubts"][0], "leave": q["leave"],
                  "options": {**{o["letter"]: plain(o["text"]) + (
                      " (recomendada)" if n == q.get("recommended") else "")
                      for n, o in enumerate(q["options"])}, q["leave"]: uq.LEAVE}}
                 for q in data["questions"]]
    return {"version": data["version"], "questions": questions}


def _after_text(c) -> str:
    """'Safeguards: ahora está relacionado con Safeguard' (plain words, no SKOS)."""
    now = [f"{RELATION_VERBS.get(up.local(m.get('relation') or ''), 'se relaciona con')} "
           f"{m.get('dpv_name')}" for m in c.after]
    return f"{c.concept_name}: ahora " + (" y ".join(now) if now else "no tiene equivalente")


def adjust_options(ws_id: str, iri: str, *, targets=None) -> list[dict]:
    """What another concept can become: one of its DPV candidates (the decider's, best first)
    or no correspondence; each option already passed the fixed rule of S10-T03."""
    from src.alignment import user_decisions as ud
    from src.alignment.propose import load_result as decisions_result
    ws = load_workspace(ws_id, cfg=workspace_cfg())
    cfg = workspace_cfg()
    current = [r for r in _rows(ws) if r.get("concept_iri") == iri
               and r.get("status") in ua.WRITTEN and r.get("dpv_iri")]
    entry = next((e for e in (decisions_result(ws) or {}).get("entries", [])
                  if e.get("key") == iri), {})
    replaces = current[0]["dpv_iri"] if len(current) == 1 else None

    def valid(change) -> bool:      # alone: the rule takes one change per concept in a list
        return ud.check(ws, [change], targets=targets, cfg=cfg)[0].ok
    options = [(f"Es lo mismo que {up.local(term)}",
                ud.Change(iri, ud.SET, relation="skos:exactMatch", dpv=term, replaces=replaces))
               for term in entry.get("candidates", [])]
    options = [o for o in options if valid(o[1])][:ADJUST_MAX]   # not what it already is
    remove = ("No tiene equivalente (quitar la correspondencia)", ud.Change(iri, ud.REMOVE))
    options += [remove] if valid(remove[1]) else []              # only when it has one
    return [{"text": text, "change": {"action": c.action, "relation": c.relation,
                                      "dpv": c.dpv, "replaces": c.replaces}}
            for text, c in options]


def apply_answers(ws_id: str, answers: dict[int, str], extra: list[dict] = (), *,
                  targets=None, llm=None, use_llm: bool | None = None, sleep_fn=None) -> dict:
    """The answers ({question: letter}) and the other concepts the person adjusted become one
    new version (S10-T03); "Dejar como está" is remembered and not asked again."""
    from src.alignment import user_decisions as ud
    from src.alignment import user_questions as uq
    ws = load_workspace(ws_id, cfg=workspace_cfg())
    cfg = workspace_cfg()
    data = uq.load_questions(ws) or {"version": ud.current_version(ws), "questions": []}
    version = ud.current_version(ws)
    if data["questions"] and data["version"] != version:
        raise WorkspaceError(OLD_QUESTIONS.format(old=data["version"], new=version))
    asked = {q["n"]: q for q in data["questions"]}
    changes, left = [], []
    for n, letter in sorted(answers.items()):
        q = asked[n]
        if letter == q["leave"]:
            left += [{"concept_key": key, "version": version, "question": q["text"],
                      "at": ud._now()} for key in q["doubts"]]
            continue
        option = next(o for o in q["options"] if o["letter"] == letter)
        changes += [ud.Change(c["concept"], c["action"], relation=c["relation"], dpv=c["dpv"],
                              replaces=c["replaces"], note=f"{q['text']} → {option['text']}")
                    for c in option["apply"]]
    changes += [ud.Change(e["concept"], e["action"], relation=e.get("relation"),
                          dpv=e.get("dpv"), replaces=e.get("replaces"), note=e.get("note"))
                for e in extra]
    if left:
        uq.keep(ws, left)
    out = {"applied": False, "version": version, "left": len(left), "changes": [],
           "refused": []}
    if changes:
        result = ud.apply(ws, changes, cfg=cfg, targets=targets, llm=llm, use_llm=use_llm,
                          sleep_fn=sleep_fn)
        out.update(applied=result["applied"], version=result["version"],
                   changes=[_after_text(c) for c in result["checked"] if c.ok],
                   refused=[f"{c.concept_name or up.local(c.change.concept)}: {c.reason}"
                            for c in result["checked"]
                            if not c.ok])
    if not out["applied"] and left:
        uq.refresh_questions(ws, cfg, targets=targets, llm=llm, use_llm=use_llm)
    return out


# ------------------------------------------------------------------ the download (S10-T15)

FINAL_TITLE = "Descarga tu ontología"
FINAL_CAPTION = ("Un solo archivo con todo: tus conceptos, sus equivalentes en el DPV, los "
                 "nombres en español e inglés y sus definiciones{core}. Ábrelo directamente en "
                 "Protégé.")
FINAL_CORE = ", junto con lo que usa de la base del sistema (OntoPriv-Core)"
FINAL_NOTE = "Protégé 5.6 no abre JSON-LD: para Protégé usa RDF/XML, OWL o Turtle."
DETAILS_TOGGLE = "Ver los detalles técnicos"


def _source(ws) -> dict:
    """What the person uploaded: the law in "Una ley" (the ontology is built from it), else
    the ontology."""
    inputs = ws.manifest["inputs"]
    return inputs["law" if inputs["ontology"].get("generated_from") == "law" else "ontology"]


def source_name(ws) -> str:
    """'gdpr-en.pdf' (not "Perfil de gdpr-en.pdf"), 'Mi GDPR.ttl'."""
    return _source(ws)["original_name"]


def aligned_stem(ws) -> str:
    """'gdpr-en-dpv-aligned', 'mi-gdpr-dpv-aligned': the name of the file the person gets."""
    return f"{Path(_source(ws)['stored_as']).stem}-dpv-aligned"


def final_bytes(ws_id: str, key: str, cfg: dict | None = None) -> bytes:
    """The aligned ontology as one file in ``key`` (what a button calls when clicked)."""
    return ua.single_file(load_workspace(ws_id, cfg=cfg), key)


def previous_bytes(ws_id: str, version: int, cfg: dict | None = None) -> bytes:
    return ua.previous_single_file(load_workspace(ws_id, cfg=cfg), version, "turtle")


def final_downloads(ws_id: str) -> dict:
    """The four buttons of the last screen (one file each, built only when clicked) and, from
    version 2 on, the previous version in Turtle when history/ kept its package."""
    from src.alignment import user_decisions as ud
    cfg = workspace_cfg()
    ws = load_workspace(ws_id, cfg=cfg)
    manifest = ua.load_manifest(ws) or {}
    version = ud.current_version(ws)
    stem = aligned_stem(ws)
    tag = f"-v{version}" if version > 1 else ""
    buttons = [{"key": f.key, "label": f.label.split(" (")[0],
                "help": TRANSFORM_JSONLD if f.key == "jsonld" else f.label,
                "file_name": f"{stem}{tag}{f.extension}", "mime": f.media_type,
                "data": partial(final_bytes, ws_id, f.key, cfg)}
               for f in OUTPUT_FORMATS.values()]
    previous = None
    old = version - 1
    if old >= 1 and (ud.history_dir(ws, old) / ua.package_name(ws, "turtle", old)).exists():
        previous = {"label": f"Descargar la versión anterior (versión {old}, Turtle)",
                    "file_name": f"{stem}-v{old}.ttl",
                    "data": partial(previous_bytes, ws_id, old, cfg)}
    core = FINAL_CORE if (manifest.get("modules") or {}).get("core") else ""
    return {"version": version, "buttons": buttons, "previous": previous,
            "caption": FINAL_CAPTION.format(core=core)}
