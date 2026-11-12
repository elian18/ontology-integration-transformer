# Asistente de alineación: three steps kept in st.session_state, one screen each.
import hashlib
from functools import partial

import pandas as pd
import streamlit as st
from services import align_ontology as svc

ss = st.session_state


@st.cache_resource(show_spinner=False)
def _base():
    from src.alignment.inherit import load_base
    return load_base()


@st.cache_resource(show_spinner=False)
def _targets():
    from src.alignment.propose import load_dpv_targets
    return load_dpv_targets()


def _restart(mode: str | None = None) -> None:
    """Go to step 1 (``mode`` None) or to step 2 of ``mode``: new uploaders, nothing computed."""
    ss["align_round"] = ss.get("align_round", 0) + 1
    for key in ("align_key", "align_ws", "align_failed"):
        ss.pop(key, None)
    if mode is None:
        ss.pop("align_mode", None)
    else:
        ss["align_mode"] = mode


def _retry() -> None:
    ss.pop("align_failed", None)


def _data_note() -> None:
    with st.expander("¿Qué sucede con mis datos?", icon=":material/lock:"):
        st.markdown(svc.DATA_NOTE)


# ------------------------------------------------------------------ step 1

def step_choose() -> None:
    st.title("Transforma y alinea tu ontología de privacidad")
    st.caption("Elige qué tienes. El resto es automático: alineación con el Data Privacy "
               "Vocabulary (DPV) y etiquetas en español e inglés.")
    for col, (code, label) in zip(st.columns(len(svc.MODES)), svc.MODES.items()):
        with col.container(border=True, height="stretch", vertical_alignment="distribute"):
            with st.container():                          # text on top, button at the bottom
                st.markdown(f"### {svc.MODE_ICONS[code]} {label}")
                st.write(svc.MODE_HELP[code])
            st.button("Empezar", key=f"align_pick_{code}", type="primary", width="stretch",
                      icon=":material/arrow_forward:", on_click=_restart, args=(code,))
    _data_note()


# ------------------------------------------------------------------ step 2

def _uploads(mode: str) -> dict:
    """The uploaders of ``mode`` side by side; their keys change with every new round."""
    round_ = ss.get("align_round", 0)
    kinds = svc.MODE_FILES[mode]
    files = {}
    for col, kind in zip(st.columns(len(kinds)), kinds):
        types = svc.ACCEPTED_TYPES if kind == svc.KIND_ONTOLOGY else svc.LAW_TYPES
        with col:
            up = st.file_uploader(svc.UPLOAD_LABELS[kind], type=types,
                                  key=f"align_{kind}_{round_}")
        files[kind] = (up.name, up.getvalue()) if up is not None else None
    return files


def _files_key(mode: str, files: dict) -> str:
    digest = hashlib.sha256(mode.encode("utf-8"))
    for kind in svc.MODE_FILES[mode]:
        name, data = files[kind]
        digest.update(kind.encode("utf-8") + name.encode("utf-8") + data)
    return digest.hexdigest()


def _failed(message: str) -> None:
    st.error(message)                                     # outside the progress box
    st.button("Intentar de nuevo", key="align_retry", icon=":material/refresh:",
              on_click=_retry)


def step_upload(mode: str) -> None:
    st.button("Cambiar de opción", key="align_back", type="tertiary",
              icon=":material/arrow_back:", on_click=_restart)
    st.title(svc.MODES[mode])
    files = _uploads(mode)
    if not svc.ready(mode, files):
        st.caption(svc.missing_text(mode, files))
        _data_note()
        return
    key = _files_key(mode, files)
    failed = ss.get("align_failed")
    if failed and failed[0] == key:                       # the same files failed before
        _failed(failed[1])
        return
    error = None
    with st.status(svc.MODE_RUNNING[mode], expanded=True) as status:
        try:
            ws_id = svc.run(mode, files, progress=lambda label: status.write(f"{label}…"),
                            base=_base(), targets=_targets())
        except Exception as exc:                          # noqa: BLE001 - shown to the person
            error = svc.error_message(exc)
            status.update(label="No se pudo terminar la alineación", state="error",
                          expanded=False)
        else:
            status.update(label="Alineación lista", state="complete", expanded=False)
    if error is not None:
        ss["align_failed"] = (key, error)
        _failed(error)
        return
    ss["align_key"], ss["align_ws"] = key, ws_id
    st.rerun()


# ------------------------------------------------------------------ step 3

def _summary_tab(v: dict) -> None:
    for col, (label, value) in zip(st.columns(len(v["metrics"])), v["metrics"].items()):
        col.metric(label, value, help=svc.METRIC_HELP.get(label))
    for line in svc.summary_captions(v):
        st.caption(line)
    with st.expander("¿Qué significa cada estado?"):
        for state, text in svc.STATE_HELP.items():
            st.markdown(f"**{state}.** {text}")


def _concepts_tab(v: dict) -> None:
    present = list(dict.fromkeys(r["Estado"] for r in v["rows"]))
    state = st.selectbox("Mostrar", ["Todos"] + present, key="align_filter")
    rows = svc.filter_rows(v["rows"], state)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"{svc.counted(len(rows), 'fila', 'filas')}. Un concepto con varias "
               "correspondencias ocupa una fila por cada una.")
    _provenance(v)


def _provenance(v: dict) -> None:
    """The card of the chosen concept, read from its provenance module (S10-T08)."""
    st.markdown(f"**{svc.PROVENANCE_TITLE}**")
    options = v["concept_options"]
    iri = st.selectbox("Concepto", list(options), index=None, format_func=options.get,
                       placeholder=svc.PROVENANCE_PLACEHOLDER, key="align_concept",
                       help=svc.PROVENANCE_HELP)
    if iri is None:
        return
    c = svc.provenance_card(v["id"], iri)
    with st.container(border=True, key="align_card"):
        if c["missing"]:
            st.caption(c["missing"])
            return
        st.markdown(f"#### {c['name']}" + (f" :gray-badge[{c['state']}]" if c["state"] else ""))
        st.caption(c["iri"])
        st.markdown(c["headline"])
        st.markdown(c["origin"])
        st.markdown("**Versiones**")
        if c["versions"]:
            st.dataframe(pd.DataFrame(c["versions"]), hide_index=True, width="stretch")
        else:
            st.caption("Ninguna versión le dio una correspondencia.")
        st.markdown("**Cómo se produjo**")
        for s in c["steps"]:
            head = (f"{s['badge']} " if s["badge"] else "") + f"**{s['title']}**"
            st.markdown("  \n".join([head, *([s["text"]] if s["text"] else []), *s["extra"]]))
            if s["meta"]:
                st.caption(s["meta"])
        if not c["steps"]:
            st.caption("La procedencia no dice nada más de este concepto.")


def _review_tab(v: dict) -> None:
    flagged = (v["validation"] or {}).get("flagged") or []
    if flagged:
        st.markdown(f"**Marcadas por la IA revisora ({len(flagged)})**")
        st.caption("Están en tu ontología (la regla las aceptó); la IA explica por qué el "
                   "término del DPV podría no tratar de lo mismo.")
        st.dataframe(pd.DataFrame(flagged), hide_index=True, width="stretch")
    suggested = svc.suggested_rows(v["rows"])
    if suggested:
        st.markdown(f"**Sugeridas por la IA ({len(suggested)})**")
        st.caption("La IA las propuso con confianza alta, pero la regla no las aceptó: no están "
                   "en tu ontología.")
        st.dataframe(pd.DataFrame(suggested), hide_index=True, width="stretch")
    if not flagged and not suggested:
        st.info("No hay nada pendiente de revisar.")
    st.caption("Todo esto también está en alignment.csv, dentro del .zip.")


def _law_tab(v: dict) -> None:
    st.caption(v["evidence"])
    evidence = svc.evidence_rows(v["rows"])
    if evidence:
        st.dataframe(pd.DataFrame(evidence), hide_index=True, width="stretch")


def _questions_tab(v: dict) -> None:
    q = v["validation"]["questions"]
    st.caption(q["caption"])
    for r in q["rows"]:
        with st.container(border=True):
            st.markdown(f"{r['badge']} **{r['id']}** · {r['question']}")
            st.markdown(f"**Respuesta:** {r['answer']}")
            if r["why"]:
                st.markdown(f"**{r['why_label']}:** {r['why']}")
            st.caption(r["criterion"])
            with st.expander("Ver la consulta SPARQL"):
                if r["query"]:
                    st.code(r["query"], language="sparql")
                else:
                    st.caption("Esta consulta ya no está en el catálogo de preguntas.")


def _validation(v: dict) -> None:
    val = v["validation"]
    st.markdown("**Validación**")
    with st.container(border=True):
        if val is None:
            st.caption(svc.NO_VALIDATION)
            return
        st.markdown(f"{val['badge']} {val['headline']}")
        for col, card in zip(st.columns(len(val["cards"])), val["cards"]):
            with col:
                st.caption(f"Criterio: {card['criterion']}")
                st.markdown(f"**{card['title']}** {card['badge']}")
                st.markdown(card["text"])
                st.caption(card["caption"])
        with st.expander("¿Qué revisa cada uno?"):
            for title, text in svc.VALIDATION_HELP.items():
                st.markdown(f"**{title}.** {text}")
        with st.expander("Ver el detalle"):
            if val["rules"]:
                st.markdown("**Forma: reglas con hallazgos**")
                st.dataframe(pd.DataFrame(val["rules"]), hide_index=True, width="stretch")
            if val["kinds"]:
                st.markdown("**Lógica: tipos de hallazgo**")
                st.dataframe(pd.DataFrame(val["kinds"]), hide_index=True, width="stretch")
            if not val["rules"] and not val["kinds"]:
                st.caption("La forma y la lógica no tienen hallazgos.")
            st.download_button(f"Descargar el detalle ({val['log_file']})",
                               data=partial(svc.validation_log, v["id"], svc.workspace_cfg()),
                               file_name=val["log_file"], mime="text/plain", key="align_dl_log",
                               on_click="ignore", icon=":material/description:")


def step_result(ws_id: str) -> None:
    try:
        v = svc.view(ws_id)
        downloads = svc.download_options(ws_id)
    except Exception as exc:                              # noqa: BLE001 - shown to the person
        st.error(svc.error_message(exc))
        st.button("Empezar de nuevo", key="align_again", icon=":material/restart_alt:",
                  on_click=_restart)
        return
    warnings = svc.notes(v)
    st.success(svc.summary_line(v), title="Tu ontología está lista",
               icon=":material/check_circle:")
    for note in warnings:
        st.warning(note, icon=":material/warning:")
    st.markdown("**Descárgala en el formato que necesites**")
    for col, d in zip(st.columns(len(downloads)), downloads):
        col.download_button(d["label"], data=d["data"], file_name=d["file_name"],
                            mime="application/zip", key=f"align_dl_{d['key']}", help=d["help"],
                            on_click="ignore", icon=":material/download:", width="stretch")
    st.caption(svc.package_caption(v))
    _validation(v)
    pages = {"Resumen": _summary_tab, "Conceptos": _concepts_tab, "Para revisar": _review_tab}
    if (v["validation"] or {}).get("questions"):
        pages[svc.QUESTIONS_TAB] = _questions_tab
    if v["evidence"]:
        pages[svc.LAW_TAB] = _law_tab
    for tab, show in zip(st.tabs(list(pages)), pages.values()):
        with tab:
            show(v)
    st.button("Alinear otro archivo", key="align_again", icon=":material/restart_alt:",
              on_click=_restart)


# ------------------------------------------------------------------ which step

if ss.get("align_ws"):
    step_result(ss["align_ws"])
elif ss.get("align_mode") in svc.MODES:
    step_upload(ss["align_mode"])
else:
    step_choose()