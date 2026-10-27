"""Page "Asistente de alineación" (S8-T01; before S8 the page "Alinear con el DPV" of S6-T08,
S7-T06, S7-T08 and S7-T12): the user's flow, as little to do as possible, in three steps kept
in st.session_state, one screen each.

1. "¿Qué tienes?": one card per mode ("Mi ontología", "Una ley", "Mi ontología y una ley").
2. The uploader(s) of that mode. The alignment starts on its own when the files of the mode are
   there, with a progress box that names each step. An error is shown outside the box, and the
   same files do not start again until the person asks for it ("Intentar de nuevo"), so a
   failure never spends AI calls in a loop.
3. The result: a banner, the AI warnings (always visible: they change what the person gets),
   four download buttons (one zip per format, each built only when clicked) and, below, tabs
   with the numbers, the table per concept and what is left to review. The validation panel
   (S8-T09) goes between the downloads and the tabs.

"Cambiar de opción" and "Alinear otro archivo" go back to step 1 and the next uploaders are NEW
ones (S7-T12): Streamlit keeps the file of an uploader that was hidden, and with the same key
coming back would start a long run (and AI calls) with a file the person no longer sees.
Uploading the same files again reuses what was already computed (S6-T04). Nothing to review:
the ontology holds what a person decided while building the base and what the calibrated rule
accepted; the suggestions and the articles of the law are only shown (comments: Sprint 10).
"""
import hashlib

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
    for key in ("align_key", "align_ws", "align_failed", "align_new"):
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
    ss["align_key"], ss["align_ws"], ss["align_new"] = key, ws_id, True
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
    st.caption(f"{len(rows)} filas. Un concepto con varias correspondencias ocupa una fila por "
               "cada una.")


def _review_tab(v: dict) -> None:
    suggested = svc.suggested_rows(v["rows"])
    if suggested:
        st.markdown(f"**Sugeridas por la IA ({len(suggested)})**")
        st.caption("La IA las propuso con confianza alta, pero la regla no las aceptó: no están "
                   "en tu ontología.")
        st.dataframe(pd.DataFrame(suggested), hide_index=True, width="stretch")
    if v["evidence"]:
        st.markdown("**Artículos de la ley para cada concepto**")
        st.caption(v["evidence"])
        evidence = svc.evidence_rows(v["rows"])
        if evidence:
            st.dataframe(pd.DataFrame(evidence), hide_index=True, width="stretch")
    if not suggested and not v["evidence"]:
        st.info("No hay nada pendiente de revisar.")
    st.caption("Todo esto también está en alignment.csv, dentro del .zip.")


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
    if ss.pop("align_new", False) and not warnings:      # once, right after a clean alignment
        st.balloons()
    st.markdown("**Descárgala en el formato que necesites**")
    for col, d in zip(st.columns(len(downloads)), downloads):
        col.download_button(d["label"], data=d["data"], file_name=d["file_name"],
                            mime="application/zip", key=f"align_dl_{d['key']}", help=d["help"],
                            on_click="ignore", icon=":material/download:", width="stretch")
    st.caption(svc.package_caption(v))
    # S8-T09: the validation panel (SHACL + reasoner) goes here.
    summary, concepts, review = st.tabs(["Resumen", "Conceptos", "Para revisar"])
    with summary:
        _summary_tab(v)
    with concepts:
        _concepts_tab(v)
    with review:
        _review_tab(v)
    st.button("Alinear otro archivo", key="align_again", icon=":material/restart_alt:",
              on_click=_restart)


# ------------------------------------------------------------------ which step

if ss.get("align_ws"):
    step_result(ss["align_ws"])
elif ss.get("align_mode") in svc.MODES:
    step_upload(ss["align_mode"])
else:
    step_choose()