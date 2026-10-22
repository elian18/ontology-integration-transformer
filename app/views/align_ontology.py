"""Page "Alinear con el DPV" (S6-T08, S7-T06, S7-T08, S7-T12): the user's flow, as little to do
as possible, in the three modes of the system.

1. Pick what to upload (S7-T12): "Mi ontología", "Una ley" or "Mi ontología y una ley".
2. Upload: the flow starts on its own when the files of the mode are there (no extra button),
   with a progress box that names each step.
3. See the result: five numbers, the rule that decided, what was done with the labels in
   Spanish and English and, with a law, what was built from it ("Una ley") or the articles to
   review ("Mi ontología y una ley"); if wanted, a read-only table.
4. Pick the format and download the zip.
Nothing to review: the ontology holds what a person decided while building the base and what
the calibrated rule accepted; the suggestions and the articles of the law are only shown
(comments come in Sprint 10). Uploading the same files again reuses what was already computed
(S6-T04), so changing the format costs nothing.
"""
import hashlib

import pandas as pd
import streamlit as st
from services import align_ontology as svc

st.header("Alinear con el DPV")
st.write(
    "Sube tu ontología de protección de datos, una ley o las dos, y el sistema te devuelve una "
    "ontología alineada con el **Data Privacy Vocabulary (DPV)**. No tienes que revisar nada: "
    "lo que ya fue validado en la base del sistema (OntoPriv + LOPDP) se aplica solo; para lo "
    "demás, la IA elige entre los términos más parecidos del DPV y una regla calibrada decide "
    "qué entra. Lo que la IA sugiere y la regla no acepta se muestra aparte, sin entrar a la "
    "ontología. Además, completa las etiquetas en español e inglés que falten."
)


@st.cache_resource(show_spinner=False)
def _base():
    from src.alignment.inherit import load_base
    return load_base()


@st.cache_resource(show_spinner=False)
def _targets():
    from src.alignment.propose import load_dpv_targets
    return load_dpv_targets()


@st.cache_data(show_spinner=False, max_entries=8)
def _download(ws_id: str, fmt: str, created: str) -> dict:
    return svc.download(ws_id, fmt)


codes = list(svc.MODES)
mode = st.radio("¿Qué vas a subir?", codes, format_func=svc.MODES.get, horizontal=True,
                captions=[svc.MODE_HELP[c] for c in codes], key="align_mode")

# New uploaders every time the mode changes: a file uploaded before (in this or another mode)
# never starts a long run, with its AI calls, without the person seeing it.
if st.session_state.get("align_last_mode") != mode:
    st.session_state["align_last_mode"] = mode
    st.session_state["align_round"] = st.session_state.get("align_round", 0) + 1
round_ = st.session_state["align_round"]
files = {}
if svc.KIND_ONTOLOGY in svc.MODE_FILES[mode]:
    onto = st.file_uploader("Tu ontología: RDF/XML (.rdf, .owl), Turtle (.ttl) o JSON-LD "
                            "(.jsonld)", type=svc.ACCEPTED_TYPES, key=f"align_file_{round_}")
    files[svc.KIND_ONTOLOGY] = (onto.name, onto.getvalue()) if onto is not None else None
if svc.KIND_LAW in svc.MODE_FILES[mode]:
    law = st.file_uploader("Tu ley: texto (.txt) o PDF", type=svc.LAW_TYPES,
                           key=f"align_law_{round_}")
    files[svc.KIND_LAW] = (law.name, law.getvalue()) if law is not None else None

if svc.ready(mode, files):
    digest = hashlib.sha256(mode.encode("utf-8"))
    for kind in svc.MODE_FILES[mode]:
        name, data = files[kind]
        digest.update(kind.encode("utf-8") + name.encode("utf-8") + data)
    key = digest.hexdigest()
    if st.session_state.get("align_key") != key:
        error = None
        with st.status(svc.MODE_RUNNING[mode], expanded=True) as status:
            try:
                ws_id = svc.run(mode, files,
                                progress=lambda label: status.write(f"{label}…"),
                                base=_base(), targets=_targets())
            except Exception as exc:                      # noqa: BLE001 - shown to the person
                error = exc
                status.update(label="No se pudo terminar la alineación", state="error",
                              expanded=False)
            else:
                status.update(label="Alineación lista", state="complete", expanded=False)
        if error is not None:
            st.error(svc.error_message(error))            # outside the box, always visible
            st.stop()
        st.session_state["align_key"] = key
        st.session_state["align_ws"] = ws_id
elif st.session_state.get("align_key"):
    st.session_state.pop("align_key")                     # a file was removed or the mode changed
    st.session_state.pop("align_ws", None)

ws_id = st.session_state.get("align_ws")
if not ws_id:
    st.info(svc.missing_text(mode, files))
    with st.expander("¿Qué hace el sistema con lo que subes?"):
        for code, label in svc.MODES.items():
            st.markdown(f"**{label}.** {svc.MODE_HELP[code]}")
        for state, text in svc.STATE_HELP.items():
            st.markdown(f"**{state}.** {text}")
    st.stop()

try:
    v = svc.view(ws_id)
except Exception as exc:                                  # noqa: BLE001 - shown to the person
    st.error(svc.error_message(exc))
    st.stop()

st.subheader(v["file"])
cols = st.columns(len(v["metrics"]))
for col, (label, value) in zip(cols, v["metrics"].items()):
    col.metric(label, value, help=svc.METRIC_HELP.get(label))
st.caption(f"{v['mode_label']} · {v['concepts']} conceptos · {v['mappings']} correspondencias "
           f"en la ontología ({v['validated_mappings']} de la base y {v['accepted_mappings']} "
           f"aceptadas por la regla) · {v['suggested']} sugeridas fuera de la ontología · "
           f"versión {v['version']}")
if v["profile"]:
    st.caption(v["profile"])
if v["rule"]:
    st.caption(f"Regla: {v['rule']}.")
if v["labels"]:
    st.caption(v["labels"])
if v["evidence"]:
    st.caption(v["evidence"])
for note in (v["extraction_note"], v["ai_note"], v["labels_note"]):
    if note:
        st.warning(note)
if v["iri_generated"]:
    st.caption("Tu ontología no declaraba un IRI de ontología: se le asignó uno en la copia "
               "del paquete (tu archivo original no cambia).")

labels = {label: key for key, label in svc.FORMATS}
default = [k for k, _ in svc.FORMATS].index(svc.DEFAULT_FORMAT)
choice = st.radio("Formato de descarga", list(labels), index=default, horizontal=True,
                  key="align_fmt")
package = _download(ws_id, labels[choice], v["created"])
st.download_button(f"Descargar ontología alineada ({package['label']}, .zip)",
                   data=package["data"], file_name=package["file_name"],
                   mime="application/zip", type="primary", key="align_dl")
if package["note"]:
    st.info(package["note"])
else:
    core = (" El paquete incluye OntoPriv-Core, que tu ontología importa." if v["core"] else "")
    apart = (" Las sugeridas no están en la ontología: las ves abajo y en alignment.csv."
             if v["suggested"] else "")
    st.caption(f"Descomprímelo y abre {package['main_file']} en Protégé: el catálogo incluido "
               f"resuelve las importaciones.{core}{apart}")

with st.expander("Ver el detalle por concepto"):
    present = list(dict.fromkeys(r["Estado"] for r in v["rows"]))
    state = st.selectbox("Mostrar", ["Todos"] + present, key="align_filter")
    rows = svc.filter_rows(v["rows"], state)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"{len(rows)} filas. Un concepto con varias correspondencias ocupa una fila "
               "por cada una.")

with st.expander("¿Qué significa cada estado?"):
    for state_label, text in svc.STATE_HELP.items():
        st.markdown(f"**{state_label}.** {text}")