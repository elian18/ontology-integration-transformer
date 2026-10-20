"""Page "Alinear mi ontología" (S6-T08, S7-T06): the user's flow, as little to do as possible.

1. Upload the ontology: the alignment starts on its own (no extra button), with a progress
   box that names each step.
2. See the result: five numbers, the rule that decided, a short explanation and, if wanted, a
   read-only table (suggestions included).
3. Pick the format and download the zip.
Nothing to review: the ontology holds what a person decided while building the base and what
the calibrated rule accepted; the suggestions are only shown (comments come in Sprint 10).
Uploading the same file again reuses what was already computed (S6-T04), so changing the format
costs nothing.
"""
import hashlib

import pandas as pd
import streamlit as st
from services import align_ontology as svc

st.header("Alinear mi ontología con el DPV")
st.write(
    "Sube tu ontología de protección de datos y el sistema la alinea con el **Data Privacy "
    "Vocabulary (DPV)**. No tienes que revisar nada: lo que ya fue validado en la base del "
    "sistema (OntoPriv + LOPDP) se aplica solo; para lo demás, la IA elige entre los términos "
    "más parecidos del DPV y una regla calibrada decide qué entra. Lo que la IA sugiere y la "
    "regla no acepta se muestra aparte, sin entrar a la ontología."
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


uploaded = st.file_uploader("Tu ontología: RDF/XML (.rdf, .owl), Turtle (.ttl) o JSON-LD "
                            "(.jsonld)", type=svc.ACCEPTED_TYPES, key="align_file")

if uploaded is not None:
    data = uploaded.getvalue()
    key = hashlib.sha256(uploaded.name.encode("utf-8") + data).hexdigest()
    if st.session_state.get("align_key") != key:
        with st.status("Alineando tu ontología…", expanded=True) as status:
            try:
                ws_id = svc.align(uploaded.name, data,
                                  progress=lambda label: status.write(f"{label}…"),
                                  base=_base(), targets=_targets())
            except Exception as exc:                      # noqa: BLE001 - shown to the person
                status.update(label="No se pudo alinear la ontología", state="error")
                st.error(svc.error_message(exc))
                st.stop()
            status.update(label="Alineación lista", state="complete", expanded=False)
        st.session_state["align_key"] = key
        st.session_state["align_ws"] = ws_id
elif st.session_state.get("align_key"):
    st.session_state.pop("align_key")                     # the file was removed
    st.session_state.pop("align_ws", None)

ws_id = st.session_state.get("align_ws")
if not ws_id:
    st.info("Sube un archivo para empezar. La alineación arranca sola y tarda desde unos "
            "segundos hasta pocos minutos, según el tamaño de tu ontología.")
    with st.expander("¿Qué hace el sistema con tu ontología?"):
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
st.caption(f"{v['concepts']} conceptos · {v['mappings']} correspondencias en la ontología "
           f"({v['validated_mappings']} de la base y {v['accepted_mappings']} aceptadas por la "
           f"regla) · {v['suggested']} sugeridas fuera de la ontología · "
           f"versión {v['version']}")
if v["rule"]:
    st.caption(f"Regla: {v['rule']}.")
if v["ai_note"]:
    st.warning(v["ai_note"])
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
    apart = (" Las sugeridas no están en la ontología: las ves abajo y en alignment.csv."
             if v["suggested"] else "")
    st.caption(f"Descomprímelo y abre {package['main_file']} en Protégé: el catálogo incluido "
               f"resuelve las importaciones.{apart}")

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