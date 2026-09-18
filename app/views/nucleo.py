"""Modular-core view (Sprint 3): structure of the core/profile and downloadable modules.

Presentation only; the counts, families and file bytes come from services.nucleo. The
AI-proposed concepts are S3-T08 — this view shows the structure and the downloads."""
import streamlit as st
from services import nucleo

st.header("Núcleo modular")
st.write(
    "Estructura del núcleo reutilizable (OntoPriv-Core) y del perfil de Ecuador (LOPDP): "
    "cuántos conceptos hay en cada uno y a qué familias pertenecen. El núcleo es la base "
    "general que se reutiliza para cualquier ley; el perfil es lo propio de la jurisdicción."
)

structure = nucleo.core_structure()
if structure is None:
    st.warning("No se encontró la ontología base. Revisa 'inputs.ontology' en config/config.yaml.")
    st.stop()

counts = structure["counts"]
col_core, col_profile = st.columns(2)
with col_core:
    st.subheader("Núcleo · OntoPriv-Core")
    a, b = st.columns(2)
    a.metric("Clases", counts["core"]["classes"])
    b.metric("Propiedades", counts["core"]["properties"])
with col_profile:
    st.subheader("Perfil · Ecuador (LOPDP)")
    a, b = st.columns(2)
    a.metric("Clases", counts["profile"]["classes"])
    b.metric("Propiedades", counts["profile"]["properties"])

st.divider()
col_cf, col_pf = st.columns(2)
with col_cf:
    st.caption("Familias en el núcleo (clases por familia)")
    st.table([{"Familia": f, "Clases": n} for f, n in structure["core_families"].items()])
with col_pf:
    st.caption("Familias en el perfil (clases por familia)")
    st.table([{"Familia": f, "Clases": n} for f, n in structure["profile_families"].items()])

st.divider()
st.subheader("Descargar")
st.write("Genera el núcleo modular y descarga los dos archivos en RDF/XML. El perfil importa "
         "el núcleo (owl:imports).")

if st.button("Generar núcleo modular", type="primary", use_container_width=True):
    with st.spinner("Generando núcleo y perfil..."):
        st.session_state["nucleo_downloads"] = nucleo.build_downloads()

downloads = st.session_state.get("nucleo_downloads")
if downloads:
    core, profile = downloads["core"], downloads["profile"]
    st.caption(
        f"Archivo real — Núcleo: {core['entities']} entidades, {core['triples']} tripletas · "
        f"Perfil: {profile['entities']} entidades, {profile['triples']} tripletas · "
        f"{len(downloads['moved_to_profile'])} propiedades movidas al perfil para que el "
        f"núcleo sea autónomo (por eso difiere un poco de los conteos de arriba)."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Descargar núcleo (RDF/XML)", data=core["bytes"],
                           file_name=core["filename"], mime="application/rdf+xml",
                           use_container_width=True)
    with c2:
        st.download_button("Descargar perfil (RDF/XML)", data=profile["bytes"],
                           file_name=profile["filename"], mime="application/rdf+xml",
                           use_container_width=True)