"""Modular-core view (Sprint 3): structure of the core and the jurisdiction profile.

Presentation only; the counts and families come from services.nucleo. Downloading the files
is S3-T07 and the AI-proposed concepts are S3-T08 — this view just shows the structure."""
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
    st.warning(
        "No se encontró la ontología base. Revisa 'inputs.ontology' en config/config.yaml."
    )
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