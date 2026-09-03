"""Inputs view (Sprint 1): project base inputs (auto-loaded) + optional upload."""
import streamlit as st
from services import ingest

st.header("Insumos")
st.write(
    "Los insumos base del proyecto (OntoPriv y la LOPDP) se cargan automáticamente y "
    "alimentan las fases siguientes. Abajo puedes, opcionalmente, procesar otra ontología o ley."
)

# ── Zona 1: insumos base del proyecto ─────────────────────────────────────────
st.subheader("Insumos base del proyecto")

col_onto, col_law = st.columns(2)

with col_onto:
    st.markdown("**Ontología base — OntoPriv**")
    r = ingest.base_ontology()
    if r is None:
        st.warning("No se encontró la ontología base (revisa config.yaml).")
    else:
        st.success(f"{r.n_classes} clases · {r.n_object_props} obj · {r.n_data_props} datos · "
                   f"{r.n_individuals} indiv · {r.n_triples} tripletas.")
        flavor = "OWL Full" if r.ontology_flavor == "owl-full" else r.ontology_flavor
        st.info(f"Tipo: **{flavor}** — {r.flavor_detail}")
        st.caption(f"formato={r.source_format} · sha256={r.sha256}")

with col_law:
    st.markdown("**Texto normativo — LOPDP**")
    t = ingest.base_legal_text()
    if t is None:
        st.warning("No se encontró el texto base (revisa config.yaml).")
    else:
        st.success(f"{t.source} · {t.n_chars} caracteres · {t.n_lines} líneas · "
                   f"{t.candidate_articles} artículos detectados.")
        st.caption(f"sha256={t.sha256}")

st.divider()

# ── Zona 2: procesar otra ontología / ley (opcional) ──────────────────────────
st.subheader("Procesar otra ontología o ley (opcional)")
st.caption("Para caracterizar una ontología o ley distinta a las del proyecto.")

col_left, col_right = st.columns(2)
with col_left:
    onto_file = st.file_uploader(
        "Otra ontología (RDF/XML, Turtle o JSON-LD)",
        type=["rdf", "owl", "xml", "ttl", "jsonld", "json"],
        help="OWL/XML (.owx) debe convertirse antes a RDF/XML con Protégé.",
    )
with col_right:
    law_file = st.file_uploader(
        "Otro texto normativo (.txt o PDF)",
        type=["txt", "pdf"],
        help="Del PDF se extrae el texto. No se segmenta por artículo (eso es el Sprint 2).",
    )

if st.button("Procesar", type="primary"):
    if onto_file is None and law_file is None:
        st.warning("Sube una ontología o un texto para procesar.")
    if onto_file is not None:
        try:
            with st.spinner("Cargando y caracterizando la ontología..."):
                r2 = ingest.ingest_ontology(onto_file.name, onto_file.getvalue())
            st.success(f"Ontología: {r2.n_classes} clases · formato {r2.source_format}.")
            flavor = "OWL Full" if r2.ontology_flavor == "owl-full" else r2.ontology_flavor
            st.info(f"Tipo: **{flavor}** — {r2.flavor_detail}")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"No se pudo cargar la ontología: {type(e).__name__}: {e}")
    if law_file is not None:
        try:
            with st.spinner("Extrayendo el texto..."):
                t2 = ingest.ingest_legal_text(law_file.name, law_file.getvalue())
            st.success(f"Texto ({t2.source}): {t2.n_chars} caracteres · "
                       f"{t2.candidate_articles} artículos detectados.")
            if t2.n_chars < 500:
                st.warning("Texto muy corto: si es PDF, podría estar escaneado (necesitaría OCR).")
        except Exception as e:
            st.error(f"No se pudo leer el texto: {type(e).__name__}: {e}")

st.divider()

# ── DPV (referencia en memoria) ───────────────────────────────────────────────
st.subheader("DPV (vocabulario de referencia en memoria)")
d = ingest.dpv_status()
if d:
    c1, c2, c3 = st.columns(3)
    c1.metric("Tripletas", f"{d.n_triples:,}")
    c2.metric("Conceptos", f"{d.n_concepts:,}")
    c3.metric("Propiedades", f"{d.n_properties:,}")
else:
    st.warning("DPV no encontrado en vocab/dpv.ttl (tarea S1-T05).")