"""Inputs view (Sprint 1): user upload is the focus; base inputs shown as a status line."""
import streamlit as st
from services import ingest

st.header("Insumos")
st.write("Sube tu ontología y/o el texto de ley para procesarlos. Puedes cargar uno de los dos o ambos.")

# ── Protagonista: carga del usuario ───────────────────────────────────────────
col_onto, col_law = st.columns(2)
with col_onto:
    st.markdown("### Tu ontología")
    st.caption("RDF/XML, Turtle o JSON-LD")
    onto_file = st.file_uploader(
        "Tu ontología", type=["rdf", "owl", "xml", "ttl", "jsonld", "json"],
        label_visibility="collapsed",
    )
with col_law:
    st.markdown("### Tu texto de ley")
    st.caption(".txt o PDF")
    law_file = st.file_uploader(
        "Tu texto de ley", type=["txt", "pdf"],
        label_visibility="collapsed",
    )

process = st.button("Procesar", type="primary", use_container_width=True)

# ── Resultados ────────────────────────────────────────────────────────────────
if process:
    if onto_file is None and law_file is None:
        st.warning("Sube una ontología o un texto para procesar.")

    if onto_file is not None:
        try:
            with st.spinner("Cargando y caracterizando la ontología..."):
                r = ingest.ingest_ontology(onto_file.name, onto_file.getvalue())
            flavor = "OWL Full" if r.ontology_flavor == "owl-full" else r.ontology_flavor
            with st.container(border=True):
                st.markdown(f"**Ontología procesada** · {flavor}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Clases", r.n_classes)
                c2.metric("Prop. objeto", r.n_object_props)
                c3.metric("Prop. datos", r.n_data_props)
                c4.metric("Individuos", r.n_individuals)
                st.caption(f"{r.flavor_detail} · formato {r.source_format} · {r.n_triples} tripletas")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"No se pudo cargar la ontología: {type(e).__name__}: {e}")

    if law_file is not None:
        try:
            with st.spinner("Extrayendo el texto..."):
                t = ingest.ingest_legal_text(law_file.name, law_file.getvalue())
            with st.container(border=True):
                st.markdown(f"**Texto procesado** · {t.source.upper()}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Artículos", t.candidate_articles)
                c2.metric("Caracteres", f"{t.n_chars:,}")
                c3.metric("Líneas", f"{t.n_lines:,}")
                if t.n_chars < 500:
                    st.warning("Texto muy corto: si es un PDF, podría estar escaneado.")
        except Exception as e:
            st.error(f"No se pudo leer el texto: {type(e).__name__}: {e}")

# ── Estado del sistema (base) al pie, tenue ───────────────────────────────────
st.divider()
r_base = ingest.base_ontology()
t_base = ingest.base_legal_text()
d_base = ingest.dpv_status()

parts = ["**Sistema listo**"]
if r_base:
    flavor = "OWL Full" if r_base.ontology_flavor == "owl-full" else r_base.ontology_flavor
    parts.append(f"OntoPriv ({r_base.n_classes} clases, {flavor})")
if t_base:
    parts.append(f"LOPDP ({t_base.candidate_articles} artículos)")
if d_base:
    parts.append(f"DPV ({d_base.n_concepts:,} conceptos)")

st.caption(" · ".join(parts))