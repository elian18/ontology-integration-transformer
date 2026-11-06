"""AI proposals view (Sprint 3, S3-T08): the concepts the AI proposed for the profile.

Presentation only; the data comes from services.core_module.proposed_concepts(), which reads
the file written by the extraction (S3-T05). The list is shown as the AI wrote it; the person's
decision on each concept (Sprint 5) is on the "Aprobar conceptos IA" page."""
import streamlit as st
from services import core_module

st.header("Conceptos propuestos por la IA")
st.write(
    "Conceptos que la IA extrajo del texto de la ley y propuso para el perfil, cada uno con el "
    "artículo del que proviene. Es la lista tal como la escribió la IA: la decisión sobre cada "
    "uno (aprobarlo, descartarlo o marcarlo como ya existente en OntoPriv) se tomó una sola vez "
    "al construir la base y se ve en **Aprobar conceptos IA**."
)

proposed = core_module.proposed_concepts()
if proposed is None:
    st.info("Aún no hay propuestas. Genera la extracción con:  `py -m src.core.extract`")
    st.stop()

counts = proposed["counts"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Propuestos", counts.get("concepts_proposed", len(proposed["concepts"])))
c2.metric("Clases", counts.get("classes", 0))
c3.metric("Propiedades", counts.get("properties", 0))
c4.metric("Artículos procesados", counts.get("articles_processed", len(proposed["processed_articles"])))

if proposed["quota_exhausted"]:
    st.warning(
        "La extracción quedó a medias por la cuota diaria del modelo. Vuelve a correr "
        "`py -m src.core.extract` más tarde para completar los artículos que faltan."
    )

st.divider()
query = st.text_input("Filtrar por concepto, etiqueta o número de artículo",
                      placeholder="p. ej. titular, consent, 17")

concepts = proposed["concepts"]
if query:
    q = query.strip().lower()
    concepts = [
        c for c in concepts
        if q in c["name"].lower()
        or q in c.get("label", "").lower()
        or q in ",".join(str(a) for a in c.get("articles", []))
    ]
    st.caption(f"{len(concepts)} coincidencia(s)")

rows = [
    {
        "Concepto": c["name"],
        "Etiqueta": c.get("label", ""),
        "Tipo": "clase" if c.get("type") == "class" else "propiedad",
        "Artículos": ", ".join(str(a) for a in c.get("articles", [])),
        "Definición": c.get("definition", ""),
    }
    for c in concepts
]
st.dataframe(rows, use_container_width=True, hide_index=True)
st.caption("Lista original de la IA · fuente: " + (proposed["source"] or "ley"))