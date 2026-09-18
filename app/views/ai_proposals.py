"""AI proposals view (Sprint 3, S3-T08): concepts the AI proposes for the profile.

Presentation only; the data comes from services.core_module.proposed_concepts(), which reads
the file written by the extraction (S3-T05). These are proposals in a "por validar" state:
nothing here has entered any ontology. Approving and aligning them is the alignment sprint."""
import streamlit as st
from services import core_module

st.header("Conceptos propuestos por la IA")
st.write(
    "Conceptos que la IA extrajo del texto de la ley y propone para el perfil, cada uno con el "
    "artículo del que proviene. Están **por validar**: nada de esto ha entrado a la ontología. "
    "Revisarlos y alinearlos es el siguiente sprint."
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
st.caption("Estado: por validar · fuente: " + (proposed["source"] or "ley"))