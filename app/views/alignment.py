"""Alignment view (Sprint 4, S4-T10): DPV candidates per concept, with the AI's type.

Presentation only; the data comes from services.alignment, which reads the files written by
the Sprint 4 pipeline. Nothing is approved here: every row is "por validar" (Sprint 5)."""
import streamlit as st
from services import alignment

st.header("Candidatos de alineación con el DPV")
st.write(
    "Para cada concepto de OntoPriv y cada concepto propuesto por la IA, los términos del DPV "
    "más parecidos (por escritura y por significado) y el tipo de relación SKOS que propone la "
    "IA con su justificación. Todo está **por validar**: aprobar o descartar es el Sprint 5."
)

data = alignment.candidates()
if data is None:
    st.info("Aún no hay candidatos. Genéralos con:  `py -m src.alignment.export`  y luego "
            "`py -m src.alignment.justify`")
    st.stop()

counts = data["counts"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Conceptos", counts.get("concepts", 0))
c2.metric("Filas (concepto · candidato)", counts.get("rows", len(data["rows"])))
c3.metric("IA: ya existen en OntoPriv", counts.get("ai_possible_duplicates", 0))
c4.metric("IA: nuevos", counts.get("ai_new", 0))

evaluation = alignment.evaluation()
if evaluation:
    m = evaluation.get("metrics", {})

    def pct(v):
        return "-" if v is None else f"{v * 100:.0f} %"

    with st.expander("Medición contra la muestra etiquetada a mano (S4-T09)"):
        e1, e2, e3 = st.columns(3)
        e1.metric("Hit@1", pct(m.get("hit_at_1_rate")),
                  help="El término correcto sale primero")
        e2.metric(f"Hit@{m.get('k', 3)}", pct(m.get("hit_at_k_rate")),
                  help="El término correcto está entre los candidatos")
        e3.metric("Acierto del tipo (IA)", pct(m.get("type_accuracy")))
        st.caption(f"{m.get('labelled', 0)} conceptos etiquetados · "
                   f"muestra: {evaluation.get('reference', '')}")

st.divider()
f1, f2, f3 = st.columns([1, 1, 2])
origin = f1.selectbox("Origen", ["Todos", "OntoPriv", "IA"])
duplicate = f2.selectbox("Marca (solo IA)", ["Todas", "nuevo", "ya existe en OntoPriv"])
relations = f3.multiselect("Tipo propuesto", data["relations"])
g1, g2 = st.columns([3, 1])
query = g1.text_input("Buscar concepto o término DPV", placeholder="p. ej. consent, titular")
best_only = g2.checkbox("Solo el mejor candidato", value=False)

rows = alignment.filter_rows(
    data["rows"],
    origin={"OntoPriv": "ontology", "IA": "ai"}.get(origin),
    duplicate={"nuevo": "new", "ya existe en OntoPriv": "possible_duplicate"}.get(duplicate),
    relations=relations,
    query=query,
    best_only=best_only,
)
st.caption(f"{len(rows)} fila(s)")
st.dataframe(
    alignment.table_rows(rows),
    width="stretch",
    hide_index=True,
    column_config={
        "Similitud": st.column_config.ProgressColumn("Similitud", min_value=0.0, max_value=1.0,
                                                     format="%.2f"),
        "Justificación": st.column_config.TextColumn("Justificación", width="large"),
    },
)
settings = data["metadata"].get("settings", {})
st.caption(
    f"Estado: por validar · similitud = {settings.get('weight_lexical', 0.4)} × léxico + "
    f"{settings.get('weight_semantic', 0.6)} × semántico · modelo de embeddings: "
    f"{data['metadata'].get('embedding_model') or '-'}"
)