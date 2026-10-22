"""Home view (S6-T08; S7-T12: the three modes of "Alinear con el DPV")."""
import streamlit as st

st.title("Ontology Integration Transformer")
st.caption("Integración y Transformación de Ontologías de Privacidad (LOPDP) · TIC-EPN")
st.markdown(
    "Obtén una ontología de protección de datos alineada con el **Data Privacy Vocabulary "
    "(DPV)**, con etiquetas en español e inglés, y descárgala en RDF/XML, OWL, Turtle o "
    "JSON-LD.\n\n"
    "**Alinear con el DPV**: eliges qué subes y el resto es automático.\n"
    "- **Mi ontología**: se alinea con el DPV tal como está. Lo ya validado en la base del "
    "sistema se aplica solo; para lo demás, la IA elige entre los términos más parecidos y una "
    "regla calibrada decide qué entra.\n"
    "- **Una ley** (.txt o PDF): la IA saca sus conceptos, una regla los acepta y el sistema "
    "arma con ellos una ontología sobre OntoPriv-Core, ya alineada.\n"
    "- **Mi ontología y una ley**: tu ontología se alinea igual y cada concepto trae los "
    "artículos de la ley más cercanos, para que los revises (no entran a la ontología).\n\n"
    "**Construcción de la base**: herramientas con las que se construyó, una sola vez, la base "
    "del sistema (OntoPriv + LOPDP): insumos, artículos, núcleo modular, propuestas de la IA y "
    "la alineación revisada a mano."
)
st.page_link("views/align_ontology.py", label="Ir a Alinear con el DPV",
             icon=":material/auto_awesome:")