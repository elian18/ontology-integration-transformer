"""Home view (updated in S6-T08: points the end user to "Alinear mi ontología")."""
import streamlit as st

st.title("Ontology Integration Transformer")
st.caption("Integración y Transformación de Ontologías de Privacidad (LOPDP) · TIC-EPN")
st.markdown(
    "Alinea tu ontología de protección de datos con el **Data Privacy Vocabulary (DPV)** y "
    "descárgala en RDF/XML, OWL, Turtle o JSON-LD.\n\n"
    "- **Alinear mi ontología**: subes tu archivo y recibes la ontología alineada. Lo ya "
    "validado en la base del sistema se aplica solo; lo demás se propone de forma automática, "
    "en un archivo aparte y marcado como no validado.\n"
    "- **Construcción de la base**: herramientas con las que se construyó, una sola vez, la "
    "base del sistema (OntoPriv + LOPDP): insumos, artículos, núcleo modular, propuestas de la "
    "IA y la alineación revisada a mano."
)
st.page_link("views/align_ontology.py", label="Ir a Alinear mi ontología",
             icon=":material/auto_awesome:")