"""Streamlit entrypoint. Run from the project root:
    py -m streamlit run app/app.py

The menu has what the end user does, the "Asistente de alineación" (S8-T01: the default page,
three steps for the three modes, ontology, law or both), and below it the tools used once to
build the system's base in Sprints 1-5 ("Construcción de la base").
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent          # .../app
ROOT = HERE.parent                                       # project root
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(page_title="Ontology Integration Transformer", layout="wide")

pages = {
    "": [
        st.Page("views/align_ontology.py", title="Asistente de alineación",
                icon=":material/auto_awesome:", default=True),
    ],
    "Construcción de la base": [
        st.Page("views/inputs.py", title="Insumos", icon=":material/upload_file:"),
        st.Page("views/articles.py", title="Artículos", icon=":material/article:"),
        st.Page("views/core_module.py", title="Núcleo", icon=":material/account_tree:"),
        st.Page("views/ai_proposals.py", title="Propuestas IA", icon=":material/fact_check:"),
        st.Page("views/concept_review.py", title="Aprobar conceptos IA", icon=":material/how_to_reg:"),
        st.Page("views/alignment.py", title="Alineación DPV", icon=":material/compare_arrows:"),
    ],
}
st.navigation(pages).run()