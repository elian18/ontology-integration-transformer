"""Streamlit entrypoint. Run from the project root:
    py -m streamlit run app/app.py
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

pages = [
    st.Page("views/home.py", title="Inicio", icon=":material/home:", default=True),
    st.Page("views/inputs.py", title="Insumos", icon=":material/upload_file:"),
    st.Page("views/articles.py", title="Artículos", icon=":material/article:"),
    st.Page("views/core_module.py", title="Núcleo", icon=":material/account_tree:"),
    st.Page("views/ai_proposals.py", title="Propuestas IA", icon=":material/fact_check:"),
]
st.navigation(pages).run()