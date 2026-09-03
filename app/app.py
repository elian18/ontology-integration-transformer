"""Streamlit entrypoint. Run from the project root:
    py -m streamlit run app/app.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Ontology Integration Transformer", layout="wide")

pages = [
    st.Page("views/home.py", title="Inicio", icon=":material/home:", default=True)
]
st.navigation(pages).run()