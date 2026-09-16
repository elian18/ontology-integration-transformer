"""Articles view (Sprint 2): show a law split by article.

Uploads take priority; with nothing uploaded, the base LOPDP is shown. The
heavy lifting lives in services.segmentation, so this view is just presentation.
"""
import streamlit as st
from services import segmentation

st.header("Artículos")
st.write(
    "Sube un texto de ley para verlo partido por artículo. "
    "Si no subes nada, se muestra la LOPDP base del proyecto."
)

law_file = st.file_uploader(
    "Texto de ley", type=["txt", "pdf"], label_visibility="collapsed",
)
show = st.button("Ver artículos", type="primary", use_container_width=True)

if show:
    try:
        with st.spinner("Segmentando la ley por artículo..."):
            if law_file is not None:
                view = segmentation.segment_uploaded_law(law_file.name, law_file.getvalue())
            else:
                view = segmentation.segment_base_law()
        if view is None:
            st.warning("No subiste ningún archivo y no hay ley base disponible.")
            st.session_state.pop("articles_view", None)
        else:
            st.session_state["articles_view"] = view
    except Exception as e:
        st.error(f"No se pudo segmentar la ley: {type(e).__name__}: {e}")

view = st.session_state.get("articles_view")
if view:
    st.divider()
    st.success(f"{view['n_articles']} artículos · fuente: {view['source']}")

    query = st.text_input(
        "Filtrar por número o título", placeholder="p. ej. 8, consentimiento",
    )
    articles = view["articles"]
    if query:
        q = query.strip().lower()
        articles = [
            a for a in articles
            if q in str(a["number"]) or q in a["title"].lower()
        ]
        st.caption(f"{len(articles)} coincidencia(s)")

    for a in articles:
        with st.expander(f"Art. {a['number']} — {a['title']}"):
            st.write(a["text"])