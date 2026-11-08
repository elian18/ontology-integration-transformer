"""Page "Preguntas de la base" (S9-T10): what the base answers, read only.

Presentation only; services.base_questions reads what the commands saved. It has no button that
writes: it looks the same with and without EDITABLE_BASE."""
import streamlit as st
from services import base_questions as svc

st.header("Preguntas de la base")
st.write(
    "Las preguntas de competencia con las que se valida la base del sistema (OntoPriv + LOPDP "
    "alineada con el DPV): 5 del artículo de OntoPriv y 10 de la LOPDP, con la respuesta que "
    "se espera según la ley. Las preguntas son fijas: aquí solo se ven sus respuestas.")

v = svc.view(svc.base_cfg())


def _missing(what: str, command: str) -> None:
    st.info(f"Todavía no hay {what}. Se genera con:")
    st.code(command, language="bash")


def _question(r: dict) -> None:
    with st.container(border=True):
        st.markdown(f"{r['badge']} **{r['id']}** · {r['question']}")
        if r["origin"]:
            st.caption(r["origin"])
        if r["expected"]:
            st.markdown(f"**Esperada:** {r['expected']}")
        st.markdown(f"**Respuesta:** {r['answer']}")
        if r["why"]:
            st.markdown(f"**{r['why_label']}:** {r['why']}")
        st.caption(r["criterion"])
        with st.expander("Ver la consulta SPARQL"):
            if r["query"]:
                st.code(r["query"], language="sparql")
            else:
                st.caption("Esta consulta ya no está en el catálogo de preguntas.")


tabs = st.tabs(["Preguntas", "OntoPriv original", "Formatos", "Puente con el GDPR"])
q = v["questions"]
with tabs[0]:
    if q is None:
        _missing("respuestas de las preguntas de la base", v["commands"]["questions"])
    else:
        for line in q["header"]:
            st.caption(line)
        if q["stale"]:
            st.warning("El catálogo de preguntas cambió después de esta corrida. Vuelve a "
                       "correrla:")
            st.code(v["commands"]["questions"], language="bash")
        base, *others = q["groups"]
        st.subheader(base["title"])
        st.caption(base["caption"])
        for r in base["rows"]:
            _question(r)
        for group in others:
            with st.expander(f"{group['title']}: las mismas que se responden en cada subida"):
                st.caption(group["caption"])
                for r in group["rows"]:
                    _question(r)
with tabs[1]:
    if v["original"] is None:
        _missing("la comparación con el OntoPriv original", v["commands"]["questions"])
    else:
        st.write(v["original"]["message"])
        st.table(v["original"]["rows"])
with tabs[2]:
    if v["formats"] is None:
        _missing("la comparación de los cuatro formatos", v["commands"]["questions"])
    else:
        st.write(v["formats"]["message"])
        st.table(v["formats"]["rows"])
with tabs[3]:
    b = v["bridge"]
    if b is None:
        _missing("puente con el GDPR", v["commands"]["bridge"])
    else:
        for line in b["lines"]:
            st.markdown(line)
        if b["flagged"]:
            st.subheader(f"Para revisar ({len(b['flagged'])})")
            for f in b["flagged"]:
                with st.container(border=True):
                    st.markdown(f["title"])
                    st.markdown(f["explanation"])
        st.subheader(f"Los {len(b['rows'])} pares")
        st.dataframe(b["rows"], hide_index=True, use_container_width=True)
        for d in b["downloads"]:
            st.download_button(d["label"], data=d["data"], file_name=d["file_name"],
                               mime=d["mime"], use_container_width=True)
