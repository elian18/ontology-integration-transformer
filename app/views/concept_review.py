"""AI concepts review view (Sprint 5, S5-T03): a person decides on each AI-proposed concept.

One card at a time: the concept, its law article and, if it looks like an OntoPriv entity, which
one. The person approves it as a new concept (confirming its type), discards it, or says it
already exists in OntoPriv and picks which entity (also for concepts marked "new": the automatic
check misses some, e.g. PrincipleOfLawfulness vs Juridicity). Every click is written to the decisions log (S5-T01) through
services.concept_review; nothing enters the ontology here (the aligned graph is S5-T06)."""
import streamlit as st
from services import concept_review as cr
from src.alignment.ai_concepts import ACTION_APPROVE, ACTION_REJECT, ACTION_DUPLICATE, ACTION_UNDO

st.header("Aprobar conceptos propuestos por la IA")
st.write(
    "La IA extrajo estos conceptos del texto de la ley. Ninguno entra a la ontología sin tu "
    "decisión: **apruébalo** como concepto nuevo (confirmando su tipo), **descártalo**, o, si "
    "ya está en OntoPriv, márcalo como **ya existe en OntoPriv**. Cada decisión queda "
    "registrada con tu nombre y la fecha. Ojo: el chequeo automático de duplicados no detecta "
    "todo (p. ej. *PrincipleOfLawfulness* es *Juridicity* en OntoPriv); si un concepto marcado "
    "como nuevo ya existe, elige la entidad y márcalo como **ya existe en OntoPriv**."
)

state = cr.load()
if state is None:
    st.info("Aún no hay candidatos. Genéralos con:  `py -m src.alignment.export`  y luego "
            "`py -m src.alignment.justify`")
    st.stop()

flash = st.session_state.pop("cr_flash", None)
if flash:
    st.success(flash)

p = state["progress"]
decided = p["total"] - p["pending"]
st.progress(decided / p["total"] if p["total"] else 0.0,
            text=f"{decided} de {p['total']} conceptos revisados")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Por validar", p["pending"])
m2.metric("Aprobados como nuevos", p["approved"])
m3.metric("Descartados", p["rejected"])
m4.metric("Ya existen en OntoPriv", p["duplicate"])

if "reviewer" not in st.session_state:
    st.session_state["reviewer"] = state["settings"]["reviewer"]
st.text_input("Revisor (queda registrado en cada decisión)", key="reviewer")

st.divider()
f1, f2, f3 = st.columns([1, 1, 2])
status_label = f1.selectbox("Estado", list(cr.STATUS_FILTERS))
mark_label = f2.selectbox("Marca", list(cr.MARK_FILTERS))
query = f3.text_input("Buscar", placeholder="nombre, etiqueta o número de artículo")
rows = cr.filter_rows(state["rows"], status=cr.STATUS_FILTERS[status_label],
                      mark=cr.MARK_FILTERS[mark_label], query=query)

if not rows:
    if p["pending"] == 0:
        st.success("Revisaste todos los conceptos propuestos por la IA. El siguiente paso es "
                   "aprobar sus correspondencias con el DPV.")
    else:
        st.info("Ningún concepto coincide con el filtro.")
    st.stop()

keys = [r["key"] for r in rows]
by_key = {r["key"]: r for r in rows}
selected = st.session_state.get("cr_selected")
if selected not in keys:
    selected = keys[0]
col_sel, col_next = st.columns([4, 1])
selected = col_sel.selectbox(f"Concepto ({len(rows)} con este filtro)", keys,
                             index=keys.index(selected),
                             format_func=lambda k: cr.option_label(by_key[k]))
st.session_state["cr_selected"] = selected
col_next.write("")
if col_next.button("Siguiente pendiente", width="stretch"):
    nxt = cr.next_pending(state["rows"], after_key=selected)
    if nxt:
        st.session_state["cr_selected"] = nxt
    st.rerun()

row = by_key[selected]
with st.container(border=True):
    st.subheader(row["name"])
    if row["label"]:
        st.markdown(f"**{row['label']}**")
    st.write(row["definition"] or "_(sin definición)_")
    articles = ", ".join(str(a) for a in row["articles"]) or "-"
    st.caption(f"Artículo(s) de la ley: {articles} · Marca: {cr.MARK_LABELS.get(row['mark'], row['mark'])}"
               f" · IRI si se aprueba: {row['iri']}")
    if row["mark"] == "possible_duplicate":
        reason = "mismo nombre" if row.get("name_clash") else "parecido por puntaje"
        score = f"{row['duplicate_score']:.2f}" if row["duplicate_score"] is not None else "-"
        family = f", familia {row['duplicate_of_family']}" if row.get("duplicate_of_family") else ""
        st.info(f"Se parece a **{row['duplicate_of_name']}** de OntoPriv{family} "
                f"(similitud {score}, {reason}). Si es lo mismo, márcalo como "
                f"'ya existe en OntoPriv'.")
    if row["name_clash"]:
        st.warning("OntoPriv ya tiene una entidad con este mismo nombre: no se puede aprobar "
                   "como nuevo. Márcalo como 'ya existe en OntoPriv' o descártalo.")
    if row["status"] != "pending":
        kind = f" como {cr.ENTITY_KIND_LABELS.get(row['entity_kind'])}" if row["entity_kind"] else ""
        same = f" (= {row['same_as_name']})" if row.get("same_as_name") else ""
        st.success(f"Estado actual: **{row['status_label']}**{same}{kind} · por "
                   f"{row['reviewer']} el {row['decided_at']}")

    kinds = cr.kind_options()
    current_kind = row["entity_kind"] or row["suggested_kind"]
    kind_label = st.radio("Tipo del concepto (se usa solo si lo apruebas)", list(kinds),
                          index=list(kinds.values()).index(current_kind), horizontal=True,
                          key=f"kind_{selected}")
    entity_keys = [e["key"] for e in state["ontology"]]
    entity_by_key = {e["key"]: e for e in state["ontology"]}
    preset = row.get("same_as") or state["concepts"][selected].duplicate_of
    same_as = st.selectbox(
        "Entidad de OntoPriv equivalente (solo para 'Ya existe en OntoPriv'; escribe para buscar)",
        entity_keys, index=entity_keys.index(preset) if preset in entity_keys else None,
        format_func=lambda k: cr.entity_label(entity_by_key[k]),
        placeholder="elige la entidad de OntoPriv", key=f"same_{selected}")
    note = st.text_input("Nota (opcional)", key=f"note_{selected}")

    b1, b2, b3, b4 = st.columns(4)
    clicked = None
    if b1.button("Aprobar como nuevo", type="primary", width="stretch",
                 disabled=bool(row["name_clash"])):
        clicked = ACTION_APPROVE
    if b2.button("Descartar", width="stretch"):
        clicked = ACTION_REJECT
    if b3.button("Ya existe en OntoPriv", width="stretch", disabled=same_as is None):
        clicked = ACTION_DUPLICATE
    if b4.button("Deshacer", width="stretch", disabled=row["status"] == "pending"):
        clicked = ACTION_UNDO

    if clicked:
        ok, message = cr.apply(state, selected, clicked, st.session_state.get("reviewer", ""),
                               entity_kind=kinds[kind_label], note=note, same_as=same_as)
        if ok:
            st.session_state["cr_flash"] = message
            if clicked != ACTION_UNDO:
                nxt = cr.next_pending(state["rows"], after_key=selected)
                if nxt:
                    st.session_state["cr_selected"] = nxt
            st.rerun()
        else:
            st.error(message)

with st.expander("Todos los conceptos y su estado"):
    st.dataframe(
        [{"Concepto": r["name"], "Etiqueta": r["label"] or "",
          "Marca": cr.MARK_LABELS.get(r["mark"], r["mark"]), "Estado": r["status_label"],
          "Equivale a (OntoPriv)": r.get("same_as_name") or "",
          "Tipo": cr.ENTITY_KIND_LABELS.get(r["entity_kind"], "") if r["entity_kind"] else "",
          "Revisor": r["reviewer"] or "", "Artículos": ", ".join(str(a) for a in r["articles"])}
         for r in state["rows"]],
        width="stretch", hide_index=True,
    )
st.caption(f"Registro de decisiones: {state['log_path']}")