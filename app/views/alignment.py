"""Alignment view: review the DPV correspondences (Sprint 5, S5-T05) + candidates table (S4-T10).

Tab "Revisar correspondencias": one card per concept with its DPV candidates. For each one the
person CHOOSES the SKOS type (empty by default: the AI type is shown only as a suggestion) and
approves or discards it; they can also search another DPV term or mark the concept as having no
DPV counterpart. Every click goes to the decisions log (S5-T01) through
services.mapping_review. Tab "Confirmar propuestas del asistente" (S5-T08): the assistant's
proposals for the concepts still open, as an editable table per family; nothing is recorded
until the person confirms the batch, and then in their name with the proposal as provenance.
Tab "Tabla de candidatos": the Sprint 4 table, unchanged."""
import pandas as pd
import streamlit as st
from services import alignment
from services import mapping_review as mr

st.header("Alineación con el DPV")
st.write(
    "Para cada concepto (de OntoPriv o nuevo de la IA) están los 3 términos del DPV más "
    "parecidos. **Tú eliges el tipo SKOS** y apruebas o descartas cada uno; el tipo que "
    "propone la IA es solo una sugerencia (en la medición del Sprint 4 acertó el 25 %). Si "
    "ninguno sirve, busca otro término del DPV o marca el concepto como **sin "
    "correspondencia**. Cada decisión queda registrada con tu nombre y la fecha."
)


@st.cache_resource(show_spinner="Cargando el DPV para la búsqueda…")
def _dpv_targets():
    return mr.load_dpv_targets()


state = mr.load()
if state is None:
    st.info("Aún no hay candidatos. Genéralos con:  `py -m src.alignment.export`  y luego "
            "`py -m src.alignment.justify`")
    st.stop()

flash = st.session_state.pop("mr_flash", None)
if flash:
    st.success(flash)
flash_warn = st.session_state.pop("mr_flash_warn", None)
if flash_warn:
    st.warning(flash_warn)

if "reviewer" not in st.session_state:
    st.session_state["reviewer"] = state["settings"]["reviewer"]
st.text_input("Revisor (queda registrado en cada decisión)", key="reviewer")
reviewer = st.session_state.get("reviewer", "")

tab_review, tab_assist, tab_table = st.tabs(["Revisar correspondencias",
                                             "Confirmar propuestas del asistente",
                                             "Tabla de candidatos"])


def _done(ok: bool, message: str, next_key: str | None = None):
    if ok:
        st.session_state["mr_flash"] = message
        if next_key:
            st.session_state["mr_selected"] = next_key
        st.rerun()
    else:
        st.error(message)


with tab_review:
    p = state["progress"]
    st.progress(p["reviewed"] / p["total"] if p["total"] else 0.0,
                text=f"{p['reviewed']} de {p['total']} conceptos revisados")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Por validar", p["pending"])
    m2.metric("Alineados", p["aligned"])
    m3.metric("Sin correspondencia", p["no_match"])
    m4.metric("Correspondencias aprobadas", p["mappings"])
    a = state["agreement"]
    if a["type_agreement"] is not None:
        st.caption(f"Tipo SKOS igual al que sugirió la IA: {a['same_type_as_ai']} de "
                   f"{a['approved_with_ai_type']} ({a['type_agreement'] * 100:.0f} %)")

    st.divider()
    f1, f2, f3, f4 = st.columns([1, 1, 1.4, 1.6])
    status_label = f1.selectbox("Estado", list(mr.STATUS_FILTERS))
    origin_label = f2.selectbox("Origen", list(mr.ORIGIN_FILTERS))
    family = f3.selectbox("Familia (OntoPriv)", ["Todas"] + mr.families(state))
    query = f4.text_input("Buscar", placeholder="concepto, etiqueta o término del DPV")
    concepts = mr.filter_concepts(state, status=mr.STATUS_FILTERS[status_label],
                                  origin=mr.ORIGIN_FILTERS[origin_label],
                                  family=None if family == "Todas" else family, query=query)

    selected = st.session_state.get("mr_selected")
    keys = [c.key for c in concepts]
    filters = (status_label, origin_label, family, query)
    same_filters = st.session_state.get("mr_filters") == filters
    st.session_state["mr_filters"] = filters
    if same_filters and selected in state["by_key"] and selected not in keys:
        keys = [selected] + keys            # after approving, keep the card you are working on
    if not keys:
        if p["pending"] == 0:
            st.success("Revisaste todos los conceptos. El siguiente paso es escribir el "
                       "grafo alineado.")
        else:
            st.info("Ningún concepto coincide con el filtro.")
    else:
        if selected not in keys:
            selected = keys[0]
        col_sel, col_next = st.columns([4, 1])
        selected = col_sel.selectbox(
            f"Concepto ({len(concepts)} con este filtro)", keys, index=keys.index(selected),
            format_func=lambda k: mr.option_label(state, state["by_key"][k]))
        st.session_state["mr_selected"] = selected
        col_next.write("")
        if col_next.button("Siguiente pendiente", width="stretch"):
            nxt = mr.next_pending(state, after_key=selected)
            if nxt:
                st.session_state["mr_selected"] = nxt
            st.rerun()

        concept = state["by_key"][selected]
        status = mr.status_of(state, selected)
        with st.container(border=True):
            st.subheader(concept.name)
            if concept.label:
                st.markdown(f"**{concept.label}**")
            if concept.definition:
                st.write(concept.definition)
            kinds = "/".join({"class": "clase", "property": "propiedad",
                              "individual": "individuo"}.get(k, k) for k in concept.kinds)
            articles = ", ".join(str(x) for x in concept.articles) or "-"
            where = concept.family or "perfil (concepto nuevo de la IA)"
            st.caption(f"{mr.ORIGIN_LABELS[concept.origin]} · {where} · tipo: {kinds} · "
                       f"artículo(s): {articles} · {concept.iri}")
            st.markdown(f"Estado del concepto: **{mr.STATUS_LABELS[status]}**")

            relations = mr.relation_options()
            for i, card in enumerate(mr.candidate_cards(state, selected)):
                with st.container(border=True):
                    head = f"**dpv:{card['dpv_name']}**"
                    if card["dpv_label"]:
                        head += f" · {card['dpv_label']}"
                    if card["found_by"] == "search":
                        head += " · _(encontrado con la búsqueda)_"
                    elif card["rank"]:
                        head += f" · candidato {card['rank']} · similitud {card['score']:.2f}"
                    st.markdown(head)
                    if card["dpv_definition"]:
                        st.caption(card["dpv_definition"])
                    if card["dpv_parents"]:
                        st.caption("Padres en el DPV: " + ", ".join(card["dpv_parents"]))
                    if card["ai_text"]:
                        art = (f" (art. {card['evidence_article']})"
                               if card["evidence_article"] else "")
                        st.markdown(f"La IA sugiere: _{card['ai_text']}_{art}")
                        if card["justification"]:
                            st.caption(card["justification"])
                    if card["state"] != "pending":
                        rel = f" como {mr.relation_text(card['relation'])}" if card["relation"] else ""
                        st.success(f"{card['state_label'].capitalize()}{rel} · por {card['reviewer']}")

                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    choice = c1.selectbox("Tipo SKOS", list(relations), index=None,
                                          placeholder="elige el tipo SKOS",
                                          key=f"rel_{selected}_{i}",
                                          label_visibility="collapsed")
                    if c2.button("Aprobar", key=f"ok_{selected}_{i}", type="primary",
                                 width="stretch", disabled=choice is None):
                        ok, msg = mr.apply_approve(state, selected, card["dpv_iri"],
                                                   relations[choice], reviewer)
                        _done(ok, msg)
                    if c3.button("Descartar", key=f"no_{selected}_{i}", width="stretch",
                                 disabled=card["state"] == "rejected"):
                        ok, msg = mr.apply_reject(state, selected, card["dpv_iri"], reviewer)
                        _done(ok, msg)
                    if c4.button("Deshacer", key=f"undo_{selected}_{i}", width="stretch",
                                 disabled=card["state"] == "pending"):
                        ok, msg = mr.apply_undo(state, selected, reviewer, card["dpv_iri"])
                        _done(ok, msg)

            with st.expander("Buscar otro término del DPV (fuera de los 3 candidatos)"):
                text = st.text_input("Texto a buscar (en inglés)", key=f"q_{selected}",
                                     placeholder="p. ej. impact assessment, turnover")
                if text.strip():
                    targets = _dpv_targets()
                    hits = mr.search(state, targets, selected, text) if targets else []
                    if not hits:
                        st.caption("Sin resultados compatibles con el tipo del concepto.")
                    else:
                        by_iri = {t.iri: t for t in hits}
                        iri = st.selectbox("Término del DPV", list(by_iri),
                                           format_func=lambda x: f"{by_iri[x].name} · "
                                                                 f"{by_iri[x].label}",
                                           key=f"hit_{selected}")
                        if by_iri[iri].definition:
                            st.caption(by_iri[iri].definition)
                        s1, s2 = st.columns([2, 1])
                        s_choice = s1.selectbox("Tipo SKOS", list(relations), index=None,
                                                placeholder="elige el tipo SKOS",
                                                key=f"srel_{selected}")
                        if s2.button("Aprobar este término", key=f"sok_{selected}",
                                     type="primary", width="stretch", disabled=s_choice is None):
                            ok, msg = mr.apply_approve(state, selected, iri,
                                                       relations[s_choice], reviewer,
                                                       target=by_iri[iri])
                            _done(ok, msg)

            note = st.text_input("Nota (opcional, se guarda con 'Sin correspondencia')",
                                 key=f"note_{selected}")
            n1, n2, n3 = st.columns(3)
            if status == mr.STATUS_NO_MATCH:
                if n1.button("Deshacer 'sin correspondencia'", width="stretch"):
                    ok, msg = mr.apply_undo(state, selected, reviewer)
                    _done(ok, msg)
            elif n1.button("Sin correspondencia en el DPV", width="stretch",
                           disabled=status == mr.STATUS_ALIGNED):
                ok, msg = mr.apply_no_match(state, selected, reviewer, note=note)
                _done(ok, msg, mr.next_pending(state, after_key=selected))
            if n2.button("Listo · siguiente pendiente", type="primary", width="stretch",
                         disabled=status == mr.STATUS_PENDING):
                nxt = mr.next_pending(state, after_key=selected)
                if nxt:
                    st.session_state["mr_selected"] = nxt
                st.rerun()
        st.caption(f"Registro de decisiones: {state['log_path']}")

with tab_assist:
    assist = mr.load_assistant(state)
    if assist is None:
        st.info("No hay propuestas del asistente (data/review/assistant-proposals.json).")
    else:
        st.warning(
            "Estas son **propuestas del asistente, no decisiones**. Nada entra al registro "
            "hasta que confirmas el lote; entonces queda **a tu nombre**, con la nota "
            "«propuesta del asistente confirmada» y la propuesta original como procedencia. "
            "Lee cada fila (término del DPV, tipo SKOS y razón): si no te convence, cambia el "
            "tipo, márcala «Sin correspondencia» o déjala en «Omitir» y revísala luego en la "
            "pestaña «Revisar correspondencias»."
        )
        meta, prov = assist["metadata"], assist["provenance"]
        st.caption(f"Propuestas de: {meta.get('assistant') or '-'} · "
                   f"{meta.get('created_at') or '-'} · {assist['path']}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Propuestas abiertas", len(assist["open"]))
        k2.metric("Revisados por ti solo", prov["reviewed_by_person"])
        k3.metric("Confirmados desde propuestas", prov["confirmed_from_assistant"])
        k4.metric("Cambiados por ti", prov["changed_by_person"])
        if not assist["open"]:
            st.success("No quedan propuestas abiertas: todos sus conceptos ya tienen decisión.")
        else:
            fams = mr.proposal_families(assist)
            fam_labels = {f"{f} ({n})": f for f, n in fams}
            family = fam_labels[st.selectbox("Familia", list(fam_labels), key="ap_family")]
            prefill = st.checkbox("Marcar todas las filas de esta familia como «Aceptar» "
                                  "(después cambia las que no aceptes)",
                                  key=f"ap_prefill_{family}")
            table = pd.DataFrame(mr.proposal_table(state, assist, family, prefill))
            table["Tipo SKOS"] = table["Tipo SKOS"].astype(
                pd.CategoricalDtype(list(mr.relation_options())))   # empty cell, not "None"
            edited = st.data_editor(
                table,
                key=f"ap_editor_{family}_{prefill}_{len(assist['open'])}",
                hide_index=True,
                width="stretch",
                column_order=["Acción", "Concepto", "Propuesta del asistente", "Tipo SKOS",
                              "Razón", "Etiqueta", "Definición en el DPV",
                              "Definición del concepto"],
                disabled=["Concepto", "Etiqueta", "Propuesta del asistente", "Razón",
                          "Definición en el DPV", "Definición del concepto"],
                column_config={
                    "Acción": st.column_config.SelectboxColumn(
                        "Acción", options=mr.action_options(), required=True),
                    "Tipo SKOS": st.column_config.SelectboxColumn(
                        "Tipo SKOS", options=list(mr.relation_options()),
                        help="Del concepto hacia el DPV; en las filas sin correspondencia "
                             "se ignora"),
                    "Razón": st.column_config.TextColumn("Razón", width="large"),
                    "Etiqueta": st.column_config.TextColumn(width="medium"),
                    "Definición en el DPV": st.column_config.TextColumn(width="medium"),
                    "Definición del concepto": st.column_config.TextColumn(width="medium"),
                },
            )
            rows = edited.to_dict("records")
            counts = mr.count_actions(rows)
            n = len(rows) - counts["Omitir"]
            st.caption(f"{len(rows)} fila(s) · Aceptar: {counts['Aceptar']} · Sin "
                       f"correspondencia: {counts['Sin correspondencia']} · Omitir: "
                       f"{counts['Omitir']}")
            if st.button(f"Confirmar lote ({n} fila(s))", type="primary",
                         disabled=n == 0 or not reviewer.strip(), key="ap_confirm"):
                ok, msg, summary = mr.confirm_table(state, assist, rows, reviewer)
                if ok:
                    st.session_state["mr_flash_warn" if summary["errors"] else "mr_flash"] = msg
                    st.rerun()
                else:
                    st.error(msg)

with tab_table:
    data = alignment.candidates()
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

    t1, t2, t3 = st.columns([1, 1, 2])
    origin = t1.selectbox("Origen ", ["Todos", "OntoPriv", "IA"])
    duplicate = t2.selectbox("Marca (solo IA)", ["Todas", "nuevo", "ya existe en OntoPriv"])
    rel_filter = t3.multiselect("Tipo propuesto", data["relations"])
    g1, g2 = st.columns([3, 1])
    t_query = g1.text_input("Buscar concepto o término DPV", placeholder="p. ej. consent, titular")
    best_only = g2.checkbox("Solo el mejor candidato", value=False)

    rows = alignment.filter_rows(
        data["rows"],
        origin={"OntoPriv": "ontology", "IA": "ai"}.get(origin),
        duplicate={"nuevo": "new", "ya existe en OntoPriv": "possible_duplicate"}.get(duplicate),
        relations=rel_filter,
        query=t_query,
        best_only=best_only,
    )
    st.caption(f"{len(rows)} fila(s) · tipo propuesto por la IA, sin validar")
    st.dataframe(
        alignment.table_rows(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Similitud": st.column_config.ProgressColumn("Similitud", min_value=0.0,
                                                         max_value=1.0, format="%.2f"),
            "Justificación": st.column_config.TextColumn("Justificación", width="large"),
        },
    )
    settings = data["metadata"].get("settings", {})
    st.caption(
        f"Similitud = {settings.get('weight_lexical', 0.4)} × léxico + "
        f"{settings.get('weight_semantic', 0.6)} × semántico · modelo de embeddings: "
        f"{data['metadata'].get('embedding_model') or '-'}"
    )