# Asistente de alineación: three steps kept in st.session_state, one screen each.
import hashlib
from functools import partial

import pandas as pd
import streamlit as st
from services import align_ontology as svc

ss = st.session_state


@st.cache_resource(show_spinner=False)
def _base():
    from src.alignment.inherit import load_base
    return load_base()


@st.cache_resource(show_spinner=False)
def _targets():
    from src.alignment.propose import load_dpv_targets
    return load_dpv_targets()


def _restart(mode: str | None = None) -> None:
    """Go to the three modes (``mode`` None) or to the upload of ``mode``: new uploaders,
    nothing computed."""
    ss["align_round"] = ss.get("align_round", 0) + 1
    for key in ("align_key", "align_ws", "align_failed", "align_review", "align_step",
                "align_applied", "align_answers", "align_extra", "align_q"):
        ss.pop(key, None)
    if mode is None:
        ss.pop("align_mode", None)
    else:
        ss["align_mode"] = mode


def _home(path: str | None = None) -> None:
    """Go to the start (``path`` None) or into one of its two paths: the files go too."""
    _restart()
    for key in ("align_path", "align_files", "align_done"):
        ss.pop(key, None)
    if path is not None:
        ss["align_path"] = path


def _to_upload() -> None:
    """From the result back to the files of its mode: they stay, nothing is computed again."""
    ss.pop("align_ws", None)


def _show(key: str, ws_id: str) -> None:
    ss["align_key"], ss["align_ws"] = key, ws_id


def _change(kind: str) -> None:
    """Put the uploader of ``kind`` back, with a new key (S7-T12)."""
    ss.get("align_files", {}).pop(kind, None)
    ss.pop("align_failed", None)
    ss["align_round"] = ss.get("align_round", 0) + 1


def _retry() -> None:
    ss.pop("align_failed", None)


def _back(on_click, key: str) -> None:
    st.button("Atrás", key=key, type="tertiary", icon=":material/arrow_back:", on_click=on_click)


def _cards(items: dict, helps: dict, icons: dict, prefix: str, on_click) -> None:
    """One card per option, its text on top and its button at the bottom."""
    for col, (code, label) in zip(st.columns(len(items)), items.items()):
        with col.container(border=True, height="stretch", vertical_alignment="distribute"):
            with st.container():
                st.markdown(f"### {icons[code]} {label}")
                st.write(helps[code])
            st.button("Empezar", key=f"{prefix}{code}", type="primary", width="stretch",
                      icon=":material/arrow_forward:", on_click=on_click, args=(code,))


# ------------------------------------------------------------------ the start (S10-T09)

def step_start() -> None:
    st.title(svc.START_TITLE)
    st.caption(svc.START_CAPTION)
    _cards(svc.PATHS, svc.PATH_HELP, svc.PATH_ICONS, "align_path_", _home)


def step_transform() -> None:
    _back(_home, "align_home")
    st.title(svc.PATHS[svc.PATH_TRANSFORM])
    st.caption(svc.PATH_HELP[svc.PATH_TRANSFORM])
    up = st.file_uploader(svc.UPLOAD_LABELS[svc.KIND_ONTOLOGY], type=svc.ACCEPTED_TYPES,
                          key=f"transform_{ss.get('align_round', 0)}", help=svc.TRANSFORM_NOTE)
    if up is None:
        return
    try:
        downloads = svc.transform_downloads(up.name, up.getvalue())
    except Exception as exc:                              # noqa: BLE001 - shown to the person
        st.error(svc.error_message(exc))
        return
    st.success(up.name, title=svc.TRANSFORM_READY, icon=":material/check_circle:")
    st.markdown("**Descárgala en el formato que necesites**")
    for col, d in zip(st.columns(len(downloads)), downloads):
        col.download_button(d["label"], data=d["data"], file_name=d["file_name"], mime=d["mime"],
                            key=f"transform_dl_{d['key']}", help=d["help"], on_click="ignore",
                            icon=":material/download:", width="stretch")
    st.caption(svc.TRANSFORM_CAPTION)


# ------------------------------------------------------------------ step 1

def step_choose() -> None:
    _back(_home, "align_home")
    st.title(svc.PATHS[svc.PATH_ALIGN])
    st.caption(svc.MODES_CAPTION)
    _cards(svc.MODES, svc.MODE_HELP, svc.MODE_ICONS, "align_pick_", _restart)


# ------------------------------------------------------------------ step 2

def _uploads(mode: str) -> dict:
    """The files of ``mode`` side by side: the ones already uploaded are kept (and shown), the
    others get an uploader whose key changes with every new round."""
    round_ = ss.get("align_round", 0)
    kept = ss.setdefault("align_files", {})
    kinds = svc.MODE_FILES[mode]
    for col, kind in zip(st.columns(len(kinds)), kinds):
        types = svc.ACCEPTED_TYPES if kind == svc.KIND_ONTOLOGY else svc.LAW_TYPES
        with col:
            if kind in kept:
                _kept(kind, *kept[kind])
                continue
            up = st.file_uploader(svc.UPLOAD_LABELS[kind], type=types,
                                  key=f"align_{kind}_{round_}", help=svc.DATA_NOTE)
            if up is not None:
                kept[kind] = (up.name, up.getvalue())
    return {kind: kept.get(kind) for kind in kinds}


def _kept(kind: str, name: str, data: bytes) -> None:
    st.markdown(svc.UPLOAD_LABELS[kind])
    with st.container(border=True, horizontal=True, vertical_alignment="center",
                      key=f"align_kept_{kind}"):
        st.markdown(f":material/description: **{svc.plain(name)}** · {svc.file_size(len(data))}")
        st.button("Cambiar", key=f"align_change_{kind}", type="tertiary",
                  icon=":material/swap_horiz:", on_click=_change, args=(kind,))


def _files_key(mode: str, files: dict) -> str:
    digest = hashlib.sha256(mode.encode("utf-8"))
    for kind in svc.MODE_FILES[mode]:
        name, data = files[kind]
        digest.update(kind.encode("utf-8") + name.encode("utf-8") + data)
    return digest.hexdigest()


def _failed(message: str) -> None:
    st.error(message)                                     # outside the progress box
    st.button("Intentar de nuevo", key="align_retry", icon=":material/refresh:",
              on_click=_retry)


def step_upload(mode: str) -> None:
    _back(_restart, "align_back")
    st.title(svc.MODES[mode])
    files = _uploads(mode)
    if not svc.ready(mode, files):
        st.caption(svc.missing_text(mode, files))
        return
    key = _files_key(mode, files)
    done = ss.setdefault("align_done", {})
    if key in done:                                       # back from the result: no new run
        st.button(svc.NEXT_LABEL, key="align_next", type="primary",
                  icon=":material/arrow_forward:", on_click=_show, args=(key, done[key]))
        return
    failed = ss.get("align_failed")
    if failed and failed[0] == key:                       # the same files failed before
        _failed(failed[1])
        return
    error = None
    with st.status(svc.MODE_RUNNING[mode], expanded=True) as status:
        bar = st.progress(0.0, text=svc.PROGRESS_START)

        def say(label: str) -> None:
            text, fraction = svc.progress_of(mode, label)
            bar.progress(fraction, text=text)
            status.write(text)
        try:
            ws_id = svc.run(mode, files, progress=say, base=_base(), targets=_targets())
        except Exception as exc:                          # noqa: BLE001 - shown to the person
            error = svc.error_message(exc)
            status.update(label="No se pudo terminar la alineación", state="error",
                          expanded=False)
        else:
            status.update(label="Alineación lista", state="complete", expanded=False)
    if error is not None:
        ss["align_failed"] = (key, error)
        _failed(error)
        return
    done[key] = ws_id
    _show(key, ws_id)
    st.rerun()


# ------------------------------------------------------------------ step 3

def _summary_tab(v: dict) -> None:
    for col, (label, value) in zip(st.columns(len(v["metrics"])), v["metrics"].items()):
        col.metric(label, value, help=svc.METRIC_HELP.get(label))
    for line in svc.summary_captions(v):
        st.caption(line)
    with st.expander("¿Qué significa cada estado?"):
        for state, text in svc.STATE_HELP.items():
            st.markdown(f"**{state}.** {text}")


def _concepts_tab(v: dict) -> None:
    present = list(dict.fromkeys(r["Estado"] for r in v["rows"]))
    state = st.selectbox("Mostrar", ["Todos"] + present, key="align_filter")
    rows = svc.filter_rows(v["rows"], state)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"{svc.counted(len(rows), 'fila', 'filas')}. Un concepto con varias "
               "correspondencias ocupa una fila por cada una.")
    _provenance(v)


def _provenance(v: dict) -> None:
    """The card of the chosen concept, read from its provenance module (S10-T08)."""
    st.markdown(f"**{svc.PROVENANCE_TITLE}**")
    options = v["concept_options"]
    iri = st.selectbox("Concepto", list(options), index=None, format_func=options.get,
                       placeholder=svc.PROVENANCE_PLACEHOLDER, key="align_concept",
                       help=svc.PROVENANCE_HELP)
    if iri is None:
        return
    c = svc.provenance_card(v["id"], iri)
    with st.container(border=True, key="align_card"):
        if c["missing"]:
            st.caption(c["missing"])
            return
        st.markdown(f"#### {c['name']}" + (f" :gray-badge[{c['state']}]" if c["state"] else ""))
        st.caption(c["iri"])
        st.markdown(c["headline"])
        st.markdown(c["origin"])
        st.markdown("**Versiones**")
        if c["versions"]:
            st.dataframe(pd.DataFrame(c["versions"]), hide_index=True, width="stretch")
        else:
            st.caption("Ninguna versión le dio una correspondencia.")
        st.markdown("**Cómo se produjo**")
        for s in c["steps"]:
            head = (f"{s['badge']} " if s["badge"] else "") + f"**{s['title']}**"
            st.markdown("  \n".join([head, *([s["text"]] if s["text"] else []), *s["extra"]]))
            if s["meta"]:
                st.caption(s["meta"])
        if not c["steps"]:
            st.caption("La procedencia no dice nada más de este concepto.")


def _review_tab(v: dict) -> None:
    flagged = (v["validation"] or {}).get("flagged") or []
    if flagged:
        st.markdown(f"**Marcadas por la IA revisora ({len(flagged)})**")
        st.caption("Están en tu ontología (la regla las aceptó); la IA explica por qué el "
                   "término del DPV podría no tratar de lo mismo.")
        st.dataframe(pd.DataFrame(flagged), hide_index=True, width="stretch")
    suggested = svc.suggested_rows(v["rows"])
    if suggested:
        st.markdown(f"**Sugeridas por la IA ({len(suggested)})**")
        st.caption("La IA las propuso con confianza alta, pero la regla no las aceptó: no están "
                   "en tu ontología.")
        st.dataframe(pd.DataFrame(suggested), hide_index=True, width="stretch")
    if not flagged and not suggested:
        st.info("No hay nada pendiente de revisar.")
    st.caption("Todo esto también está en alignment.csv, dentro del .zip.")


def _law_tab(v: dict) -> None:
    st.caption(v["evidence"])
    evidence = svc.evidence_rows(v["rows"])
    if evidence:
        st.dataframe(pd.DataFrame(evidence), hide_index=True, width="stretch")


def _questions_tab(v: dict) -> None:
    q = v["validation"]["questions"]
    st.caption(q["caption"])
    for r in q["rows"]:
        with st.container(border=True):
            st.markdown(f"{r['badge']} **{r['id']}** · {r['question']}")
            st.markdown(f"**Respuesta:** {r['answer']}")
            if r["why"]:
                st.markdown(f"**{r['why_label']}:** {r['why']}")
            st.caption(r["criterion"])
            with st.expander("Ver la consulta SPARQL"):
                if r["query"]:
                    st.code(r["query"], language="sparql")
                else:
                    st.caption("Esta consulta ya no está en el catálogo de preguntas.")


def _validation(v: dict) -> None:
    val = v["validation"]
    st.markdown("**Validación**")
    with st.container(border=True):
        if val is None:
            st.caption(svc.NO_VALIDATION)
            return
        st.markdown(f"{val['badge']} {val['headline']}")
        for col, card in zip(st.columns(len(val["cards"])), val["cards"]):
            with col:
                st.caption(f"Criterio: {card['criterion']}")
                st.markdown(f"**{card['title']}** {card['badge']}")
                st.markdown(card["text"])
                st.caption(card["caption"])
        with st.expander("¿Qué revisa cada uno?"):
            for title, text in svc.VALIDATION_HELP.items():
                st.markdown(f"**{title}.** {text}")
        with st.expander("Ver el detalle"):
            if val["rules"]:
                st.markdown("**Forma: reglas con hallazgos**")
                st.dataframe(pd.DataFrame(val["rules"]), hide_index=True, width="stretch")
            if val["kinds"]:
                st.markdown("**Lógica: tipos de hallazgo**")
                st.dataframe(pd.DataFrame(val["kinds"]), hide_index=True, width="stretch")
            if not val["rules"] and not val["kinds"]:
                st.caption("La forma y la lógica no tienen hallazgos.")
            st.download_button(f"Descargar el detalle ({val['log_file']})",
                               data=partial(svc.validation_log, v["id"], svc.workspace_cfg()),
                               file_name=val["log_file"], mime="text/plain", key="align_dl_log",
                               on_click="ignore", icon=":material/description:")


_COLORS = {"aligned": "#4C78A8", "none": "#BAB0AC", "dpv": "#F58518", "root": "#31333F"}
_SHAPES = {"dpv": "diamond", "root": "box"}
_EDGES = {"dpv": "#F58518", "subclass": "#4C78A8", "root": "#D5DAE5"}


def _preview(ws_id: str) -> None:
    """The small interactive graph of the ready screen (streamlit-agraph, offline)."""
    from streamlit_agraph import Config, Edge, Node, agraph
    p = svc.preview(ws_id)
    nodes = [Node(id=n["id"], label=n["label"], title=n["title"], color=_COLORS[n["kind"]],
                  shape=_SHAPES.get(n["kind"], "dot"), size=14,
                  font={"size": 16, "color": "#FFFFFF" if n["kind"] == "root" else "#31333F"})
             for n in p["nodes"]]
    edges = [Edge(source=e["source"], target=e["target"], label=e["label"],
                  color=_EDGES[e["kind"]], font={"size": 13}) for e in p["edges"]]
    config = Config(height=460, directed=True, physics=True)
    config.width = "100%"                                 # Config always appends "px"
    config.physics["barnesHut"] = {"springLength": 130, "avoidOverlap": 0.4}
    with st.container(border=True, key="align_preview"):
        agraph(nodes=nodes, edges=edges, config=config)
    st.caption(svc.preview_caption(p))


DECIDE = len(svc.WALK_TITLES) + 1                         # the step of "Tú decides"


def _walk_to(step: int) -> None:
    """Step 0 is "¡Tu ontología está lista!", 1 to 3 "Lo que hicimos", 4 "Tú decides" and
    after them the review."""
    ss.pop("align_review", None)
    ss.pop("align_step", None)
    ss.pop("align_applied", None)
    if step > DECIDE:
        ss["align_review"] = True
    elif step > 0:
        ss["align_step"] = step


def _open_review() -> None:
    _walk_to(1)


def _close_review() -> None:
    _walk_to(DECIDE)


def _broken(exc: Exception) -> None:
    st.error(svc.error_message(exc))
    st.button("Empezar de nuevo", key="align_again", icon=":material/restart_alt:",
              on_click=_home)


def step_ready(ws_id: str) -> None:
    """¡Tu ontología está lista!: a preview and one button to review it (S10-T12)."""
    if ss.get("align_mode") in svc.MODES:                 # its files are kept: go back to them
        _back(_to_upload, "align_result_back")
    try:
        v = svc.view(ws_id)
    except Exception as exc:                              # noqa: BLE001 - shown to the person
        _broken(exc)
        return
    st.title(svc.READY_TITLE)
    st.caption(v["source"])
    _preview(ws_id)
    st.button(svc.REVIEW_BUTTON, key="align_review_open", type="primary",
              icon=":material/fact_check:", on_click=_open_review)
    st.caption(svc.REVIEW_HINT)


def step_walk(ws_id: str, step: int) -> None:
    """Lo que hicimos, paso a paso: three short screens in plain words (S10-T13)."""
    _back(lambda: _walk_to(step - 1), "align_walk_back")
    try:
        s = svc.walkthrough(ws_id)[step - 1]
    except Exception as exc:                              # noqa: BLE001 - shown to the person
        _broken(exc)
        return
    total = len(svc.WALK_TITLES)
    st.progress(step / total, text=f"Paso {step} de {total}")
    st.title(s["title"])
    st.markdown(s["lead"])
    if s["points"]:
        st.markdown("\n".join(f"- {point}" for point in s["points"]))
    if s["examples"]:
        with st.container(border=True):
            st.markdown(f"**{s['examples_title']}**")
            st.markdown("\n".join(f"- {example}" for example in s["examples"]))
    st.button("Siguiente" if step < total else "Continuar", key="align_walk_next",
              type="primary", icon=":material/arrow_forward:", on_click=_walk_to,
              args=(step + 1,))


def _record(ws_id: str, iri: str) -> None:
    """The provenance of a concept (S10-T08), only when the person opens it."""
    c = svc.provenance_card(ws_id, iri)
    if c["missing"]:
        st.caption(c["missing"])
        return
    st.markdown(c["headline"])
    st.markdown(c["origin"])
    if c["versions"]:
        st.dataframe(pd.DataFrame(c["versions"]), hide_index=True, width="stretch")
    for s in c["steps"]:
        head = (f"{s['badge']} " if s["badge"] else "") + f"**{s['title']}**"
        st.markdown("  \n".join([head, *([s["text"]] if s["text"] else []), *s["extra"]]))


def _answer(n: int, key: str) -> None:
    ss.setdefault("align_answers", {})[n] = ss[key]


def _goto_question(i: int) -> None:
    ss["align_q"] = i


def _add_extra(iri: str, name: str, key: str, options: list[dict]) -> None:
    chosen = options[ss[key]]
    extra = [e for e in ss.get("align_extra", []) if e["concept"] != iri]
    ss["align_extra"] = [*extra, {"concept": iri, "name": name, "text": chosen["text"],
                                  "note": f"{name} → {chosen['text']}", **chosen["change"]}]


def _drop_extra(iri: str) -> None:
    ss["align_extra"] = [e for e in ss.get("align_extra", []) if e["concept"] != iri]


def _apply() -> None:
    ss["align_applying"] = True


def _keep_all() -> None:
    for key in ("align_answers", "align_extra", "align_q"):
        ss.pop(key, None)
    _walk_to(DECIDE + 1)


def _question(ws_id: str, d: dict) -> None:
    qs = d["questions"]
    i = min(ss.get("align_q", 0), len(qs) - 1)
    q = qs[i]
    st.progress((i + 1) / len(qs), text=f"Pregunta {i + 1} de {len(qs)}")
    with st.container(border=True, key="align_question"):
        st.markdown(f"**{q['text']}**")
        letters = list(q["options"])
        chosen = ss.get("align_answers", {}).get(q["n"], q["leave"])    # nothing until chosen
        key = f"align_answer_{d['version']}_{q['n']}"
        st.radio("Tu respuesta", letters, index=letters.index(chosen),
                 format_func=q["options"].get, key=key, on_change=_answer, args=(q["n"], key),
                 label_visibility="collapsed")
        if q["why"]:
            st.caption(f"¿Por qué te lo preguntamos? {q['why']}")
        with st.expander("Ver el registro completo"):
            _record(ws_id, q["concept"])
    left, right = st.columns(2)
    if i > 0:
        left.button("Anterior", key="align_q_prev", icon=":material/arrow_back:",
                    on_click=_goto_question, args=(i - 1,))
    if i < len(qs) - 1:
        right.button("Siguiente pregunta", key="align_q_next", type="primary",
                     icon=":material/arrow_forward:", on_click=_goto_question, args=(i + 1,))


def _adjust(ws_id: str) -> None:
    """Ajustar otro concepto: what was "¿Está mal? Ver opciones"."""
    with st.expander("Ajustar otro concepto"):
        options = svc.view(ws_id)["concept_options"]
        iri = st.selectbox("Concepto", list(options), index=None, format_func=options.get,
                           placeholder="Elige el concepto que quieres cambiar",
                           key="align_adjust_concept")
        if iri is not None:
            choices = svc.adjust_options(ws_id, iri, targets=_targets())
            if not choices:
                st.caption("No tenemos otra opción para este concepto.")
            else:
                key = "align_adjust_choice"
                st.radio("Cámbialo a", range(len(choices)),
                         format_func=lambda k: choices[k]["text"], key=key)
                st.button("Agregar este cambio", key="align_adjust_add",
                          icon=":material/add:", on_click=_add_extra,
                          args=(iri, options[iri].split(" · ")[0], key, choices))
        for e in ss.get("align_extra", []):
            col, drop = st.columns([5, 1])
            col.markdown(f"- {svc.plain(e['name'])}: {svc.plain(e['text'])}")
            drop.button("Quitar", key=f"align_extra_drop_{e['concept']}", type="tertiary",
                        on_click=_drop_extra, args=(e["concept"],))


def _applied(ws_id: str, applied: dict) -> None:
    st.title(svc.APPLIED_TITLE if applied["applied"] else svc.DECIDE_TITLE)
    if not applied["applied"]:
        st.info(svc.NOTHING_APPLIED, icon=":material/info:")
    if applied["changes"]:
        st.markdown("\n".join(f"- {svc.plain(line)}" for line in applied["changes"]))
    for line in applied["refused"]:
        st.warning(f"No se aplicó: {svc.plain(line)}", icon=":material/warning:")
    if applied["applied"]:
        _preview(ws_id)
    st.button("Continuar", key="align_decide_done", type="primary",
              icon=":material/arrow_forward:", on_click=_walk_to, args=(DECIDE + 1,))


def step_decide(ws_id: str) -> None:
    """Tú decides: the questions one at a time, other concepts, and the new version (S10-T14)."""
    if ss.pop("align_applying", False):
        with st.status(svc.RECALCULATING, expanded=False) as status:
            try:
                applied = svc.apply_answers(ws_id, ss.get("align_answers", {}),
                                            ss.get("align_extra", []), targets=_targets())
            except Exception as exc:                      # noqa: BLE001 - shown to the person
                status.update(label="No se pudo recalcular", state="error")
                st.error(svc.error_message(exc))
                return
            status.update(label="Listo", state="complete")
        for key in ("align_answers", "align_extra", "align_q"):
            ss.pop(key, None)
        ss["align_applied"] = applied
        st.rerun()                                        # the result, without the status box
    applied = ss.get("align_applied")
    if applied is not None:
        _applied(ws_id, applied)
        return
    _back(lambda: _walk_to(DECIDE - 1), "align_decide_back")
    try:
        d = svc.decisions_view(ws_id)
    except Exception as exc:                              # noqa: BLE001 - shown to the person
        _broken(exc)
        return
    st.title(svc.DECIDE_TITLE)
    if d["questions"]:
        st.markdown(svc.DECIDE_LEAD.format(n=svc.counted(len(d["questions"]), "detalle",
                                                         "detalles")))
        _question(ws_id, d)
    else:
        st.success(svc.NOTHING_TO_DECIDE, icon=":material/check_circle:")
    _adjust(ws_id)
    answered = ss.get("align_answers") or ss.get("align_extra")
    if answered:
        st.button("Aplicar mis respuestas", key="align_apply", type="primary",
                  icon=":material/check:", on_click=_apply)
    st.button("Mantener todo y continuar", key="align_keep_all",
              type="tertiary" if answered else "primary", icon=":material/arrow_forward:",
              on_click=_keep_all)


def step_result(ws_id: str) -> None:
    """Descarga tu ontología: one file per format (S10-T15); the technical details only when
    the person asks for them."""
    _back(_close_review, "align_review_back")
    try:
        v = svc.view(ws_id)
        final = svc.final_downloads(ws_id)
    except Exception as exc:                              # noqa: BLE001 - shown to the person
        _broken(exc)
        return
    st.title(svc.FINAL_TITLE)
    st.caption(final["caption"])
    for note in svc.notes(v):
        st.warning(note, icon=":material/warning:")
    for col, d in zip(st.columns(len(final["buttons"])), final["buttons"]):
        col.download_button(d["label"], data=d["data"], file_name=d["file_name"], mime=d["mime"],
                            key=f"align_dl_{d['key']}", help=d["help"], on_click="ignore",
                            icon=":material/download:", width="stretch", type="primary")
    st.caption(svc.FINAL_NOTE)
    if final["previous"]:
        p = final["previous"]
        st.download_button(p["label"], data=p["data"], file_name=p["file_name"],
                           mime="text/turtle", key="align_dl_previous", on_click="ignore",
                           type="tertiary", icon=":material/history:")
    if st.toggle(svc.DETAILS_TOGGLE, key="align_details"):
        _validation(v)
        pages = {"Resumen": _summary_tab, "Conceptos": _concepts_tab,
                 "Para revisar": _review_tab}
        if (v["validation"] or {}).get("questions"):
            pages[svc.QUESTIONS_TAB] = _questions_tab
        if v["evidence"]:
            pages[svc.LAW_TAB] = _law_tab
        for tab, show in zip(st.tabs(list(pages)), pages.values()):
            with tab:
                show(v)
    st.button("Alinear otro archivo", key="align_again", icon=":material/restart_alt:",
              on_click=_home)


# ------------------------------------------------------------------ which step

if ss.get("align_ws") and ss.get("align_review"):
    step_result(ss["align_ws"])
elif ss.get("align_ws") and ss.get("align_step") == DECIDE:
    step_decide(ss["align_ws"])
elif ss.get("align_ws") and ss.get("align_step"):
    step_walk(ss["align_ws"], ss["align_step"])
elif ss.get("align_ws"):
    step_ready(ss["align_ws"])
elif ss.get("align_mode") in svc.MODES:
    step_upload(ss["align_mode"])
elif ss.get("align_path") == svc.PATH_ALIGN:
    step_choose()
elif ss.get("align_path") == svc.PATH_TRANSFORM:
    step_transform()
else:
    step_start()