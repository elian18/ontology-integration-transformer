"""S7-T05: before/after evaluation of the decider and calibration of its fixed rule.

Fast tests with the fakes of test_alignment_decider (embedder, LLM, temporary ChromaDB): the
three configurations run in leave-one-out, the call budget is shared and resumed from the
cache, the rule is chosen with the grid fixed beforehand (or, when it has none, with the second
grid, whose optimism is measured with cross-validation), and only confirmed elsewhere.
"""
import json

import pytest

from src.alignment import decider_eval as ev
from src.alignment.decider import (CACHE_FILE, SKOS_EXACT, SKOS_STRONG, STATUS_DECIDED, Decision,
                                   apply_rule)
from src.alignment.reference import (REF_BASE, REF_BLIND, REF_GDPRTEXT, RefItem, Reference,
                                     base_reference)
from src.alignment.sources import SourceConcept
from tests.test_alignment_decider import (B, D, FakeEmbed, FakeLLM, _concepts, base,  # noqa: F401
                                          chroma, targets)

CFG = {"decider": {"candidates": 3, "precedents": 2, "batch_size": 2, "pause_seconds": 0}}
BASELINE = {REF_BASE: {"proposals": 2, "correct": 1, "precision": 0.5, "coverage": 0.3333,
                       "f1": 0.4, "none_claimed": 0, "none_ok": 0, "none_wrong": 0,
                       "relation_ok": 0, "relation_typed": 1}}


@pytest.fixture
def inputs(base, targets):  # noqa: F811
    base_ref = base_reference(base)
    concepts = {c.key: SourceConcept(key=c.key, origin=c.origin, name=c.name, kinds=c.kinds,
                                     label=c.label, iri=c.iri) for c in base.concepts}
    for c in _concepts():
        concepts[c.key] = c
    by_key = base_ref.by_key()
    blind = Reference(REF_BLIND, [by_key[B + "Sanction"], by_key[B + "Right_of_access"]])
    gdpr = Reference(REF_GDPRTEXT, [
        RefItem("u#Right_to_rectification", "Right_to_rectification", "aligned",
                {D + "DataSubjectRight": "skos:broadMatch"}),
        RefItem("u#Penalty", "Penalty", "no_match")])
    return ev.EvalInputs(references={REF_BASE: base_ref, REF_BLIND: blind, REF_GDPRTEXT: gdpr},
                         concepts=concepts, targets=targets, baseline=BASELINE)


def _evaluate(inputs, chroma, llm=None, **kwargs):  # noqa: F811
    llm = llm or FakeLLM()
    kwargs.setdefault("budget", 100)
    result = ev.evaluate(inputs, llm=llm, embed_fn=FakeEmbed(), chroma_path=chroma,
                         embedding_model="fake-model", sleep_fn=lambda s: None, cfg=CFG,
                         **kwargs)
    return result, llm


# --------------------------------------------------------------------------- the grid

def test_the_rule_grid_is_fixed_before_running():
    assert len(ev.RULE_GRID) == 6
    assert [ev.rule_label(r) for r in ev.RULE_GRID[:2]] == ["alta, puntaje >= 0.0",
                                                           "alta, puntaje >= 0.4"]
    assert ev.RULE_GRID[-1] == {"accept": ("alta", "media"), "min_score": 0.5}


def test_the_second_grid_separates_relations_and_kinds():
    grid = ev.RULE_GRID_2
    labels = [ev.rule_label(r) for r in grid]
    assert len(grid) == 24 and len(set(labels)) == 24
    assert all(r["accept"] == ("alta",) for r in grid)
    assert "alta, puntaje >= 0.5 | clases: sin related | propiedades: no responde" in labels
    chosen = grid[labels.index("alta, puntaje >= 0.5 | clases: sin related | propiedades: "
                               "no responde")]
    assert chosen["relations"] == SKOS_STRONG and chosen["property_relations"] == ()
    assert "skos:relatedMatch" not in SKOS_STRONG and "skos:narrowMatch" in SKOS_STRONG
    exact = [r for r in grid if r["relations"] == SKOS_EXACT]
    assert {r["properties"] for r in exact} == {"igual", "no responde"}      # no duplicate
    assert ev.rule_json(chosen, ev.GRID_SECOND) == {
        "grid": "segunda", "label": labels[labels.index(ev.rule_label(chosen))],
        "accept": ["alta"], "min_score": 0.5, "relations": list(SKOS_STRONG),
        "property_relations": []}
    first = ev.rule_json(ev.RULE_GRID[0], ev.GRID_FIRST)                     # any relation
    assert first["relations"] is None and first["property_relations"] is None
    same = ev.rule_json({"accept": ("alta",), "min_score": 0.5, "relations": SKOS_EXACT}, "x")
    assert same["property_relations"] == ["skos:exactMatch"]       # as apply_rule: the same


def test_first_rule_of_both_grids_gives_the_same_answers():
    decisions = [Decision(key=f"k{i}", name="x", status=STATUS_DECIDED, dpv_iri=D + "X",
                          relation=rel, confidence="alta", score=0.3, kinds=kinds)
                 for i, (rel, kinds) in enumerate([("skos:relatedMatch", ("class",)),
                                                   (None, ("property",)),
                                                   ("skos:closeMatch", ("property",))])]
    assert (apply_rule(decisions, **ev.rule_kwargs(ev.RULE_GRID[0]))
            == apply_rule(decisions, **ev.rule_kwargs(ev.RULE_GRID_2[0])))


def _row(accept, min_score, proposals, precision, coverage, label=""):
    return {"accept": accept, "min_score": min_score, "label": label,
            "score": {"proposals": proposals, "precision": precision, "coverage": coverage}}


def test_choose_rule_keeps_precision_then_maximizes_coverage():
    table = [_row(("alta",), 0.0, 40, 0.80, 0.30), _row(("alta",), 0.4, 30, 0.85, 0.25),
             _row(("alta", "media"), 0.0, 60, 0.70, 0.40), _row(("alta", "media"), 0.4, 50, 0.76,
                                                                0.30),
             _row(("alta",), 0.5, 5, 1.0, 0.05)]
    chosen = ev.choose_rule(table, 0.75)
    assert chosen["accept"] == ("alta",) and chosen["min_score"] == 0.0   # tie 0.30: fewer levels
    assert ev.choose_rule(table, 0.95) is None                           # 1.0 has < 10 proposals
    assert ev.choose_rule(table, 0.75, min_proposals=100) is None
    same = [_row(("alta",), 0.5, 20, 0.80, 0.30, "a"), _row(("alta",), 0.5, 20, 0.90, 0.30, "b")]
    assert ev.choose_rule(same, 0.75)["label"] == "b"                     # then precision


def test_the_second_grid_is_searched_only_when_the_first_has_no_rule():
    good = [_row(("alta",), 0.0, 40, 0.80, 0.30, "primera")]
    bad = [_row(("alta",), 0.0, 40, 0.50, 0.60, "primera")]
    second = [_row(("alta",), 0.5, 30, 0.76, 0.25, "segunda")]
    assert ev.choose_final(good, second, 0.75)[1] == ev.GRID_FIRST
    rule, grid = ev.choose_final(bad, second, 0.75)
    assert grid == ev.GRID_SECOND and rule["label"] == "segunda"
    assert ev.choose_final(bad, second, 0.80) == (None, None)
    assert ev.choose_final(bad, None, 0.75) == (None, None)


# --------------------------------------------------------------------------- cross-validation

def _cv_case(n=40, related_right=0):
    """Half aligned and decided right with exactMatch; half 'sin correspondencia' where the
    LLM still proposes a relatedMatch: only the rules without relatedMatch are precise.
    ``related_right`` more aligned concepts are decided right but with relatedMatch."""
    items, decisions = [], []
    for i in range(n + related_right):
        aligned = i % 2 == 0 or i >= n
        relation = "skos:exactMatch" if aligned and i < n else "skos:relatedMatch"
        items.append(RefItem(f"k{i:02d}", f"c{i}", "aligned" if aligned else "no_match",
                             {D + f"T{i}": relation} if aligned else {}))
        decisions.append(Decision(key=f"k{i:02d}", name=f"c{i}", status=STATUS_DECIDED,
                                  dpv_iri=D + f"T{i}", relation=relation, confidence="alta",
                                  score=0.8, kinds=("class",)))
    return decisions, Reference(REF_BASE, items)


def test_nearest_rank_percentiles():
    values = [float(v) for v in range(10, 0, -1)]
    assert [ev._nearest_rank(values, q) for q in (0.1, 0.5, 0.9, 1.0)] == [1.0, 5.0, 9.0, 10.0]


def test_cross_validation_measures_on_the_other_half():
    decisions, ref = _cv_case()
    cv = ev.cross_validate(decisions, ref, repeats=10, seed=3)
    assert cv["folds"] == 20 and cv["folds_without_rule"] == 0
    assert cv["precision_mean"] == 1.0 and cv["precision_p10"] == 1.0
    assert cv["coverage_mean"] == 1.0 and cv["min_proposals_per_half"] == 5
    assert sum(n for _, n in cv["most_chosen"]) <= 20
    assert all("clases: todas" not in label for label, _ in cv["most_chosen"])   # not precise
    assert ev.cross_validate(decisions, ref, repeats=10, seed=3) == cv      # deterministic
    none = ev.cross_validate(decisions, ref, min_precision=1.01, repeats=5)
    assert none["folds_without_rule"] == 10 and none["precision_mean"] is None


def test_cross_validation_chooses_on_one_half_and_measures_on_the_other(monkeypatch):
    decisions, ref = _cv_case()
    seen = []
    real = ev.score

    def spy(outputs, reference):
        seen.append(frozenset(i.key for i in reference.items))
        return real(outputs, reference)

    monkeypatch.setattr(ev, "score", spy)
    ev.cross_validate(decisions, ref, repeats=1, seed=1)
    n = len(ev.RULE_GRID_2)
    a, b = seen[0], seen[n + 1]
    assert not a & b and len(a | b) == 40 and len(a) == 20
    assert seen == [a] * n + [b] + [b] * n + [a]       # choose on one half, measure the other


def test_summary_uses_the_second_grid_and_counts_the_suggestions():
    decisions, ref = _cv_case(related_right=4)
    data = {"counts": ev._counts(decisions), "complete": True, "calls": 0, "cached": 44,
            "errors": [], "rules": ev.rule_table(decisions, ref),
            "rules_2": ev.rule_table(decisions, ref, ev.RULE_GRID_2), "_decisions": decisions}
    results = {ev.CONFIG_PRECEDENTS: {"label": "x", "sets": {REF_BASE: data}}}
    inputs = ev.EvalInputs(references={REF_BASE: ref}, concepts={}, targets=None, baseline=None)
    result = ev._summarize(results, inputs, 0, False)
    chosen = result["chosen_rule"]                       # first grid: 24 of 44 right = 54,5 %
    assert chosen["grid"] == ev.GRID_SECOND and "skos:relatedMatch" not in chosen["relations"]
    assert result["suggestions"] == {"total": 24, "correct": 4, "on_no_match": 20}
    assert result["proposals_on_no_match_total"] == 0
    assert result["cross_validation"]["folds"] == 2 * ev.CV_REPEATS


# --------------------------------------------------------------------------- running

def test_three_configurations_run_in_leave_one_out(inputs, chroma):
    article = {"number": 9, "title": "Sanciones", "text": "Las sanciones..."}
    result, llm = _evaluate(inputs, chroma, evidence_fn=lambda c: article)
    configs = result["configs"]
    assert set(configs) == {ev.CONFIG_PLAIN, ev.CONFIG_PRECEDENTS, ev.CONFIG_LAW}
    assert set(configs[ev.CONFIG_LAW]["sets"]) == {REF_BASE, REF_BLIND}          # no GDPRtEXT
    assert set(configs[ev.CONFIG_PLAIN]["sets"]) == {REF_BASE, REF_BLIND, REF_GDPRTEXT}
    contexts = [c["context"] for c in llm.calls]
    assert any("Precedentes" not in c and "Articulo" not in c for c in contexts)    # plain
    assert any("Articulo 9" in c for c in contexts)                                 # law
    for c in contexts:                                    # leave one out: never itself
        for name in ("Right_of_access", "Sanction"):
            if f"Concepto: {name} " in c and "Precedentes" in c:
                block = c.split(f"Concepto: {name} ")[1].split("===")[0]
                assert f"- {name} " not in block
    base = configs[ev.CONFIG_PRECEDENTS]["sets"][REF_BASE]
    assert base["complete"] and base["counts"]["concepts"] == 4
    assert len(base["rules"]) == 6
    assert result["pending"] == 0 and result["llm_calls_this_run"] == len(llm.calls)


def test_blind_concepts_already_in_the_base_come_from_the_cache(inputs, chroma, tmp_path):
    result, _ = _evaluate(inputs, chroma, cache_path=tmp_path / CACHE_FILE,
                          configs=(ev.CONFIG_PRECEDENTS,))
    blind = result["configs"][ev.CONFIG_PRECEDENTS]["sets"][REF_BLIND]
    assert blind["calls"] == 0 and blind["cached"] == 2


def test_no_rule_with_too_few_proposals_and_a_rule_when_allowed(inputs, chroma, monkeypatch):
    result, _ = _evaluate(inputs, chroma)
    assert result["chosen_rule"] is None                                    # 4 concepts < 10
    text = ev.render_console(result)
    assert "Ninguna regla llega" in text and "en ninguna de las dos grillas" in text
    assert "Segunda grilla" in text and "Confirmación con la regla mostrada (provisional)" in text
    assert result["suggestions"] is None
    monkeypatch.setattr(ev, "MIN_PROPOSALS", 1)
    inputs.min_precision = 0.0
    result, _ = _evaluate(inputs, chroma)
    assert result["chosen_rule"]["grid"] == ev.GRID_FIRST
    assert "Segunda grilla" not in ev.render_console(result)
    names = {e["name"] for e in result["proposals_on_no_match"]}
    assert result["proposals_on_no_match_total"] == len(names) and names <= {"Sanction"}


def test_the_second_grid_rule_is_used_when_the_first_grid_fails(inputs, chroma, monkeypatch):
    monkeypatch.setattr(ev, "RULE_GRID", ({"accept": ("baja",), "min_score": 0.0},))
    monkeypatch.setattr(ev, "MIN_PROPOSALS", 1)
    inputs.min_precision = 0.0
    result, _ = _evaluate(inputs, chroma)
    chosen = result["chosen_rule"]
    assert chosen["grid"] == ev.GRID_SECOND and "| clases:" in chosen["label"]
    base = result["configs"][ev.CONFIG_PRECEDENTS]["sets"][REF_BASE]
    assert len(base["rules_2"]) == 24
    assert set(result["suggestions"]) == {"total", "correct", "on_no_match"}
    text = ev.render_console(result)
    assert "Ninguna regla de esta grilla" in text and text.count("<- elegida") == 1
    assert "Confirmación con la regla elegida" in text and "Sugeridas" in text
    blind = result["configs"][ev.CONFIG_PRECEDENTS]["sets"][REF_BLIND]
    assert ev._rule_of(blind, chosen)["label"] == chosen["label"]          # confirmed, same rule


def test_the_budget_is_shared_and_the_next_run_resumes(inputs, chroma, tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "MIN_PROPOSALS", 1)          # any rule would qualify...
    inputs.min_precision = 0.0
    cache = tmp_path / CACHE_FILE
    first, _ = _evaluate(inputs, chroma, budget=1, cache_path=cache)
    assert first["llm_calls_this_run"] == 1 and first["pending"] > 0
    assert first["chosen_rule"] is None                  # ...but not on incomplete results
    base = first["configs"][ev.CONFIG_PLAIN]["sets"][REF_BASE]
    assert not base["complete"] and base["counts"]["pending"] == 2
    assert "incompleta" in ev.render_console(first)
    second, llm = _evaluate(inputs, chroma, budget=100, cache_path=cache)
    assert second["pending"] == 0 and len(llm.calls) == second["llm_calls_this_run"]
    assert second["chosen_rule"] is not None
    third, llm = _evaluate(inputs, chroma, budget=100, cache_path=cache)
    assert third["llm_calls_this_run"] == 0 and llm.calls == []


def test_no_rule_is_chosen_on_incomplete_results(inputs, chroma, monkeypatch):
    monkeypatch.setattr(ev, "MIN_PROPOSALS", 1)          # any rule with a proposal qualifies...
    inputs.min_precision = 0.0
    partial, _ = _evaluate(inputs, chroma, budget=1, configs=(ev.CONFIG_PRECEDENTS,))
    base = partial["configs"][ev.CONFIG_PRECEDENTS]["sets"][REF_BASE]
    assert not base["complete"]
    assert any(r["score"]["proposals"] for r in base["rules"])
    assert partial["chosen_rule"] is None                # ...but not while results are partial


def test_quota_stops_every_following_set(inputs, chroma):
    quota = Exception("Error code: 429 RESOURCE_EXHAUSTED quota")
    result, llm = _evaluate(inputs, chroma, FakeLLM(fail=[quota, quota]))
    assert result["quota_exhausted"] and len(llm.calls) == 2
    assert result["pending"] == sum(s["counts"]["concepts"] for r in result["configs"].values()
                                    for s in r["sets"].values())
    assert "CUOTA AGOTADA" in ev.render_console(result)


# --------------------------------------------------------------------------- outputs

def test_json_output_is_serializable_without_the_decisions(inputs, chroma):
    result, _ = _evaluate(inputs, chroma, configs=(ev.CONFIG_PRECEDENTS,))
    data = ev.to_json(result)
    text = json.dumps(data)
    assert "_decisions" not in text and data["prompt"]["name"] == "decide"
    assert data["configs"][ev.CONFIG_PRECEDENTS]["sets"][REF_BASE]["rules"][0]["score"]
    assert len(data["grid_2"]) == 24 and data["cross_validation"]["folds"] == 2 * ev.CV_REPEATS


def test_console_is_in_spanish(inputs, chroma):
    result, _ = _evaluate(inputs, chroma)
    text = ev.render_console(result)
    assert "Evaluación del decisor (S7-T05)" in text and "Línea base (Sprint 6)" in text
    assert "Decisor con precedentes + ley" in text and "Confirmación con la regla" in text
    assert "Validación cruzada de la segunda grilla" in text
    assert "Llamadas al LLM en esta corrida" in text


def test_reference_round_trip(inputs):
    ref = inputs.references[REF_BASE]
    again = ev.reference_from_dict(ref.to_dict())
    assert [i.key for i in again.items] == [i.key for i in ref.items]
    assert again.aligned() == ref.aligned()
    assert again.by_key()[B + "Sanction"].forbidden == ref.by_key()[B + "Sanction"].forbidden


def test_lopdp_evidence_like_sprint4():
    articles = {17: {"number": 17, "title": "Portabilidad", "text": "..."}}
    retrieved = []

    def retrieve(query):
        retrieved.append(query)
        return [{"number": 4, "title": "Definiciones", "text": "...", "distance": 0.1}]

    evidence = ev.lopdp_evidence_fn(articles=articles, retrieve_fn=retrieve)
    ai = SourceConcept(key="ai:PortabilityRight", origin="ai", name="PortabilityRight",
                       kinds=("class",), articles=[17])
    onto = SourceConcept(key="o#Consent", origin="ontology", name="Consent", kinds=("class",),
                         label="consentimiento")
    assert evidence(ai)["number"] == 17 and retrieved == []          # its own source article
    assert evidence(onto)["number"] == 4 and "Consent" in retrieved[0]