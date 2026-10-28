from __future__ import annotations

import json

import pytest

from src.ai.prompts import PromptNotFound, load_prompt
from src.alignment import decider as dc
from src.alignment import prompt_lab as lab
from src.alignment.decider import Decision, decide_prompt
from src.alignment.decider_eval import EvalInputs
from src.alignment.prompt_lab import (ACCEPTED_WRONG, DEV, INVALID, NONE_WRONG, OTHER_TERM,
                                      RIGHT_LOW, RIGHT_RULE, TERM_ON_NONE, TEST, check_prompt,
                                      classify, run_half, save_run, split, verdict)
from src.alignment.reference import REF_BASE, Reference, RefItem
from src.alignment.sources import SourceConcept
from tests.test_alignment_propose import DeciderLLM, Embed, make_env, targets  # noqa: F401

EX = "http://example.org/lab#"
DPV = "https://w3id.org/dpv#"
# name: (kinds, definition, what the person decided, what the fake LLM answers)
CASES = {
    "PersonalData": (("class",), "Information about an identified person, personal.",
                     {DPV + "PersonalData": "skos:exactMatch"},
                     ("PersonalData", "exactMatch", "alta")),
    "DataController": (("class",), "The one who decides the processing, a controller.",
                       {DPV + "DataController": "skos:closeMatch"},
                       ("DataController", "relatedMatch", "alta")),
    "PersonalInfo": (("class",), "Personal information of someone.",
                     {DPV + "PersonalData": "skos:closeMatch"},
                     ("PersonalData", "closeMatch", "media")),
    "SensitiveInfo": (("class",), "Sensitive personal data such as health.",
                      {DPV + "SensitivePersonalData": "skos:exactMatch"},
                      ("PersonalData", "broadMatch", "alta")),
    "PersonalDetail": (("class",), "A personal detail kept in a form.", {},
                       ("PersonalData", "closeMatch", "media")),
    "Holder": (("class",), "The holder of the data.", {DPV + "DataSubject": "skos:exactMatch"},
               ("ninguno", "", "alta")),
    "Pizza": (("class",), "Food.", {}, ("ninguno", "", "alta")),
    "hasRecipient": (("property",), "Links data with its recipient.",
                     {DPV + "hasRecipient": "skos:exactMatch"},
                     ("hasRecipient", "exactMatch", "alta")),
    "hasOwner": (("property",), "Links data with its owner.", {}, ("ninguno", "", "alta")),
}


def _concepts():
    return {EX + n: SourceConcept(key=EX + n, origin="ontology", name=n, kinds=kinds,
                                  definition=definition, iri=EX + n)
            for n, (kinds, definition, _, _) in CASES.items()}


def _reference():
    return Reference("base", [RefItem(key=EX + n, name=n,
                                      status="aligned" if terms else "no_match", terms=terms)
                              for n, (_, _, terms, _) in CASES.items()])


class LabLLM(DeciderLLM):
    def __init__(self, **kw):
        super().__init__(answers={n: a for n, (_, _, _, a) in CASES.items()}, **kw)
        self.prompts = []

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        self.prompts.append(prompt)
        return super().ask(prompt, context, max_tokens, json_mode)


@pytest.fixture
def env(tmp_path, targets):  # noqa: F811
    env = make_env(tmp_path)
    env.inputs = EvalInputs(references={REF_BASE: _reference()}, concepts=_concepts(),
                            targets=targets, baseline=None)
    env.cache = tmp_path / "decider-cache.jsonl"
    env.folder = tmp_path / "lab"
    return env


def _run(env, version=1, half=DEV, llm=None, budget=5):
    return run_half(env.inputs, version, half, llm=llm or LabLLM(), embed_fn=Embed(),
                    budget=budget, cache_path=env.cache, chroma_path=env.chroma,
                    sleep_fn=lambda s: None, cfg=env.cfg, base=env.base)


def _where(name):
    halves = split(_reference(), _concepts())
    return DEV if EX + name in {i.key for i in halves[DEV].items} else TEST


# ------------------------------------------------------------------ the halves

def test_the_halves_are_fixed_balanced_and_disjoint():
    ref, concepts = _reference(), _concepts()
    halves = split(ref, concepts)
    dev, test = ({i.key for i in halves[h].items} for h in (DEV, TEST))
    assert not dev & test and dev | test == {i.key for i in ref.items}
    assert abs(len(dev) - len(test)) <= 1
    assert abs(halves[DEV].aligned() - halves[TEST].aligned()) <= 1
    props = {h: sum(lab._only_property(concepts[i.key]) for i in halves[h].items)
             for h in halves}
    assert abs(props[DEV] - props[TEST]) <= 1
    assert split(ref, concepts) == halves                       # same seed, same halves
    many = Reference("base", [RefItem(key=f"k{n}", name=f"k{n}", status="no_match")
                              for n in range(40)])
    assert split(many, {})[DEV] != split(many, {}, seed=1)[DEV]
    assert len(split(many, {})[DEV].items) == 20


def test_the_halves_of_an_odd_group_go_to_the_smaller_half():
    ref = Reference("base", [RefItem(key=f"a{n}", name="a", status="aligned") for n in range(3)]
                    + [RefItem(key=f"b{n}", name="b", status="no_match") for n in range(3)])
    halves = split(ref, {})
    assert (len(halves[DEV].items), len(halves[TEST].items)) == (3, 3)


# ------------------------------------------------------------------ the prompt

def test_the_published_versions_keep_the_output():
    for version in (1, 2):
        assert check_prompt(load_prompt("decide", version).text) == ([], [])
    v2 = load_prompt("decide", 2).text
    for part in ("ROL", "FUENTES", "TAREA", "REGLAS", "SALIDA"):
        assert f"\n{part}\n" in f"\n{v2}\n"
    v1 = load_prompt("decide", 1).text
    for rule in ("No inventes terminos.", '"ninguno" es una respuesta valida y frecuente',
                 "Tambien se usa \"alta\" cuando estas seguro de que no hay correspondencia."):
        assert rule in v1 and rule in v2                       # v2 only reorganizes v1


def test_check_prompt_finds_what_a_new_version_lost():
    text = load_prompt("decide", 2).text
    errors, _ = check_prompt(text.replace('"justification"', '"razon"'))
    assert errors == ['falta "justification"']
    errors, _ = check_prompt(text.replace("relatedMatch", "related"))
    assert errors == ["falta relatedMatch"]
    _, warnings = check_prompt(text + " Definición.")
    assert warnings == ["tiene tildes o eñes (los prompts del proyecto se escriben sin ellas)"]
    _, warnings = check_prompt(text + "x" * lab.MAX_PROMPT_CHARS)
    assert warnings[0].startswith("es muy largo")


def test_a_missing_version_says_where_to_save_it():
    with pytest.raises(ValueError, match=r"src/ai/prompts/decide\.v99\.txt"):
        lab.prompt_version(99)


# ------------------------------------------------------------------ the errors

def _d(**kw):
    base = {"key": "k", "name": "n", "status": "decided", "kinds": ("class",)}
    return Decision(**{**base, **kw})


def test_an_answer_outside_the_candidates_shows_the_term_it_named():
    item = RefItem(key="k", name="Application", status="no_match")
    d = _d(status="invalid", confidence="alta", candidates=[DPV + "Applicability"],
           justification="Sigue al precedente.",
           issues=["el término 'hasApplicability' no está entre los candidatos"])
    e = lab.example(d, item, None, {DPV + "Applicability": "Applicability"}, {})
    assert e["answer"]["dpv"] == "hasApplicability" and e["answer"]["outside_candidates"]
    assert ("   Respondió: hasApplicability (no está entre sus candidatos), confianza alta. "
            "«Sigue al precedente.»") in lab._example_lines(1, e)
    assert lab.written_term(_d(status="invalid", issues=[])) is None
    plain = lab.example(_d(none=True, confidence="alta"), item, None, {}, {})
    assert plain["answer"]["dpv"] == "ninguno" and not plain["answer"]["outside_candidates"]


def test_classify_every_kind_of_error():
    aligned = RefItem(key="k", name="n", status="aligned", terms={"T": "skos:exactMatch"})
    no_match = RefItem(key="k", name="n", status="no_match")
    yes = {"term": "T", "relation": "skos:exactMatch", "none": False}
    assert classify(_d(dpv_iri="T", confidence="alta"), aligned, yes) is None
    assert classify(_d(dpv_iri="X", confidence="alta"), aligned,
                    {**yes, "term": "X"}) == ACCEPTED_WRONG
    assert classify(_d(dpv_iri="X", confidence="alta"), no_match,
                    {**yes, "term": "X"}) == ACCEPTED_WRONG
    assert classify(_d(none=True, confidence="alta"), aligned, None) == NONE_WRONG
    assert classify(_d(none=True, confidence="alta"), no_match, None) is None
    assert classify(_d(dpv_iri="X", confidence="alta"), aligned, None) == OTHER_TERM
    assert classify(_d(dpv_iri="T", confidence="media"), aligned, None) == RIGHT_LOW
    assert classify(_d(dpv_iri="T", confidence="alta", relation="skos:relatedMatch"), aligned,
                    None) == RIGHT_RULE
    assert classify(_d(dpv_iri="X", confidence="media"), no_match, None) == TERM_ON_NONE
    assert classify(_d(status="invalid"), aligned, None) == INVALID
    assert classify(_d(status="pending"), aligned, None) is None
    assert classify(None, aligned, None) is None
    prop = {"kinds": ("property",)}                           # never automatic
    assert classify(_d(dpv_iri="T", confidence="alta", **prop), aligned, None) is None
    assert classify(_d(dpv_iri="X", confidence="media", **prop), no_match, None) is None
    assert classify(_d(none=True, confidence="alta", **prop), aligned, None) == NONE_WRONG


# ------------------------------------------------------------------ a run

def test_both_halves_decide_every_concept_with_the_production_rule(env):
    llm = LabLLM()
    dev, test = _run(env, llm=llm), _run(env, half=TEST, llm=llm)
    assert dev["concepts"] + test["concepts"] == len(CASES)
    assert dev["complete"] and test["complete"] and dev["prompt"]["version"] == 1
    assert all(p == load_prompt("decide", 1).text for p in llm.prompts)
    total = {k: dev["score"][k] + test["score"][k] for k in ("proposals", "correct", "none_ok",
                                                              "none_wrong")}
    assert total == {"proposals": 2, "correct": 1, "none_ok": 2, "none_wrong": 1}
    kinds = {e["name"]: k for run in (dev, test) for k, found in run["examples"].items()
             for e in found}
    assert kinds == {"SensitiveInfo": ACCEPTED_WRONG, "Holder": NONE_WRONG,
                     "PersonalInfo": RIGHT_LOW, "DataController": RIGHT_RULE,
                     "PersonalDetail": TERM_ON_NONE}
    assert sum(r["errors"][ACCEPTED_WRONG] for r in (dev, test)) == 1
    llm_total = {k: dev["llm"][k] + test["llm"][k] for k in dev["llm"]}
    assert llm_total == {"right_term": 4, "aligned": 6, "none_right": 2, "no_match": 3}
    assert dev["rule"] == dc.rule_settings(env.cfg).label()


def test_an_example_has_what_gemini_needs(env):
    run = _run(env, half=_where("Holder"))
    e = run["examples"][NONE_WRONG][0]
    assert e["name"] == "Holder" and e["kinds"] == ["class"]
    assert e["definition"] == "The holder of the data." and e["candidates"]
    assert e["answer"]["dpv"] == "ninguno" and e["answer"]["confidence"] == "alta"
    assert e["person"] == "DataSubject (exactMatch)"
    shown = {p.split(":")[0]: p for p in e["precedents"]}
    assert shown and set(shown) <= {c.name for c in env.base.concepts}
    assert "Consent" not in shown or shown["Consent"] == ("Consent: Consent (exactMatch); "
                                                          "ConsentRecord (relatedMatch)")
    assert "Turnover" not in shown or shown["Turnover"] == "Turnover: sin correspondencia"


def test_without_a_base_the_lab_loads_the_real_one(env, monkeypatch):
    from src.alignment import inherit
    monkeypatch.setattr(inherit, "load_base", lambda **kw: env.base)
    run = run_half(env.inputs, 1, _where("Holder"), llm=LabLLM(), embed_fn=Embed(), budget=5,
                   cache_path=env.cache, chroma_path=env.chroma, sleep_fn=lambda s: None,
                   cfg=env.cfg)
    assert run["examples"][NONE_WRONG][0]["precedents"]


def test_the_cache_answers_again_and_another_version_asks_again(env):
    llm = LabLLM()
    first = _run(env, llm=llm)
    calls = first["calls"]
    again = _run(env, llm=llm)
    assert calls >= 1 and again["calls"] == 0 and again["cached"] == first["concepts"]
    assert again["score"] == first["score"]
    v2 = _run(env, version=2, llm=llm)
    assert v2["calls"] == calls and v2["prompt"]["version"] == 2
    assert llm.prompts[-1] == load_prompt("decide", 2).text


def test_the_call_limit_leaves_a_run_incomplete_and_without_a_message(env):
    run = _run(env, budget=0)
    assert not run["complete"] and run["pending"] == run["concepts"] and run["calls"] == 0
    save_run(run, env.folder)
    assert not list(env.folder.glob("*gemini*"))
    text = lab.render_run(run, env.folder, {}, 1)
    assert "Errores: ninguno" in text and "cuando la corrida esté completa" in text
    assert f"pendientes: {run['pending']}" in text


def test_a_prompt_that_lost_the_output_is_refused_before_any_call(env, tmp_path, monkeypatch):
    broken = lab.load_prompt("decide", 2)
    monkeypatch.setattr(lab, "prompt_version",
                        lambda v: broken.__class__(**{**broken.__dict__,
                                                      "text": broken.text.replace('"dpv"', "")}))
    llm = LabLLM()
    with pytest.raises(ValueError, match='no sirve: falta "dpv"'):
        _run(env, version=2, llm=llm)
    assert llm.calls == 0


# ------------------------------------------------------------------ saved files

def test_save_writes_the_run_the_halves_and_the_message_for_gemini(env):
    dev = _run(env)
    path = save_run(dev, env.folder, env.inputs.references[REF_BASE], env.inputs.concepts)
    assert path.name == "decide-v1-desarrollo.json"
    assert json.loads(path.read_text(encoding="utf-8"))["score"] == dev["score"]
    halves = json.loads((env.folder / "split.json").read_text(encoding="utf-8"))
    assert halves["seed"] == lab.SPLIT_SEED and len(halves[DEV]) + len(halves[TEST]) == 9
    text = (env.folder / "decide-v1-desarrollo-para-gemini.txt").read_text(encoding="utf-8")
    assert "PROMPT ACTUAL (decide.v1)\n```\n" + load_prompt("decide", 1).text + "\n```" in text
    assert f"Propón decide.v{lab.next_version()} con UN SOLO cambio" in text
    assert lab.next_version() == max(lab.available()["decide"]) + 1 >= 3
    assert f"una regla fija decide qué entra a la ontología: {dev['rule']}" in text
    assert "No nombres conceptos ni términos de estos ejemplos" in text
    test = _run(env, half=TEST)
    save_run(test, env.folder)
    assert (env.folder / "decide-v1-desarrollo-para-gemini.txt").read_text(
        encoding="utf-8") == text                              # the test half never goes there
    assert sorted(p.name for p in env.folder.glob("*gemini*")) == [
        "decide-v1-desarrollo-para-gemini.txt"]


def test_the_message_shows_a_sample_of_each_kind_of_error():
    found = [{"key": f"k{n:02d}", "name": f"C{n}", "kinds": ["class"], "label": None,
              "definition": None, "candidates": ["A", "B"], "precedents": [],
              "answer": {"dpv": "ninguno", "relation": "", "confidence": "alta",
                         "justification": "No hay."}, "person": "A (exactMatch)"}
             for n in range(20)]
    result = {"prompt": {"version": 3}, "model": "m", "rule": "regla", "concepts": 276,
              "aligned": 87, "score": {"proposals": 10, "correct": 8, "precision": 0.8,
                                       "coverage": 0.092, "f1": 0.165},
              "llm": {"right_term": 50, "aligned": 87, "none_right": 150, "no_match": 189},
              "errors": {k: (20 if k == NONE_WRONG else 0) for k in lab.ERROR_LABELS},
              "examples": {k: (found if k == NONE_WRONG else []) for k in lab.ERROR_LABELS}}
    text = lab.gemini_message(result, "texto del prompt", 7)
    assert text.count("   La persona decidió: A (exactMatch)") == lab.EXAMPLES
    assert f"### {lab.ERROR_LABELS[NONE_WRONG]} (20)" in text
    assert "Con la regla: 10 propuestas, 8 correctas: precisión 80,0 %" in text
    assert "Propón decide.v7" in text and lab.gemini_message(result, "texto del prompt", 7) == text
    assert "Propón decide.v4" in lab.gemini_message(result, "texto del prompt")


# ------------------------------------------------------------------ adopting

def _result(version, half, precision, f1, complete=True, proposals=10, correct=8, aligned=87,
            variant=""):
    prompt = load_prompt("decide", version)
    return {"prompt": prompt.manifest(), "half": half, "complete": complete, "variant": variant,
            "score": {"precision": precision, "f1": f1, "proposals": proposals,
                      "correct": correct, "coverage": round(correct / aligned, 4),
                      "aligned": aligned}}


def test_verdict_needs_the_test_half_the_floor_and_three_points_more():
    v1, v2 = _result(1, TEST, 0.75, 0.42), _result(2, TEST, 0.76, 0.45)
    assert verdict(v1, v2, 0.75)["adopt"] is True
    assert verdict(v1, v2, 0.75)["reason"] == ("precisión 76,0 % (>= 75,0 %); F1 +3,0 puntos "
                                               "(>= +3)")
    assert verdict(v1, _result(2, TEST, 0.74, 0.50), 0.75)["adopt"] is False
    assert verdict(v1, _result(2, TEST, 0.80, 0.449), 0.75)["adopt"] is False
    assert verdict(v1, None, 0.75)["adopt"] is None
    assert verdict(v1, _result(2, TEST, 0.8, 0.5, complete=False), 0.75)["adopt"] is None
    stale = {**v2, "prompt": {**v2["prompt"], "sha256": "otro"}}
    assert verdict(v1, stale, 0.75)["reason"].startswith("un prompt cambió")


def test_the_table_compares_the_versions_and_says_what_to_adopt(env):
    runs = {(1, "", DEV): _result(1, DEV, 0.74, 0.40), (1, "", TEST): _result(1, TEST, 0.75, 0.42),
            (2, "", DEV): _result(2, DEV, 0.78, 0.46), (2, "", TEST): _result(2, TEST, 0.77, 0.46)}
    text = lab.render_table(runs, 1, 0.75)
    assert "v1 (en uso)" in text and "\n  v2 " in text
    assert ("v2 frente a v1: SE ADOPTA (precisión 77,0 % (>= 75,0 %); F1 +4,0 puntos (>= +3)). "
            "Cambia decider.prompt_version a 2 en config/config.yaml.") in text
    runs[(2, "", TEST)] = _result(2, TEST, 0.70, 0.46)
    assert "v2 frente a v1: no se adopta" in lab.render_table(runs, 1, 0.75)
    del runs[(2, "", TEST)]
    assert "v2 frente a v1" not in lab.render_table(runs, 1, 0.75)
    runs[(2, "", DEV)]["prompt"]["sha256"] = "editado"
    assert "(el prompt cambió: vuelve a correrla)" in lab.render_table(runs, 1, 0.75)
    assert "Definiciones de la ley" not in lab.render_table(runs, 1, 0.75)


def test_load_runs_reads_what_save_wrote(env):
    save_run(_run(env), env.folder)
    save_run(_run(env, version=2), env.folder)
    assert sorted(lab.load_runs(env.folder)) == [(1, "", DEV), (2, "", DEV)]
    old = env.folder / "decide-v1-desarrollo.json"               # saved before the variants
    data = json.loads(old.read_text(encoding="utf-8"))
    del data["variant"]
    old.write_text(json.dumps(data), encoding="utf-8")
    assert (1, "", DEV) in lab.load_runs(env.folder)


# ------------------------------------------------------------------ production

def test_production_uses_the_configured_version():
    assert decide_prompt({}) is dc.PROMPT
    assert decide_prompt({"decider": {"prompt_version": 2}}).version == 2
    with pytest.raises(PromptNotFound):
        decide_prompt({"decider": {"prompt_version": 99}})
    assert decide_prompt().version == 1                       # config.yaml: v1 until adopted


def test_adopting_a_version_decides_again_with_it(env, targets):  # noqa: F811
    from src.alignment.propose import is_current
    from src.alignment.user_alignment import align_upload
    from tests.test_alignment_propose import TTL
    kw = dict(base=env.base, targets=targets, embed_fn=Embed(), chroma_path=env.chroma,
              sleep_fn=lambda s: None)
    ws, m, _ = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), llm=DeciderLLM(), cfg=env.cfg,
                            **kw)
    assert m["settings"]["prompts"]["decide"]["version"] == 1
    adopted = {**env.cfg, "decider": {**env.cfg["decider"], "prompt_version": 2}}
    from src.alignment.propose import load_result
    assert is_current(load_result(ws), env.cfg) and not is_current(load_result(ws), adopted)
    llm = LabLLM()
    _, m2, _ = align_upload("Mi GDPR.ttl", TTL.encode("utf-8"), llm=llm, cfg=adopted, **kw)
    assert m2["recomputed"]["decider"] and m2["settings"]["prompts"]["decide"]["version"] == 2
    assert load_prompt("decide", 2).text in llm.prompts


# ------------------------------------------------------------------ CLI

def test_cli_checks_a_version_without_calling_the_ai(capsys):
    assert lab._run(["--revisar", "2"]) == 0
    assert "decide.v2: mantiene la salida" in capsys.readouterr().out
    assert lab._run(["--revisar", "99"]) == 1
    assert "Guárdalo como src/ai/prompts/decide.v99.txt" in capsys.readouterr().out


def test_cli_prints_the_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lab, "lab_dir", lambda cfg=None: tmp_path)
    save_run(_result(1, DEV, 0.75, 0.42) | {"examples": {k: [] for k in lab.ERROR_LABELS},
                                             "errors": {k: 0 for k in lab.ERROR_LABELS},
                                             "llm": {"right_term": 1, "aligned": 2,
                                                     "none_right": 1, "no_match": 2},
                                             "concepts": 4, "aligned": 2, "rule": "regla",
                                             "model": "m"}, tmp_path)
    assert lab._run(["--tabla"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Banco de prompts (decide): tus 552 decisiones en dos mitades")
    assert "v1 (en uso)" in out


# ------------------------------------------------------------------ the definitions of the law

LAW_DEFINITIONS = {"Holder": "Persona natural titular de los datos personales.",
                   "PersonalDetail": "Dato que identifica a una persona en un formulario."}


def _without_definitions(env):
    for name in LAW_DEFINITIONS:
        env.inputs.concepts[EX + name].definition = None


def _definitions(entries=None):
    entries = [{"key": EX + n, "name": n, "kinds": ["class"], "for_decider": False,
                "status": "articles", "definition": d, "article": 4, "articles": [4, 1],
                "law_concept": None} for n, d in LAW_DEFINITIONS.items()] \
        if entries is None else entries
    return {"law": {"file": "lopdp.pdf", "language": "es", "segmenter_version": 1},
            "skipped": None, "summary": {"pending": 0}, "entries": entries}


def test_with_the_definitions_the_decider_sees_them(env):
    _without_definitions(env)
    half = _where("Holder")
    llm = LabLLM()
    plain = _run(env, half=half, llm=llm)
    calls = llm.calls
    run = run_half(env.inputs, 1, half, llm=llm, embed_fn=Embed(), budget=5,
                   cache_path=env.cache, chroma_path=env.chroma, sleep_fn=lambda s: None,
                   cfg=env.cfg, base=env.base, definitions=_definitions())
    assert run["variant"] == lab.DEFINITIONS and plain["variant"] == ""
    assert llm.calls > calls                                    # new bodies: asked again
    assert f"Definicion: {LAW_DEFINITIONS['Holder']}" in llm.contexts[-1]
    assert run["definitions"]["law"] == "lopdp.pdf" and run["definitions"]["added"] == 2
    assert run["definitions"]["in_half"] == sum(_where(n) == half for n in LAW_DEFINITIONS)
    holder = run["examples"][NONE_WRONG][0]
    assert holder["definition"] == LAW_DEFINITIONS["Holder"]
    path = save_run(run, env.folder)
    assert path.name == f"decide-v1-definiciones-{half}.json"
    assert not list(env.folder.glob("*gemini*"))                # never iterated on
    assert (1, lab.DEFINITIONS, half) in lab.load_runs(env.folder)
    text = lab.render_run(run, env.folder, {(1, "", half): plain}, 1)
    assert text.startswith(f"Banco de prompts: decide.v1 con las definiciones de la ley en la "
                           f"mitad de {half}")
    assert "Frente a v1 sin definiciones en esta mitad: precisión" in text


def test_definitions_that_add_nothing_are_refused(env):
    with pytest.raises(ValueError, match="no dio ninguna definición"):
        run_half(env.inputs, 1, DEV, llm=LabLLM(), embed_fn=Embed(), budget=5,
                 cache_path=env.cache, chroma_path=env.chroma, sleep_fn=lambda s: None,
                 cfg=env.cfg, base=env.base, definitions=_definitions([]))


def test_the_definitions_are_judged_on_the_552_decisions():
    plain = [_result(1, DEV, 0.84, 0.44, proposals=31, correct=26, aligned=87),
             _result(1, TEST, 0.68, 0.41, proposals=37, correct=25, aligned=86)]
    assert lab.combined(plain) == {"proposals": 68, "correct": 51, "aligned": 173,
                                   "precision": 0.75, "coverage": 0.2948, "f1": 0.4232}
    better = [_result(1, DEV, 0, 0, proposals=34, correct=28, variant=lab.DEFINITIONS),
              _result(1, TEST, 0, 0, proposals=36, correct=27, aligned=86,
                      variant=lab.DEFINITIONS)]
    result = lab.definitions_verdict(plain, better, 0.75)
    assert result["adopt"] is True and result["after"]["correct"] == 55
    assert result["reason"] == "precisión 78,6 % (>= 75,0 %); cobertura 29,5 % -> 31,8 %"
    worse = [_result(1, DEV, 0, 0, proposals=40, correct=28),
             _result(1, TEST, 0, 0, proposals=40, correct=27, aligned=86)]
    assert lab.definitions_verdict(plain, worse, 0.75)["adopt"] is False   # 68,8 %
    same = [_result(1, DEV, 0, 0, proposals=31, correct=26),
            _result(1, TEST, 0, 0, proposals=37, correct=25, aligned=86)]
    assert lab.definitions_verdict(plain, same, 0.75)["adopt"] is False    # no more coverage
    assert lab.definitions_verdict(plain, [better[0], None], 0.75)["adopt"] is None
    late = [better[0], {**better[1], "complete": False}]
    assert lab.definitions_verdict(plain, late, 0.75)["adopt"] is None


def test_the_table_says_whether_the_definitions_help():
    runs = {(1, "", DEV): _result(1, DEV, 0.84, 0.44, proposals=31, correct=26),
            (1, "", TEST): _result(1, TEST, 0.68, 0.41, proposals=37, correct=25, aligned=86),
            (1, lab.DEFINITIONS, DEV): _result(1, DEV, 0.82, 0.46, proposals=34, correct=28,
                                               variant=lab.DEFINITIONS)}
    text = lab.render_table(runs, 1, 0.75)
    assert "\n  v1 + definiciones " in text
    assert ("Definiciones de la ley con v1 (las 552 decisiones): faltan mitades: corre las dos, "
            "con y sin definiciones") in text
    runs[(1, lab.DEFINITIONS, TEST)] = _result(1, TEST, 0.75, 0.43, proposals=36, correct=27,
                                               aligned=86, variant=lab.DEFINITIONS)
    assert "(las 552 decisiones): AYUDAN (precisión 78,6 %" in lab.render_table(runs, 1, 0.75)
    runs[(1, lab.DEFINITIONS, TEST)] = _result(1, TEST, 0.6, 0.4, proposals=46, correct=27,
                                               aligned=86, variant=lab.DEFINITIONS)
    assert ("no ayudan (precisión 68,8 % (< 75,0 %); cobertura 29,5 % -> 31,8 %): quedan en el "
            "módulo de definiciones, pero el decisor no debería recibirlas.") in \
        lab.render_table(runs, 1, 0.75)
    assert "v1 + definiciones frente" not in lab.render_table(runs, 1, 0.75)


def test_prepare_the_definitions_with_the_steps_of_mode_3(env, targets):  # noqa: F811
    from src.workspace.store import KIND_LAW, KIND_ONTOLOGY, open_upload
    from tests.test_law_definitions import LAW, TTL, LawEmbed, LawLLM
    cfg = {**env.cfg, "law_definitions": {"pause_seconds": 0},       # off: the lab turns it on
           "extraction": {"pause_seconds": 0}}
    ws = open_upload({KIND_ONTOLOGY: ("onto.ttl", TTL.encode("utf-8")),
                      KIND_LAW: ("ley.txt", LAW.encode("utf-8"))}, cfg=cfg)
    seen = []
    _, waiting = lab.prepare_definitions(cfg, budget=0, ws=ws, llm=LawLLM(),
                                         embed_fn=LawEmbed(), sleep_fn=lambda s: None,
                                         progress=seen.append)
    assert waiting["summary"]["pending"] == 2 and waiting["settings"]["max_calls"] == 0
    with pytest.raises(ValueError, match="Faltan 2 definiciones: vuelve a correr"):
        lab.load_definitions(ws)
    assert "Faltan definiciones: vuelve a correr" in lab.render_definitions(waiting, ws)
    llm = LawLLM()
    _, ready = lab.prepare_definitions(cfg, budget=5, ws=ws, llm=llm, embed_fn=LawEmbed(),
                                       sleep_fn=lambda s: None)
    assert ready["summary"]["added"] == 3 and lab.load_definitions(ws) == ready
    assert ready["settings"]["enabled"] and ready["settings"]["max_calls"] == 5
    assert llm.asked == ["Pizza", "Erasure"]                    # extraction: from the cache
    text = lab.render_definitions(ready, ws)
    assert "a 4 de 5 conceptos les faltaba; 2 de conceptos de la ley" in text
    assert "Ahora: py -m src.alignment.prompt_lab --con-definiciones" in text
    calls = llm.calls
    lab.prepare_definitions(cfg, budget=5, ws=ws, llm=llm, embed_fn=LawEmbed())
    assert llm.calls == calls                                   # current: nothing again


def test_definitions_must_be_prepared_first(env, targets):  # noqa: F811
    from src.workspace.store import KIND_LAW, KIND_ONTOLOGY, open_upload
    from tests.test_law_definitions import LAW, TTL
    ws = open_upload({KIND_ONTOLOGY: ("onto.ttl", TTL.encode("utf-8")),
                      KIND_LAW: ("ley.txt", LAW.encode("utf-8"))}, cfg=env.cfg)
    with pytest.raises(ValueError, match="Primero prepara las definiciones"):
        lab.load_definitions(ws)


def test_the_definitions_workspace_is_ontopriv_and_the_lopdp(tmp_path):
    (tmp_path / "onto.ttl").write_text(
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n<http://e.org/o> a owl:Ontology .\n",
        encoding="utf-8")
    (tmp_path / "ley.txt").write_text("Artículo 1°. Objeto. Algo.\n", encoding="utf-8")
    cfg = {"workspaces": {"dir": str(tmp_path / "ws")},
           "inputs": {"ontology": str(tmp_path / "onto.ttl"),
                      "legal_text": str(tmp_path / "ley.txt")}}
    ws = lab.definitions_workspace(cfg)
    assert ws.manifest["mode"] == "ontology+law" and lab.definitions_workspace(cfg).id == ws.id
    with pytest.raises(FileNotFoundError, match="Falta"):
        lab.definitions_workspace({**cfg, "inputs": {"ontology": str(tmp_path / "x.ttl")}})


def test_cli_asks_to_prepare_the_definitions_first(env, monkeypatch, capsys):
    from src.workspace.store import KIND_LAW, KIND_ONTOLOGY, open_upload
    from tests.test_law_definitions import LAW, TTL
    ws = open_upload({KIND_ONTOLOGY: ("onto.ttl", TTL.encode("utf-8")),
                      KIND_LAW: ("ley.txt", LAW.encode("utf-8"))}, cfg=env.cfg)
    monkeypatch.setattr(lab, "definitions_workspace", lambda cfg=None: ws)
    assert lab._run(["--con-definiciones"]) == 1
    assert "Error: Primero prepara las definiciones" in capsys.readouterr().out