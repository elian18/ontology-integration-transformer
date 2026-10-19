"""S7-T03: baseline of the Sprint 6 design and where the decider's candidates come from.

Fast tests with a fake embedder and a temporary ChromaDB. Two tests read the REAL base files
(no LLM, fake embeddings): the baseline must repeat the calibration of S6-T06 (34 proposals,
28 correct) and the blind sample must agree with the Sprint 5 decisions in 10 of 23 concepts.
They are skipped if the base files are not on this machine.
"""
import hashlib

import numpy as np
import pytest

from src.alignment import benchmark as bm
from src.alignment import knowledge as kn
from src.alignment.dpv_targets import DpvTarget, DpvTargets
from src.alignment.inherit import BaseConcept, BaseIndex
from src.alignment.reference import REF_BASE, REF_BLIND, RefItem, Reference, base_reference, score
from src.alignment.sources import SourceConcept

D = "https://w3id.org/dpv#"


class FakeEmbed:
    def __init__(self, dims=32):
        self.dims = dims
        self.texts = 0

    def __call__(self, texts):
        self.texts += len(texts)
        out = []
        for t in texts:
            v = np.zeros(self.dims)
            for w in str(t).lower().replace(".", " ").split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


def _targets():
    return DpvTargets(dpv_path="t", targets=[
        DpvTarget(D + "Consent", "Consent", "class", "Consent", None, []),
        DpvTarget(D + "DataSubjectRight", "DataSubjectRight", "class", "Data Subject Right",
                  None, []),
        DpvTarget(D + "hasPurpose", "hasPurpose", "property", "has purpose", None, [])])


def _src(key, name, kinds=("class",), label=None):
    return SourceConcept(key=key, origin="ontology", name=name, kinds=tuple(kinds), label=label)


def _stored(tmp_path, targets, vectors):
    """A "dpv" collection with chosen vectors (as if built by S7-T01)."""
    records = kn.dpv_records(targets)
    table = dict(zip([t.semantic_text() for t in targets.targets], vectors))
    kn.ensure_collection("dpv", "fp", lambda: records, embed_fn=lambda docs: [table[d] for d in docs],
                         model_name="fake-model", path=tmp_path)


# --------------------------------------------------------------------------- settings

def test_decider_settings_defaults_and_project_config():
    assert bm.decider_settings({}) == {"candidate_source": "hybrid", "candidates": 10}
    assert bm.decider_settings() == {"candidate_source": "hybrid", "candidates": 10}


@pytest.mark.parametrize("block", [{"candidate_source": "otro"}, {"candidates": 2},
                                   {"candidates": 51}])
def test_invalid_decider_settings(block):
    with pytest.raises(ValueError, match="bloque decider"):
        bm.decider_settings({"decider": block})


# --------------------------------------------------------------------------- baseline

def test_baseline_proposes_rank1_above_threshold_unless_the_ai_said_none():
    data = {"rows": [
        {"concept_key": "a", "rank": 1, "score": 0.80, "dpv_iri": D + "X",
         "proposed_relation": "skos:closeMatch"},
        {"concept_key": "a", "rank": 2, "score": 0.79, "dpv_iri": D + "Y"},
        {"concept_key": "b", "rank": 1, "score": 0.74, "dpv_iri": D + "X"},
        {"concept_key": "c", "rank": 1, "score": 0.90, "dpv_iri": D + "X",
         "proposed_relation": "none"},
        {"concept_key": "d", "rank": 1, "score": 0.75, "dpv_iri": D + "Z",
         "proposed_relation": "untyped"}]}
    out = bm.baseline_outputs(data, 0.75)
    assert out == {"a": {"term": D + "X", "relation": "skos:closeMatch", "none": False},
                   "d": {"term": D + "Z", "relation": None, "none": False}}
    assert "c" in bm.baseline_outputs(data, 0.75, ai_filter=False)


# --------------------------------------------------------------------------- candidates

def test_the_three_sources_rank_differently_and_respect_the_kind(tmp_path):
    targets = _targets()
    tvec = np.array([[0.8, 0.6, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    _stored(tmp_path, targets, tvec.tolist())
    concept = _src("x#Consent", "Consent")
    cvec = np.array([[1.0, 0.0, 0.0]])          # by meaning: DataSubjectRight; by name: Consent
    dpv = bm.candidate_lists("dpv", [concept], targets, cvec, tvec, k=3, path=tmp_path)
    rerank = bm.candidate_lists("dpv_rerank", [concept], targets, cvec, tvec, k=3, path=tmp_path)
    hybrid = bm.candidate_lists("hybrid", [concept], targets, cvec, tvec, k=3, path=tmp_path)
    assert dpv["x#Consent"] == [D + "DataSubjectRight", D + "Consent"]      # no property
    assert rerank["x#Consent"] == [D + "Consent", D + "DataSubjectRight"]
    assert hybrid["x#Consent"] == rerank["x#Consent"]   # same formula, all terms in the pool


def test_a_property_only_gets_property_candidates(tmp_path):
    targets = _targets()
    tvec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _stored(tmp_path, targets, tvec.tolist())
    prop = _src("x#hasGoal", "hasGoal", kinds=("property",))
    lists = bm.candidate_lists("dpv", [prop], targets, np.array([[1.0, 0.0, 0.0]]), tvec, k=3,
                               path=tmp_path)
    assert lists == {"x#hasGoal": [D + "hasPurpose"]}


def test_unknown_source_is_rejected(tmp_path):
    targets = _targets()
    tvec = np.eye(3)
    _stored(tmp_path, targets, tvec.tolist())
    with pytest.raises(ValueError, match="desconocido"):
        bm.candidate_lists("otro", [_src("x#A", "A")], targets, np.eye(3)[:1], tvec,
                           path=tmp_path)


def test_hits_counts_the_best_rank_of_any_approved_term():
    ref = Reference("t", [RefItem("a", "A", "aligned", {D + "X": "skos:exactMatch"}),
                          RefItem("b", "B", "aligned", {D + "Y": "r", D + "Z": "r"}),
                          RefItem("c", "C", "aligned", {D + "W": "r"}),
                          RefItem("n", "N", "no_match")])
    lists = {"a": [D + "X"], "b": [D + "Q", D + "R", D + "Y", D + "Z"], "c": [D + "Q"]}
    assert bm.hits(lists, ref) == {"aligned": 3, "1": 1, "3": 2, "10": 2, "missing": 1}


def test_winner_follows_the_fixed_criterion():
    def row(base):
        return {REF_BASE: {"1": 0, "3": base[0], "10": base[1]}}
    tie = {"hybrid": row((5, 9)), "dpv": row((5, 9)), "dpv_rerank": row((5, 9))}
    assert bm.choose_source(tie) == "hybrid"
    more = {"hybrid": row((6, 9)), "dpv": row((5, 10)), "dpv_rerank": row((7, 9))}
    assert bm.choose_source(more) == "dpv"                     # more top-10 hits wins
    top3 = {"hybrid": row((6, 9)), "dpv": row((5, 9)), "dpv_rerank": row((7, 9))}
    assert bm.choose_source(top3) == "dpv_rerank"              # tie on top 10 -> top 3


def test_stored_vectors_must_match_the_model(tmp_path):
    targets = _targets()
    with pytest.raises(ValueError, match="knowledge"):
        bm.stored_dpv_vectors(targets, "fake-model", tmp_path)      # collection missing
    _stored(tmp_path, targets, np.eye(3).tolist())
    assert bm.stored_dpv_vectors(targets, "fake-model", tmp_path).shape == (3, 3)
    with pytest.raises(ValueError, match="knowledge"):
        bm.stored_dpv_vectors(targets, "another-model", tmp_path)


def test_memo_embed_computes_each_text_once():
    inner = FakeEmbed()
    memo = bm.MemoEmbed(inner)
    memo(["a b", "c"])
    memo(["c", "a b", "d"])
    assert inner.texts == 3


# --------------------------------------------------------------------------- whole run

def _inputs():
    targets = _targets()
    base = BaseIndex(concepts=[
        BaseConcept("o#Consent", "o#Consent", "ontology", "Consent", "consentimiento", None,
                    ("class",), "aligned",
                    [{"dpv_iri": D + "Consent", "dpv_name": "Consent",
                      "relation": "skos:exactMatch"}]),
        BaseConcept("o#Right_of_access", "o#Right_of_access", "ontology", "Right_of_access",
                    None, None, ("class",), "aligned",
                    [{"dpv_iri": D + "DataSubjectRight", "dpv_name": "DataSubjectRight",
                      "relation": "skos:broadMatch"}]),
        BaseConcept("o#Sanction", "o#Sanction", "ontology", "Sanction", None, None, ("class",),
                    "no_match")], source_id="ontopriv+lopdp")
    s4 = [_src("o#Consent", "Consent", label="consentimiento"),
          _src("o#Right_of_access", "Right_of_access"), _src("o#Sanction", "Sanction")]
    data = {"rows": [
        {"concept_key": "o#Consent", "rank": 1, "score": 0.9, "dpv_iri": D + "Consent",
         "proposed_relation": "skos:closeMatch"},
        {"concept_key": "o#Right_of_access", "rank": 1, "score": 0.6,
         "dpv_iri": D + "DataSubjectRight", "proposed_relation": "skos:broadMatch"},
        {"concept_key": "o#Sanction", "rank": 1, "score": 0.8, "dpv_iri": D + "hasPurpose",
         "proposed_relation": "skos:relatedMatch"}]}
    external = [_src("g#Consent", "Consent"), _src("g#Purpose", "Purpose")]
    sample = [{"concept_key": "o#Right_of_access", "concept_name": "Right_of_access",
               "gold_dpv_name": "DataSubjectRight", "gold_relation": "broadMatch"},
              {"concept_key": "o#Sanction", "concept_name": "Sanction",
               "gold_dpv_name": "none", "gold_relation": "none"}]
    return bm.BenchmarkInputs(data=data, base=base, targets=targets, s4_concepts=s4,
                              external_concepts=external, sample=sample, threshold=0.75)


def test_full_run_writes_both_files_and_makes_no_llm_call(tmp_path):
    inputs, embed = _inputs(), FakeEmbed()
    kn.ensure_collection("dpv", "fp", lambda: kn.dpv_records(inputs.targets), embed_fn=embed,
                         model_name="fake-model", path=tmp_path / "chroma")
    result = bm.run_benchmark(inputs, embed_fn=embed, model_name="fake-model",
                              path=tmp_path / "chroma", cfg={"decider": {"candidates": 3}})
    assert result["llm_calls"] == 0 and result["candidates"] == 3
    base = result["baseline"][REF_BASE]
    assert (base["proposals"], base["correct"], base["on_no_match"]) == (2, 1, 1)
    assert base["relation_ok"] == 0 and base["relation_typed"] == 1
    assert result["references"]["gdprtext"]["concepts"] == 1          # g#Consent inherits
    assert result["reference_consistency"] == {"shared": 2, "agree": 2, "rate": 1.0}
    assert set(result["candidate_sources"]) == {"hybrid", "dpv", "dpv_rerank"}
    assert result["candidate_sources"]["hybrid"][REF_BASE]["aligned"] == 2
    assert result["winner"] in bm.SOURCES
    paths = bm.write_results(result, tmp_path / "out")
    assert paths["benchmark"].exists() and paths["reference"].exists()
    text = bm.render_console(result)
    assert "0 llamadas al LLM" in text and "Origen de los 3 candidatos" in text
    assert "precisión 50,0 %" in text


# --------------------------------------------------------------------------- real base

@pytest.fixture(scope="module")
def real():
    """The real base files, loaded once (skipped where they are not present)."""
    try:
        return bm.load_inputs()
    except (FileNotFoundError, OSError):
        pytest.skip("faltan los archivos de la base en data/")


def test_real_baseline_repeats_the_sprint6_calibration(real):
    ref = base_reference(real.base)
    s = score(bm.baseline_outputs(real.data, real.threshold), ref)
    assert (s["concepts"], s["aligned"], s["no_match"]) == (552, 173, 379)
    assert (s["proposals"], s["correct"], s["on_no_match"]) == (34, 28, 3)
    assert s["coverage"] == pytest.approx(28 / 173, abs=1e-4)
    assert s["none_claimed"] == 0                 # the Sprint 6 design never claims "none"


def test_real_run_with_fake_embeddings(real, tmp_path):
    embed = FakeEmbed()
    kn.ensure_collection("dpv", "fp", lambda: kn.dpv_records(real.targets), embed_fn=embed,
                         model_name="fake-model", path=tmp_path)
    result = bm.run_benchmark(real, embed_fn=embed, model_name="fake-model", path=tmp_path)
    refs = result["references"]
    assert (refs[REF_BASE]["concepts"], refs[REF_BASE]["aligned"]) == (552, 173)
    assert (refs[REF_BLIND]["concepts"], refs[REF_BLIND]["aligned"]) == (30, 19)
    assert refs["gdprtext"]["concepts"] >= 18                    # 18 by name, any embeddings
    assert result["reference_consistency"]["shared"] == 23
    assert result["reference_consistency"]["agree"] == 10
    assert result["baseline"][REF_BLIND]["proposals"] == 1
    assert result["baseline"][REF_BLIND]["correct"] == 0
    assert result["candidate_sources"]["hybrid"][REF_BASE]["aligned"] == 173