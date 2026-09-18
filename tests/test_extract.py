"""Tests for the AI concept extraction that proposes profile concepts (S3-T05).

The fast tests inject a fake LLM that returns JSONL keyed by which articles are in the batch,
so they assert the batching / JSONL parsing / provenance / quota / resume logic exactly and
never hit the network. One slow test uses the real LLMClient and is excluded by
``-m "not slow"``."""
from __future__ import annotations
import json

import pytest

from src.core.extract import (
    propose_profile_concepts, extraction_summary,
)

_ARTS = [
    {"number": 1, "title": "Objeto", "text": "articulo uno sobre el objeto y finalidad"},
    {"number": 2, "title": "Ambito", "text": "articulo dos sobre el ambito de aplicacion"},
    {"number": 3, "title": "Consentimiento", "text": "articulo tres sobre el consentimiento"},
]

# One concept per article, tagged with its own number.
_BY_ART = {
    1: {"name": "Objective", "label": "objeto", "type": "class", "definition": "d1", "article": 1},
    2: {"name": "Scope", "label": "ambito", "type": "class", "definition": "d2", "article": 2},
    3: {"name": "Consent", "label": "consentimiento", "type": "class", "definition": "d3", "article": 3},
}


class FakeLLM:
    """Returns JSONL for whichever article markers appear in the batch context."""
    def __init__(self, fail_articles=None, quota_articles=None, raw_override=None):
        self.fail_articles = fail_articles or set()      # raise 503 once
        self.quota_articles = quota_articles or set()    # raise 429 (quota)
        self.raw_override = raw_override or {}            # article -> raw string to return
        self.calls = 0

    def ask(self, prompt: str, context: str = "", max_tokens: int = 1024) -> str:
        self.calls += 1
        present = [n for n in _BY_ART if f"ARTICULO {n} " in context]
        for n in present:
            if n in self.quota_articles:
                raise RuntimeError("Error code: 429 - RESOURCE_EXHAUSTED quota")
            if n in self.fail_articles:
                raise RuntimeError("503 Service Unavailable")
        lines = []
        for n in present:
            if n in self.raw_override:
                lines.append(self.raw_override[n])
            else:
                lines.append(json.dumps(_BY_ART[n], ensure_ascii=False))
        return "\n".join(lines)


def test_batches_articles_into_fewer_calls():
    llm = FakeLLM()
    report = propose_profile_concepts(_ARTS, llm=llm, batch_size=2)
    assert llm.calls == 2                                # 3 articles / batch 2 -> 2 calls
    assert report.counts()["articles_processed"] == 3


def test_provenance_comes_from_the_article_tag():
    llm = FakeLLM()
    report = propose_profile_concepts(_ARTS, llm=llm, batch_size=8)
    by_name = {c.name: c for c in report.concepts}
    assert by_name["Objective"].articles == [1]
    assert by_name["Consent"].articles == [3]


def test_bad_line_is_skipped_others_survive():
    # Article 1 returns a broken line; article 2 is fine (same batch).
    llm = FakeLLM(raw_override={1: '{"name": "Broken", "label": "x"'})   # unterminated
    report = propose_profile_concepts(_ARTS[:2], llm=llm, batch_size=8)
    names = {c.name for c in report.concepts}
    assert "Scope" in names and "Broken" not in names


def test_control_chars_in_values_are_tolerated():
    llm = FakeLLM(raw_override={
        1: json.dumps({"name": "Objective", "label": "obj", "type": "class",
                       "definition": "linea1", "article": 1})})
    report = propose_profile_concepts(_ARTS[:1], llm=llm)
    assert [c.name for c in report.concepts] == ["Objective"]


def test_quota_error_stops_and_keeps_partial():
    # First batch (art 1,2) ok; second batch (art 3) hits quota.
    llm = FakeLLM(quota_articles={3})
    report = propose_profile_concepts(_ARTS, llm=llm, batch_size=2)
    assert report.quota_exhausted is True
    assert report.counts()["articles_processed"] == 2   # only the first batch survived
    assert {c.name for c in report.concepts} == {"Objective", "Scope"}


def test_transient_error_is_retried_then_batch_skipped():
    llm = FakeLLM(fail_articles={1})                     # batch with art 1 always 503s
    report = propose_profile_concepts(_ARTS, llm=llm, batch_size=1, retries=1)
    # art1 batch: 2 attempts fail; art2, art3 batches ok.
    assert llm.calls == 4                                # 2 (art1) + 1 (art2) + 1 (art3)
    assert 1 not in report.processed_articles
    assert {c.name for c in report.concepts} == {"Scope", "Consent"}


def test_resume_skips_already_processed_articles(tmp_path):
    out = tmp_path / "props.json"
    llm1 = FakeLLM()
    propose_profile_concepts(_ARTS[:2], llm=llm1, batch_size=8, out_path=out)   # process art 1,2
    assert out.exists()
    llm2 = FakeLLM()
    report = propose_profile_concepts(_ARTS, llm=llm2, batch_size=8, out_path=out, resume=True)
    assert llm2.calls == 1                               # only art 3 remained
    assert report.counts()["articles_processed"] == 3
    assert {c.name for c in report.concepts} == {"Objective", "Scope", "Consent"}


def test_summary_is_json_safe():
    llm = FakeLLM()
    report = propose_profile_concepts(_ARTS, llm=llm)
    summary = extraction_summary(report)
    json.dumps(summary)
    assert summary["counts"]["concepts_proposed"] == 3
    assert set(summary["processed_articles"]) == {1, 2, 3}


@pytest.mark.slow
def test_real_llm_extracts_a_batch():
    try:
        from src.ai.llm_client import LLMClient
        llm = LLMClient()
    except Exception as exc:                             # pragma: no cover - sin credenciales
        pytest.skip(f"LLM no disponible: {exc}")
    report = propose_profile_concepts(_ARTS, llm=llm, source="LOPDP", batch_size=8)
    assert report.counts()["articles_processed"] >= 1
    for c in report.concepts:
        assert all(a in (1, 2, 3) for a in c.articles)