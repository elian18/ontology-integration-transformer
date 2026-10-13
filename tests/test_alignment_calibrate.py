"""S6-T06: the threshold of the automatic proposals is calibrated on the validated base."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.alignment.calibrate import (THRESHOLDS, auto_settings, blind_check, calibrate,
                                     choose_threshold, render_console, sweep, top1_points)
from src.alignment.inherit import load_base
from tests.test_alignment_write import DPV, ONTO, _decide_all

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base_data(tmp_path):
    """Consent rank 1 = Consent (approved, 0.80); Turnover rank 1 = Fee (no match, 0.78, the AI
    said none); ImpactAssessment rank 1 = DPIA (approved, 0.60)."""
    data, log = _decide_all(tmp_path)
    scores = {(ONTO + "Consent", 1): 0.80, (ONTO + "Turnover", 1): 0.78,
              ("ai:ImpactAssessment", 1): 0.60}
    for r in data["rows"]:
        r["score"] = scores.get((r["concept_key"], r["rank"]), 0.3)
    cand = tmp_path / "c.json"
    cand.write_text(json.dumps(data), encoding="utf-8")
    return data, load_base(cand, log)


def test_points_mark_what_the_person_approved(base_data):
    data, base = base_data
    points = {p["name"]: p for p in top1_points(data, base)}
    assert points["Consent"]["correct"] and points["ImpactAssessment"]["correct"]
    assert not points["Turnover"]["correct"] and points["Turnover"]["status"] == "no_match"
    assert points["Turnover"]["ai_relation"] == "none"


def test_sweep_precision_and_coverage(base_data):
    data, base = base_data
    rows = {r["threshold"]: r for r in sweep(top1_points(data, base), (0.55, 0.75, 0.80))}
    assert (rows[0.55]["proposals"], rows[0.55]["correct"]) == (3, 2)
    assert rows[0.75]["precision"] == 0.5 and rows[0.75]["on_no_match"] == 1
    assert rows[0.80]["precision"] == 1.0 and rows[0.80]["coverage"] == 0.5


def test_ai_filter_drops_what_the_ai_rejected(base_data):
    data, base = base_data
    points = top1_points(data, base)
    plain = choose_threshold(sweep(points), 0.75, min_proposals=1)
    filtered = choose_threshold(sweep(points, ai_filter=True), 0.75, min_proposals=1)
    assert (plain, filtered) == (0.80, 0.50)          # Turnover (AI: none) drops out
    assert choose_threshold(sweep(points), 0.99, min_proposals=3) is None
    assert choose_threshold(sweep(points), 0.5, min_proposals=10) is None


def test_blind_sample_check():
    data = {"rows": [{"concept_key": "a", "rank": 1, "score": 0.9, "dpv_name": "Consent"},
                     {"concept_key": "b", "rank": 1, "score": 0.8, "dpv_name": "Fee"},
                     {"concept_key": "c", "rank": 1, "score": 0.2, "dpv_name": "Law"}]}
    ref = [{"concept_key": "a", "gold_dpv_name": "consent"},
           {"concept_key": "b", "gold_dpv_name": "none"},
           {"concept_key": "c", "gold_dpv_name": "Law"},
           {"concept_key": "d", "gold_dpv_name": ""}]
    out = blind_check(data, ref, 0.75)
    assert (out["labelled"], out["proposals"], out["correct"], out["gold_none_proposed"]) == \
        (3, 2, 1, 1)
    assert out["precision"] == 0.5


def test_calibrate_report_and_console(base_data):
    data, base = base_data
    result = calibrate(data, base, reference=None, min_precision=0.75)
    assert result["concepts"] == 3 and result["reachable"] == 2
    assert [r["threshold"] for r in result["sweep"]] == list(THRESHOLDS)
    assert result["chosen_threshold"] is None             # 3 concepts < 10 proposals
    text = render_console(result)
    assert "Calibración del umbral" in text and "Ningún umbral llega" in text


def test_auto_settings_defaults_and_validation():
    assert auto_settings({})["threshold"] == 0.75
    assert auto_settings({"auto": {"use_llm": False}})["use_llm"] is False
    with pytest.raises(ValueError, match="fuera de rango"):
        auto_settings({"auto": {"threshold": 1.5}})


def test_config_threshold_is_the_calibrated_one():
    """The threshold in config.yaml is the one the real base gives (regression of S6-T06)."""
    from src.config import load_config
    from src.alignment.export import load_candidate_file
    from src.alignment.inherit import base_paths
    paths = base_paths()
    if not paths["candidates"].exists():
        pytest.skip("Falta data/output/alignment-candidates.json")
    cfg = load_config()
    result = calibrate(load_candidate_file(paths["candidates"]), load_base(cfg=cfg),
                       min_precision=auto_settings(cfg)["min_precision"])
    assert result["chosen_threshold"] == auto_settings(cfg)["threshold"]