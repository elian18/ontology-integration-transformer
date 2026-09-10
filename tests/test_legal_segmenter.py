import json
from pathlib import Path

import pytest

from src.ingest.legal_segmenter import segment_articles
from src.ingest.text_loader import load_legal_text

# The LOPDP is a base input of the project. This number is anchored to the file
# currently in data/input/; if the file is replaced, update it on purpose.
LOPDP_PATH = Path(__file__).resolve().parents[1] / "data" / "input" / "lopdp.pdf"
LOPDP_EXPECTED_ARTICLES = 77


# --------------------------------------------------------------------------- #
# Unit tests: synthetic text, fast and deterministic                          #
# --------------------------------------------------------------------------- #

def test_counts_three_numbered_articles():
    text = (
        "Art. 1.- Objeto. Esta ley regula la proteccion de datos.\n"
        "Art. 2.- Ambito. Se aplica a toda persona.\n"
        "Art. 3.- Principios. Los datos se tratan con licitud.\n"
    )
    summary = segment_articles(text)
    assert summary["n_articles"] == 3
    assert [a["number"] for a in summary["articles"]] == [1, 2, 3]


def test_cross_reference_inside_body_is_not_a_header():
    # "Art. 1.-" appears mid-line, quoted inside the body of article 2.
    text = (
        "Art. 1.- Objeto de la ley.\n"
        "Art. 2.- El tratamiento se rige por el Art. 1.- de esta ley.\n"
    )
    summary = segment_articles(text)
    assert summary["n_articles"] == 2
    assert [a["number"] for a in summary["articles"]] == [1, 2]


def test_reference_without_dash_is_ignored():
    # An index line or a reference at line start but WITHOUT ".-" must not count.
    text = (
        "Art. 1 Objeto y finalidad ................ 3\n"   # index-like, no dash
        "Art. 1.- Objeto y finalidad de la ley.\n"          # the real header
    )
    summary = segment_articles(text)
    assert summary["n_articles"] == 1
    assert summary["articles"][0]["number"] == 1


def test_article_spans_are_contiguous():
    text = (
        "Art. 1.- Uno.\n"
        "Art. 2.- Dos.\n"
        "Art. 3.- Tres.\n"
    )
    arts = segment_articles(text)["articles"]
    for current, nxt in zip(arts, arts[1:]):
        assert current["char_end"] == nxt["char_start"]
    assert arts[-1]["char_end"] == len(text)


def test_title_is_the_caption_after_the_marker():
    text = "Art. 5.- Consentimiento. El titular debe consentir.\n"
    art = segment_articles(text)["articles"][0]
    assert art["title"].startswith("Consentimiento")


def test_text_with_no_articles_returns_empty():
    summary = segment_articles("Un texto cualquiera sin encabezados de articulo.")
    assert summary["n_articles"] == 0
    assert summary["articles"] == []


def test_summary_is_json_serializable():
    summary = segment_articles("Art. 1.- Objeto.\n", source="prueba")
    dumped = json.loads(json.dumps(summary))
    assert dumped["source"] == "prueba"
    assert dumped["articles"][0]["number"] == 1


# --------------------------------------------------------------------------- #
# Integration tests: the real LOPDP                                           #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def lopdp_summary():
    if not LOPDP_PATH.exists():
        pytest.skip(f"LOPDP no encontrada en {LOPDP_PATH}")
    report = load_legal_text(LOPDP_PATH)
    summary = segment_articles(report.text, source=report.path)
    return report, summary


def test_lopdp_has_expected_article_count(lopdp_summary):
    _, summary = lopdp_summary
    assert summary["n_articles"] == LOPDP_EXPECTED_ARTICLES


def test_lopdp_segmenter_matches_tentative_count(lopdp_summary):
    # Real segmentation must reproduce the Sprint 1 tentative count exactly.
    report, summary = lopdp_summary
    assert summary["n_articles"] == report.candidate_articles


def test_lopdp_numbers_are_contiguous(lopdp_summary):
    _, summary = lopdp_summary
    numbers = [a["number"] for a in summary["articles"]]
    assert numbers == list(range(1, LOPDP_EXPECTED_ARTICLES + 1))


def test_lopdp_every_article_has_number_and_text(lopdp_summary):
    _, summary = lopdp_summary
    for article in summary["articles"]:
        assert article["number"] >= 1
        assert article["text"].strip() != ""