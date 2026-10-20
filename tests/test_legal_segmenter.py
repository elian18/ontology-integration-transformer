"""S2-T01 + S7-T07: the article segmenter.

Sprint 2 knew one heading (``Art. 1.-``, the LOPDP); since S7-T07 it also reads the Spanish
headings of other laws (``Artículo 1°.``, ``ARTÍCULO 1o.``, ``Artículo 10.``) and the English
ones (``Article 1`` + title on the next line), and the numbers must follow each other. The base
LOPDP must come out EXACTLY as in Sprint 2 (its index and caches depend on it)."""
import json
import re
from pathlib import Path

import pytest

from src.ingest import legal_segmenter as seg
from src.ingest.legal_segmenter import count_articles, detect_language, segment_articles
from src.ingest.text_loader import load_legal_text

ROOT = Path(__file__).resolve().parents[1]
# The LOPDP is a base input of the project. This number is anchored to the file
# currently in data/input/; if the file is replaced, update it on purpose.
LOPDP_PATH = ROOT / "data" / "input" / "lopdp.pdf"
LOPDP_EXPECTED_ARTICLES = 77
GDPR_PATH = ROOT / "data" / "input" / "laws" / "gdpr-en.pdf"
COLOMBIA_PATH = ROOT / "data" / "input" / "laws" / "ley-1581-2012-colombia.pdf"


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
    assert summary["style"] == "sprint2" and summary["missing"] == []


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
    assert summary["articles"] == [] and summary["style"] is None


def test_summary_is_json_serializable():
    summary = segment_articles("Art. 1.- Objeto.\n", source="prueba")
    dumped = json.loads(json.dumps(summary))
    assert dumped["source"] == "prueba"
    assert dumped["articles"][0]["number"] == 1


# --------------------------------------------------------------------------- other laws

SPANISH = (
    "LEY ESTATUTARIA 1581 DE 2012\n"
    "Artículo 1°. Objeto. La presente ley tiene por objeto desarrollar el derecho.\n"
    "Artículo 2º. Ámbito de aplicación. Los principios se aplican a los datos.\n"
    "conforme a lo previsto en el\n"
    "artículo 2°. de la Constitución.\n"                 # cross reference, lowercase
    "ARTÍCULO 3o. DEFINICIONES. Para los efectos de la presente ley.\n"
    "Artículo 4. Principios para el Tratamiento de datos personales.\n"
    "a) Principio de legalidad.\n"
    "Artículo 5\n"
    "Datos sensibles\n"
    "Se entiende por datos sensibles aquellos que afectan la intimidad.\n"
)


def test_spanish_headings_of_other_laws():
    summary = segment_articles(SPANISH)
    arts = summary["articles"]
    assert [a["number"] for a in arts] == [1, 2, 3, 4, 5] and summary["style"] == "es"
    assert [a["title"] for a in arts] == ["Objeto", "Ámbito de aplicación", "DEFINICIONES",
                                          "Principios para el Tratamiento de datos personales",
                                          "Datos sensibles"]
    assert "artículo 2°. de la Constitución" in arts[1]["text"]     # stays inside article 2
    assert summary["language"] == "es" and summary["rejected"] == 0


ENGLISH = (
    "CHAPTER I\nGeneral provisions\n"
    "Article 1\nSubject-matter and objectives\n1. This Regulation lays down rules.\n"
    "Article 2\nMaterial scope\n1. This Regulation applies to the processing of data.\n"
    "Article 1. In the case of accreditation the requirements shall apply.\n"   # reference
    "Article 3 - Territorial scope. Where it applies\nThis Regulation applies to the data.\n"
    "Article 4\n(1) 'personal data' means any information relating to a person;\n"
)


def test_english_headings_and_titles_on_the_next_line():
    summary = segment_articles(ENGLISH)
    arts = summary["articles"]
    assert [a["number"] for a in arts] == [1, 2, 3, 4] and summary["style"] == "en"
    assert [a["title"] for a in arts] == ["Subject-matter and objectives", "Material scope",
                                          "Territorial scope", ""]   # no title: body follows
    assert summary["rejected"] == 1 and summary["language"] == "en"
    assert arts[0]["text"].startswith("Article 1\nSubject-matter")


def test_windows_line_endings_given_directly_are_read_too():
    for text in (SPANISH, ENGLISH):
        crlf = segment_articles(text.replace("\n", "\r\n"))
        lf = segment_articles(text)
        assert [(a["number"], a["title"]) for a in crlf["articles"]] == [
            (a["number"], a["title"]) for a in lf["articles"]]


def test_a_table_of_contents_is_not_taken_for_the_body():
    toc = "Artículo 1. Objeto ........ 3\nArtículo 2. Ámbito ........ 4\n"
    body = "Artículo 1. Objeto. Regula los datos.\nArtículo 2. Ámbito. Aplica a todos.\n"
    summary = segment_articles(toc + body)
    assert summary["n_articles"] == 2 and summary["articles"][0]["char_start"] == len(toc)
    short = "Artículo 1. Objeto\nArtículo 2. Ámbito\n"             # index without dot leaders
    summary = segment_articles(short + body)
    assert [a["char_start"] for a in summary["articles"]][0] == len(short)   # the later chain
    assert summary["rejected"] == 2


def test_a_longer_table_of_contents_with_dot_leaders_is_ignored():
    toc = ("Artículo 1. Objeto ........ 3\nArtículo 2. Ámbito ........ 4\n"
           "Artículo 3. Fin .......... 5\n")
    body = "Artículo 1. Objeto. Regula los datos.\nArtículo 2. Ámbito. Aplica a todos.\n"
    summary = segment_articles(toc + body)
    assert summary["n_articles"] == 2 and summary["rejected"] == 0


def test_headings_of_another_family_do_not_break_the_chain():
    text = ("Art. 1.- Objeto.\nArt. 2.- Ámbito. Se aplica lo previsto en el\n"
            "Artículo 3. de la Constitución.\nArt. 3.- Principios.\n")
    arts = segment_articles(text)["articles"]
    assert [a["number"] for a in arts] == [1, 2, 3] and arts[2]["text"] == "Art. 3.- Principios."
    assert "Artículo 3. de la Constitución." in arts[1]["text"]


def test_numbers_must_follow_each_other():
    text = ("Art. 1.- Uno.\nArt. 2.- Dos.\nArt. 2.- Repetido.\nArt. 4.- Cuatro (falta el 3).\n"
            "Art. 9.- Salto demasiado grande.\nArt. 5.- Cinco.\n")
    summary = segment_articles(text)
    assert [a["number"] for a in summary["articles"]] == [1, 2, 4, 5]
    assert summary["missing"] == [3] and summary["rejected"] == 2
    assert count_articles(text) == 4
    assert seg.MAX_GAP == 3
    assert [a["number"] for a in segment_articles("Art. 1.- a\nArt. 4.- b\n")["articles"]] == [
        1, 4]                                                            # gap of 3: still in
    assert [a["number"] for a in segment_articles("Art. 1.- a\nArt. 5.- b\n")["articles"]] == [1]


def test_a_law_that_does_not_start_at_one():
    summary = segment_articles("Art. 7.- Siete.\nArt. 8.- Ocho.\n")
    assert [a["number"] for a in summary["articles"]] == [7, 8]


def test_language_is_detected():
    assert detect_language("The controller shall be responsible for the processing of data.") \
        == "en"
    assert detect_language("El responsable del tratamiento de los datos personales.") == "es"


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


def _sprint2_segmenter(text: str) -> list[dict]:
    """The segmenter of Sprint 2, copied as it was (v0.6.0)."""
    header = re.compile(r"(?im)^\s*art[íi]?(?:culo|\.)\s*(\d+)\s*\.?-")
    matches = list(header.finditer(text))
    starts = [m.start() + (len(m.group(0)) - len(m.group(0).lstrip())) for m in matches]
    out = []
    for i, m in enumerate(matches):
        end = starts[i + 1] if i + 1 < len(matches) else len(text)
        block = text[starts[i]:end].strip()
        marker = m.group(0).strip()
        line = block.split("\n", 1)[0]
        out.append({"number": int(m.group(1)),
                    "title": line[len(marker):].lstrip(" .–-\t").strip(),
                    "text": block, "char_start": starts[i], "char_end": end})
    return out


def test_lopdp_is_segmented_exactly_as_in_sprint_2(lopdp_summary):
    report, summary = lopdp_summary
    assert summary["articles"] == _sprint2_segmenter(report.text)
    assert summary["style"] == "sprint2" and summary["rejected"] == 0
    assert summary["language"] == "es"


# --------------------------------------------------------------------------- #
# Integration tests: the test laws of Sprint 7 (data/input/laws)              #
# --------------------------------------------------------------------------- #

def _uploaded(path):
    if not path.exists():
        pytest.skip(f"Falta {path}")
    from src.ingest.law_text import clean_law_file
    return segment_articles(clean_law_file(path)[0], source=path.name)


def test_gdpr_in_english_has_its_99_articles():
    summary = _uploaded(GDPR_PATH)
    arts = {a["number"]: a for a in summary["articles"]}
    assert list(arts) == list(range(1, 100)) and summary["missing"] == []
    assert summary["language"] == "en" and summary["style"] == "en"
    assert arts[5]["title"] == "Principles relating to processing of personal data"
    assert arts[17]["title"].startswith("Right to erasure")
    assert arts[99]["title"] == "Entry into force and application"


def test_colombian_law_1581_has_its_30_articles():
    summary = _uploaded(COLOMBIA_PATH)
    arts = {a["number"]: a for a in summary["articles"]}
    assert list(arts) == list(range(1, 31)) and summary["rejected"] == 0
    assert summary["language"] == "es" and summary["style"] == "es"
    assert arts[3]["title"] == "Definiciones"                     # ligature "ﬁ" cleaned
    assert arts[9]["title"] == "Autorización del Titular"
    assert arts[30]["title"] == "Vigencia"