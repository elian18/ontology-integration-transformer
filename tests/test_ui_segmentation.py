"""Tests for the UI segmentation service (Sprint 2, S2-T08)."""

import pytest

from app.services import segmentation


def test_segment_uploaded_law_returns_view_shape():
    data = (
        "Art. 1.- Objeto. Esta ley regula la proteccion de datos.\n"
        "Art. 2.- Ambito de aplicacion. Se aplica a toda persona.\n"
        "Art. 3.- Definiciones. Para efectos de esta ley.\n"
    ).encode("utf-8")

    view = segmentation.segment_uploaded_law("mi_ley.txt", data)

    assert view["source"] == "mi_ley.txt"
    assert view["n_articles"] == 3
    assert [a["number"] for a in view["articles"]] == [1, 2, 3]
    first = view["articles"][0]
    assert first["title"].startswith("Objeto")
    assert first["preview"] != ""
    assert first["text"].startswith("Art. 1.-")


def test_segment_base_law_if_present():
    view = segmentation.segment_base_law()
    if view is None:
        pytest.skip("LOPDP base no disponible en data/input/")
    assert view["n_articles"] >= 1
    assert all("number" in a and "title" in a for a in view["articles"])