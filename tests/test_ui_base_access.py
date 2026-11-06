import re
from pathlib import Path

from app.services import base_access as access

ROOT = Path(__file__).resolve().parents[1]
REMOVED = ["app/views/inputs.py", "app/views/articles.py", "app/services/ingest.py",
           "app/services/segmentation.py"]


def test_the_switch_only_opens_with_1(monkeypatch):
    monkeypatch.delenv(access.FLAG, raising=False)
    assert not access.editable()
    for value, expected in (("1", True), (" 1 ", True), ("0", False), ("true", False),
                            ("", False)):
        monkeypatch.setenv(access.FLAG, value)
        assert access.editable() is expected


def test_the_menu_has_the_assistant_and_the_construction_tools():
    text = (ROOT / "app/app.py").read_text(encoding="utf-8")
    main, tools = text.split('"Herramientas de construcción": [')
    assert 'title="Asistente de alineación"' in main and "default=True" in main
    assert re.findall(r'title="([^"]+)"', tools) == ["Núcleo", "Propuestas IA",
                                                     "Aprobar conceptos IA", "Alineación DPV"]
    assert "Insumos" not in text and "Artículos" not in text
    assert not any((ROOT / f).exists() for f in REMOVED)


def test_the_example_env_documents_the_switch():
    assert "EDITABLE_BASE=0" in (ROOT / ".env.example").read_text(encoding="utf-8")


def test_the_example_env_names_the_whole_embedding_model():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert ("EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            in lines)


def test_the_ai_proposals_page_points_to_the_review(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.syspath_prepend(str(ROOT / "app"))
    at = AppTest.from_file(str(ROOT / "app/views/ai_proposals.py"), default_timeout=60).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "**Aprobar conceptos IA**" in text and "siguiente sprint" not in text
    assert "por validar" not in text
