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