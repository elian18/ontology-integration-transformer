"""S7-T02: the prompts live in versioned text files (src/ai/prompts/).

The prompts of justify (v2) and extract were moved out of the code WITHOUT changing a letter:
the sha256 below were computed from the Python constants of v0.6.0, before the move, so these
tests prove the AI receives exactly the same text as in Sprints 3-6. Every run now records
which prompt produced it (name, version and sha256): in the candidates file (justify), in the
extraction summary and in the manifest of an upload.
"""
import json

import pytest

from src.ai.prompts import PromptNotFound, available, load_prompt, prompts_manifest
from src.alignment import justify
from src.alignment.export import CANDIDATES_JSON
from src.alignment.user_alignment import write_user_alignment
from src.core import extract
from tests.test_alignment_justify import FakeLLM as JustifyLLM, _data, _run
from tests.test_alignment_propose import targets  # noqa: F401  (fixture)
from tests.test_alignment_propose import env  # noqa: F401  (fixture)
from tests.test_user_alignment import _ws  # noqa: F401  (helper)

# sha256 of the constants _INSTRUCTION of v0.6.0 (src/alignment/justify.py, src/core/extract.py)
JUSTIFY_V2_SHA256 = "3d05c891ba7decd4814a4ed0d7b9ab914619ecff630a722988adb34a0fe3ff05"
EXTRACT_V1_SHA256 = "11955077384c5c0d7550d00bc243f55d41b9f7dd35da44ec9fcef8715fa1c337"


# --------------------------------------------------------------------------- the moved prompts

def test_the_folder_holds_the_moved_prompts():
    found = available()
    assert 2 in found["justify"] and 1 in found["extract"]


def test_justify_prompt_moved_without_changing_a_letter():
    p = load_prompt("justify", 2)
    assert p.sha256 == JUSTIFY_V2_SHA256 and len(p.text) == 1802
    assert justify.PROMPT_VERSION == 2
    assert justify._INSTRUCTION == p.text and justify.PROMPT.sha256 == JUSTIFY_V2_SHA256


def test_extract_prompt_moved_without_changing_a_letter():
    p = load_prompt("extract", 1)
    assert p.sha256 == EXTRACT_V1_SHA256 and len(p.text) == 666
    assert extract.PROMPT_VERSION == 1
    assert extract._INSTRUCTION == p.text and extract.PROMPT.sha256 == EXTRACT_V1_SHA256


# --------------------------------------------------------------------------- the loader

def test_latest_version_is_the_default(tmp_path):
    (tmp_path / "decide.v1.txt").write_text("uno\n", encoding="utf-8")
    (tmp_path / "decide.v3.txt").write_text("tres\n", encoding="utf-8")
    latest = load_prompt("decide", directory=tmp_path)
    assert (latest.version, latest.text) == (3, "tres")
    assert load_prompt("decide", 1, directory=tmp_path).text == "uno"


def test_missing_prompt_or_version_is_reported_in_spanish(tmp_path):
    (tmp_path / "decide.v1.txt").write_text("uno", encoding="utf-8")
    with pytest.raises(PromptNotFound, match="No existe el prompt 'otro'"):
        load_prompt("otro", directory=tmp_path)
    with pytest.raises(PromptNotFound, match="no tiene la versión 2"):
        load_prompt("decide", 2, directory=tmp_path)


def test_windows_line_endings_and_final_newline_keep_the_fingerprint(tmp_path):
    variants = {"a": b"linea 1\nlinea 2", "b": b"linea 1\r\nlinea 2\r\n",
                "c": b"linea 1\nlinea 2\n"}
    for folder, raw in variants.items():
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "p.v1.txt").write_bytes(raw)
    loaded = [load_prompt("p", directory=tmp_path / f) for f in variants]
    assert {p.sha256 for p in loaded} == {loaded[0].sha256}
    assert loaded[0].text == "linea 1\nlinea 2"


def test_only_one_final_newline_is_removed(tmp_path):
    (tmp_path / "p.v1.txt").write_bytes(b"texto\n\n")
    assert load_prompt("p", directory=tmp_path).text == "texto\n"


def test_empty_prompt_is_rejected(tmp_path):
    (tmp_path / "p.v1.txt").write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="vacío"):
        load_prompt("p", directory=tmp_path)


def test_files_with_other_names_are_ignored(tmp_path):
    # Names differ in letters, not only in case: Windows folders ignore case, so
    # "Justify.v1.txt" and "justify.v1.txt" would be the same file there.
    for name in ("notas.txt", "Decide.v1.txt", "justify.v1.md", "justify_v1.txt", "justify.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "justify.v1.txt").write_text("x", encoding="utf-8")
    assert available(tmp_path) == {"justify": [1]}


def test_manifest_block_names_version_and_fingerprint():
    assert prompts_manifest(("justify", 2), ("extract", 1)) == {
        "justify": {"name": "justify", "version": 2, "sha256": JUSTIFY_V2_SHA256},
        "extract": {"name": "extract", "version": 1, "sha256": EXTRACT_V1_SHA256}}


# --------------------------------------------------------------------------- runs record it

def test_justify_run_records_prompt_version_and_fingerprint(tmp_path):
    _run(_data(), JustifyLLM(), out_dir=tmp_path)
    saved = json.loads((tmp_path / CANDIDATES_JSON).read_text(encoding="utf-8"))
    meta = saved["metadata"]["justification"]
    assert meta["prompt_version"] == 2 and meta["prompt_sha256"] == JUSTIFY_V2_SHA256


def test_justify_sends_the_loaded_text(tmp_path):
    seen = []

    class Spy(JustifyLLM):
        def ask(self, prompt, context="", max_tokens=1024):
            seen.append(prompt)
            return super().ask(prompt, context, max_tokens)

    _run(_data(), Spy())
    assert seen and all(p == load_prompt("justify", 2).text for p in seen)


def test_extraction_summary_records_its_prompt():
    summary = extract.extraction_summary(extract.ExtractionReport(source="ley"))
    assert summary["prompt"] == {"name": "extract", "version": 1, "sha256": EXTRACT_V1_SHA256}


def test_upload_manifest_records_the_prompt(env, targets):  # noqa: F811
    from src.alignment.decider import PROMPT as DECIDE_PROMPT
    manifest = write_user_alignment(_ws(env, targets), cfg=env.cfg)   # S7-T06: the decider
    assert manifest["settings"]["prompts"] == {"decide": {
        "name": "decide", "version": 1, "sha256": DECIDE_PROMPT.sha256}}