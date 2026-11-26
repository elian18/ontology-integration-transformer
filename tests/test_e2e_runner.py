from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import align_ontology as svc
from src.e2e import runner as rn
from tests.test_alignment_propose import TTL, DeciderLLM, Embed, env, targets  # noqa: F401
from tests.test_law_profile import LAW, LawLLM
from tests.test_law_profile import Embed as LawEmbed
from tests.test_ui_align_ontology import law_cfg  # noqa: F401


@pytest.fixture
def files(tmp_path):
    folder = tmp_path / "material"
    folder.mkdir()
    (folder / "mi-gdpr.ttl").write_bytes(TTL.encode("utf-8"))     # the same bytes on Windows
    (folder / "ley.txt").write_bytes(LAW.encode("utf-8"))
    (folder / "vacia.ttl").write_bytes(b"")
    return folder


@pytest.fixture
def fakes(env, targets, monkeypatch):  # noqa: F811
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    return dict(base=env.base, targets=targets, embed_fn=Embed(), llm=DeciderLLM(),
                chroma_path=env.chroma, sleep_fn=lambda s: None)


def test_the_cases_come_from_a_yaml(tmp_path):
    path = tmp_path / "casos.yaml"
    path.write_text("cases:\n  - {id: pizza, mode: ontology, ontology: m/pizza.owl, "
                    "control: true}\n  - {id: mx, mode: law, law: m/mx.pdf}\n", encoding="utf-8")
    assert rn.load_cases(path, tmp_path) == [
        rn.Case("pizza", "ontology", {"ontology": tmp_path / "m/pizza.owl"}, control=True),
        rn.Case("mx", "law", {"law": tmp_path / "m/mx.pdf"})]
    path.write_text("cases:\n  - {id: x, mode: otra}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="caso x: modo desconocido otra"):
        rn.load_cases(path, tmp_path)


def test_an_ontology_gives_the_numbers_the_web_shows(files, fakes, env, tmp_path):  # noqa: F811
    case = rn.Case("gdpr", svc.MODE_ONTOLOGY, {"ontology": files / "mi-gdpr.ttl"})
    r = rn.run_case(case, **fakes)
    assert r["result"] == "ok" and r["workspace"].startswith("mi-gdpr-")
    m = svc.view(r["workspace"])["metrics"]
    assert (r["concepts"], r["base"], r["rule"], r["suggested"], r["no_match"],
            r["no_proposal"]) == (svc.view(r["workspace"])["concepts"],
                                  m["Validadas en la base"], m["Aceptadas automáticamente"],
                                  m["Sugeridas"], m["Sin correspondencia"], m["Sin propuesta"])
    assert r["validation"]["status"] in ("verde", "amarillo", "rojo")
    assert set(r["validation"]) == {"status", "shapes", "logic", "questions", "audit"}
    assert r["download"]["same"] and len(set(r["download"]["triples"].values())) == 1
    assert r["articles"] is None and r["ai_calls"] == 0 and "inherit" in r["recomputed"]
    assert r["ai_errors"] == {} and r["without_ai"] == []
    assert r["files"] == [{"kind": "ontology", "name": "mi-gdpr.ttl",
                           "bytes": len(TTL.encode("utf-8")), "sha256": r["files"][0]["sha256"]}]


def test_the_same_case_again_reuses_its_space_unless_it_is_cleaned(files, fakes):
    case = rn.Case("gdpr", svc.MODE_ONTOLOGY, {"ontology": files / "mi-gdpr.ttl"})
    rn.run_case(case, **fakes)
    assert "inherit" not in rn.run_case(case, **fakes)["recomputed"]
    assert "inherit" in rn.run_case(case, fresh=True, **fakes)["recomputed"]


def test_a_law_says_how_many_articles_it_found(files, env, targets, law_cfg):  # noqa: F811
    case = rn.Case("ley", svc.MODE_LAW, {"law": files / "ley.txt"})
    r = rn.run_case(case, base=env.base, targets=targets, embed_fn=LawEmbed(), llm=LawLLM(),
                    chroma_path=env.chroma, sleep_fn=lambda s: None)
    assert r["result"] == "ok" and r["articles"] == 4 and r["download"]["same"]


def test_a_failure_is_a_result_with_what_the_person_reads(files, fakes):
    r = rn.run_case(rn.Case("vacia", svc.MODE_ONTOLOGY, {"ontology": files / "vacia.ttl"}),
                    **fakes)
    assert r["result"] == "error" and r["error"] == "El archivo vacia.ttl está vacío."
    assert "concepts" not in r and r["seconds"] >= 0


def test_only_transforming_keeps_the_content_in_the_four_formats(files):
    r = rn.run_case(rn.Case("tf", rn.TRANSFORM, {"ontology": files / "mi-gdpr.ttl"}))
    assert r["result"] == "ok" and r["download"]["same"]
    assert set(r["download"]["triples"]) == {"rdfxml", "owl", "turtle", "jsonld"}


def test_the_requests_to_the_ai_are_counted(monkeypatch):
    from src.ai import llm_client

    class Client:
        def ask(self, prompt, **kw):
            return "{}"
    monkeypatch.setattr(llm_client, "LLMClient", Client)
    with rn.counting_ai() as calls:
        Client().ask("a")
        llm_client.LLMClient().ask("b")
    assert calls == [2] and Client.ask.__name__ == "ask" and Client().ask("c") == "{}"


def _record(case="pizza", **kw):
    base = {"case": case, "mode": "ontology", "control": False, "result": "ok", "concepts": 10,
            "base": 1, "rule": 2, "suggested": 3, "no_match": 4, "no_proposal": 0,
            "validation": {"status": "amarillo", "shapes": "amarillo", "logic": "verde"},
            "ai_calls": 5, "articles": None, "seconds": 12.5,
            "download": {"same": True, "triples": {"turtle": 90}}}
    return {**base, **kw}


def test_the_summary_keeps_the_last_run_of_each_case(tmp_path):
    rn.save(_record(seconds=99.0), tmp_path)
    records = rn.save(_record(), tmp_path)
    rn.save(_record("vacia", result="error", error="El archivo está vacío.", ai_calls=0,
                    seconds=0.1), tmp_path)
    assert len(records) == 2 and len((tmp_path / rn.RUNS_FILE).read_text(
        encoding="utf-8").splitlines()) == 3
    lines = (tmp_path / rn.SUMMARY_FILE).read_text(encoding="utf-8").splitlines()
    assert lines[2] == "| " + " | ".join(rn.COLUMNS) + " |"
    assert lines[4:] == [
        "| pizza | Mi ontología | ok | 10 | 1 | 2 | 3 | 4 | 0 | amarillo (forma: amarillo, lógica: "
        "verde) | 5 | — | iguales (90 tripletas) | 12.5 |",
        "| vacia | Mi ontología | error: El archivo está vacío. | — | — | — | — | — | — | — | 0 | "
        "— "
        "| — | 0.1 |"]
    assert rn.row(_record(mode=rn.TRANSFORM))[1] == "Solo transformar"


def test_a_control_case_counts_its_false_positives():
    text = rn.render(_record(control=True))
    assert text.endswith("; posibles falsos positivos: 3")
    assert text.startswith("  pizza: 10 conceptos; base 1, regla 2, sugeridas 3; validación "
                           "amarillo; 5 llamadas a la IA; 12.5 s")


def test_the_command_runs_the_cases_and_says_what_is_missing(files, tmp_path, capsys,
                                                             monkeypatch):
    monkeypatch.setattr(rn, "workspace_settings", lambda cfg=None: {"root": tmp_path / "ws"})
    cases = tmp_path / "casos.yaml"
    cases.write_text(f"cases:\n  - {{id: tf, mode: transform, ontology: {files / 'mi-gdpr.ttl'}}}"
                     f"\n  - {{id: falta, mode: ontology, ontology: {files / 'no.ttl'}}}\n",
                     encoding="utf-8")
    out = tmp_path / "e2e"
    spaces = ["--salida", str(out), "--espacios", str(tmp_path / "e2e")]
    assert rn._run([str(cases), *spaces]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "Pruebas del Sprint 11 (S11-T09)"
    assert printed[1].startswith("  tf: los cuatro formatos dicen lo mismo que el archivo (")
    assert printed[2] == f"  falta: falta {files / 'no.ttl'}; no se corrió"
    assert [json.loads(x)["case"] for x in (out / rn.RUNS_FILE).read_text(
        encoding="utf-8").splitlines()] == ["tf"]
    assert rn._run([str(cases), *spaces, "--solo", "otro"]) == 1
    assert "casos que no están" in capsys.readouterr().out


def test_cleaning_does_not_clean_a_case_that_comes_again(files, tmp_path, monkeypatch):
    """--limpio cleans each space once: "dos-veces-2" has to find the one of "dos-veces-1"."""
    onto = files / "mi-gdpr.ttl"
    cases = tmp_path / "casos.yaml"
    cases.write_text(f"cases:\n  - {{id: uno, mode: ontology, ontology: {onto}}}\n"
                     f"  - {{id: dos, mode: ontology, ontology: {onto}}}\n"
                     f"  - {{id: ley, mode: law, law: {files / 'ley.txt'}}}\n", encoding="utf-8")
    seen = []

    def run_case(case, fresh=False, cfg=None):
        seen.append((case.id, fresh))
        return {"case": case.id}
    monkeypatch.setattr(rn, "run_case", run_case)
    monkeypatch.setattr(rn, "save", lambda record, out: None)
    monkeypatch.setattr(rn, "render", lambda record: "")
    spaces = ["--salida", str(tmp_path / "e2e"), "--espacios", str(tmp_path / "e2e")]
    assert rn._run([str(cases), *spaces, "--limpio"]) == 0
    assert seen == [("uno", True), ("dos", False), ("ley", True)]
    seen.clear()
    assert rn._run([str(cases), *spaces]) == 0
    assert seen == [("uno", False), ("dos", False), ("ley", False)]


def test_the_cases_run_in_their_own_spaces(files, fakes, env, tmp_path):  # noqa: F811
    ours = {**env.cfg, "workspaces": {**env.cfg["workspaces"], "dir": str(tmp_path / "e2e")}}
    r = rn.run_case(rn.Case("gdpr", svc.MODE_ONTOLOGY, {"ontology": files / "mi-gdpr.ttl"}),
                    cfg=ours, **fakes)
    assert (tmp_path / "e2e" / r["workspace"] / "workspace.json").exists()
    assert not (Path(env.cfg["workspaces"]["dir"]) / r["workspace"]).exists()
    assert svc.workspace_cfg() == env.cfg                      # the web's, as before
    from src.workspace.store import check_root
    check_root(rn.e2e_cfg(Path(rn.SPACES))["workspaces"]["dir"])    # not in the base


def test_the_answers_the_ai_already_gave_are_shared(tmp_path, monkeypatch):
    main = tmp_path / "workspaces"
    (main / "_cache").mkdir(parents=True)
    (main / "_cache" / "decider-cache.jsonl").write_text("vieja\n", encoding="utf-8")
    (main / "_cache" / "labels-cache.jsonl").write_text("de la persona\n", encoding="utf-8")
    monkeypatch.setattr(rn, "workspace_settings", lambda cfg=None: {"root": main})
    ours = tmp_path / "e2e"
    (ours / "_cache").mkdir(parents=True)
    (ours / "_cache" / "labels-cache.jsonl").write_text("de las pruebas\n", encoding="utf-8")
    assert rn.share_cache(ours) == 1
    assert (ours / "_cache" / "decider-cache.jsonl").read_text(encoding="utf-8") == "vieja\n"
    assert (ours / "_cache" / "labels-cache.jsonl").read_text(
        encoding="utf-8") == "de las pruebas\n"                     # what it has stays


def test_the_cases_of_the_sprint_are_well_written():
    root = Path(__file__).resolve().parents[1]
    cases = rn.load_cases(root / "data/reference/e2e-s11.yaml", root)
    assert len({c.id for c in cases}) == len(cases) == 27
    assert [c.id for c in cases if c.control] == ["pizza"]
    assert {c.mode for c in cases} == set(rn.MODES)
    assert all(set(c.files) == {"ontology", "law"} for c in cases if c.mode == svc.MODE_BOTH)


def test_a_download_that_lost_something_is_seen(files):
    from rdflib import Graph
    source = Graph().parse(files / "mi-gdpr.ttl", format="turtle")
    blobs = {k: source.serialize(format=f.rdflib_name).encode("utf-8")
             for k, f in rn.OUTPUT_FORMATS.items()}
    assert rn.same_downloads(blobs, source)["same"]
    smaller = Graph()
    for t in list(source)[1:]:
        smaller.add(t)
    blobs["jsonld"] = smaller.serialize(format="json-ld").encode("utf-8")
    d = rn.same_downloads(blobs, source)
    assert not d["same"] and d["triples"]["jsonld"] == len(source) - 1


def test_each_number_counts_the_states_the_web_counts():
    assert sorted(rn.STATES.values()) == sorted(svc.METRICS.values())   # every state, once


def test_a_busy_ai_is_written_down(files, fakes):
    class Busy:
        def ask(self, *args, **kwargs):
            raise RuntimeError("Error code: 503 - This model is currently experiencing high "
                               "demand.")
    r = rn.run_case(rn.Case("gdpr", svc.MODE_ONTOLOGY, {"ontology": files / "mi-gdpr.ttl"}),
                    **{**fakes, "llm": Busy()})
    assert r["result"] == "ok" and r["ai_errors"]["decider"].startswith("Error code: 503")
