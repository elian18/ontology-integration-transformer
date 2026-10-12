"""S6-T02: round trip (convert, read back, compare) on the fixture and on the real files.

The fixture is the one from S6-T01 (decimal "15", typed SWRL-like list, blank nodes, es/en).
The real-file tests use the base modules in data/output when they exist (skipped otherwise);
the full OntoPriv check takes a while (1472 blank-node triples), so it is marked slow."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Graph, Literal
from rdflib.namespace import XSD

from src.transform import roundtrip
from src.transform.convert import serialize
from src.transform.roundtrip import (CHECKED_FORMATS, REPORT_FILE, check_chain, check_file,
                                     check_plain, check_round_trip, compare, default_files,
                                     render_console, run, summary, write_report)
from tests.test_transform_convert import EX, FIXTURE_TTL

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "output"


@pytest.fixture
def graph() -> Graph:
    g = Graph()
    g.parse(data=FIXTURE_TTL, format="turtle")
    return g


@pytest.fixture
def fixture_file(tmp_path, graph):
    path = tmp_path / "fixture.ttl"
    path.write_bytes(serialize(graph, "turtle"))
    return path


# ------------------------------------------------------------------ comparison

@pytest.mark.parametrize("key", CHECKED_FORMATS)
def test_each_format_gives_the_same_graph(graph, key):
    check = check_round_trip(graph, key)
    assert check.same_graph
    assert check.triples_before == check.triples_after == len(graph)
    assert check.missing == check.extra == 0 and check.examples == []


def test_chain_gives_the_same_graph(graph):
    check = check_chain(graph)
    assert check.format == "chain" and check.same_graph


def test_compare_reports_a_changed_literal(graph):
    changed = Graph()
    for t in graph:
        changed.add(t)
    changed.remove((EX.Controller, EX.deadline, Literal("15", datatype=XSD.decimal)))
    changed.add((EX.Controller, EX.deadline, Literal("15.0", datatype=XSD.decimal)))
    check = compare(graph, changed, "turtle")
    assert not check.same_graph
    assert (check.missing, check.extra) == (1, 1)
    assert any('"15"' in e for e in check.examples if e.startswith("falta"))
    assert any('"15.0"' in e for e in check.examples if e.startswith("sobra"))


def test_plain_rdflib_still_fails_on_the_fixture(graph):
    # Documents what S6-T01 fixes: if rdflib ever fixes it, this test tells us.
    turtle = check_plain(graph, "turtle")
    assert not turtle.same_graph and (turtle.missing, turtle.extra) == (1, 1)
    jsonld = check_plain(graph, "jsonld")
    assert not jsonld.same_graph
    assert jsonld.triples_before - jsonld.triples_after == 2     # the two rdf:type rdf:List


def test_owl_output_is_the_same_bytes_as_rdfxml(graph):
    assert serialize(graph, "owl") == serialize(graph, "rdfxml")


# ------------------------------------------------------------------ files and report

def test_check_file_counts_blank_nodes_and_runs_everything(fixture_file, graph):
    report = check_file(fixture_file)
    assert report.source_format == "Turtle"
    assert report.triples == len(graph)
    assert report.blank_node_triples > 0
    assert [c.format for c in report.checks] == list(CHECKED_FORMATS)
    assert report.chain is not None and report.ok
    assert [p.format for p in report.plain] == ["turtle", "jsonld"]


def test_run_skips_missing_files(fixture_file, tmp_path):
    reports, missing = run([fixture_file, tmp_path / "no-existe.rdf"], with_plain=False)
    assert len(reports) == 1 and reports[0].plain == []
    assert missing == [str(tmp_path / "no-existe.rdf")]


def test_report_json_and_console_in_spanish(fixture_file, tmp_path):
    reports, missing = run([fixture_file, tmp_path / "falta.ttl"])
    path = write_report(reports, missing, tmp_path / REPORT_FILE)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["round_trips"] == 4 and data["summary"]["differences"] == 0
    assert data["summary"]["plain_rdflib_failures"] == 2
    assert data["files"][0]["ok"] is True
    assert data["missing_files"] == [str(tmp_path / "falta.ttl")]
    text = render_console(reports, missing, path)
    assert "4 de 4 viajes de ida y vuelta dan el mismo grafo (1 archivo," in text
    assert "rdflib falla en 2 de 2 viajes" in text
    assert "(no encontrado, se omite)" in text


def test_summary_counts_differences():
    ok = roundtrip.RoundTrip("turtle", 3, 3, True)
    bad = roundtrip.RoundTrip("jsonld", 3, 2, False)
    report = roundtrip.FileReport("x.ttl", "x.ttl", "Turtle", 3, 0, checks=[ok, bad])
    assert summary([report])["differences"] == 1 and not report.ok


def test_default_files_follow_config(tmp_path):
    cfg = {"inputs": {"ontology": "in/onto.rdf", "dpv": "voc/dpv.ttl"},
           "outputs": {"dir": "out"}}
    files = default_files(cfg, root=tmp_path)
    names = [f.relative_to(tmp_path).as_posix() for f in files]
    assert names == ["in/onto.rdf", "out/ontopriv-core.rdf", "out/profile-ecuador-lopdp.rdf",
                     "out/ontopriv-dpv-alignment.rdf", "data/input/sample.ttl", "voc/dpv.ttl"]
    assert len(default_files(cfg, include_dpv=False, root=tmp_path)) == 5


def test_cli_with_one_file(fixture_file, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("src.config.load_config", lambda: {"outputs": {"dir": str(tmp_path)}})
    code = roundtrip._run([str(fixture_file), "--sin-comparar"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Prueba de ida y vuelta (S6-T02)" in out and "mismo grafo" in out
    assert (tmp_path / REPORT_FILE).exists()


# ------------------------------------------------------------------ real files

@pytest.mark.parametrize("name", ["ontopriv-core.rdf", "profile-ecuador-lopdp.rdf",
                                  "ontopriv-dpv-alignment.rdf"])
def test_real_base_module_round_trip(name):
    path = OUTPUT / name
    if not path.exists():
        pytest.skip(f"{name} no generado todavía (data/output)")
    report = check_file(path, with_plain=False)
    assert report.ok, [c for c in report.checks if not c.same_graph]


@pytest.mark.slow
def test_real_ontopriv_round_trip():
    path = ROOT / "data" / "input" / "ontopriv.rdf"
    if not path.exists():
        pytest.skip("OntoPriv no está en data/input")
    report = check_file(path)
    assert report.ok
    assert report.triples == 5480 and report.blank_node_triples == 1472
    plain = {p.format: p for p in report.plain}
    assert plain["jsonld"].triples_before - plain["jsonld"].triples_after == 12