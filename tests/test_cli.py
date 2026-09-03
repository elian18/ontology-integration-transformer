from src.ingest.cli import main


def test_cli_runs_and_reports_ok(capsys):
    code = main([])   # usa las rutas del config.yaml
    out = capsys.readouterr().out
    assert "SPRINT 1 - CARGA DE INSUMOS" in out
    assert "[ONTOLOGIA]" in out and "[DPV]" in out
    assert code == 0