from src.ingest.cli import main


def test_cli_runs_and_reports_ok(capsys):
    code = main([])   # usa las rutas del config.yaml
    out = capsys.readouterr().out
    assert "SPRINT 1 - CARGA DE INSUMOS" in out
    assert "[ONTOLOGIA]" in out and "[DPV]" in out
    assert code == 0


def test_cli_index_segments_and_indexes(tmp_path, capsys):
    law = tmp_path / "mini_ley.txt"
    law.write_text(
        "Art. 1.- Objeto de la ley.\n"
        "Art. 2.- Ambito de aplicacion.\n"
        "Art. 3.- Definiciones.\n",
        encoding="utf-8",
    )
    code = main(["--index", "--reset", "--law", str(law)])
    out = capsys.readouterr().out
    assert code == 0
    assert "SEGMENTACION E INDEXADO" in out
    assert "articulos_segmentados=3" in out
    assert "articulos_indexados=3" in out