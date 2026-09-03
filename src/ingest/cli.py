"""Inputs demo (Sprint 1): load ontology + legal text + DPV and print a console report.

Reads paths from config/config.yaml. Console output is in Spanish; identifiers in English.

Usage:
    py -m src.ingest.cli
    py -m src.ingest.cli --onto data/input/ontopriv.rdf --law data/input/lopdp.pdf --dpv vocab/dpv.ttl
    py -m src.ingest.cli --normalize     (also write the canonical RDF/XML for later stages)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml

from src.ingest.ontology_loader import load_ontology, normalize_to_rdfxml, characterization_summary
from src.ingest.text_loader import load_legal_text
from src.ingest.dpv_loader import load_dpv

ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    path = ROOT / "config" / "config.yaml"
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def main(argv=None) -> int:
    cfg = _config()
    inputs = cfg.get("inputs", {})
    interim = cfg.get("interim", {})

    parser = argparse.ArgumentParser(description="Ingesta de insumos - Sprint 1")
    parser.add_argument("--onto", default=inputs.get("ontology", "data/input/ontopriv.rdf"))
    parser.add_argument("--law", default=inputs.get("legal_text", "data/input/lopdp.pdf"))
    parser.add_argument("--dpv", default=inputs.get("dpv", "vocab/dpv.ttl"))
    parser.add_argument("--normalize", action="store_true",
                        help="Escribe el RDF/XML canónico en interim.ontology_rdfxml")
    args = parser.parse_args(argv)

    ok = True
    print("=" * 66)
    print(" SPRINT 1 - CARGA DE INSUMOS")
    print("=" * 66)

    # 1) Base ontology + characterization
    try:
        r = load_ontology(args.onto)
        print(f"[ONTOLOGIA]  {r.path}")
        print(f"             formato={r.source_format}  sha256={r.sha256}  tripletas={r.n_triples}")
        print(f"             clases={r.n_classes}  obj_props={r.n_object_props}  "
              f"data_props={r.n_data_props}  individuos={r.n_individuals}")
        print(f"             tipo de ontología: {r.ontology_flavor.upper()} -> {r.flavor_detail}")
        if args.normalize:
            out = interim.get("ontology_rdfxml", "data/interim/ontopriv.rdf")
            saved = normalize_to_rdfxml(r.graph, out)
            print(f"             normalizado a RDF/XML -> {saved}")
    except Exception as e:
        print(f"[ONTOLOGIA]  ERROR: {type(e).__name__}: {e}")
        ok = False

    # 2) Legal text (.txt or .pdf; not segmented yet)
    try:
        t = load_legal_text(args.law)
        print(f"[LEY]        {t.path}")
        print(f"             fuente={t.source}  sha256={t.sha256}  chars={t.n_chars}  "
              f"lineas={t.n_lines} (no vacias={t.n_nonempty_lines})")
        print(f"             codificacion={t.encoding}  articulos_detectados={t.candidate_articles}")
    except FileNotFoundError as e:
        print(f"[LEY]        (opcional en esta demo) no cargada: {e}")
    except Exception as e:
        print(f"[LEY]        ERROR: {type(e).__name__}: {e}")
        ok = False

    # 3) DPV in memory
    try:
        d = load_dpv(args.dpv)
        print(f"[DPV]        {d.path}")
        print(f"             tripletas={d.n_triples}  conceptos={d.n_concepts}  "
              f"clases={d.n_classes}  propiedades={d.n_properties}  con_etiqueta={d.n_labeled}")
    except Exception as e:
        print(f"[DPV]        ERROR: {type(e).__name__}: {e}")
        ok = False

    print("=" * 66)
    print(" RESULTADO:", "OK - insumos reconocidos" if ok else "FALLO - revisar errores")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())