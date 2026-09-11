"""Inputs demo (Sprint 1) + law segmentation and indexing (Sprint 2).

Reads paths from config/config.yaml via src.config. Console output is in
Spanish; identifiers in English.

Usage:
    py -m src.ingest.cli                       (Sprint 1 report: ontology + law + DPV)
    py -m src.ingest.cli --normalize           (also write the canonical RDF/XML)
    py -m src.ingest.cli --index               (Sprint 2: segment the law and index it)
    py -m src.ingest.cli --index --reset       (rebuild the 'normativa' collection first)
    py -m src.ingest.cli --index --law data/input/lopdp.pdf
"""
from __future__ import annotations
import argparse
import sys

from src.config import load_config
from src.ingest.ontology_loader import load_ontology, normalize_to_rdfxml, characterization_summary
from src.ingest.text_loader import load_legal_text
from src.ingest.dpv_loader import load_dpv


def _index_and_report(law_path: str, reset: bool) -> int:
    """Segment the law and index its articles into ChromaDB, then report."""
    # Imported here (not at module top) so the Sprint 1 report does not load
    # the embedding model.
    from src.ingest.legal_segmenter import segment_articles
    from src.ai.rag import vector_store
    from src.ai.rag.indexer import index_law

    print("=" * 66)
    print(" SPRINT 2 - SEGMENTACION E INDEXADO DE LA LEY")
    print("=" * 66)

    try:
        t = load_legal_text(law_path)
    except Exception as e:
        print(f"[LEY]        ERROR: {type(e).__name__}: {e}")
        print("=" * 66)
        print(" RESULTADO: FALLO - no se pudo cargar la ley")
        print("=" * 66)
        return 1

    summary = segment_articles(t.text, source=t.path)
    print(f"[LEY]        {t.path}")
    print(f"             fuente={t.source}  chars={t.n_chars}  "
          f"articulos_segmentados={summary['n_articles']}")

    if reset:
        vector_store.reset()
        print("[INDEX]      coleccion 'normativa' reiniciada")

    report = index_law(summary)
    print(f"[INDEX]      articulos_indexados={report['n_indexed']}  "
          f"coleccion='normativa'  documentos={report['collection_count']}")

    print("=" * 66)
    print(" RESULTADO: OK - ley segmentada e indexada")
    print("=" * 66)
    return 0


def main(argv=None) -> int:
    cfg = load_config() or {}
    inputs = cfg.get("inputs", {})
    interim = cfg.get("interim", {})

    parser = argparse.ArgumentParser(description="Ingesta de insumos y segmentacion")
    parser.add_argument("--onto", default=inputs.get("ontology", "data/input/ontopriv.rdf"))
    parser.add_argument("--law", default=inputs.get("legal_text", "data/input/lopdp.pdf"))
    parser.add_argument("--dpv", default=inputs.get("dpv", "vocab/dpv.ttl"))
    parser.add_argument("--normalize", action="store_true",
                        help="Escribe el RDF/XML canónico en interim.ontology_rdfxml")
    parser.add_argument("--index", action="store_true",
                        help="Segmenta la ley e indexa los articulos en ChromaDB (coleccion 'normativa')")
    parser.add_argument("--reset", action="store_true",
                        help="Vacia la coleccion 'normativa' antes de indexar (reconstruye desde cero)")
    args = parser.parse_args(argv)

    # Sprint 2 path: segment + index, then stop.
    if args.index:
        return _index_and_report(args.law, reset=args.reset)

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

    # 2) Legal text (.txt or .pdf; not segmented here)
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