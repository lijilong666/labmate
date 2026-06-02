from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "paper_rag" / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_rag.cli_io import configure_utf8_stdio  # noqa: E402
from paper_rag.indexing import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL_CACHE_DIR  # noqa: E402
from paper_rag.llm_client import DEFAULT_LLM_TIMEOUT  # noqa: E402
from paper_rag.pipeline import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_STORAGE_DIR,
    PipelineOptions,
    PipelinePaths,
    build_workspace,
    format_pipeline_results,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or refresh the local paper_rag workspace.")

    parser.add_argument("--all", action="store_true", help="Run non-LLM stages: ingest, index, cards, cleanup.")
    parser.add_argument("--run_ingest", action="store_true", help="Run PDF inventory and chunk ingestion.")
    parser.add_argument("--run_index", action="store_true", help="Run FAISS index building.")
    parser.add_argument("--run_cards", action="store_true", help="Generate heuristic paper cards.")
    parser.add_argument("--run_enrich", action="store_true", help="Run LLM-assisted paper card enrichment.")
    parser.add_argument("--run_cleanup", action="store_true", help="Run paper-card metadata cleanup.")

    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing PDF files.")
    parser.add_argument("--storage_dir", type=Path, default=DEFAULT_STORAGE_DIR, help="Runtime storage directory.")
    parser.add_argument("--inventory", type=Path, default=None, help="Inventory CSV path.")
    parser.add_argument("--chunks", type=Path, default=None, help="Chunks JSONL path.")
    parser.add_argument("--index_dir", type=Path, default=None, help="FAISS index directory.")
    parser.add_argument("--cards", type=Path, default=None, help="Heuristic paper cards JSONL path.")
    parser.add_argument("--enriched_cards", type=Path, default=None, help="LLM-enriched paper cards JSONL path.")
    parser.add_argument("--cleaned_cards", type=Path, default=None, help="Cleaned paper cards JSONL path.")
    parser.add_argument("--title_overrides", type=Path, default=None, help="Optional title override JSON path.")

    parser.add_argument("--skip_existing", action="store_true", help="Skip stages whose outputs already exist.")
    parser.add_argument("--force", action="store_true", help="Rebuild selected stages even if outputs exist.")
    parser.add_argument("--limit", type=int, default=None, help="Optional card/enrichment/cleanup limit.")
    parser.add_argument("--paper_id", default=None, help="Optional paper id for enrichment.")
    parser.add_argument("--only_missing", action="store_true", help="Do not overwrite non-empty enrichment fields.")

    parser.add_argument("--chunk_size", type=int, default=1200, help="Maximum chunk length in characters.")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Character overlap between chunks.")
    parser.add_argument("--model_name", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name or local path.")
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR, help="Embedding model cache dir.")
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding batch size.")

    parser.add_argument("--llm_model", default=None, help="Override LABMATE_LLM_MODEL for enrichment.")
    parser.add_argument("--llm_base_url", default=None, help="Override LABMATE_LLM_BASE_URL for enrichment.")
    parser.add_argument("--llm_timeout", type=float, default=DEFAULT_LLM_TIMEOUT, help="LLM request timeout in seconds.")
    return parser


def resolve_paths(args: argparse.Namespace) -> PipelinePaths:
    storage_dir = args.storage_dir
    return PipelinePaths(
        input_dir=args.input_dir,
        storage_dir=storage_dir,
        inventory_path=args.inventory or storage_dir / "paper_inventory.csv",
        chunks_path=args.chunks or storage_dir / "chunks.jsonl",
        index_dir=args.index_dir or storage_dir / "vector_store",
        cards_path=args.cards or storage_dir / "paper_cards.jsonl",
        enriched_cards_path=args.enriched_cards or storage_dir / "paper_cards_enriched.jsonl",
        cleaned_cards_path=args.cleaned_cards or storage_dir / "paper_cards_cleaned.jsonl",
    )


def resolve_stage_flags(args: argparse.Namespace) -> tuple[bool, bool, bool, bool, bool]:
    run_ingest = args.run_ingest or args.all
    run_index = args.run_index or args.all
    run_cards = args.run_cards or args.all
    run_enrich = args.run_enrich
    run_cleanup = args.run_cleanup or args.all
    return run_ingest, run_index, run_cards, run_enrich, run_cleanup


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    paths = resolve_paths(args)
    options = PipelineOptions(
        force=args.force,
        skip_existing=args.skip_existing,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        paper_id=args.paper_id,
        only_missing=args.only_missing,
        title_overrides_path=args.title_overrides,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_timeout=args.llm_timeout,
    )
    run_ingest, run_index, run_cards, run_enrich, run_cleanup = resolve_stage_flags(args)

    try:
        results = build_workspace(
            paths=paths,
            options=options,
            run_ingest=run_ingest,
            run_index=run_index,
            run_cards=run_cards,
            run_enrich=run_enrich,
            run_cleanup=run_cleanup,
            repo_root=Path.cwd(),
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    print(format_pipeline_results(results, paths))
    if run_enrich:
        print("")
        print("Note: --run_enrich calls the configured LLM API and may consume tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
