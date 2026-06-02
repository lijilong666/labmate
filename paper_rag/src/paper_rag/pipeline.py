from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_rag.indexing import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_CACHE_DIR,
    INDEX_FILE_NAME,
    METADATA_FILE_NAME,
    build_faiss_index,
)
from paper_rag.ingestion import ingest_pdfs
from paper_rag.llm_client import DEFAULT_LLM_TIMEOUT
from paper_rag.paper_card_cleanup import cleanup_paper_cards
from paper_rag.paper_card_enricher import enrich_paper_cards
from paper_rag.paper_cards import generate_paper_cards


DEFAULT_INPUT_DIR = Path("data/raw_papers")
DEFAULT_STORAGE_DIR = Path("paper_rag/storage")


@dataclass(frozen=True)
class PipelinePaths:
    input_dir: Path = DEFAULT_INPUT_DIR
    storage_dir: Path = DEFAULT_STORAGE_DIR
    inventory_path: Path = DEFAULT_STORAGE_DIR / "paper_inventory.csv"
    chunks_path: Path = DEFAULT_STORAGE_DIR / "chunks.jsonl"
    index_dir: Path = DEFAULT_STORAGE_DIR / "vector_store"
    cards_path: Path = DEFAULT_STORAGE_DIR / "paper_cards.jsonl"
    enriched_cards_path: Path = DEFAULT_STORAGE_DIR / "paper_cards_enriched.jsonl"
    cleaned_cards_path: Path = DEFAULT_STORAGE_DIR / "paper_cards_cleaned.jsonl"


@dataclass(frozen=True)
class PipelineOptions:
    force: bool = False
    skip_existing: bool = False
    chunk_size: int = 1200
    chunk_overlap: int = 150
    model_name: str = DEFAULT_EMBEDDING_MODEL
    cache_dir: Path = DEFAULT_MODEL_CACHE_DIR
    batch_size: int = 32
    limit: int | None = None
    paper_id: str | None = None
    only_missing: bool = False
    title_overrides_path: Path | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_timeout: float = DEFAULT_LLM_TIMEOUT


def path_exists(path: Path) -> bool:
    return path.exists()


def index_exists(index_dir: Path) -> bool:
    return (index_dir / INDEX_FILE_NAME).exists() and (index_dir / METADATA_FILE_NAME).exists()


def should_run(output_exists: bool, force: bool, skip_existing: bool) -> bool:
    if force:
        return True
    if skip_existing and output_exists:
        return False
    return True


def stage_result(stage: str, status: str, message: str) -> dict[str, str]:
    return {"stage": stage, "status": status, "message": message}


def run_ingest_stage(paths: PipelinePaths, options: PipelineOptions, repo_root: Path | None = None) -> dict[str, str]:
    if not should_run(path_exists(paths.chunks_path), options.force, options.skip_existing):
        return stage_result("ingest", "skipped", f"Existing chunks found: {paths.chunks_path}")

    paper_count, chunk_count, failed_count = ingest_pdfs(
        input_dir=paths.input_dir,
        inventory_path=paths.inventory_path,
        output_path=paths.chunks_path,
        chunk_size=options.chunk_size,
        chunk_overlap=options.chunk_overlap,
        repo_root=repo_root,
    )
    return stage_result(
        "ingest",
        "completed",
        f"Ingested {paper_count} PDF(s), wrote {chunk_count} chunk(s), recorded {failed_count} failure(s).",
    )


def run_index_stage(paths: PipelinePaths, options: PipelineOptions) -> dict[str, str]:
    if not should_run(index_exists(paths.index_dir), options.force, options.skip_existing):
        return stage_result("index", "skipped", f"Existing index found: {paths.index_dir}")

    manifest = build_faiss_index(
        chunks_path=paths.chunks_path,
        index_dir=paths.index_dir,
        model_name=options.model_name,
        cache_dir=options.cache_dir,
        batch_size=options.batch_size,
    )
    return stage_result(
        "index",
        "completed",
        "Built FAISS index with {chunk_count} chunk(s), dim={embedding_dim}, model={embedding_model}.".format(
            **manifest
        ),
    )


def run_cards_stage(paths: PipelinePaths, options: PipelineOptions) -> dict[str, str]:
    if not should_run(path_exists(paths.cards_path), options.force, options.skip_existing):
        return stage_result("cards", "skipped", f"Existing paper cards found: {paths.cards_path}")

    count = generate_paper_cards(
        inventory_path=paths.inventory_path,
        output_path=paths.cards_path,
        limit=options.limit,
    )
    return stage_result("cards", "completed", f"Wrote {count} heuristic paper card(s): {paths.cards_path}")


def run_enrich_stage(paths: PipelinePaths, options: PipelineOptions) -> dict[str, str]:
    if not should_run(path_exists(paths.enriched_cards_path), options.force, options.skip_existing):
        return stage_result("enrich", "skipped", f"Existing enriched cards found: {paths.enriched_cards_path}")

    processed, failed = enrich_paper_cards(
        cards_path=paths.cards_path,
        chunks_path=paths.chunks_path,
        output_path=paths.enriched_cards_path,
        paper_id=options.paper_id,
        limit=options.limit,
        only_missing=options.only_missing,
        llm_model=options.llm_model,
        llm_base_url=options.llm_base_url,
        llm_timeout=options.llm_timeout,
    )
    return stage_result(
        "enrich",
        "completed",
        f"Processed {processed} paper card(s), {failed} failed. Wrote {paths.enriched_cards_path}.",
    )


def run_cleanup_stage(paths: PipelinePaths, options: PipelineOptions, use_enriched_input: bool = False) -> dict[str, str]:
    if not should_run(path_exists(paths.cleaned_cards_path), options.force, options.skip_existing):
        return stage_result("cleanup", "skipped", f"Existing cleaned cards found: {paths.cleaned_cards_path}")

    cleanup_input = paths.enriched_cards_path if use_enriched_input and paths.enriched_cards_path.exists() else paths.cards_path
    stats = cleanup_paper_cards(
        cards_path=cleanup_input,
        output_path=paths.cleaned_cards_path,
        title_overrides_path=options.title_overrides_path,
        limit=options.limit,
    )
    return stage_result(
        "cleanup",
        "completed",
        (
            "Processed {processed}/{total} card(s): {updated} updated, {unchanged} unchanged, "
            "{needs_review} need review. Wrote {output}."
        ).format(output=paths.cleaned_cards_path, **stats),
    )


def build_workspace(
    paths: PipelinePaths,
    options: PipelineOptions,
    run_ingest: bool = False,
    run_index: bool = False,
    run_cards: bool = False,
    run_enrich: bool = False,
    run_cleanup: bool = False,
    repo_root: Path | None = None,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    if run_ingest:
        results.append(run_ingest_stage(paths, options, repo_root=repo_root))
    if run_index:
        results.append(run_index_stage(paths, options))
    if run_cards:
        results.append(run_cards_stage(paths, options))
    if run_enrich:
        results.append(run_enrich_stage(paths, options))
    if run_cleanup:
        results.append(run_cleanup_stage(paths, options, use_enriched_input=run_enrich))

    if not results:
        results.append(stage_result("pipeline", "skipped", "No stages selected. Use --all or --run_* flags."))
    return results


def recommended_cards_path(paths: PipelinePaths) -> Path:
    if paths.cleaned_cards_path.exists():
        return paths.cleaned_cards_path
    if paths.enriched_cards_path.exists():
        return paths.enriched_cards_path
    return paths.cards_path


def format_pipeline_results(results: list[dict[str, Any]], paths: PipelinePaths) -> str:
    lines = []
    for result in results:
        lines.append("[{stage}] {status}: {message}".format(**result))
    lines.append("")
    lines.append("Recommended downstream cards: " + str(recommended_cards_path(paths)))
    return "\n".join(lines)
