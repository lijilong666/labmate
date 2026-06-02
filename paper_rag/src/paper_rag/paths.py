from __future__ import annotations

from pathlib import Path


DEFAULT_STORAGE_DIR = Path("paper_rag/storage")
DEFAULT_VECTOR_STORE_DIR = DEFAULT_STORAGE_DIR / "vector_store"
DEFAULT_PAPER_CARDS_FILE = "paper_cards.jsonl"
DEFAULT_ENRICHED_PAPER_CARDS_FILE = "paper_cards_enriched.jsonl"
DEFAULT_CLEANED_PAPER_CARDS_FILE = "paper_cards_cleaned.jsonl"
DEFAULT_CHUNK_METADATA_FILE = "metadata.jsonl"


def _as_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def resolve_cards_path(
    cards_path: str | Path | None = None,
    storage_dir: str | Path = DEFAULT_STORAGE_DIR,
    prefer_cleaned: bool = True,
    require_exists: bool = True,
) -> Path:
    """Resolve the best available paper-card file for downstream tools."""
    explicit_path = _as_path(cards_path)
    if explicit_path is not None:
        if require_exists and not explicit_path.exists():
            raise FileNotFoundError(f"Paper cards file not found: {explicit_path}")
        return explicit_path

    storage = Path(storage_dir)
    candidates = []
    if prefer_cleaned:
        candidates.append(storage / DEFAULT_CLEANED_PAPER_CARDS_FILE)
    candidates.extend(
        [
            storage / DEFAULT_ENRICHED_PAPER_CARDS_FILE,
            storage / DEFAULT_PAPER_CARDS_FILE,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = storage / DEFAULT_PAPER_CARDS_FILE
    if require_exists:
        raise FileNotFoundError(
            "Paper cards file not found. Run paper_rag/scripts/build_workspace.py --all first "
            f"or pass --cards_path explicitly. Checked: {', '.join(str(path) for path in candidates)}"
        )
    return fallback


def resolve_chunk_metadata_path(
    metadata_path: str | Path | None = None,
    storage_dir: str | Path = DEFAULT_STORAGE_DIR,
    require_exists: bool = True,
) -> Path:
    """Resolve the chunk metadata JSONL path used by evidence-grounded tools."""
    explicit_path = _as_path(metadata_path)
    if explicit_path is not None:
        if require_exists and not explicit_path.exists():
            raise FileNotFoundError(f"Chunk metadata file not found: {explicit_path}")
        return explicit_path

    path = Path(storage_dir) / "vector_store" / DEFAULT_CHUNK_METADATA_FILE
    if require_exists and not path.exists():
        raise FileNotFoundError(
            "Chunk metadata file not found. Run paper_rag/scripts/build_workspace.py --run_index first "
            f"or pass --metadata_path explicitly. Checked: {path}"
        )
    return path
