from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_rag.indexing import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_CACHE_DIR,
    INDEX_FILE_NAME,
    METADATA_FILE_NAME,
    load_embedding_model,
    load_faiss,
)


DEFAULT_INDEX_DIR = Path("paper_rag/storage/vector_store")


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid metadata JSON at line {line_number}: {exc}") from exc
    return records


def load_vector_store(index_dir: Path):
    faiss = load_faiss()
    index_path = index_dir / INDEX_FILE_NAME
    metadata_path = index_dir / METADATA_FILE_NAME

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")

    index = faiss.read_index(str(index_path))
    metadata = load_metadata(metadata_path)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            "FAISS index and metadata size mismatch: "
            f"index has {index.ntotal} vectors, metadata has {len(metadata)} records."
        )

    return index, metadata


def search_papers(
    query: str,
    top_k: int = 5,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: str | Path = DEFAULT_MODEL_CACHE_DIR,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    index_path = Path(index_dir)
    cache_path = Path(cache_dir) if cache_dir is not None else None

    index, metadata = load_vector_store(index_path)
    model = load_embedding_model(model_name, cache_path)

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependency: numpy. Install numpy before searching.") from exc

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")

    search_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, search_k)

    results: list[dict[str, Any]] = []
    for rank, (score, vector_id) in enumerate(zip(scores[0], indices[0]), start=1):
        if vector_id < 0:
            continue

        record = metadata[int(vector_id)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": record.get("chunk_id", ""),
                "paper_id": record.get("paper_id", ""),
                "source_file": record.get("source_file") or record.get("source_path", ""),
                "file_name": record.get("file_name", ""),
                "page_number": record.get("page_number", ""),
                "text": record.get("text", ""),
            }
        )

    return results


def snippet(text: str, max_length: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def format_result(result: dict[str, Any], snippet_length: int) -> str:
    display_file = result.get("file_name") or result.get("source_file")
    return (
        f"[{result['rank']}] score={result['score']:.4f} "
        f"file={display_file} page={result['page_number']} "
        f"chunk={result['chunk_id']}\n"
        f"source={result['source_file']}\n"
        f"{snippet(str(result['text']), snippet_length)}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search local paper chunks with a FAISS vector index.")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--index_dir", type=Path, default=DEFAULT_INDEX_DIR, help="FAISS index directory.")
    parser.add_argument(
        "--model_name",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Hugging Face model name or local model directory used for query embeddings.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE_DIR,
        help=f"Model cache directory for sentence-transformers. Default: {DEFAULT_MODEL_CACHE_DIR}",
    )
    parser.add_argument("--snippet_length", type=int, default=500, help="Maximum displayed text length.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        results = search_papers(
            query=args.query,
            top_k=args.top_k,
            index_dir=args.index_dir,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    if not results:
        print("No results found.")
        return 0

    for result in results:
        print(format_result(result, args.snippet_length))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
