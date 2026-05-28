from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_CACHE_DIR = Path("paper_rag/model_cache")
INDEX_FILE_NAME = "index.faiss"
METADATA_FILE_NAME = "metadata.jsonl"
MANIFEST_FILE_NAME = "manifest.json"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    paper_id: str
    source_path: str
    file_name: str
    page_number: int
    chunk_index: int
    text: str


def load_chunks(chunks_path: Path) -> Iterator[ChunkRecord]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
                text = str(payload.get("text", "")).strip()
                if not text:
                    continue
                yield ChunkRecord(
                    chunk_id=str(payload["chunk_id"]),
                    paper_id=str(payload["paper_id"]),
                    source_path=str(payload["source_path"]),
                    file_name=str(payload["file_name"]),
                    page_number=int(payload["page_number"]),
                    chunk_index=int(payload["chunk_index"]),
                    text=text,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid chunk record at line {line_number}: {exc}") from exc


def chunk_batches(records: Iterable[ChunkRecord], batch_size: int) -> Iterator[list[ChunkRecord]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    batch: list[ChunkRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def metadata_from_chunk(chunk: ChunkRecord) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "source_file": chunk.source_path,
        "source_path": chunk.source_path,
        "file_name": chunk.file_name,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
    }


def load_embedding_model(model_name: str, cache_dir: Path | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install it before building the index."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - dependency stacks may fail while importing torch.
        raise RuntimeError(
            "Failed to import sentence-transformers. It is installed, but one of its runtime "
            "dependencies could not be loaded. Check your PyTorch installation and native DLL "
            f"dependencies. Original error: {exc}"
        ) from exc

    try:
        kwargs = {}
        if cache_dir is not None:
            kwargs["cache_folder"] = str(cache_dir)
        return SentenceTransformer(model_name, **kwargs)
    except Exception as exc:  # noqa: BLE001 - provide actionable model loading guidance.
        raise RuntimeError(
            "Failed to load embedding model. You can pass a local model directory with "
            "`--model_name`, pass a custom cache directory with `--cache_dir`, or configure "
            "your own Hugging Face / ModelScope mirror source and retry. "
            f"Original error: {exc}"
        ) from exc


def load_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: faiss. Install faiss-cpu before building the index.") from exc
    return faiss


def build_faiss_index(
    chunks_path: Path,
    index_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: Path = DEFAULT_MODEL_CACHE_DIR,
    batch_size: int = 32,
) -> dict[str, object]:
    faiss = load_faiss()
    model = load_embedding_model(model_name, cache_dir)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependency: numpy. Install numpy before building the index.") from exc

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / INDEX_FILE_NAME
    metadata_path = index_dir / METADATA_FILE_NAME
    manifest_path = index_dir / MANIFEST_FILE_NAME

    index = None
    total_chunks = 0
    embedding_dim = None

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for batch in chunk_batches(load_chunks(chunks_path), batch_size):
            texts = [record.text for record in batch]
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings = np.ascontiguousarray(embeddings, dtype="float32")

            if embeddings.ndim != 2:
                raise RuntimeError(f"Expected 2D embeddings, got shape {embeddings.shape}.")

            if index is None:
                embedding_dim = int(embeddings.shape[1])
                index = faiss.IndexFlatIP(embedding_dim)

            index.add(embeddings)

            for record in batch:
                metadata_file.write(json.dumps(metadata_from_chunk(record), ensure_ascii=False) + "\n")
                total_chunks += 1

    if index is None or embedding_dim is None:
        raise RuntimeError(f"No valid chunks found in {chunks_path}.")

    faiss.write_index(index, str(index_path))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks_path": chunks_path.as_posix(),
        "index_file": index_path.name,
        "metadata_file": metadata_path.name,
        "embedding_model": model_name,
        "embedding_dim": embedding_dim,
        "chunk_count": total_chunks,
        "faiss_index_type": "IndexFlatIP",
        "normalized_embeddings": True,
    }
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local FAISS index from paper chunks.")
    parser.add_argument("--chunks", type=Path, required=True, help="Input chunks JSONL path.")
    parser.add_argument("--index_dir", type=Path, required=True, help="Output FAISS index directory.")
    parser.add_argument(
        "--model_name",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Hugging Face model name or local model directory.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE_DIR,
        help=f"Model cache directory for sentence-transformers. Default: {DEFAULT_MODEL_CACHE_DIR}",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding batch size.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        manifest = build_faiss_index(
            chunks_path=args.chunks,
            index_dir=args.index_dir,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    print(
        "Built FAISS index with "
        f"{manifest['chunk_count']} chunk(s), dim={manifest['embedding_dim']}, "
        f"model={manifest['embedding_model']}."
    )
    print(f"Index directory: {args.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
