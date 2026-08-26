from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_QUERY_CACHE_PATH = Path("paper_rag/storage/query_cache.jsonl")


def load_query_cache(cache_path: str | Path = DEFAULT_QUERY_CACHE_PATH) -> list[dict[str, Any]]:
    path = Path(cache_path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid query cache JSON at line {line_number}: {exc}") from exc
    return records


def find_cached_query(
    query: str,
    cache_path: str | Path = DEFAULT_QUERY_CACHE_PATH,
    *,
    cache_key: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    for record in reversed(load_query_cache(cache_path)):
        if record.get("query") != query:
            continue
        if cache_key is not None and any(record.get(key) != value for key, value in cache_key.items()):
            continue
        return record
    return None


def append_query_cache(
    query: str,
    mode: str,
    answer: str,
    results: list[dict[str, Any]],
    cache_path: str | Path = DEFAULT_QUERY_CACHE_PATH,
    search_query: str = "",
    filters: dict[str, Any] | None = None,
    cache_key: dict[str, Any] | None = None,
) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "query": query,
        "mode": mode,
        "answer": answer,
        "results": results,
        "search_query": search_query,
        "filters": filters or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if cache_key:
        reserved = (set(record) | {"cache_schema_version"}) & set(cache_key)
        if reserved:
            raise ValueError(f"cache_key uses reserved fields: {', '.join(sorted(reserved))}")
        record["cache_schema_version"] = 2
        record.update(cache_key)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_request_fingerprint(options: dict[str, Any]) -> str:
    serialized = json.dumps(options, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_paper_revision(
    *,
    mode: str,
    cards_path: str | Path,
    index_dir: str | Path,
) -> str:
    paths: list[Path]
    if mode == "metadata":
        paths = [Path(cards_path)]
    else:
        root = Path(index_dir)
        paths = [root / "manifest.json", root / "index.faiss", root / "metadata.jsonl"]
    digest = hashlib.sha256()
    digest.update(mode.encode("utf-8"))
    for path in paths:
        resolved = path.resolve(strict=False)
        digest.update(str(resolved).encode("utf-8"))
        if not path.exists():
            digest.update(b"missing")
            continue
        stat = path.stat()
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        if path.name == "manifest.json":
            digest.update(path.read_bytes())
    return digest.hexdigest()
