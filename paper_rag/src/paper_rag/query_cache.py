from __future__ import annotations

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
) -> dict[str, Any] | None:
    for record in load_query_cache(cache_path):
        if record.get("query") == query:
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
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
