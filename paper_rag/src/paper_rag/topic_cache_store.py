from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_TOPIC_CACHE_PATH = Path("paper_rag/storage/topic_cache.jsonl")


def load_topic_cache(cache_path: str | Path = DEFAULT_TOPIC_CACHE_PATH) -> list[dict[str, Any]]:
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
                raise ValueError(f"Invalid topic cache JSON at line {line_number}: {exc}") from exc
    return records


def find_cached_topic(
    topic: str,
    cache_path: str | Path = DEFAULT_TOPIC_CACHE_PATH,
) -> dict[str, Any] | None:
    for record in load_topic_cache(cache_path):
        if record.get("topic") == topic:
            return record
    return None


def upsert_topic_cache(
    record: dict[str, Any],
    cache_path: str | Path = DEFAULT_TOPIC_CACHE_PATH,
) -> None:
    topic = str(record.get("topic", "")).strip()
    if not topic:
        raise ValueError("Cached topic record must include a non-empty topic.")

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = load_topic_cache(path)
    updated = False
    for index, existing in enumerate(records):
        if existing.get("topic") == topic:
            merged = dict(record)
            if existing.get("created_at") and not merged.get("created_at"):
                merged["created_at"] = existing["created_at"]
            records[index] = merged
            updated = True
            break

    if not updated:
        records.append(record)

    with path.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
