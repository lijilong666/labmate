from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


MEMORY_KINDS = {"task_state", "user_fact", "episode"}
MEMORY_STATUSES = {"active", "superseded", "archived"}
SOURCE_TYPES = {"user", "paper_chunk", "paper_card", "episode"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_json_object(value: dict[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable: {exc}") from exc


@dataclass(frozen=True)
class MemorySource:
    source_type: str
    paper_id: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    source_path: str | None = None

    def validate(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of: {', '.join(sorted(SOURCE_TYPES))}")
        if self.page_number is not None and self.page_number <= 0:
            raise ValueError("page_number must be greater than 0 when provided.")
        if self.source_type == "paper_chunk":
            if not (self.paper_id and self.chunk_id):
                raise ValueError("paper_chunk sources require paper_id and chunk_id.")
        if self.source_type == "paper_card" and not self.paper_id:
            raise ValueError("paper_card sources require paper_id.")


@dataclass(frozen=True)
class MemorySession:
    session_id: str
    project_id: str
    state: dict[str, Any] = field(default_factory=dict)
    memory_revision: int = 0
    created_at: str = ""
    updated_at: str = ""

    def validate(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty.")
        if not self.project_id.strip():
            raise ValueError("project_id must not be empty.")
        if self.memory_revision < 0:
            raise ValueError("memory_revision must not be negative.")
        validate_json_object(self.state, "state")


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    kind: str
    content: str
    project_id: str
    session_id: str | None = None
    canonical_key: str | None = None
    status: str = "active"
    confidence: float = 1.0
    importance: float = 0.5
    observed_at: str = ""
    valid_from: str | None = None
    valid_to: str | None = None
    supersedes_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    sources: tuple[MemorySource, ...] = ()

    def validate(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory_id must not be empty.")
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(sorted(MEMORY_KINDS))}")
        if not self.content.strip():
            raise ValueError("content must not be empty.")
        if not self.project_id.strip():
            raise ValueError("project_id must not be empty.")
        if self.status not in MEMORY_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(MEMORY_STATUSES))}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be between 0 and 1.")
        if self.kind in {"task_state", "user_fact"} and not str(self.canonical_key or "").strip():
            raise ValueError(f"{self.kind} memories require canonical_key.")
        if self.kind in {"task_state", "episode"} and not str(self.session_id or "").strip():
            raise ValueError(f"{self.kind} memories require session_id.")
        validate_json_object(self.metadata, "metadata")
        for source in self.sources:
            source.validate()


@dataclass(frozen=True)
class MemorySearchHit:
    item: MemoryItem
    lexical_rank: int
    match_source: str
    raw_lexical_score: float | None = None

    def validate(self) -> None:
        if self.lexical_rank <= 0:
            raise ValueError("lexical_rank must be greater than 0.")
        if self.match_source not in {"fts5", "substring", "pinned"}:
            raise ValueError("match_source must be fts5, substring, or pinned.")
