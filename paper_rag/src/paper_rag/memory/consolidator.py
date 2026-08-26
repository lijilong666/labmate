from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from paper_rag.memory.models import MemoryItem, utc_now_iso
from paper_rag.memory.store import MemoryStore
from paper_rag.memory.writer import MemoryWriter


@dataclass(frozen=True)
class ConsolidationReport:
    project_id: str
    session_id: str
    analyzed_episode_count: int
    retained_episode_count: int
    duplicate_groups: tuple[tuple[str, ...], ...]
    archive_candidates: tuple[str, ...]
    archived_memory_ids: tuple[str, ...]
    session_summary: dict[str, Any]
    applied: bool


class MemoryConsolidator:
    """Deterministic, offline compaction for lightweight RAG memory.

    The first version deliberately avoids LLM-based fact extraction. It only
    identifies exact semantic duplicates from structured episode fields and
    writes a descriptive session summary when explicitly applied.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.writer = MemoryWriter(store)

    def consolidate_session(
        self,
        *,
        project_id: str,
        session_id: str,
        apply: bool = False,
        limit: int = 10_000,
        recent_query_limit: int = 10,
    ) -> ConsolidationReport:
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")
        if recent_query_limit <= 0:
            raise ValueError("recent_query_limit must be greater than 0.")
        session = self.store.get_session(session_id)
        if session.project_id != project_id:
            raise ValueError("Session does not belong to the requested project.")

        episodes = self.store.list_memories(
            project_id=project_id,
            session_id=session_id,
            kinds=["episode"],
            statuses=["active"],
            limit=limit,
        )
        grouped: dict[tuple[Any, ...], list[MemoryItem]] = defaultdict(list)
        for episode in episodes:
            grouped[self._episode_signature(episode)].append(episode)

        duplicate_groups: list[tuple[str, ...]] = []
        archive_candidates: list[str] = []
        retained: list[MemoryItem] = []
        for group in grouped.values():
            # list_memories returns newest first, so the first record is retained.
            retained.append(group[0])
            if len(group) > 1:
                ids = tuple(item.memory_id for item in group)
                duplicate_groups.append(ids)
                archive_candidates.extend(ids[1:])

        retained.sort(key=lambda item: (item.observed_at, item.created_at), reverse=True)
        summary = self._build_summary(retained, recent_query_limit=recent_query_limit)
        archived: list[str] = []
        if apply:
            for memory_id in archive_candidates:
                result = self.writer.archive(
                    target_id=memory_id,
                    project_id=project_id,
                    session_id=session_id,
                    explicit_user_request=True,
                )
                if result.changed and result.item is not None:
                    archived.append(result.item.memory_id)

            new_state = dict(session.state)
            new_state["memory_consolidation"] = {
                **summary,
                "duplicate_episode_count": len(archive_candidates),
                "archived_memory_ids": archived,
                "consolidated_at": utc_now_iso(),
                "policy": "exact_structured_episode_dedup_v1",
            }
            self.store.update_session_state(session_id, new_state)

        return ConsolidationReport(
            project_id=project_id,
            session_id=session_id,
            analyzed_episode_count=len(episodes),
            retained_episode_count=len(retained),
            duplicate_groups=tuple(duplicate_groups),
            archive_candidates=tuple(archive_candidates),
            archived_memory_ids=tuple(archived),
            session_summary=summary,
            applied=apply,
        )

    @classmethod
    def _episode_signature(cls, episode: MemoryItem) -> tuple[Any, ...]:
        metadata = episode.metadata
        structured = any(
            key in metadata
            for key in ("query", "route", "outcome", "result_summary", "retrieved_chunk_ids")
        )
        if not structured:
            return ("content", cls._normalize(episode.content))
        chunk_ids = metadata.get("retrieved_chunk_ids", [])
        if not isinstance(chunk_ids, list):
            chunk_ids = []
        return (
            "structured",
            cls._normalize(str(metadata.get("query", ""))),
            str(metadata.get("route", "")).strip().casefold(),
            str(metadata.get("outcome", "")).strip().casefold(),
            cls._normalize(str(metadata.get("result_summary", ""))),
            tuple(sorted(str(chunk_id) for chunk_id in chunk_ids)),
        )

    @staticmethod
    def _build_summary(episodes: list[MemoryItem], *, recent_query_limit: int) -> dict[str, Any]:
        routes = Counter(str(item.metadata.get("route", "unknown")) for item in episodes)
        outcomes = Counter(str(item.metadata.get("outcome", "unknown")) for item in episodes)
        recent_queries: list[str] = []
        for item in episodes:
            query = str(item.metadata.get("query", "")).strip()
            if query and query not in recent_queries:
                recent_queries.append(query)
            if len(recent_queries) >= recent_query_limit:
                break
        return {
            "episode_count": len(episodes),
            "routes": dict(sorted(routes.items())),
            "outcomes": dict(sorted(outcomes.items())),
            "evidence_backed_episode_count": sum(
                bool(item.metadata.get("evidence_sufficient")) for item in episodes
            ),
            "recent_queries": recent_queries,
            "automatic_fact_promotions": 0,
        }

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split()).casefold()
