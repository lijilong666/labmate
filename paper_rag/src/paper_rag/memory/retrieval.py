from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from paper_rag.memory.models import MEMORY_KINDS, MemoryItem, MemorySearchHit
from paper_rag.memory.store import MemoryStore


@dataclass(frozen=True)
class MemoryRetrievalConfig:
    top_k: int = 6
    candidate_k: int = 50
    min_score: float = 0.0
    kinds: tuple[str, ...] | None = None
    pinned_kinds: tuple[str, ...] = ("task_state", "user_fact")
    pinned_limit: int = 10
    include_history: bool = False
    include_archived: bool = False
    recency_half_life_days: float = 90.0
    lexical_weight: float = 0.45
    scope_weight: float = 0.15
    confidence_weight: float = 0.15
    importance_weight: float = 0.10
    source_weight: float = 0.10
    recency_weight: float = 0.05

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1.")
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be greater than 0.")
        if self.kinds is not None:
            invalid = set(self.kinds) - MEMORY_KINDS
            if invalid:
                raise ValueError(f"Unsupported memory kinds: {', '.join(sorted(invalid))}")
        invalid_pinned = set(self.pinned_kinds) - MEMORY_KINDS
        if invalid_pinned:
            raise ValueError(f"Unsupported pinned memory kinds: {', '.join(sorted(invalid_pinned))}")
        if self.pinned_limit < 0:
            raise ValueError("pinned_limit must not be negative.")
        weights = self.weights()
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("retrieval weights must not be negative.")
        if sum(weights.values()) <= 0:
            raise ValueError("at least one retrieval weight must be positive.")

    def weights(self) -> dict[str, float]:
        return {
            "lexical": self.lexical_weight,
            "scope": self.scope_weight,
            "confidence": self.confidence_weight,
            "importance": self.importance_weight,
            "source_quality": self.source_weight,
            "recency": self.recency_weight,
        }


@dataclass(frozen=True)
class RetrievedMemory:
    item: MemoryItem
    rank: int
    final_score: float
    score_components: dict[str, float]
    lexical_rank: int
    match_source: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.item.memory_id,
            "memory_type": self.item.kind,
            "content": self.item.content,
            "status": self.item.status,
            "project_id": self.item.project_id,
            "session_id": self.item.session_id,
            "rank": self.rank,
            "final_score": self.final_score,
            "score_components": dict(self.score_components),
            "lexical_rank": self.lexical_rank,
            "match_source": self.match_source,
            "reasons": list(self.reasons),
            "sources": [source.__dict__ for source in self.item.sources],
        }


class MemoryRetriever:
    """Two-stage lexical retrieval plus deterministic value/validity ranking."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        query: str,
        *,
        project_id: str,
        session_id: str | None = None,
        as_of: str | datetime | None = None,
        config: MemoryRetrievalConfig | None = None,
    ) -> list[RetrievedMemory]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if not project_id.strip():
            raise ValueError("project_id must not be empty.")
        options = config or MemoryRetrievalConfig()
        options.validate()
        as_of_time = self._as_datetime(as_of)
        statuses = ["active"]
        if options.include_history:
            statuses.append("superseded")
        if options.include_archived:
            statuses.append("archived")

        candidates = self.store.search_memory_hits(
            query,
            project_id=project_id,
            session_id=session_id,
            include_global=session_id is not None,
            kinds=options.kinds,
            statuses=statuses,
            top_k=options.candidate_k,
            global_only=session_id is None,
        )
        candidates = self._add_pinned_candidates(
            candidates,
            project_id=project_id,
            session_id=session_id,
            statuses=statuses,
            config=options,
        )
        scored: list[tuple[MemorySearchHit, float, dict[str, float], tuple[str, ...]]] = []
        for hit in candidates:
            if not self._temporally_visible(hit.item, as_of_time):
                continue
            components, reasons = self._score(hit, session_id, as_of_time, options)
            final_score = self._weighted_score(components, options.weights())
            if final_score < options.min_score:
                continue
            scored.append((hit, final_score, components, reasons))

        scored.sort(
            key=lambda value: (
                -value[1],
                value[0].lexical_rank,
                -value[0].item.importance,
                value[0].item.memory_id,
            )
        )
        results: list[RetrievedMemory] = []
        for rank, (hit, final_score, components, reasons) in enumerate(scored[: options.top_k], start=1):
            results.append(
                RetrievedMemory(
                    item=hit.item,
                    rank=rank,
                    final_score=round(final_score, 6),
                    score_components={key: round(value, 6) for key, value in components.items()},
                    lexical_rank=hit.lexical_rank,
                    match_source=hit.match_source,
                    reasons=reasons,
                )
            )
        return results

    @staticmethod
    def _score(
        hit: MemorySearchHit,
        session_id: str | None,
        as_of: datetime,
        config: MemoryRetrievalConfig,
    ) -> tuple[dict[str, float], tuple[str, ...]]:
        item = hit.item
        lexical = 0.35 if hit.match_source == "pinned" else 1.0 / float(hit.lexical_rank)
        if session_id is not None and item.session_id == session_id:
            scope = 1.0
            scope_reason = "session-scoped memory"
        elif item.session_id is None:
            scope = 0.85
            scope_reason = "project-global memory"
        else:
            scope = 0.0
            scope_reason = "different session"

        observed = MemoryRetriever._as_datetime(item.observed_at)
        age_days = max(0.0, (as_of - observed).total_seconds() / 86400.0)
        recency = math.pow(0.5, age_days / config.recency_half_life_days)
        source_quality = MemoryRetriever._source_quality(item)
        components = {
            "lexical": lexical,
            "scope": scope,
            "confidence": item.confidence,
            "importance": item.importance,
            "source_quality": source_quality,
            "recency": recency,
        }
        reasons = (
            f"matched by {hit.match_source} at lexical rank {hit.lexical_rank}",
            scope_reason,
            f"status={item.status}",
            f"source_quality={source_quality:.2f}",
        )
        return components, reasons

    def _add_pinned_candidates(
        self,
        candidates: list[MemorySearchHit],
        *,
        project_id: str,
        session_id: str | None,
        statuses: list[str],
        config: MemoryRetrievalConfig,
    ) -> list[MemorySearchHit]:
        if not config.pinned_kinds or config.pinned_limit == 0:
            return candidates
        allowed_kinds = set(config.pinned_kinds)
        if config.kinds is not None:
            allowed_kinds &= set(config.kinds)
        if not allowed_kinds:
            return candidates

        pinned: list[MemoryItem] = []
        if session_id is not None:
            pinned.extend(
                self.store.list_memories(
                    project_id=project_id,
                    session_id=session_id,
                    kinds=sorted(allowed_kinds),
                    statuses=statuses,
                    limit=config.pinned_limit,
                )
            )
        remaining = config.pinned_limit - len(pinned)
        if remaining > 0:
            pinned.extend(
                self.store.list_memories(
                    project_id=project_id,
                    kinds=sorted(allowed_kinds),
                    statuses=statuses,
                    global_only=True,
                    limit=remaining,
                )
            )

        merged = list(candidates)
        seen = {hit.item.memory_id for hit in merged}
        for item in pinned:
            if item.memory_id in seen:
                continue
            merged.append(
                MemorySearchHit(
                    item=item,
                    lexical_rank=len(merged) + 1,
                    match_source="pinned",
                )
            )
            seen.add(item.memory_id)
        return merged

    @staticmethod
    def _weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
        total_weight = sum(weights.values())
        return sum(components[name] * weight for name, weight in weights.items()) / total_weight

    @staticmethod
    def _source_quality(item: MemoryItem) -> float:
        if not item.sources:
            return 0.4
        quality = {
            "user": 1.0,
            "paper_chunk": 1.0,
            "paper_card": 0.8,
            "episode": 0.6,
        }
        return max(quality.get(source.source_type, 0.4) for source in item.sources)

    @staticmethod
    def _temporally_visible(item: MemoryItem, as_of: datetime) -> bool:
        observed = MemoryRetriever._as_datetime(item.observed_at)
        if observed > as_of:
            return False
        if item.valid_from is not None and MemoryRetriever._as_datetime(item.valid_from) > as_of:
            return False
        if item.valid_to is not None and as_of >= MemoryRetriever._as_datetime(item.valid_to):
            return False
        return True

    @staticmethod
    def _as_datetime(value: str | datetime | None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"Invalid ISO datetime: {value}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
