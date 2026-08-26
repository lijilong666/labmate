from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from paper_rag.memory.models import MemoryItem, MemorySource
from paper_rag.memory.policy import MemoryOperation, MemoryPolicy
from paper_rag.memory.store import MemoryStore


@dataclass(frozen=True)
class MemoryWriteResult:
    action: str
    changed: bool
    reason: str
    item: MemoryItem | None = None


class MemoryWriter:
    """High-level deterministic write API; no LLM is used on this path."""

    def __init__(self, store: MemoryStore, policy: MemoryPolicy | None = None) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy()

    def execute(self, operation: MemoryOperation) -> MemoryWriteResult:
        validated = self.policy.validate(operation)
        if validated.action == "NOOP":
            return MemoryWriteResult(
                action="NOOP",
                changed=False,
                reason=validated.reason,
            )
        if validated.action == "ADD":
            item = self.store.add_memory(
                kind=str(validated.kind),
                content=str(validated.content),
                project_id=str(validated.project_id),
                session_id=validated.session_id,
                canonical_key=validated.canonical_key,
                confidence=validated.confidence,
                importance=validated.importance,
                metadata=validated.metadata,
                sources=validated.sources,
            )
        elif validated.action == "UPDATE":
            target = self._target_in_scope(validated)
            if target.kind == "user_fact" and validated.metadata.get("explicit_user_correction") is not True:
                raise ValueError("user_fact updates require an explicit user correction.")
            if target.kind == "task_state" and validated.metadata.get("deterministic_task_state_update") is not True:
                raise ValueError("task_state updates require a deterministic state transition.")
            if target.kind == "episode":
                raise ValueError("Episodes are append-only and cannot be updated.")
            item = self.store.supersede_memory(
                target.memory_id,
                content=str(validated.content),
                confidence=validated.confidence,
                importance=validated.importance,
                metadata=validated.metadata,
                sources=validated.sources,
            )
        elif validated.action == "ARCHIVE":
            target = self._target_in_scope(validated)
            if target.status == "archived":
                return MemoryWriteResult(
                    action="NOOP",
                    changed=False,
                    reason="Target memory is already archived.",
                    item=target,
                )
            item = self.store.archive_memory(target.memory_id)
        else:  # pragma: no cover - policy validation makes this unreachable.
            raise RuntimeError(f"Unsupported validated action: {validated.action}")
        return MemoryWriteResult(
            action=validated.action,
            changed=True,
            reason=validated.reason,
            item=item,
        )

    def ignore_unclassified_interaction(self, reason: str = "Interaction has no explicit memory intent.") -> MemoryWriteResult:
        return self.execute(self.policy.noop(reason))

    def remember_user_fact(
        self,
        *,
        canonical_key: str,
        content: str,
        project_id: str,
        session_id: str | None = None,
        explicit_user_request: bool,
        importance: float = 0.7,
    ) -> MemoryWriteResult:
        if not explicit_user_request:
            return self.ignore_unclassified_interaction(
                "Stable user facts are written only after an explicit remember request."
            )
        existing = self.store.find_active_by_key(
            project_id=project_id,
            session_id=session_id,
            kind="user_fact",
            canonical_key=canonical_key,
        )
        if existing is not None:
            if self._normalize(existing.content) == self._normalize(content):
                return self.ignore_unclassified_interaction("An identical active user fact already exists.")
            return self.ignore_unclassified_interaction(
                "A different active value exists; an explicit correction is required."
            )
        return self.execute(
            MemoryOperation(
                action="ADD",
                reason="User explicitly requested that a stable preference or fact be remembered.",
                project_id=project_id,
                session_id=session_id,
                kind="user_fact",
                canonical_key=canonical_key,
                content=content,
                confidence=1.0,
                importance=importance,
                metadata={"explicit_user_request": True},
                sources=(MemorySource(source_type="user"),),
            )
        )

    def set_task_state(
        self,
        *,
        canonical_key: str,
        content: str,
        project_id: str,
        session_id: str,
        metadata: dict[str, object] | None = None,
        importance: float = 0.6,
    ) -> MemoryWriteResult:
        existing = self.store.find_active_by_key(
            project_id=project_id,
            session_id=session_id,
            kind="task_state",
            canonical_key=canonical_key,
        )
        operation_metadata = dict(metadata or {})
        operation_metadata["deterministic_task_state_update"] = True
        if existing is None:
            return self.execute(
                MemoryOperation(
                    action="ADD",
                    reason="A deterministic RAG workflow state was established.",
                    project_id=project_id,
                    session_id=session_id,
                    kind="task_state",
                    canonical_key=canonical_key,
                    content=content,
                    confidence=1.0,
                    importance=importance,
                    metadata=operation_metadata,
                )
            )
        if self._normalize(existing.content) == self._normalize(content):
            return self.ignore_unclassified_interaction("Task state is unchanged.")
        return self.execute(
            MemoryOperation(
                action="UPDATE",
                reason="A deterministic RAG workflow state changed.",
                project_id=project_id,
                session_id=session_id,
                target_id=existing.memory_id,
                content=content,
                confidence=1.0,
                importance=importance,
                metadata=operation_metadata,
            )
        )

    def correct_user_fact(
        self,
        *,
        target_id: str,
        content: str,
        project_id: str,
        session_id: str | None = None,
        explicit_user_correction: bool,
        importance: float = 0.8,
    ) -> MemoryWriteResult:
        if not explicit_user_correction:
            return self.ignore_unclassified_interaction(
                "Existing memory is updated only after an explicit user correction."
            )
        target = self.store.get_memory(target_id)
        if target.project_id != project_id:
            raise ValueError("Target memory does not belong to the requested project.")
        if session_id is not None and target.session_id != session_id:
            raise ValueError("Target memory does not belong to the requested session.")
        if target.kind != "user_fact":
            raise ValueError("Explicit correction supports user_fact memory only.")
        if target.status != "active":
            raise ValueError("Only active user facts can be corrected.")
        if self._normalize(target.content) == self._normalize(content):
            return self.ignore_unclassified_interaction("The corrected value is identical to active memory.")
        return self.execute(
            MemoryOperation(
                action="UPDATE",
                reason="User explicitly corrected a stored fact.",
                project_id=project_id,
                session_id=session_id,
                target_id=target_id,
                content=content,
                confidence=1.0,
                importance=importance,
                metadata={"explicit_user_correction": True},
                sources=(MemorySource(source_type="user"),),
            )
        )

    def archive(
        self,
        *,
        target_id: str,
        project_id: str,
        session_id: str | None = None,
        explicit_user_request: bool,
    ) -> MemoryWriteResult:
        if not explicit_user_request:
            return self.ignore_unclassified_interaction(
                "Memory is archived only after an explicit user request."
            )
        return self.execute(
            MemoryOperation(
                action="ARCHIVE",
                reason="User explicitly requested selective forgetting.",
                project_id=project_id,
                session_id=session_id,
                target_id=target_id,
                metadata={"explicit_user_request": True},
            )
        )

    def record_rag_episode(
        self,
        *,
        query: str,
        route: str,
        outcome: str,
        project_id: str,
        session_id: str,
        result_summary: str = "",
        retrieved_sources: Iterable[MemorySource] = (),
        cache_hit: bool = False,
        contains_research_claims: bool = False,
    ) -> MemoryWriteResult:
        clean_query = query.strip()
        clean_summary = result_summary.strip()
        sources = tuple(retrieved_sources)
        chunk_ids = [source.chunk_id for source in sources if source.source_type == "paper_chunk"]
        content_parts = [f"Query: {clean_query}", f"Route: {route}", f"Outcome: {outcome}"]
        if clean_summary:
            content_parts.append(f"Summary: {clean_summary}")
        return self.execute(
            MemoryOperation(
                action="ADD",
                reason="A completed RAG task is recorded as an episode, not as a stable fact.",
                project_id=project_id,
                session_id=session_id,
                kind="episode",
                content="\n".join(content_parts),
                confidence=1.0 if outcome == "success" else 0.5,
                importance=0.4,
                metadata={
                    "query": clean_query,
                    "route": route,
                    "outcome": outcome,
                    "result_summary": clean_summary,
                    "retrieved_chunk_ids": chunk_ids,
                    "cache_hit": bool(cache_hit),
                    "contains_research_claims": bool(contains_research_claims),
                    "evidence_sufficient": bool(chunk_ids),
                },
                sources=sources,
            )
        )

    def _target_in_scope(self, operation: MemoryOperation) -> MemoryItem:
        target = self.store.get_memory(str(operation.target_id))
        if target.project_id != operation.project_id:
            raise ValueError("Target memory does not belong to the requested project.")
        if operation.session_id is not None and target.session_id != operation.session_id:
            raise ValueError("Target memory does not belong to the requested session.")
        return target

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split()).casefold()
