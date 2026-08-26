from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from paper_rag.memory.models import MEMORY_KINDS, MemorySource, validate_json_object


MEMORY_ACTIONS = {"NOOP", "ADD", "UPDATE", "ARCHIVE"}
RAG_ROUTES = {"metadata", "search", "answer", "compare"}
EPISODE_OUTCOMES = {"success", "insufficient_evidence", "failed", "cancelled"}


@dataclass(frozen=True)
class MemoryOperation:
    action: str
    reason: str
    project_id: str | None = None
    session_id: str | None = None
    kind: str | None = None
    content: str | None = None
    canonical_key: str | None = None
    target_id: str | None = None
    confidence: float = 1.0
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    sources: tuple[MemorySource, ...] = ()


class MemoryPolicy:
    """Validates deterministic operations before they reach MemoryStore."""

    @staticmethod
    def noop(reason: str) -> MemoryOperation:
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("NOOP reason must not be empty.")
        return MemoryOperation(action="NOOP", reason=clean_reason)

    def validate(self, operation: MemoryOperation) -> MemoryOperation:
        if operation.action not in MEMORY_ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(sorted(MEMORY_ACTIONS))}")
        if not operation.reason.strip():
            raise ValueError("operation reason must not be empty.")
        validate_json_object(operation.metadata, "operation metadata")
        for source in operation.sources:
            source.validate()

        if operation.action == "NOOP":
            return operation

        if not str(operation.project_id or "").strip():
            raise ValueError(f"{operation.action} operations require project_id.")

        if operation.action == "ADD":
            self._validate_add(operation)
        elif operation.action == "UPDATE":
            if not str(operation.target_id or "").strip():
                raise ValueError("UPDATE operations require target_id.")
            if not str(operation.content or "").strip():
                raise ValueError("UPDATE operations require content.")
            explicit_correction = operation.metadata.get("explicit_user_correction") is True
            task_state_update = operation.metadata.get("deterministic_task_state_update") is True
            if not (explicit_correction or task_state_update):
                raise ValueError(
                    "UPDATE requires an explicit user correction or deterministic task-state update."
                )
            if explicit_correction and not any(source.source_type == "user" for source in operation.sources):
                raise ValueError("Explicit user corrections require a user source.")
        elif operation.action == "ARCHIVE":
            if not str(operation.target_id or "").strip():
                raise ValueError("ARCHIVE operations require target_id.")
            if operation.metadata.get("explicit_user_request") is not True:
                raise ValueError("ARCHIVE requires an explicit user request.")

        if not 0.0 <= operation.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not 0.0 <= operation.importance <= 1.0:
            raise ValueError("importance must be between 0 and 1.")
        return operation

    @staticmethod
    def _validate_add(operation: MemoryOperation) -> None:
        if operation.kind not in MEMORY_KINDS:
            raise ValueError(f"ADD kind must be one of: {', '.join(sorted(MEMORY_KINDS))}")
        if not str(operation.content or "").strip():
            raise ValueError("ADD operations require content.")
        if operation.kind in {"task_state", "episode"} and not str(operation.session_id or "").strip():
            raise ValueError(f"{operation.kind} ADD operations require session_id.")
        if operation.kind in {"task_state", "user_fact"} and not str(operation.canonical_key or "").strip():
            raise ValueError(f"{operation.kind} ADD operations require canonical_key.")

        if operation.kind == "task_state" and operation.metadata.get("deterministic_task_state_update") is not True:
            raise ValueError("task_state writes require a deterministic state transition.")

        if operation.kind == "user_fact":
            if operation.metadata.get("explicit_user_request") is not True:
                raise ValueError("user_fact writes require an explicit user request.")
            if not any(source.source_type == "user" for source in operation.sources):
                raise ValueError("user_fact writes require a user source.")

        if operation.kind == "episode":
            route = operation.metadata.get("route")
            outcome = operation.metadata.get("outcome")
            if route not in RAG_ROUTES:
                raise ValueError(f"episode route must be one of: {', '.join(sorted(RAG_ROUTES))}")
            if outcome not in EPISODE_OUTCOMES:
                raise ValueError(
                    f"episode outcome must be one of: {', '.join(sorted(EPISODE_OUTCOMES))}"
                )
            if not str(operation.metadata.get("query", "")).strip():
                raise ValueError("episode metadata requires a query.")
            requires_chunk_evidence = operation.metadata.get("contains_research_claims") is True
            requires_chunk_evidence = requires_chunk_evidence or (
                route in {"answer", "compare"}
                and outcome == "success"
                and bool(str(operation.metadata.get("result_summary", "")).strip())
            )
            if requires_chunk_evidence and not any(source.source_type == "paper_chunk" for source in operation.sources):
                raise ValueError("Research claims in an episode require paper_chunk evidence.")
