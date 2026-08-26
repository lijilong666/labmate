from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from paper_rag.memory.context import MemoryContextBuilder, MemoryContextConfig, MemoryContextPacket
from paper_rag.memory.models import MemorySource
from paper_rag.memory.retrieval import MemoryRetrievalConfig, MemoryRetriever
from paper_rag.memory.store import DEFAULT_MEMORY_DB_PATH, MemoryStore
from paper_rag.memory.writer import MemoryWriteResult, MemoryWriter


@dataclass(frozen=True)
class MemoryQueryPreparation:
    store: MemoryStore
    writer: MemoryWriter
    packet: MemoryContextPacket
    contextualized_query: str

    @property
    def prompt_context(self) -> str:
        return self.packet.text if self.packet.entries else ""


def prepare_query_memory(
    query: str,
    *,
    project_id: str,
    session_id: str,
    db_path: str | Path = DEFAULT_MEMORY_DB_PATH,
    top_k: int = 6,
    token_budget: int = 800,
) -> MemoryQueryPreparation:
    if not session_id.strip():
        raise ValueError("session_id is required when memory is enabled.")
    if not project_id.strip():
        raise ValueError("project_id is required when memory is enabled.")
    store = MemoryStore(db_path)
    try:
        session = store.get_session(session_id)
    except KeyError:
        session = store.create_session(session_id, project_id)
    if session.project_id != project_id:
        raise ValueError(
            f"Session {session_id!r} belongs to project {session.project_id!r}, not {project_id!r}."
        )

    retriever = MemoryRetriever(store)
    builder = MemoryContextBuilder(retriever)
    packet = builder.build(
        query,
        project_id=project_id,
        session_id=session_id,
        config=MemoryContextConfig(
            token_budget=token_budget,
            retrieval=MemoryRetrievalConfig(
                top_k=top_k,
                candidate_k=max(50, top_k),
                kinds=("task_state", "user_fact", "episode"),
            ),
        ),
    )
    return MemoryQueryPreparation(
        store=store,
        writer=MemoryWriter(store),
        packet=packet,
        contextualized_query=contextualize_query(query, packet),
    )


def contextualize_query(query: str, packet: MemoryContextPacket) -> str:
    lowered = query.casefold()
    reference_markers = (
        " it ",
        " them ",
        " those ",
        " these ",
        "continue",
        "previous",
        "它",
        "它们",
        "这些",
        "那些",
        "上述",
        "刚才",
        "继续",
    )
    padded = f" {lowered} "
    needs_resolution = any(marker in padded for marker in reference_markers)
    task_state = [
        entry.retrieved.item
        for entry in packet.entries
        if entry.retrieved.item.kind == "task_state"
        and (needs_resolution or entry.retrieved.match_source != "pinned")
    ]
    if not task_state:
        return query
    lines = [query, "", "Resolved active task state (not paper evidence):"]
    for item in task_state:
        key = item.canonical_key or item.memory_id
        lines.append(f"- {key}: {item.content}")
    return "\n".join(lines)


def resolve_memory_answer_language(
    requested_language: str,
    preparation: MemoryQueryPreparation | None,
) -> str:
    if requested_language != "auto" or preparation is None:
        return requested_language
    preferences = [
        entry.retrieved.item
        for entry in preparation.packet.entries
        if entry.retrieved.item.kind == "user_fact"
        and entry.retrieved.item.canonical_key == "answer_language"
    ]
    for item in preferences:
        normalized = item.content.casefold()
        if normalized.strip() in {"zh", "zh-cn"} or any(
            marker in normalized for marker in ("chinese", "中文", "简体")
        ):
            return "zh"
        if normalized.strip() in {"en", "en-us"} or any(
            marker in normalized for marker in ("english", "英文", "英语")
        ):
            return "en"
    return requested_language


def record_query_episode(
    preparation: MemoryQueryPreparation,
    *,
    original_query: str,
    selected_mode: str,
    answer: str,
    results: list[dict[str, Any]],
    project_id: str,
    session_id: str,
    cache_hit: bool = False,
) -> MemoryWriteResult:
    sources = tuple(result_sources(results))
    insufficient = not results or "evidence is insufficient" in answer.casefold()
    outcome = "insufficient_evidence" if insufficient else "success"
    if selected_mode == "answer":
        result_summary = answer.split("\n\nSources:", maxsplit=1)[0].strip()[:1200]
    else:
        result_summary = answer.strip()[:1200]
    contains_research_claims = (
        selected_mode in {"answer", "compare"}
        and outcome == "success"
        and bool(result_summary)
    )
    return preparation.writer.record_rag_episode(
        query=original_query,
        route=selected_mode,
        outcome=outcome,
        result_summary=result_summary,
        project_id=project_id,
        session_id=session_id,
        retrieved_sources=sources,
        cache_hit=cache_hit,
        contains_research_claims=contains_research_claims,
    )


def record_failed_query_episode(
    preparation: MemoryQueryPreparation,
    *,
    original_query: str,
    selected_mode: str,
    error: Exception,
    project_id: str,
    session_id: str,
) -> MemoryWriteResult:
    return preparation.writer.record_rag_episode(
        query=original_query,
        route=selected_mode,
        outcome="failed",
        result_summary=f"{type(error).__name__}: {error}"[:1200],
        project_id=project_id,
        session_id=session_id,
    )


def result_sources(results: Iterable[dict[str, Any]]) -> list[MemorySource]:
    sources: list[MemorySource] = []
    seen: set[tuple[Any, ...]] = set()
    for result in results:
        paper_id = str(result.get("paper_id", "")).strip()
        chunk_id = str(result.get("chunk_id", "")).strip()
        source_path = str(result.get("source_file") or result.get("source_path") or "").strip() or None
        if chunk_id and paper_id:
            page_number = _positive_int(result.get("page_number"))
            source = MemorySource(
                source_type="paper_chunk",
                paper_id=paper_id,
                page_number=page_number,
                chunk_id=chunk_id,
                source_path=source_path,
            )
        elif paper_id:
            source = MemorySource(
                source_type="paper_card",
                paper_id=paper_id,
                source_path=source_path,
            )
        else:
            continue
        key = (
            source.source_type,
            source.paper_id,
            source.page_number,
            source.chunk_id,
            source.source_path,
        )
        if key in seen:
            continue
        source.validate()
        sources.append(source)
        seen.add(key)
    return sources


def memory_result_payload(
    preparation: MemoryQueryPreparation,
    episode_result: MemoryWriteResult | None,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "contextualized_query": preparation.contextualized_query,
        "recalled_memory_ids": [entry.retrieved.item.memory_id for entry in preparation.packet.entries],
        "context_token_budget": preparation.packet.token_budget,
        "context_estimated_tokens": preparation.packet.estimated_tokens,
        "context_truncated": preparation.packet.truncated,
        "episode_memory_id": (
            episode_result.item.memory_id
            if episode_result is not None and episode_result.item is not None
            else None
        ),
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
