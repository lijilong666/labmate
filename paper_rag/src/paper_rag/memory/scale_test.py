from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from paper_rag.memory.consolidator import MemoryConsolidator
from paper_rag.memory.evaluation import audit_memory_store, evaluate_retrieval
from paper_rag.memory.models import MemorySource
from paper_rag.memory.store import MemoryStore
from paper_rag.memory.writer import MemoryWriter


@dataclass(frozen=True)
class MemoryScaleConfig:
    session_count: int = 40
    facts_per_session: int = 30
    episodes_per_session: int = 10
    duplicate_episode_pairs_per_session: int = 2
    global_fact_count: int = 20
    query_count: int = 400
    top_k: int = 6
    seed: int = 20260826

    def validate(self) -> None:
        positive = {
            "session_count": self.session_count,
            "facts_per_session": self.facts_per_session,
            "episodes_per_session": self.episodes_per_session,
            "query_count": self.query_count,
            "top_k": self.top_k,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0.")
        non_negative = {
            "duplicate_episode_pairs_per_session": self.duplicate_episode_pairs_per_session,
            "global_fact_count": self.global_fact_count,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must not be negative.")
        if self.facts_per_session < 2:
            raise ValueError("facts_per_session must be at least 2 for lifecycle checks.")


def run_memory_scale_test(db_path: str | Path, config: MemoryScaleConfig) -> dict[str, Any]:
    config.validate()
    resolved_db = Path(db_path)
    if resolved_db.exists():
        raise ValueError(f"Scale-test database already exists: {resolved_db}")
    project_id = "scale-project"
    store = MemoryStore(resolved_db)
    writer = MemoryWriter(store)
    source = MemorySource(source_type="user")
    randomizer = random.Random(config.seed)
    active_fact_candidates: list[tuple[str, str, str]] = []
    scope_by_memory_id: dict[str, str | None] = {}

    timings: dict[str, float] = {}
    started = perf_counter()
    for session_index in range(config.session_count):
        session_id = f"scale-session-{session_index:04d}"
        store.create_session(session_id, project_id)
        for fact_index in range(config.facts_per_session):
            marker = f"scale_s{session_index:04d}_fact_{fact_index:04d}"
            item = store.add_memory(
                kind="user_fact",
                canonical_key=f"fact-{fact_index:04d}",
                content=f"Synthetic preference marker {marker}",
                project_id=project_id,
                session_id=session_id,
                sources=[source],
            )
            scope_by_memory_id[item.memory_id] = session_id
            active_fact_candidates.append((session_id, marker, item.memory_id))

        for episode_index in range(config.episodes_per_session):
            writer.record_rag_episode(
                query=f"scale episode {session_index:04d} {episode_index:04d}",
                route="search",
                outcome="success",
                project_id=project_id,
                session_id=session_id,
                result_summary=f"Retrieved synthetic result {episode_index:04d}",
            )
        for duplicate_index in range(config.duplicate_episode_pairs_per_session):
            duplicate_kwargs = {
                "query": f"scale duplicate {session_index:04d} {duplicate_index:04d}",
                "route": "search",
                "outcome": "success",
                "project_id": project_id,
                "session_id": session_id,
                "result_summary": "Deterministic duplicate episode",
            }
            writer.record_rag_episode(**duplicate_kwargs)
            writer.record_rag_episode(**duplicate_kwargs)

    for global_index in range(config.global_fact_count):
        marker = f"scale_global_fact_{global_index:04d}"
        item = store.add_memory(
            kind="user_fact",
            canonical_key=f"global-fact-{global_index:04d}",
            content=f"Project global marker {marker}",
            project_id=project_id,
            sources=[source],
        )
        scope_by_memory_id[item.memory_id] = None
    timings["populate_ms"] = (perf_counter() - started) * 1000.0

    # Exercise append-and-supersede and archive paths across a spread of sessions.
    lifecycle_started = perf_counter()
    lifecycle_session_count = max(1, config.session_count // 5)
    excluded_ids: set[str] = set()
    for session_index in range(lifecycle_session_count):
        session_id = f"scale-session-{session_index:04d}"
        first = store.find_active_by_key(
            project_id=project_id,
            session_id=session_id,
            kind="user_fact",
            canonical_key="fact-0000",
        )
        second = store.find_active_by_key(
            project_id=project_id,
            session_id=session_id,
            kind="user_fact",
            canonical_key="fact-0001",
        )
        assert first is not None and second is not None
        corrected = writer.correct_user_fact(
            target_id=first.memory_id,
            content=f"Corrected synthetic marker scale_s{session_index:04d}_fact_0000_corrected",
            project_id=project_id,
            session_id=session_id,
            explicit_user_correction=True,
        )
        assert corrected.item is not None
        scope_by_memory_id[corrected.item.memory_id] = session_id
        active_fact_candidates.append(
            (session_id, f"scale_s{session_index:04d}_fact_0000_corrected", corrected.item.memory_id)
        )
        writer.archive(
            target_id=second.memory_id,
            project_id=project_id,
            session_id=session_id,
            explicit_user_request=True,
        )
        excluded_ids.update((first.memory_id, second.memory_id))
    active_fact_candidates = [
        candidate for candidate in active_fact_candidates if candidate[2] not in excluded_ids
    ]
    timings["lifecycle_ms"] = (perf_counter() - lifecycle_started) * 1000.0

    consolidation_started = perf_counter()
    archived_duplicates: list[str] = []
    consolidation_candidate_count = 0
    for session_index in range(config.session_count):
        report = MemoryConsolidator(store).consolidate_session(
            project_id=project_id,
            session_id=f"scale-session-{session_index:04d}",
            apply=True,
        )
        consolidation_candidate_count += len(report.archive_candidates)
        archived_duplicates.extend(report.archived_memory_ids)
    timings["consolidation_ms"] = (perf_counter() - consolidation_started) * 1000.0

    sample_count = min(config.query_count, len(active_fact_candidates))
    sampled = randomizer.sample(active_fact_candidates, sample_count)
    cases = [
        {
            "case_id": f"scale-query-{index:05d}",
            "query": marker,
            "project_id": project_id,
            "session_id": session_id,
            "relevant_memory_ids": [memory_id],
            "forbidden_memory_ids": list(excluded_ids),
        }
        for index, (session_id, marker, memory_id) in enumerate(sampled)
    ]
    retrieval_started = perf_counter()
    retrieval_report = evaluate_retrieval(
        store,
        cases,
        top_k=config.top_k,
        candidate_k=max(50, config.top_k),
    )
    timings["retrieval_eval_ms"] = (perf_counter() - retrieval_started) * 1000.0

    isolation_violations = 0
    for case, result in zip(cases, retrieval_report["cases"]):
        expected_session = case["session_id"]
        for memory_id in result["returned_memory_ids"]:
            scope = scope_by_memory_id.get(memory_id)
            if scope not in {None, expected_session}:
                isolation_violations += 1

    audit_started = perf_counter()
    audit_report = audit_memory_store(store, project_id=project_id)
    timings["audit_ms"] = (perf_counter() - audit_started) * 1000.0
    all_items = store.list_memories(project_id=project_id, statuses=None, limit=1_000_000)
    active_items = [item for item in all_items if item.status == "active"]
    expected_duplicate_archives = (
        config.session_count * config.duplicate_episode_pairs_per_session
    )
    checks = {
        "all_queries_recalled_expected_memory": retrieval_report["metrics"]["memory_recall_at_k"] == 1.0,
        "no_stale_memory_recalled": retrieval_report["metrics"]["stale_memory_error_rate"] == 0.0,
        "session_isolation_preserved": isolation_violations == 0,
        "all_duplicate_candidates_archived": (
            consolidation_candidate_count == expected_duplicate_archives
            and len(archived_duplicates) == expected_duplicate_archives
        ),
        "post_consolidation_audit_healthy": audit_report["healthy"],
        "post_consolidation_redundancy_zero": audit_report["memory_redundancy_ratio"] == 0.0,
    }
    total_ms = (perf_counter() - started) * 1000.0
    timings["total_ms"] = total_ms
    operation_count = len(all_items) + sample_count
    return {
        "passed": all(checks.values()),
        "config": asdict(config),
        "database_path": str(resolved_db),
        "database_bytes": resolved_db.stat().st_size,
        "counts": {
            "total_memory_versions": len(all_items),
            "active_memories": len(active_items),
            "sessions": config.session_count,
            "retrieval_cases": sample_count,
            "lifecycle_sessions": lifecycle_session_count,
            "duplicate_archive_candidates": consolidation_candidate_count,
            "isolation_violations": isolation_violations,
        },
        "retrieval_metrics": retrieval_report["metrics"],
        "audit": {
            "healthy": audit_report["healthy"],
            "issue_counts": audit_report["issue_counts"],
            "memory_redundancy_ratio": audit_report["memory_redundancy_ratio"],
        },
        "checks": checks,
        "timings_ms": {name: round(value, 3) for name, value in timings.items()},
        "approx_operations_per_second": round(operation_count / max(total_ms / 1000.0, 1e-9), 2),
    }
