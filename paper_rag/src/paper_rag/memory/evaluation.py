from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

from paper_rag.memory.models import MemoryItem
from paper_rag.memory.retrieval import MemoryRetrievalConfig, MemoryRetriever
from paper_rag.memory.store import MemoryStore


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} of {path} must be a JSON object.")
            records.append(value)
    return records


def evaluate_retrieval(
    store: MemoryStore,
    cases: Iterable[dict[str, Any]],
    *,
    top_k: int = 6,
    candidate_k: int = 50,
) -> dict[str, Any]:
    config = MemoryRetrievalConfig(top_k=top_k, candidate_k=candidate_k)
    config.validate()
    retriever = MemoryRetriever(store)
    results: list[dict[str, Any]] = []
    total_stale_hits = 0
    total_returned = 0
    stale_cases = 0

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id") or f"case-{index}")
        query = str(case.get("query", "")).strip()
        project_id = str(case.get("project_id", "")).strip()
        session_id = case.get("session_id")
        session_id = str(session_id) if session_id is not None else None
        if not query or not project_id:
            raise ValueError(f"{case_id} requires non-empty query and project_id.")

        relevant_ids = _string_set(case.get("relevant_memory_ids", []), "relevant_memory_ids", case_id)
        relevant_keys = _string_set(
            case.get("relevant_canonical_keys", []), "relevant_canonical_keys", case_id
        )
        if bool(relevant_ids) == bool(relevant_keys):
            raise ValueError(
                f"{case_id} must define exactly one of relevant_memory_ids or relevant_canonical_keys."
            )
        forbidden_ids = _string_set(case.get("forbidden_memory_ids", []), "forbidden_memory_ids", case_id)
        recalled = retriever.retrieve(
            query,
            project_id=project_id,
            session_id=session_id,
            as_of=case.get("as_of"),
            config=config,
        )
        returned_items = [entry.item for entry in recalled]
        relevance: list[bool] = []
        matched_targets: set[str] = set()
        for item in returned_items:
            target = item.memory_id if relevant_ids else str(item.canonical_key or "")
            expected = relevant_ids if relevant_ids else relevant_keys
            is_new_relevant = target in expected and target not in matched_targets
            relevance.append(is_new_relevant)
            if is_new_relevant:
                matched_targets.add(target)
        relevant_count = len(relevant_ids or relevant_keys)
        hit_count = len(matched_targets)
        reciprocal_rank = next(
            (1.0 / rank for rank, is_relevant in enumerate(relevance, start=1) if is_relevant),
            0.0,
        )
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, is_relevant in enumerate(relevance, start=1)
            if is_relevant
        )
        ideal_hits = min(relevant_count, top_k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        stale_ids = [item.memory_id for item in returned_items if item.memory_id in forbidden_ids]
        total_stale_hits += len(stale_ids)
        total_returned += len(returned_items)
        stale_cases += bool(stale_ids)
        results.append(
            {
                "case_id": case_id,
                "query": query,
                "returned_memory_ids": [item.memory_id for item in returned_items],
                "relevant_hit_count": hit_count,
                "relevant_count": relevant_count,
                "recall_at_k": hit_count / relevant_count,
                "precision_at_k": hit_count / top_k,
                "reciprocal_rank": reciprocal_rank,
                "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
                "stale_memory_ids": stale_ids,
            }
        )

    if not results:
        raise ValueError("Retrieval benchmark must contain at least one case.")
    return {
        "case_count": len(results),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "metrics": {
            "memory_recall_at_k": fmean(item["recall_at_k"] for item in results),
            "precision_at_k": fmean(item["precision_at_k"] for item in results),
            "mrr": fmean(item["reciprocal_rank"] for item in results),
            "ndcg_at_k": fmean(item["ndcg_at_k"] for item in results),
            "stale_memory_error_rate": total_stale_hits / total_returned if total_returned else 0.0,
            "stale_case_rate": stale_cases / len(results),
        },
        "cases": results,
    }


def audit_memory_store(
    store: MemoryStore,
    *,
    project_id: str,
    as_of: str | datetime | None = None,
    limit: int = 100_000,
) -> dict[str, Any]:
    if not project_id.strip():
        raise ValueError("project_id must not be empty.")
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")
    now = _as_datetime(as_of)
    items = store.list_memories(project_id=project_id, statuses=None, limit=limit)
    active = [item for item in items if item.status == "active"]
    issues: list[dict[str, Any]] = []

    for item in active:
        if item.valid_to is not None and _as_datetime(item.valid_to) <= now:
            issues.append(_issue("error", "expired_active_memory", item.memory_id))
        if _as_datetime(item.observed_at) > now:
            issues.append(_issue("error", "future_observation_active", item.memory_id))
        if item.kind == "user_fact" and not any(source.source_type == "user" for source in item.sources):
            issues.append(_issue("warning", "user_fact_missing_user_source", item.memory_id))
        claims_research = item.kind == "episode" and (
            item.metadata.get("contains_research_claims") is True
            or item.metadata.get("evidence_sufficient") is True
        )
        if claims_research and not any(source.source_type == "paper_chunk" for source in item.sources):
            issues.append(_issue("error", "research_episode_missing_chunk_source", item.memory_id))

    keyed: dict[tuple[str | None, str, str], list[MemoryItem]] = defaultdict(list)
    duplicate_groups: dict[tuple[str, str | None, str | None, str], list[MemoryItem]] = defaultdict(list)
    for item in active:
        if item.canonical_key:
            keyed[(item.session_id, item.kind, item.canonical_key)].append(item)
        duplicate_groups[
            (item.kind, item.session_id, item.canonical_key, _normalize(item.content))
        ].append(item)
    for group in keyed.values():
        if len(group) > 1:
            issues.append(
                {
                    "severity": "error",
                    "code": "multiple_active_canonical_versions",
                    "memory_ids": [item.memory_id for item in group],
                }
            )
    exact_duplicates = [group for group in duplicate_groups.values() if len(group) > 1]
    redundant_count = sum(len(group) - 1 for group in exact_duplicates)
    for group in exact_duplicates:
        issues.append(
            {
                "severity": "warning",
                "code": "exact_active_duplicate",
                "memory_ids": [item.memory_id for item in group],
            }
        )

    status_counts = Counter(item.status for item in items)
    kind_counts = Counter(item.kind for item in items)
    severity_counts = Counter(issue["severity"] for issue in issues)
    return {
        "project_id": project_id,
        "as_of": now.isoformat(),
        "scanned_memory_count": len(items),
        "active_memory_count": len(active),
        "status_counts": dict(sorted(status_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "memory_redundancy_ratio": redundant_count / len(active) if active else 0.0,
        "exact_duplicate_group_count": len(exact_duplicates),
        "issue_counts": dict(sorted(severity_counts.items())),
        "healthy": severity_counts["error"] == 0,
        "issues": issues,
        "scan_truncated": len(items) >= limit,
    }


def compare_result_sets(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = _records_by_case_id(baseline, "baseline")
    candidate_by_id = _records_by_case_id(candidate, "candidate")
    shared = sorted(set(baseline_by_id) & set(candidate_by_id))
    if not shared:
        raise ValueError("Baseline and candidate contain no shared case_id values.")
    missing_candidate = sorted(set(baseline_by_id) - set(candidate_by_id))
    missing_baseline = sorted(set(candidate_by_id) - set(baseline_by_id))
    metrics: dict[str, dict[str, float | int]] = {}
    for field in (
        "task_success",
        "latency_ms",
        "token_count",
        "citation_accuracy",
        "answer_faithfulness",
    ):
        pairs: list[tuple[float, float]] = []
        for case_id in shared:
            left = _numeric(baseline_by_id[case_id].get(field))
            right = _numeric(candidate_by_id[case_id].get(field))
            if left is not None and right is not None:
                pairs.append((left, right))
        if pairs:
            baseline_mean = fmean(pair[0] for pair in pairs)
            candidate_mean = fmean(pair[1] for pair in pairs)
            metrics[field] = {
                "paired_case_count": len(pairs),
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "absolute_delta": candidate_mean - baseline_mean,
            }
    return {
        "paired_case_count": len(shared),
        "paired_case_ids": shared,
        "missing_from_candidate": missing_candidate,
        "missing_from_baseline": missing_baseline,
        "metrics": metrics,
    }


def _records_by_case_id(records: Sequence[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        case_id = str(record.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"{label} record {index} requires case_id.")
        if case_id in mapped:
            raise ValueError(f"Duplicate {label} case_id: {case_id}")
        mapped[case_id] = record
    return mapped


def _string_set(value: Any, field: str, case_id: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{case_id}.{field} must be a list of non-empty strings.")
    return set(value)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _issue(severity: str, code: str, memory_id: str) -> dict[str, Any]:
    return {"severity": severity, "code": code, "memory_ids": [memory_id]}


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


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
