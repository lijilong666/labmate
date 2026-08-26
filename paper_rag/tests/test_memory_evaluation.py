from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import (
    MemorySource,
    MemoryStore,
    audit_memory_store,
    compare_result_sets,
    evaluate_retrieval,
    load_jsonl,
)


class MemoryEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = MemoryStore(self.root / "memory.sqlite3")
        self.store.create_session("session-1", "project-1")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def add_fact(self, key: str, content: str):
        return self.store.add_memory(
            kind="user_fact",
            canonical_key=key,
            content=content,
            project_id="project-1",
            session_id="session-1",
            sources=[MemorySource(source_type="user")],
        )

    def test_retrieval_metrics_include_stale_error_rate(self) -> None:
        relevant = self.add_fact("answer_language", "The preferred answer language is Chinese")
        stale = self.add_fact("old_language", "The old answer language was Chinese")

        report = evaluate_retrieval(
            self.store,
            [
                {
                    "case_id": "language",
                    "query": "Chinese answer language",
                    "project_id": "project-1",
                    "session_id": "session-1",
                    "relevant_memory_ids": [relevant.memory_id],
                    "forbidden_memory_ids": [stale.memory_id],
                }
            ],
            top_k=2,
            candidate_k=10,
        )

        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["metrics"]["memory_recall_at_k"], 1.0)
        self.assertEqual(report["metrics"]["precision_at_k"], 0.5)
        self.assertGreater(report["metrics"]["mrr"], 0.0)
        self.assertGreater(report["metrics"]["stale_memory_error_rate"], 0.0)

    def test_retrieval_can_grade_canonical_keys_and_validates_cases(self) -> None:
        self.add_fact("answer_language", "The preferred answer language is Chinese")
        report = evaluate_retrieval(
            self.store,
            [
                {
                    "case_id": "key-case",
                    "query": "preferred answer language",
                    "project_id": "project-1",
                    "session_id": "session-1",
                    "relevant_canonical_keys": ["answer_language"],
                }
            ],
            top_k=1,
        )
        self.assertEqual(report["metrics"]["memory_recall_at_k"], 1.0)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluate_retrieval(
                self.store,
                [{"query": "q", "project_id": "project-1"}],
            )

    def test_audit_reports_expiry_provenance_conflicts_and_redundancy(self) -> None:
        first = self.store.add_memory(
            kind="user_fact",
            canonical_key="language",
            content="Chinese",
            project_id="project-1",
            session_id="session-1",
            valid_to="2026-01-01T00:00:00+00:00",
        )
        second = self.store.add_memory(
            kind="user_fact",
            canonical_key="language",
            content="Chinese",
            project_id="project-1",
            session_id="session-1",
        )

        report = audit_memory_store(
            self.store,
            project_id="project-1",
            as_of="2026-08-01T00:00:00+00:00",
        )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["healthy"])
        self.assertIn("expired_active_memory", codes)
        self.assertIn("user_fact_missing_user_source", codes)
        self.assertIn("multiple_active_canonical_versions", codes)
        self.assertIn("exact_active_duplicate", codes)
        self.assertEqual(report["memory_redundancy_ratio"], 0.5)
        self.assertEqual({first.memory_id, second.memory_id}, set(report["issues"][-1]["memory_ids"]))

    def test_compare_result_sets_uses_only_paired_numeric_cases(self) -> None:
        report = compare_result_sets(
            [
                {"case_id": "a", "task_success": False, "latency_ms": 100},
                {"case_id": "b", "task_success": True, "latency_ms": 200},
            ],
            [
                {"case_id": "a", "task_success": True, "latency_ms": 120},
                {"case_id": "b", "task_success": True, "latency_ms": 220},
                {"case_id": "c", "task_success": True, "latency_ms": 10},
            ],
        )

        self.assertEqual(report["paired_case_count"], 2)
        self.assertEqual(report["missing_from_baseline"], ["c"])
        self.assertEqual(report["metrics"]["task_success"]["absolute_delta"], 0.5)
        self.assertEqual(report["metrics"]["latency_ms"]["absolute_delta"], 20.0)

    def test_jsonl_loader_reports_line_number(self) -> None:
        valid = self.root / "valid.jsonl"
        valid.write_text('{"case_id":"a"}\n\n{"case_id":"b"}\n', encoding="utf-8")
        self.assertEqual(len(load_jsonl(valid)), 2)

        invalid = self.root / "invalid.jsonl"
        invalid.write_text('{"case_id":"a"}\nnot-json\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            load_jsonl(invalid)
