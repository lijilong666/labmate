from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import MemoryConsolidator, MemorySource, MemoryStore, MemoryWriter


class MemoryConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.directory.name) / "memory.sqlite3")
        self.store.create_session("session-1", "project-1", {"topic": "RAG memory"})
        self.writer = MemoryWriter(self.store)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def add_episode(self, *, query: str = "Explain memory", chunk_id: str = "p1-c1") -> str:
        result = self.writer.record_rag_episode(
            query=query,
            route="answer",
            outcome="success",
            project_id="project-1",
            session_id="session-1",
            result_summary="Memory augments retrieval.",
            retrieved_sources=[
                MemorySource(
                    source_type="paper_chunk",
                    paper_id="p1",
                    chunk_id=chunk_id,
                    page_number=1,
                )
            ],
        )
        assert result.item is not None
        return result.item.memory_id

    def test_dry_run_reports_duplicates_without_mutating_memory(self) -> None:
        first = self.add_episode()
        second = self.add_episode()
        revision_before = self.store.get_session("session-1").memory_revision

        report = MemoryConsolidator(self.store).consolidate_session(
            project_id="project-1",
            session_id="session-1",
        )

        self.assertFalse(report.applied)
        self.assertEqual(report.analyzed_episode_count, 2)
        self.assertEqual(report.retained_episode_count, 1)
        self.assertEqual(set(report.duplicate_groups[0]), {first, second})
        self.assertEqual(len(report.archive_candidates), 1)
        self.assertEqual(report.archived_memory_ids, ())
        self.assertEqual(self.store.get_session("session-1").memory_revision, revision_before)
        self.assertEqual(len(self.store.list_memories(kinds=["episode"])), 2)

    def test_apply_archives_older_duplicate_and_persists_summary(self) -> None:
        self.add_episode()
        self.add_episode()
        self.add_episode(query="Compare approaches", chunk_id="p1-c2")

        report = MemoryConsolidator(self.store).consolidate_session(
            project_id="project-1",
            session_id="session-1",
            apply=True,
        )

        self.assertTrue(report.applied)
        self.assertEqual(len(report.archived_memory_ids), 1)
        self.assertEqual(len(self.store.list_memories(kinds=["episode"])), 2)
        archived = self.store.list_memories(kinds=["episode"], statuses=["archived"])
        self.assertEqual([item.memory_id for item in archived], list(report.archived_memory_ids))
        state = self.store.get_session("session-1").state
        summary = state["memory_consolidation"]
        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["duplicate_episode_count"], 1)
        self.assertEqual(summary["automatic_fact_promotions"], 0)
        self.assertEqual(summary["policy"], "exact_structured_episode_dedup_v1")

    def test_different_evidence_is_not_deduplicated(self) -> None:
        self.add_episode(chunk_id="p1-c1")
        self.add_episode(chunk_id="p1-c2")

        report = MemoryConsolidator(self.store).consolidate_session(
            project_id="project-1",
            session_id="session-1",
        )

        self.assertEqual(report.retained_episode_count, 2)
        self.assertEqual(report.archive_candidates, ())

    def test_project_scope_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not belong"):
            MemoryConsolidator(self.store).consolidate_session(
                project_id="other-project",
                session_id="session-1",
            )
