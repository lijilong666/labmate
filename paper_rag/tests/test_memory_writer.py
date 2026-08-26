from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import MemorySource, MemoryStore, MemoryWriter


class MemoryWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.directory.name) / "memory.sqlite3")
        self.store.create_session("session-1", "project-1")
        self.writer = MemoryWriter(self.store)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_unclassified_interaction_is_noop(self) -> None:
        result = self.writer.ignore_unclassified_interaction()

        self.assertEqual(result.action, "NOOP")
        self.assertFalse(result.changed)
        self.assertEqual(self.store.list_memories(statuses=None), [])
        self.assertEqual(self.store.get_session("session-1").memory_revision, 0)

    def test_user_fact_requires_explicit_remember_request(self) -> None:
        result = self.writer.remember_user_fact(
            canonical_key="answer_language",
            content="Chinese",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=False,
        )

        self.assertEqual(result.action, "NOOP")
        self.assertEqual(self.store.list_memories(statuses=None), [])

    def test_explicit_user_fact_is_written_with_user_provenance(self) -> None:
        result = self.writer.remember_user_fact(
            canonical_key="answer_language",
            content="Chinese",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )

        self.assertEqual(result.action, "ADD")
        self.assertTrue(result.changed)
        assert result.item is not None
        self.assertEqual(result.item.kind, "user_fact")
        self.assertEqual(result.item.sources, (MemorySource(source_type="user"),))
        self.assertTrue(result.item.metadata["explicit_user_request"])

    def test_identical_fact_and_conflicting_unconfirmed_value_are_noop(self) -> None:
        first = self.writer.remember_user_fact(
            canonical_key="answer_language",
            content="Chinese answers",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )
        duplicate = self.writer.remember_user_fact(
            canonical_key="answer_language",
            content="  chinese   answers ",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )
        conflict = self.writer.remember_user_fact(
            canonical_key="answer_language",
            content="English answers",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )

        self.assertEqual(first.action, "ADD")
        self.assertEqual(duplicate.action, "NOOP")
        self.assertEqual(conflict.action, "NOOP")
        self.assertEqual(len(self.store.list_memories(project_id="project-1")), 1)
        self.assertEqual(self.store.get_session("session-1").memory_revision, 1)

    def test_explicit_correction_supersedes_old_fact(self) -> None:
        old_result = self.writer.remember_user_fact(
            canonical_key="answer_language",
            content="English",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )
        assert old_result.item is not None

        no_change = self.writer.correct_user_fact(
            target_id=old_result.item.memory_id,
            content="Chinese",
            project_id="project-1",
            session_id="session-1",
            explicit_user_correction=False,
        )
        corrected = self.writer.correct_user_fact(
            target_id=old_result.item.memory_id,
            content="Chinese",
            project_id="project-1",
            session_id="session-1",
            explicit_user_correction=True,
        )

        self.assertEqual(no_change.action, "NOOP")
        self.assertEqual(corrected.action, "UPDATE")
        assert corrected.item is not None
        self.assertEqual(corrected.item.supersedes_id, old_result.item.memory_id)
        self.assertEqual(self.store.get_memory(old_result.item.memory_id).status, "superseded")
        self.assertEqual(corrected.item.sources, (MemorySource(source_type="user"),))

    def test_identical_correction_does_not_create_a_new_version(self) -> None:
        remembered = self.writer.remember_user_fact(
            canonical_key="answer_language",
            content="Chinese answers",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )
        assert remembered.item is not None

        result = self.writer.correct_user_fact(
            target_id=remembered.item.memory_id,
            content="  chinese  answers ",
            project_id="project-1",
            session_id="session-1",
            explicit_user_correction=True,
        )

        self.assertEqual(result.action, "NOOP")
        self.assertEqual(len(self.store.list_memories(statuses=None)), 1)

    def test_task_state_is_added_deduplicated_and_versioned(self) -> None:
        first = self.writer.set_task_state(
            canonical_key="selected_papers",
            content="p000001",
            project_id="project-1",
            session_id="session-1",
        )
        duplicate = self.writer.set_task_state(
            canonical_key="selected_papers",
            content="p000001",
            project_id="project-1",
            session_id="session-1",
        )
        updated = self.writer.set_task_state(
            canonical_key="selected_papers",
            content="p000001, p000002",
            project_id="project-1",
            session_id="session-1",
        )

        self.assertEqual(first.action, "ADD")
        self.assertEqual(duplicate.action, "NOOP")
        self.assertEqual(updated.action, "UPDATE")
        assert first.item is not None and updated.item is not None
        self.assertEqual(updated.item.supersedes_id, first.item.memory_id)
        self.assertEqual(self.store.get_session("session-1").memory_revision, 2)

        with self.assertRaisesRegex(ValueError, "user_fact"):
            self.writer.correct_user_fact(
                target_id=updated.item.memory_id,
                content="not a user fact",
                project_id="project-1",
                session_id="session-1",
                explicit_user_correction=True,
            )

    def test_archive_requires_explicit_request_and_never_deletes(self) -> None:
        remembered = self.writer.remember_user_fact(
            canonical_key="answer_language",
            content="Chinese",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )
        assert remembered.item is not None
        ignored = self.writer.archive(
            target_id=remembered.item.memory_id,
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=False,
        )
        archived = self.writer.archive(
            target_id=remembered.item.memory_id,
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )

        self.assertEqual(ignored.action, "NOOP")
        self.assertEqual(archived.action, "ARCHIVE")
        self.assertEqual(self.store.get_memory(remembered.item.memory_id).status, "archived")

        repeated = self.writer.archive(
            target_id=remembered.item.memory_id,
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )
        self.assertEqual(repeated.action, "NOOP")
        self.assertFalse(repeated.changed)

    def test_successful_rag_task_is_stored_only_as_episode(self) -> None:
        source = MemorySource(
            source_type="paper_chunk",
            paper_id="p000001",
            page_number=4,
            chunk_id="chunk-1",
            source_path="papers/example.pdf",
        )
        result = self.writer.record_rag_episode(
            query="Explain the method",
            route="answer",
            outcome="success",
            result_summary="The method uses frequency features.",
            project_id="project-1",
            session_id="session-1",
            retrieved_sources=[source],
            contains_research_claims=True,
        )

        self.assertEqual(result.action, "ADD")
        assert result.item is not None
        self.assertEqual(result.item.kind, "episode")
        self.assertEqual(result.item.sources, (source,))
        self.assertEqual(result.item.metadata["retrieved_chunk_ids"], ["chunk-1"])
        self.assertTrue(result.item.metadata["evidence_sufficient"])

    def test_research_claim_without_chunk_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "paper_chunk"):
            self.writer.record_rag_episode(
                query="Explain the method",
                route="answer",
                outcome="success",
                result_summary="Unsupported conclusion",
                project_id="project-1",
                session_id="session-1",
                contains_research_claims=True,
            )

        self.assertEqual(self.store.list_memories(statuses=None), [])

    def test_insufficient_evidence_episode_can_be_recorded_without_claims(self) -> None:
        result = self.writer.record_rag_episode(
            query="Unknown question",
            route="answer",
            outcome="insufficient_evidence",
            project_id="project-1",
            session_id="session-1",
        )

        self.assertEqual(result.action, "ADD")
        assert result.item is not None
        self.assertFalse(result.item.metadata["evidence_sufficient"])
        self.assertEqual(result.item.confidence, 0.5)

    def test_cache_hit_records_an_episode_but_never_a_stable_fact(self) -> None:
        result = self.writer.record_rag_episode(
            query="List papers from 2025",
            route="metadata",
            outcome="success",
            result_summary="Found two paper cards.",
            project_id="project-1",
            session_id="session-1",
            cache_hit=True,
        )

        self.assertEqual(result.action, "ADD")
        assert result.item is not None
        self.assertEqual(result.item.kind, "episode")
        self.assertTrue(result.item.metadata["cache_hit"])
        self.assertEqual(self.store.list_memories(kinds=["user_fact"]), [])

    def test_target_scope_is_enforced_for_mutations(self) -> None:
        remembered = self.writer.remember_user_fact(
            canonical_key="answer_language",
            content="Chinese",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )
        assert remembered.item is not None

        with self.assertRaisesRegex(ValueError, "project"):
            self.writer.archive(
                target_id=remembered.item.memory_id,
                project_id="different-project",
                explicit_user_request=True,
            )


if __name__ == "__main__":
    unittest.main()
