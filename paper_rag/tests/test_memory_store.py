from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import MemorySource, MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.directory.name) / "memory.sqlite3"
        self.store = MemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def create_session(self, session_id: str = "session-1", project_id: str = "project-1") -> None:
        self.store.create_session(session_id, project_id, {"selected_papers": []})

    def test_schema_migration_is_idempotent_and_fts5_is_available(self) -> None:
        reopened = MemoryStore(self.db_path)

        self.assertEqual(reopened.schema_version(), 1)
        with closing(sqlite3.connect(self.db_path)) as connection:
            migration_count = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
            fts_table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_items_fts'"
            ).fetchone()
        self.assertEqual(migration_count, 1)
        self.assertEqual(fts_table[0], "memory_items_fts")

    def test_session_state_round_trip_and_revision(self) -> None:
        created = self.store.create_session("session-1", "project-1", {"topic": "frequency"})
        updated = self.store.update_session_state("session-1", {"topic": "diffusion"})

        self.assertEqual(created.memory_revision, 0)
        self.assertEqual(updated.memory_revision, 1)
        self.assertEqual(updated.state, {"topic": "diffusion"})
        self.assertEqual(updated.created_at, created.created_at)

    def test_duplicate_session_is_rejected(self) -> None:
        self.create_session()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create_session()

    def test_three_memory_kinds_and_sources_round_trip(self) -> None:
        self.create_session()
        task = self.store.add_memory(
            kind="task_state",
            canonical_key="selected_papers",
            content="Selected papers p000001 and p000002.",
            project_id="project-1",
            session_id="session-1",
        )
        fact = self.store.add_memory(
            kind="user_fact",
            canonical_key="answer_language",
            content="The user prefers Chinese answers.",
            project_id="project-1",
            session_id="session-1",
            sources=[MemorySource(source_type="user")],
            metadata={"explicit": True},
        )
        episode = self.store.add_memory(
            kind="episode",
            content="Retrieved frequency-domain evidence.",
            project_id="project-1",
            session_id="session-1",
            sources=[
                MemorySource(
                    source_type="paper_chunk",
                    paper_id="p000001",
                    page_number=4,
                    chunk_id="p000001-c0003",
                    source_path="papers/example.pdf",
                )
            ],
            metadata={"route": "answer"},
        )

        self.assertEqual(task.kind, "task_state")
        self.assertEqual(fact.metadata, {"explicit": True})
        self.assertEqual(fact.sources, (MemorySource(source_type="user"),))
        self.assertEqual(episode.sources[0].chunk_id, "p000001-c0003")
        self.assertEqual(self.store.get_session("session-1").memory_revision, 3)

    def test_session_and_project_scope_must_match(self) -> None:
        self.create_session()
        with self.assertRaisesRegex(ValueError, "must match"):
            self.store.add_memory(
                kind="task_state",
                canonical_key="topic",
                content="A topic",
                project_id="different-project",
                session_id="session-1",
            )
        with self.assertRaisesRegex(KeyError, "Session not found"):
            self.store.add_memory(
                kind="episode",
                content="An episode",
                project_id="project-1",
                session_id="missing-session",
            )

    def test_list_filters_project_session_kind_and_status(self) -> None:
        self.create_session("session-1", "project-1")
        self.create_session("session-2", "project-2")
        first = self.store.add_memory(
            kind="task_state",
            canonical_key="topic",
            content="Frequency localization",
            project_id="project-1",
            session_id="session-1",
        )
        self.store.add_memory(
            kind="episode",
            content="Diffusion retrieval",
            project_id="project-2",
            session_id="session-2",
        )
        self.store.archive_memory(first.memory_id)

        self.assertEqual(self.store.list_memories(project_id="project-1"), [])
        archived = self.store.list_memories(
            project_id="project-1",
            session_id="session-1",
            kinds=["task_state"],
            statuses=["archived"],
        )
        self.assertEqual([item.memory_id for item in archived], [first.memory_id])

    def test_fts_search_filters_scope_and_excludes_archived_by_default(self) -> None:
        self.create_session("session-1", "project-1")
        self.create_session("session-2", "project-2")
        active = self.store.add_memory(
            kind="task_state",
            canonical_key="research_topic",
            content="Frequency domain image localization",
            project_id="project-1",
            session_id="session-1",
        )
        archived = self.store.add_memory(
            kind="task_state",
            canonical_key="research_topic",
            content="Frequency domain audio localization",
            project_id="project-2",
            session_id="session-2",
        )
        self.store.archive_memory(archived.memory_id)

        results = self.store.search_memories("frequency localization", project_id="project-1")

        self.assertEqual([item.memory_id for item in results], [active.memory_id])

    def test_cjk_substring_fallback_without_extra_tokenizer(self) -> None:
        self.create_session()
        item = self.store.add_memory(
            kind="user_fact",
            canonical_key="answer_language",
            content="用户更喜欢使用中文回答问题。",
            project_id="project-1",
            session_id="session-1",
        )

        results = self.store.search_memories("中文回答", project_id="project-1")

        self.assertEqual([result.memory_id for result in results], [item.memory_id])

    def test_supersede_preserves_history_and_source_chain(self) -> None:
        self.create_session()
        old = self.store.add_memory(
            kind="user_fact",
            canonical_key="answer_language",
            content="The user prefers English answers.",
            project_id="project-1",
            session_id="session-1",
            sources=[MemorySource(source_type="user")],
        )
        new = self.store.supersede_memory(
            old.memory_id,
            content="The user prefers Chinese answers.",
            sources=[MemorySource(source_type="user")],
        )

        old_after = self.store.get_memory(old.memory_id)
        self.assertEqual(old_after.status, "superseded")
        self.assertIsNotNone(old_after.valid_to)
        self.assertEqual(new.status, "active")
        self.assertEqual(new.supersedes_id, old.memory_id)
        self.assertEqual(new.canonical_key, old.canonical_key)
        self.assertEqual(self.store.get_session("session-1").memory_revision, 2)
        self.assertEqual(
            [item.memory_id for item in self.store.list_memories(project_id="project-1")],
            [new.memory_id],
        )

    def test_non_active_memory_cannot_be_superseded(self) -> None:
        self.create_session()
        item = self.store.add_memory(
            kind="user_fact",
            canonical_key="answer_language",
            content="English",
            project_id="project-1",
            session_id="session-1",
        )
        self.store.archive_memory(item.memory_id)

        with self.assertRaisesRegex(ValueError, "Only active"):
            self.store.supersede_memory(item.memory_id, content="Chinese")

    def test_transaction_failure_leaves_no_partial_memory_or_revision(self) -> None:
        self.create_session()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_test_source
                BEFORE INSERT ON memory_sources
                BEGIN
                    SELECT RAISE(ABORT, 'injected source failure');
                END;
                """
            )
            connection.commit()

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_memory(
                memory_id="mem-rollback",
                kind="user_fact",
                canonical_key="answer_language",
                content="Chinese",
                project_id="project-1",
                session_id="session-1",
                sources=[MemorySource(source_type="user")],
            )

        with self.assertRaises(KeyError):
            self.store.get_memory("mem-rollback")
        self.assertEqual(self.store.get_session("session-1").memory_revision, 0)

    def test_database_persists_after_reopening_and_releases_file_handle(self) -> None:
        self.create_session()
        item = self.store.add_memory(
            kind="task_state",
            canonical_key="topic",
            content="Frequency localization",
            project_id="project-1",
            session_id="session-1",
        )

        reopened = MemoryStore(self.db_path)

        self.assertEqual(reopened.get_memory(item.memory_id), item)


if __name__ == "__main__":
    unittest.main()
