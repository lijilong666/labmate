from __future__ import annotations

import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import MemorySource, MemoryStore, MemoryWriter, MemoryScaleConfig, run_memory_scale_test
from paper_rag.router import paper_query


class MemoryScaleTests(unittest.TestCase):
    def test_multi_session_scale_workload_passes_all_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_memory_scale_test(
                Path(directory) / "memory.sqlite3",
                MemoryScaleConfig(
                    session_count=8,
                    facts_per_session=10,
                    episodes_per_session=4,
                    duplicate_episode_pairs_per_session=2,
                    global_fact_count=5,
                    query_count=64,
                    top_k=4,
                    seed=7,
                ),
            )

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["counts"]["sessions"], 8)
        self.assertEqual(report["counts"]["retrieval_cases"], 64)
        self.assertEqual(report["counts"]["duplicate_archive_candidates"], 16)
        self.assertEqual(report["counts"]["isolation_violations"], 0)
        self.assertEqual(report["retrieval_metrics"]["memory_recall_at_k"], 1.0)
        self.assertEqual(report["retrieval_metrics"]["stale_memory_error_rate"], 0.0)
        self.assertGreater(report["database_bytes"], 0)

    def test_existing_database_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "memory.sqlite3"
            db_path.write_bytes(b"preserve-me")
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_memory_scale_test(db_path, MemoryScaleConfig())
            self.assertEqual(db_path.read_bytes(), b"preserve-me")

    def test_scale_configuration_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least 2"):
                run_memory_scale_test(
                    Path(directory) / "memory.sqlite3",
                    MemoryScaleConfig(facts_per_session=1),
                )

    def test_concurrent_writers_preserve_all_records_and_session_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            session_count = 12
            writes_per_session = 20
            for session_index in range(session_count):
                store.create_session(f"session-{session_index:02d}", "project-1")

            def write_session(session_index: int) -> None:
                local_store = MemoryStore(store.db_path)
                for item_index in range(writes_per_session):
                    local_store.add_memory(
                        kind="user_fact",
                        canonical_key=f"key-{item_index:03d}",
                        content=f"Concurrent marker s{session_index:02d} i{item_index:03d}",
                        project_id="project-1",
                        session_id=f"session-{session_index:02d}",
                        sources=[MemorySource(source_type="user")],
                    )

            with ThreadPoolExecutor(max_workers=6) as executor:
                list(executor.map(write_session, range(session_count)))

            items = store.list_memories(project_id="project-1", statuses=None, limit=10_000)
            self.assertEqual(len(items), session_count * writes_per_session)
            for session_index in range(session_count):
                session = store.get_session(f"session-{session_index:02d}")
                self.assertEqual(session.memory_revision, writes_per_session)

    def test_long_correction_chain_has_one_active_head_and_full_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            store.create_session("session-1", "project-1")
            writer = MemoryWriter(store)
            result = writer.remember_user_fact(
                canonical_key="answer_language",
                content="Version 0",
                project_id="project-1",
                session_id="session-1",
                explicit_user_request=True,
            )
            assert result.item is not None
            current_id = result.item.memory_id
            version_count = 60
            for version in range(1, version_count):
                result = writer.correct_user_fact(
                    target_id=current_id,
                    content=f"Version {version}",
                    project_id="project-1",
                    session_id="session-1",
                    explicit_user_correction=True,
                )
                assert result.item is not None
                current_id = result.item.memory_id

            active = store.list_memories(
                project_id="project-1",
                session_id="session-1",
                statuses=["active"],
                limit=100,
            )
            history = store.list_memories(
                project_id="project-1",
                session_id="session-1",
                statuses=None,
                limit=100,
            )
            self.assertEqual([item.memory_id for item in active], [current_id])
            self.assertEqual(active[0].content, "Version 59")
            self.assertEqual(len(history), version_count)
            self.assertEqual(sum(item.status == "superseded" for item in history), version_count - 1)
            self.assertEqual(store.get_session("session-1").memory_revision, version_count)

    @patch("paper_rag.router.search_papers")
    def test_many_sessions_have_isolated_cache_entries_and_one_episode_each(self, search) -> None:
        search.return_value = [
            {
                "paper_id": "p000001",
                "chunk_id": "p000001-c0001",
                "page_number": 1,
                "source_file": "paper.pdf",
                "text": "scale evidence",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory_db = root / "memory.sqlite3"
            query_cache = root / "query_cache.jsonl"
            store = MemoryStore(memory_db)
            session_count = 20
            for session_index in range(session_count):
                store.create_session(f"session-{session_index:02d}", "project-1")

            first_pass = []
            second_pass = []
            for session_index in range(session_count):
                first_pass.append(
                    paper_query(
                        "retrieve scale evidence",
                        mode="search",
                        use_memory=True,
                        memory_db=memory_db,
                        project_id="project-1",
                        session_id=f"session-{session_index:02d}",
                        query_cache=query_cache,
                    )
                )
            for session_index in range(session_count):
                second_pass.append(
                    paper_query(
                        "retrieve scale evidence",
                        mode="search",
                        use_memory=True,
                        memory_db=memory_db,
                        project_id="project-1",
                        session_id=f"session-{session_index:02d}",
                        query_cache=query_cache,
                    )
                )

            episodes = store.list_memories(
                project_id="project-1", kinds=["episode"], limit=100
            )

        self.assertTrue(all(not result["cache_hit"] for result in first_pass))
        self.assertTrue(all(result["cache_hit"] for result in second_pass))
        self.assertEqual(len(episodes), session_count)
        self.assertEqual(search.call_count, session_count)
