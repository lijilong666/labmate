from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import (
    MemoryContextBuilder,
    MemoryContextConfig,
    MemoryRetrievalConfig,
    MemoryRetriever,
    MemorySource,
    MemoryStore,
    estimate_tokens,
)


class MemoryContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.directory.name) / "memory.sqlite3")
        self.store.create_session("session-1", "project-1")
        self.builder = MemoryContextBuilder(MemoryRetriever(self.store))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_empty_recall_returns_safety_header_only(self) -> None:
        packet = self.builder.build(
            "unmatched query",
            project_id="project-1",
            session_id="session-1",
        )

        self.assertEqual(packet.entries, ())
        self.assertIn("not paper evidence", packet.text)
        self.assertLessEqual(packet.estimated_tokens, packet.token_budget)

    def test_context_contains_rank_score_scope_and_source_pointer(self) -> None:
        self.store.add_memory(
            kind="episode",
            content="Frequency evidence was retrieved for localization.",
            project_id="project-1",
            session_id="session-1",
            sources=[
                MemorySource(
                    source_type="paper_chunk",
                    paper_id="p000001",
                    page_number=4,
                    chunk_id="chunk-1",
                )
            ],
        )

        packet = self.builder.build(
            "frequency localization",
            project_id="project-1",
            session_id="session-1",
        )

        self.assertEqual(len(packet.entries), 1)
        self.assertIn("type=episode", packet.text)
        self.assertIn("score=", packet.text)
        self.assertIn("paper_chunk:p000001/4/chunk-1", packet.text)
        self.assertIn("must never be cited as a paper source", packet.text)

    def test_long_memory_is_truncated_to_fixed_token_budget(self) -> None:
        self.store.add_memory(
            kind="task_state",
            canonical_key="frequency_notes",
            content="frequency " + ("very long memory content " * 300),
            project_id="project-1",
            session_id="session-1",
        )
        config = MemoryContextConfig(
            token_budget=180,
            max_chars_per_memory=5000,
            retrieval=MemoryRetrievalConfig(top_k=1, candidate_k=5),
        )

        packet = self.builder.build(
            "frequency notes",
            project_id="project-1",
            session_id="session-1",
            config=config,
        )

        self.assertLessEqual(packet.estimated_tokens, 180)
        self.assertTrue(packet.truncated)
        self.assertEqual(len(packet.entries), 1)
        self.assertTrue(packet.entries[0].truncated)
        self.assertTrue(packet.entries[0].rendered_content.endswith("..."))

    def test_budget_drops_lower_ranked_memories_before_overflow(self) -> None:
        for index in range(4):
            self.store.add_memory(
                kind="task_state",
                canonical_key=f"frequency_note_{index}",
                content=f"frequency note {index} " + ("detail " * 100),
                project_id="project-1",
                session_id="session-1",
                importance=1.0 - index * 0.1,
            )
        config = MemoryContextConfig(
            token_budget=150,
            max_chars_per_memory=1000,
            retrieval=MemoryRetrievalConfig(top_k=4, candidate_k=10),
        )

        packet = self.builder.build(
            "frequency note",
            project_id="project-1",
            session_id="session-1",
            config=config,
        )

        self.assertLess(len(packet.entries), 4)
        self.assertTrue(packet.truncated)
        self.assertLessEqual(estimate_tokens(packet.text), 150)

    def test_token_estimator_is_conservative_for_chinese(self) -> None:
        chinese = "这是一个用于测试记忆预算的中文句子"
        english = "This is a short English sentence."

        self.assertGreaterEqual(estimate_tokens(chinese), len(chinese))
        self.assertLess(estimate_tokens(english), len(english))

    def test_invalid_context_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MemoryContextConfig(token_budget=10).validate()


if __name__ == "__main__":
    unittest.main()
