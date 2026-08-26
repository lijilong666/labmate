from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import (
    MemoryRetrievalConfig,
    MemoryRetriever,
    MemorySource,
    MemoryStore,
)


class MemoryRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.directory.name) / "memory.sqlite3")
        self.store.create_session("session-1", "project-1")
        self.store.create_session("session-2", "project-1")
        self.store.create_session("session-3", "project-2")
        self.retriever = MemoryRetriever(self.store)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def add_fact(
        self,
        content: str,
        *,
        project_id: str = "project-1",
        session_id: str | None = "session-1",
        key: str = "preference",
        observed_at: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        sources: tuple[MemorySource, ...] = (),
    ):
        return self.store.add_memory(
            kind="user_fact",
            canonical_key=key,
            content=content,
            project_id=project_id,
            session_id=session_id,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
            importance=importance,
            sources=sources,
        )

    def test_session_recall_includes_project_global_but_not_other_sessions(self) -> None:
        local = self.add_fact("Frequency preference for session one", key="local")
        global_item = self.add_fact(
            "Frequency preference shared in project",
            session_id=None,
            key="global",
        )
        self.add_fact(
            "Frequency preference from another session",
            session_id="session-2",
            key="other",
        )
        self.add_fact(
            "Frequency preference from another project",
            project_id="project-2",
            session_id="session-3",
            key="external",
        )

        results = self.retriever.retrieve(
            "frequency preference",
            project_id="project-1",
            session_id="session-1",
        )

        ids = {result.item.memory_id for result in results}
        self.assertEqual(ids, {local.memory_id, global_item.memory_id})
        scope_reasons = {reason for result in results for reason in result.reasons}
        self.assertIn("session-scoped memory", scope_reasons)
        self.assertIn("project-global memory", scope_reasons)

    def test_recall_without_session_sees_only_project_global_memory(self) -> None:
        global_item = self.add_fact(
            "Global frequency preference",
            session_id=None,
            key="global",
        )
        self.add_fact("Session frequency preference", key="local")

        results = self.retriever.retrieve(
            "frequency preference",
            project_id="project-1",
            session_id=None,
        )

        self.assertEqual([result.item.memory_id for result in results], [global_item.memory_id])

    def test_retrieval_is_explainable(self) -> None:
        source = MemorySource(
            source_type="paper_chunk",
            paper_id="p000001",
            page_number=3,
            chunk_id="chunk-1",
        )
        self.add_fact("Frequency method preference", sources=(source,))

        result = self.retriever.retrieve(
            "frequency method",
            project_id="project-1",
            session_id="session-1",
        )[0]
        payload = result.to_dict()

        self.assertEqual(result.rank, 1)
        self.assertEqual(result.match_source, "fts5")
        self.assertEqual(
            set(result.score_components),
            {"lexical", "scope", "confidence", "importance", "source_quality", "recency"},
        )
        self.assertEqual(result.score_components["source_quality"], 1.0)
        self.assertEqual(payload["sources"][0]["chunk_id"], "chunk-1")

    def test_configurable_value_ranking_can_prioritize_importance(self) -> None:
        low = self.add_fact("Frequency ranking low importance", key="low", importance=0.1)
        high = self.add_fact("Frequency ranking high importance", key="high", importance=0.9)
        config = MemoryRetrievalConfig(
            top_k=2,
            candidate_k=10,
            lexical_weight=0.0,
            scope_weight=0.0,
            confidence_weight=0.0,
            importance_weight=1.0,
            source_weight=0.0,
            recency_weight=0.0,
        )

        results = self.retriever.retrieve(
            "frequency ranking",
            project_id="project-1",
            session_id="session-1",
            config=config,
        )

        self.assertEqual([result.item.memory_id for result in results], [high.memory_id, low.memory_id])

    def test_validity_interval_and_future_observation_are_filtered(self) -> None:
        visible = self.add_fact(
            "Temporal frequency visible",
            key="visible",
            observed_at="2026-01-01T00:00:00+00:00",
            valid_from="2026-01-01T00:00:00+00:00",
            valid_to="2026-12-31T00:00:00+00:00",
        )
        self.add_fact(
            "Temporal frequency expired",
            key="expired",
            observed_at="2025-01-01T00:00:00+00:00",
            valid_to="2026-01-01T00:00:00+00:00",
        )
        self.add_fact(
            "Temporal frequency future",
            key="future",
            observed_at="2027-01-01T00:00:00+00:00",
        )

        results = self.retriever.retrieve(
            "temporal frequency",
            project_id="project-1",
            session_id="session-1",
            as_of="2026-06-01T00:00:00+00:00",
        )

        self.assertEqual([result.item.memory_id for result in results], [visible.memory_id])

    def test_historical_recall_returns_version_valid_at_as_of_time(self) -> None:
        old = self.add_fact(
            "Answer language was English",
            key="answer_language",
            observed_at="2026-01-01T00:00:00+00:00",
        )
        new = self.store.supersede_memory(
            old.memory_id,
            content="Answer language is Chinese",
            observed_at="2026-07-01T00:00:00+00:00",
            sources=[MemorySource(source_type="user")],
        )
        historical = MemoryRetrievalConfig(include_history=True)

        past = self.retriever.retrieve(
            "answer language",
            project_id="project-1",
            session_id="session-1",
            as_of="2026-03-01T00:00:00+00:00",
            config=historical,
        )
        current = self.retriever.retrieve(
            "answer language",
            project_id="project-1",
            session_id="session-1",
            as_of="2026-08-01T00:00:00+00:00",
            config=historical,
        )

        self.assertEqual([result.item.memory_id for result in past], [old.memory_id])
        self.assertEqual([result.item.memory_id for result in current], [new.memory_id])

    def test_archived_memory_requires_explicit_retrieval_option(self) -> None:
        item = self.add_fact("Archived frequency preference")
        self.store.archive_memory(item.memory_id)

        normal = self.retriever.retrieve(
            "archived frequency",
            project_id="project-1",
            session_id="session-1",
        )
        audit = self.retriever.retrieve(
            "archived frequency",
            project_id="project-1",
            session_id="session-1",
            config=MemoryRetrievalConfig(include_archived=True),
        )

        self.assertEqual(normal, [])
        self.assertEqual([result.item.memory_id for result in audit], [item.memory_id])

    def test_kind_top_k_and_min_score_filters(self) -> None:
        self.add_fact("Frequency preference fact", key="fact")
        self.store.add_memory(
            kind="episode",
            content="Frequency preference episode",
            project_id="project-1",
            session_id="session-1",
            metadata={"route": "search"},
        )
        config = MemoryRetrievalConfig(
            top_k=1,
            candidate_k=5,
            kinds=("user_fact",),
            min_score=0.1,
        )

        results = self.retriever.retrieve(
            "frequency preference",
            project_id="project-1",
            session_id="session-1",
            config=config,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].item.kind, "user_fact")

    def test_cjk_recall_reports_substring_fallback(self) -> None:
        self.add_fact("用户希望系统始终使用中文回答问题。", key="answer_language")

        result = self.retriever.retrieve(
            "中文回答",
            project_id="project-1",
            session_id="session-1",
        )[0]

        self.assertEqual(result.match_source, "substring")

    def test_invalid_config_and_time_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MemoryRetrievalConfig(top_k=5, candidate_k=2).validate()
        with self.assertRaises(ValueError):
            self.retriever.retrieve(
                "query",
                project_id="project-1",
                as_of="not-a-time",
            )
        with self.assertRaises(ValueError):
            self.retriever.retrieve(" ", project_id="project-1")


if __name__ == "__main__":
    unittest.main()
