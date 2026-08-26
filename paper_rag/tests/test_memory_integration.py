from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import MemorySource, MemoryStore, MemoryWriter
from paper_rag.memory.integration import prepare_query_memory, result_sources
from paper_rag.qa import generate_answer
from paper_rag.query_cache import append_query_cache
from paper_rag.router import paper_query


class MemoryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.memory_db = root / "memory.sqlite3"
        self.query_cache = root / "query_cache.jsonl"
        self.store = MemoryStore(self.memory_db)
        self.store.create_session("session-1", "project-1")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_pinned_task_state_contextualizes_anaphoric_query(self) -> None:
        MemoryWriter(self.store).set_task_state(
            canonical_key="selected_papers",
            content="p000001 and p000002",
            project_id="project-1",
            session_id="session-1",
        )

        preparation = prepare_query_memory(
            "continue comparing them",
            project_id="project-1",
            session_id="session-1",
            db_path=self.memory_db,
        )

        self.assertIn("selected_papers", preparation.contextualized_query)
        self.assertIn("p000001 and p000002", preparation.contextualized_query)
        self.assertIn("not paper evidence", preparation.prompt_context)
        self.assertEqual(preparation.packet.entries[0].retrieved.match_source, "pinned")

    def test_unrelated_query_is_not_rewritten_by_pinned_task_state(self) -> None:
        MemoryWriter(self.store).set_task_state(
            canonical_key="selected_papers",
            content="p000001 and p000002",
            project_id="project-1",
            session_id="session-1",
        )

        preparation = prepare_query_memory(
            "Explain an unrelated diffusion model",
            project_id="project-1",
            session_id="session-1",
            db_path=self.memory_db,
        )

        self.assertEqual(preparation.contextualized_query, "Explain an unrelated diffusion model")

    def test_user_preference_enters_prompt_context_not_retrieval_query(self) -> None:
        MemoryWriter(self.store).remember_user_fact(
            canonical_key="answer_language",
            content="The user prefers Chinese answers.",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )

        preparation = prepare_query_memory(
            "Explain frequency methods",
            project_id="project-1",
            session_id="session-1",
            db_path=self.memory_db,
        )

        self.assertEqual(preparation.contextualized_query, "Explain frequency methods")
        self.assertIn("prefers Chinese", preparation.prompt_context)

    @patch("paper_rag.router.search_papers")
    def test_memory_aware_search_uses_context_and_records_episode(self, search) -> None:
        MemoryWriter(self.store).set_task_state(
            canonical_key="selected_papers",
            content="p000001 and p000002",
            project_id="project-1",
            session_id="session-1",
        )
        search.return_value = [
            {
                "rank": 1,
                "score": 0.9,
                "paper_id": "p000001",
                "chunk_id": "chunk-1",
                "source_file": "papers/example.pdf",
                "page_number": 4,
                "text": "evidence",
            }
        ]

        result = paper_query(
            "continue retrieving them",
            mode="search",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            session_id="session-1",
            query_cache=self.query_cache,
        )

        retrieval_query = search.call_args.kwargs["query"]
        self.assertIn("p000001 and p000002", retrieval_query)
        self.assertIn("memory", result)
        self.assertFalse(result["cache_hit"])
        self.assertTrue(result["memory"]["episode_memory_id"])
        self.assertGreaterEqual(result["observability"]["recalled_memory_count"], 1)
        self.assertGreater(result["observability"]["memory_context_estimated_tokens"], 0)
        self.assertGreaterEqual(result["observability"]["stages_ms"]["memory_prepare_ms"], 0.0)
        episodes = MemoryStore(self.memory_db).list_memories(
            project_id="project-1",
            session_id="session-1",
            kinds=["episode"],
        )
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].sources[0].chunk_id, "chunk-1")

    @patch("paper_rag.router.search_papers")
    def test_memory_mode_bypasses_legacy_exact_cache(self, search) -> None:
        append_query_cache(
            query="retrieve evidence",
            mode="search",
            answer="stale cached answer",
            results=[{"chunk_id": "stale"}],
            cache_path=self.query_cache,
        )
        search.return_value = [
            {
                "paper_id": "p000001",
                "chunk_id": "fresh",
                "source_file": "paper.pdf",
                "page_number": 2,
                "text": "fresh evidence",
            }
        ]

        result = paper_query(
            "retrieve evidence",
            mode="search",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            session_id="session-1",
            query_cache=self.query_cache,
        )

        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["results"][0]["chunk_id"], "fresh")
        search.assert_called_once()

    @patch("paper_rag.router.search_papers")
    def test_memory_cache_hits_after_post_episode_revision_is_stored(self, search) -> None:
        search.return_value = [
            {
                "paper_id": "p000001",
                "chunk_id": "chunk-1",
                "source_file": "paper.pdf",
                "page_number": 2,
                "text": "evidence",
            }
        ]

        first = paper_query(
            "retrieve stable evidence",
            mode="search",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            session_id="session-1",
            query_cache=self.query_cache,
        )
        second = paper_query(
            "retrieve stable evidence",
            mode="search",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            session_id="session-1",
            query_cache=self.query_cache,
        )

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertIsNone(second["memory"]["episode_memory_id"])
        self.assertEqual(
            first["memory"]["cache_key_memory_revision"],
            second["memory"]["cache_key_memory_revision"],
        )
        self.assertEqual(len(MemoryStore(self.memory_db).list_memories(kinds=["episode"])), 1)
        search.assert_called_once()

    @patch("paper_rag.router.search_papers")
    def test_memory_update_invalidates_cached_result(self, search) -> None:
        search.return_value = [
            {
                "paper_id": "p000001",
                "chunk_id": "chunk-1",
                "source_file": "paper.pdf",
                "page_number": 2,
                "text": "evidence",
            }
        ]
        kwargs = dict(
            query="retrieve evidence",
            mode="search",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            session_id="session-1",
            query_cache=self.query_cache,
        )
        paper_query(**kwargs)
        cached = paper_query(**kwargs)
        MemoryWriter(MemoryStore(self.memory_db)).set_task_state(
            canonical_key="selected_papers",
            content="p000002",
            project_id="project-1",
            session_id="session-1",
        )
        refreshed = paper_query(**kwargs)

        self.assertTrue(cached["cache_hit"])
        self.assertFalse(refreshed["cache_hit"])
        self.assertEqual(search.call_count, 2)

    @patch("paper_rag.router.search_papers")
    def test_same_query_isolated_across_sessions(self, search) -> None:
        self.store.create_session("session-2", "project-1")
        search.return_value = [
            {
                "paper_id": "p000001",
                "chunk_id": "chunk-1",
                "source_file": "paper.pdf",
                "page_number": 2,
                "text": "evidence",
            }
        ]
        common = dict(
            query="retrieve evidence",
            mode="search",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            query_cache=self.query_cache,
        )

        first_session = paper_query(**common, session_id="session-1")
        second_session = paper_query(**common, session_id="session-2")

        self.assertFalse(first_session["cache_hit"])
        self.assertFalse(second_session["cache_hit"])
        self.assertEqual(search.call_count, 2)

    @patch("paper_rag.router.search_papers")
    def test_paper_revision_and_request_options_invalidate_cache(self, search) -> None:
        root = Path(self.directory.name) / "index"
        root.mkdir()
        (root / "index.faiss").write_bytes(b"index")
        (root / "metadata.jsonl").write_text("{}\n", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text('{"version":1}', encoding="utf-8")
        search.return_value = [
            {
                "paper_id": "p000001",
                "chunk_id": "chunk-1",
                "source_file": "paper.pdf",
                "page_number": 2,
                "text": "evidence",
            }
        ]
        kwargs = dict(
            query="retrieve evidence",
            mode="search",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            session_id="session-1",
            query_cache=self.query_cache,
            index_dir=root,
        )
        paper_query(**kwargs)
        self.assertTrue(paper_query(**kwargs)["cache_hit"])
        manifest.write_text('{"version":2}', encoding="utf-8")
        self.assertFalse(paper_query(**kwargs)["cache_hit"])
        self.assertFalse(paper_query(**dict(kwargs, top_k=8))["cache_hit"])
        self.assertEqual(search.call_count, 3)

    @patch("paper_rag.router.ask_papers")
    def test_answer_receives_memory_separately_from_paper_evidence(self, ask) -> None:
        MemoryWriter(self.store).remember_user_fact(
            canonical_key="answer_language",
            content="The user prefers Chinese answers.",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )
        ask.return_value = {
            "answer": "Grounded answer [1].\n\nSources:\n[1] paper.pdf",
            "search_query": "frequency method",
            "evidence": [
                {
                    "paper_id": "p000001",
                    "chunk_id": "chunk-1",
                    "source_file": "paper.pdf",
                    "page_number": 3,
                    "text": "paper evidence",
                }
            ],
        }

        paper_query(
            "Explain the method",
            mode="answer",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            session_id="session-1",
            use_cache=False,
        )

        kwargs = ask.call_args.kwargs
        self.assertEqual(kwargs["question"], "Explain the method")
        self.assertEqual(kwargs["answer_language"], "zh")
        self.assertIn("prefers Chinese", kwargs["memory_context"])
        self.assertNotIn("prefers Chinese", kwargs["retrieval_query"])

    @patch("paper_rag.router.ask_papers")
    def test_explicit_answer_language_overrides_memory_preference(self, ask) -> None:
        MemoryWriter(self.store).remember_user_fact(
            canonical_key="answer_language",
            content="The user prefers Chinese answers.",
            project_id="project-1",
            session_id="session-1",
            explicit_user_request=True,
        )
        ask.return_value = {
            "answer": "answer\n\nSources:\n[1] paper.pdf",
            "search_query": "query",
            "evidence": [
                {
                    "paper_id": "p000001",
                    "chunk_id": "chunk-1",
                    "source_file": "paper.pdf",
                    "page_number": 1,
                    "text": "evidence",
                }
            ],
        }

        paper_query(
            "Explain",
            mode="answer",
            answer_language="en",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            session_id="session-1",
            use_cache=False,
        )

        self.assertEqual(ask.call_args.kwargs["answer_language"], "en")

    def test_qa_prompt_labels_memory_as_non_evidence(self) -> None:
        client = Mock()
        client.chat.return_value = "answer"
        evidence = [
            {
                "rank": 1,
                "source_file": "paper.pdf",
                "page_number": 2,
                "chunk_id": "chunk-1",
                "text": "paper evidence",
            }
        ]

        generate_answer(
            "question",
            evidence,
            client,
            "en",
            memory_context="User preference only.",
        )

        messages = client.chat.call_args.args[0]
        self.assertIn("not scientific evidence", messages[0]["content"])
        self.assertIn("untrusted data", messages[0]["content"])
        self.assertIn("Memory context:\nUser preference only.", messages[1]["content"])
        self.assertIn("Evidence chunks:\n", messages[1]["content"])

    @patch("paper_rag.router.search_papers")
    def test_empty_search_records_insufficient_evidence_episode(self, search) -> None:
        search.return_value = []

        paper_query(
            "retrieve unknown evidence",
            mode="search",
            use_memory=True,
            memory_db=self.memory_db,
            project_id="project-1",
            session_id="session-1",
            use_cache=False,
        )

        episode = MemoryStore(self.memory_db).list_memories(kinds=["episode"])[0]
        self.assertEqual(episode.metadata["outcome"], "insufficient_evidence")
        self.assertFalse(episode.metadata["evidence_sufficient"])

    @patch("paper_rag.router.search_papers")
    def test_rag_failure_is_recorded_without_masking_original_error(self, search) -> None:
        search.side_effect = RuntimeError("index unavailable")

        with self.assertRaisesRegex(RuntimeError, "index unavailable"):
            paper_query(
                "retrieve evidence",
                mode="search",
                use_memory=True,
                memory_db=self.memory_db,
                project_id="project-1",
                session_id="session-1",
                use_cache=False,
            )

        episode = MemoryStore(self.memory_db).list_memories(kinds=["episode"])[0]
        self.assertEqual(episode.metadata["outcome"], "failed")
        self.assertIn("RuntimeError", episode.metadata["result_summary"])

    @patch("paper_rag.router.search_papers")
    def test_memory_requires_session_and_matching_project(self, search) -> None:
        with self.assertRaisesRegex(ValueError, "session_id"):
            paper_query(
                "retrieve",
                mode="search",
                use_memory=True,
                memory_db=self.memory_db,
                project_id="project-1",
            )
        with self.assertRaisesRegex(ValueError, "belongs to project"):
            paper_query(
                "retrieve",
                mode="search",
                use_memory=True,
                memory_db=self.memory_db,
                project_id="wrong-project",
                session_id="session-1",
            )
        search.assert_not_called()

    def test_result_source_conversion_deduplicates_and_preserves_provenance(self) -> None:
        rows = [
            {
                "paper_id": "p000001",
                "chunk_id": "chunk-1",
                "page_number": "4",
                "source_file": "paper.pdf",
            },
            {
                "paper_id": "p000001",
                "chunk_id": "chunk-1",
                "page_number": 4,
                "source_file": "paper.pdf",
            },
            {"paper_id": "p000002", "source_file": "card.pdf"},
        ]

        sources = result_sources(rows)

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].source_type, "paper_chunk")
        self.assertEqual(sources[0].page_number, 4)
        self.assertEqual(sources[1].source_type, "paper_card")


if __name__ == "__main__":
    unittest.main()
