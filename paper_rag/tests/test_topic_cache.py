from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.topic_cache import get_topic_summary
from paper_rag.topic_cache_store import upsert_topic_cache


class TopicCacheIntegrationTests(unittest.TestCase):
    @patch("paper_rag.topic_cache.search_papers")
    @patch("paper_rag.topic_cache.OpenAICompatibleClient.from_env")
    def test_cache_hit_skips_model_and_retrieval(self, from_env, search) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "topics.jsonl"
            cached = {
                "topic": "frequency",
                "query": "old query",
                "answer_language": "en",
                "answer": "cached answer",
                "sources": [{"chunk_id": "chunk-old"}],
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "model": "old-model",
                "top_k": 8,
            }
            upsert_topic_cache(cached, cache_path)

            result = get_topic_summary(
                topic="frequency",
                query="new query",
                cache_path=cache_path,
                index_dir=Path(directory) / "missing-index",
            )

        self.assertTrue(result["cache_hit"])
        self.assertEqual(result["record"], cached)
        self.assertEqual(result["evidence"], [])
        from_env.assert_not_called()
        search.assert_not_called()

    @patch("paper_rag.topic_cache.search_papers")
    @patch("paper_rag.topic_cache.OpenAICompatibleClient.from_env")
    def test_cache_miss_retrieves_answers_and_persists_compact_sources(self, from_env, search) -> None:
        evidence = [
            {
                "rank": 1,
                "chunk_id": "chunk-1",
                "paper_id": "paper-1",
                "source_file": "papers/example.pdf",
                "page_number": 5,
                "text": "Evidence text.",
            }
        ]
        client = Mock()
        client.model = "test-model"
        client.chat.return_value = "Grounded summary [1]."
        from_env.return_value = client
        search.return_value = evidence

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_dir = root / "index"
            index_dir.mkdir()
            (index_dir / "index.faiss").touch()
            (index_dir / "metadata.jsonl").touch()
            cache_path = root / "topics.jsonl"

            first = get_topic_summary(
                topic="frequency",
                query="Explain frequency methods",
                cache_path=cache_path,
                index_dir=index_dir,
                rewrite_query=False,
            )
            second = get_topic_summary(
                topic="frequency",
                query="Different wording",
                cache_path=cache_path,
                index_dir=index_dir,
            )

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["record"]["sources"], [
            {
                "source_file": "papers/example.pdf",
                "page_number": 5,
                "chunk_id": "chunk-1",
            }
        ])
        self.assertIn("chunk_id=chunk-1", first["record"]["answer"])
        self.assertEqual(second["record"]["answer"], first["record"]["answer"])
        search.assert_called_once()

    def test_invalid_inputs_are_rejected_before_io(self) -> None:
        with self.assertRaises(ValueError):
            get_topic_summary(topic=" ", query="query")
        with self.assertRaises(ValueError):
            get_topic_summary(topic="topic", query=" ")
        with self.assertRaises(ValueError):
            get_topic_summary(topic="topic", query="query", top_k=0)


if __name__ == "__main__":
    unittest.main()
