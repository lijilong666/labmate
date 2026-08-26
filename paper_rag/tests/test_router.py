from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.router import paper_query, route_query


class RouterTests(unittest.TestCase):
    def test_auto_routes_metadata_query_and_extracts_filters(self) -> None:
        result = route_query("which papers from CVPR 2025 use F1?")

        self.assertEqual(result["mode"], "metadata")
        self.assertEqual(result["filters"]["year"], "2025")
        self.assertEqual(result["filters"]["venue"], "CVPR")
        self.assertEqual(result["filters"]["metric"], "F1")

    def test_auto_routes_search_and_answer_intents(self) -> None:
        self.assertEqual(route_query("检索 frequency features")["mode"], "search")
        self.assertEqual(route_query("解释 frequency features")["mode"], "answer")
        self.assertEqual(route_query("What is frequency analysis?")["mode"], "answer")

    def test_explicit_mode_is_respected(self) -> None:
        result = route_query("explain papers from 2024", mode="metadata")

        self.assertEqual(result["mode"], "metadata")
        self.assertEqual(result["filters"]["year"], "2024")

    def test_invalid_and_empty_queries_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            route_query("query", mode="invalid")
        with self.assertRaises(ValueError):
            paper_query("   ", use_cache=False)

    @patch("paper_rag.router.search_paper_cards")
    def test_metadata_mode_preserves_card_results(self, search_cards) -> None:
        cards = [{"paper_id": "p000001", "title": "A paper"}]
        search_cards.return_value = cards

        result = paper_query(
            "which papers from 2025 use LoRA?",
            mode="metadata",
            cards="cards.jsonl",
            use_cache=False,
        )

        self.assertEqual(result["selected_mode"], "metadata")
        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["results"], cards)
        self.assertEqual(result["answer"], "Found 1 matching paper card(s).")
        self.assertFalse(result["observability"]["cache_hit"])
        self.assertEqual(result["observability"]["result_count"], 1)
        self.assertGreaterEqual(result["observability"]["total_ms"], 0.0)
        self.assertEqual(
            set(result["observability"]["stages_ms"]),
            {
                "memory_prepare_ms",
                "cache_lookup_ms",
                "rag_ms",
                "memory_write_ms",
                "cache_write_ms",
            },
        )
        search_cards.assert_called_once()

    @patch("paper_rag.router.search_papers")
    def test_search_mode_preserves_chunk_provenance(self, search_chunks) -> None:
        chunks = [
            {
                "chunk_id": "p000001-c0001",
                "paper_id": "p000001",
                "source_file": "paper.pdf",
                "page_number": 3,
                "text": "evidence",
            }
        ]
        search_chunks.return_value = chunks

        result = paper_query("retrieve evidence", mode="search", use_cache=False)

        self.assertEqual(result["selected_mode"], "search")
        self.assertEqual(result["results"], chunks)
        self.assertEqual(result["results"][0]["chunk_id"], "p000001-c0001")

    @patch("paper_rag.router.ask_papers")
    def test_answer_mode_preserves_qa_answer_and_evidence(self, ask) -> None:
        evidence = [{"chunk_id": "chunk-1", "paper_id": "paper-1"}]
        ask.return_value = {
            "answer": "Grounded answer.\n\nSources:\n[1] paper.pdf",
            "evidence": evidence,
            "search_query": "grounded search query",
        }

        result = paper_query("explain the method", mode="answer", use_cache=False)

        self.assertEqual(result["selected_mode"], "answer")
        self.assertEqual(result["answer"], ask.return_value["answer"])
        self.assertEqual(result["results"], evidence)
        self.assertEqual(result["search_query"], "grounded search query")

    @patch("paper_rag.router.search_papers")
    def test_exact_query_cache_skips_retrieval_on_second_call(self, search_chunks) -> None:
        search_chunks.return_value = [{"chunk_id": "chunk-1", "text": "evidence"}]

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "query_cache.jsonl"
            first = paper_query(
                "retrieve evidence",
                mode="search",
                query_cache=cache_path,
            )
            second = paper_query(
                "retrieve evidence",
                mode="search",
                query_cache=cache_path,
            )

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertTrue(second["observability"]["cache_hit"])
        self.assertEqual(second["observability"]["recalled_memory_count"], 0)
        self.assertEqual(second["results"], first["results"])
        search_chunks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
