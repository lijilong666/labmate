from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.qa import (
    ask_papers,
    build_evidence_context,
    citation_lines,
    format_answer,
    resolve_answer_language,
)


class QaContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = [
            {
                "rank": 1,
                "score": 0.91,
                "chunk_id": "p000001-c0001",
                "paper_id": "p000001",
                "source_file": "papers/example.pdf",
                "page_number": 7,
                "text": "Evidence text.",
            }
        ]

    def test_answer_language_follows_question(self) -> None:
        self.assertEqual(resolve_answer_language("这是什么？", "auto"), "zh")
        self.assertEqual(resolve_answer_language("What is this?", "auto"), "en")
        self.assertEqual(resolve_answer_language("What is this?", "zh"), "zh")

    def test_evidence_context_and_citations_keep_source_identifiers(self) -> None:
        context = build_evidence_context(self.evidence)
        citations = citation_lines(self.evidence)

        self.assertIn("source_file=papers/example.pdf", context)
        self.assertIn("page=7", context)
        self.assertIn("chunk_id=p000001-c0001", context)
        self.assertEqual(
            citations,
            ["[1] papers/example.pdf, page 7, chunk_id=p000001-c0001"],
        )

    def test_format_answer_always_adds_sources_section(self) -> None:
        answer = format_answer("Grounded answer [1].", citation_lines(self.evidence))

        self.assertEqual(
            answer,
            "Grounded answer [1].\n\nSources:\n[1] papers/example.pdf, page 7, chunk_id=p000001-c0001",
        )

    @patch("paper_rag.qa.search_papers", return_value=[])
    @patch("paper_rag.qa.OpenAICompatibleClient.from_env")
    def test_no_retrieval_returns_insufficient_evidence_without_chat(self, from_env, search) -> None:
        client = Mock()
        from_env.return_value = client

        result = ask_papers("What is missing?", rewrite_query=False)

        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["citations"], [])
        self.assertIn("evidence is insufficient", result["answer"])
        self.assertTrue(result["answer"].endswith("Sources:"))
        client.chat.assert_not_called()
        search.assert_called_once()

    @patch("paper_rag.qa.search_papers")
    @patch("paper_rag.qa.OpenAICompatibleClient.from_env")
    def test_answer_result_preserves_evidence_and_citations(self, from_env, search) -> None:
        client = Mock()
        client.chat.return_value = "Grounded answer [1]."
        from_env.return_value = client
        search.return_value = self.evidence

        result = ask_papers("Explain the evidence", rewrite_query=False)

        self.assertEqual(result["evidence"], self.evidence)
        self.assertEqual(result["citations"], citation_lines(self.evidence))
        self.assertIn("Grounded answer [1].", result["answer"])
        self.assertIn("chunk_id=p000001-c0001", result["answer"])
        client.chat.assert_called_once()

    @patch("paper_rag.qa.search_papers")
    @patch("paper_rag.qa.OpenAICompatibleClient.from_env")
    def test_non_english_query_rewrite_is_used_for_retrieval(self, from_env, search) -> None:
        client = Mock()
        client.chat.side_effect = ["frequency domain localization", "有依据的回答 [1]。"]
        from_env.return_value = client
        search.return_value = self.evidence

        result = ask_papers("频域方法有什么作用？", rewrite_query=True)

        self.assertEqual(result["search_query"], "frequency domain localization")
        search.assert_called_once_with(
            query="frequency domain localization",
            top_k=5,
            index_dir=Path("paper_rag/storage/vector_store"),
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=Path("paper_rag/model_cache"),
        )
        self.assertEqual(client.chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
