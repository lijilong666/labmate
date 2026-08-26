from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.search import load_metadata, search_papers


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        return np.array([[0.6, 0.8]], dtype="float32")


class FakeIndex:
    ntotal = 2

    def search(self, embedding, top_k):
        assert embedding.dtype == np.float32
        assert embedding.flags["C_CONTIGUOUS"]
        return (
            np.array([[0.95, 0.72]], dtype="float32")[:, :top_k],
            np.array([[1, 0]], dtype="int64")[:, :top_k],
        )


class SearchContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = [
            {
                "chunk_id": "chunk-0",
                "paper_id": "paper-0",
                "source_path": "papers/zero.pdf",
                "file_name": "zero.pdf",
                "page_number": 2,
                "text": "zero text",
            },
            {
                "chunk_id": "chunk-1",
                "paper_id": "paper-1",
                "source_file": "papers/one.pdf",
                "file_name": "one.pdf",
                "page_number": 4,
                "text": "one text",
            },
        ]

    @patch("paper_rag.search.load_embedding_model")
    @patch("paper_rag.search.load_vector_store")
    def test_ranked_results_preserve_chunk_provenance(self, load_store, load_model) -> None:
        model = FakeModel()
        load_store.return_value = (FakeIndex(), self.metadata)
        load_model.return_value = model

        results = search_papers("test query", top_k=2)

        self.assertEqual([item["rank"] for item in results], [1, 2])
        self.assertEqual(results[0]["chunk_id"], "chunk-1")
        self.assertEqual(results[0]["paper_id"], "paper-1")
        self.assertEqual(results[0]["source_file"], "papers/one.pdf")
        self.assertEqual(results[0]["page_number"], 4)
        self.assertAlmostEqual(results[0]["score"], 0.95, places=5)
        self.assertEqual(results[1]["source_file"], "papers/zero.pdf")
        self.assertTrue(model.calls[0]["normalize_embeddings"])

    @patch("paper_rag.search.load_embedding_model")
    @patch("paper_rag.search.load_vector_store")
    def test_top_k_is_capped_by_index_size(self, load_store, load_model) -> None:
        load_store.return_value = (FakeIndex(), self.metadata)
        load_model.return_value = FakeModel()

        results = search_papers("test query", top_k=10)

        self.assertEqual(len(results), 2)

    def test_invalid_query_and_top_k_are_rejected_before_model_loading(self) -> None:
        with self.assertRaises(ValueError):
            search_papers("   ")
        with self.assertRaises(ValueError):
            search_papers("query", top_k=0)

    def test_metadata_loader_reports_invalid_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.jsonl"
            path.write_text('{"chunk_id":"ok"}\ninvalid\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                load_metadata(path)

    def test_metadata_loader_requires_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_metadata(Path(directory) / "missing.jsonl")


if __name__ == "__main__":
    unittest.main()
