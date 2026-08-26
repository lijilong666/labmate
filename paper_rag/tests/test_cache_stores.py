from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.query_cache import (
    append_query_cache,
    build_paper_revision,
    build_request_fingerprint,
    find_cached_query,
    load_query_cache,
)
from paper_rag.topic_cache_store import find_cached_topic, load_topic_cache, upsert_topic_cache


class QueryCacheTests(unittest.TestCase):
    def test_append_and_exact_lookup_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.jsonl"
            append_query_cache(
                query="Exact Query",
                mode="search",
                answer="Found 1 chunk.",
                results=[{"chunk_id": "chunk-1"}],
                cache_path=path,
                search_query="exact query",
                filters={"year": "2025"},
            )

            record = find_cached_query("Exact Query", path)

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["mode"], "search")
            self.assertEqual(record["results"][0]["chunk_id"], "chunk-1")
            self.assertEqual(record["filters"], {"year": "2025"})
            self.assertIsNone(find_cached_query("exact query", path))

    def test_missing_cache_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.jsonl"
            self.assertEqual(load_query_cache(path), [])
            self.assertIsNone(find_cached_query("query", path))

    def test_invalid_query_cache_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.jsonl"
            path.write_text('{}\n{"broken"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                load_query_cache(path)

    def test_composite_cache_key_matches_all_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.jsonl"
            key = {
                "project_id": "project-1",
                "session_id": "session-1",
                "memory_revision": 3,
                "paper_revision": "paper-v1",
                "request_fingerprint": "request-v1",
            }
            append_query_cache(
                query="same query",
                mode="answer",
                answer="versioned answer",
                results=[],
                cache_path=path,
                cache_key=key,
            )

            self.assertEqual(find_cached_query("same query", path, cache_key=key)["answer"], "versioned answer")
            changed = dict(key, memory_revision=4)
            self.assertIsNone(find_cached_query("same query", path, cache_key=changed))
            changed = dict(key, session_id="session-2")
            self.assertIsNone(find_cached_query("same query", path, cache_key=changed))
            changed = dict(key, paper_revision="paper-v2")
            self.assertIsNone(find_cached_query("same query", path, cache_key=changed))

    def test_latest_matching_record_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.jsonl"
            append_query_cache("query", "search", "old", [], path)
            append_query_cache("query", "search", "new", [], path)

            self.assertEqual(find_cached_query("query", path)["answer"], "new")

    def test_request_fingerprint_is_order_independent(self) -> None:
        first = build_request_fingerprint({"top_k": 5, "mode": "answer"})
        second = build_request_fingerprint({"mode": "answer", "top_k": 5})
        changed = build_request_fingerprint({"mode": "answer", "top_k": 8})

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_paper_revision_changes_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.faiss").write_bytes(b"index")
            (root / "metadata.jsonl").write_text("{}\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text('{"version":1}', encoding="utf-8")
            first = build_paper_revision(mode="search", cards_path="unused", index_dir=root)
            manifest.write_text('{"version":2}', encoding="utf-8")
            second = build_paper_revision(mode="search", cards_path="unused", index_dir=root)

            self.assertNotEqual(first, second)


class TopicCacheTests(unittest.TestCase):
    @staticmethod
    def record(topic: str, answer: str, *, created_at: str | None = None) -> dict[str, object]:
        record: dict[str, object] = {
            "topic": topic,
            "query": "Explain the topic",
            "answer_language": "en",
            "answer": answer,
            "sources": [{"chunk_id": "chunk-1"}],
            "updated_at": "2026-01-02T00:00:00+00:00",
            "model": "test-model",
            "top_k": 5,
        }
        if created_at is not None:
            record["created_at"] = created_at
        return record

    def test_upsert_updates_in_place_and_preserves_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topics.jsonl"
            upsert_topic_cache(
                self.record("frequency", "old", created_at="2026-01-01T00:00:00+00:00"),
                path,
            )
            upsert_topic_cache(self.record("frequency", "new"), path)

            records = load_topic_cache(path)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["answer"], "new")
            self.assertEqual(records[0]["created_at"], "2026-01-01T00:00:00+00:00")

    def test_topic_lookup_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topics.jsonl"
            upsert_topic_cache(self.record("Frequency", "answer"), path)

            self.assertIsNotNone(find_cached_topic("Frequency", path))
            self.assertIsNone(find_cached_topic("frequency", path))

    def test_empty_topic_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topics.jsonl"
            with self.assertRaises(ValueError):
                upsert_topic_cache({"topic": "   "}, path)

    def test_invalid_topic_cache_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topics.jsonl"
            path.write_text(json.dumps({"topic": "ok"}) + "\nnot-json\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                load_topic_cache(path)


if __name__ == "__main__":
    unittest.main()
