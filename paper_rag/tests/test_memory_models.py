from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory.models import MemoryItem, MemorySession, MemorySource


class MemoryModelTests(unittest.TestCase):
    def test_session_requires_scope_and_json_state(self) -> None:
        with self.assertRaises(ValueError):
            MemorySession(session_id="", project_id="project").validate()
        with self.assertRaises(ValueError):
            MemorySession(session_id="session", project_id="").validate()
        with self.assertRaises(ValueError):
            MemorySession(
                session_id="session",
                project_id="project",
                state={"invalid": {1, 2}},
            ).validate()

    def test_memory_kind_specific_requirements(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical_key"):
            MemoryItem(
                memory_id="mem-1",
                kind="user_fact",
                content="Prefers Chinese answers.",
                project_id="project",
            ).validate()
        with self.assertRaisesRegex(ValueError, "session_id"):
            MemoryItem(
                memory_id="mem-2",
                kind="episode",
                content="A retrieval episode.",
                project_id="project",
            ).validate()

    def test_memory_scores_are_bounded(self) -> None:
        with self.assertRaises(ValueError):
            MemoryItem(
                memory_id="mem-1",
                kind="user_fact",
                canonical_key="answer_language",
                content="Chinese",
                project_id="project",
                confidence=1.1,
            ).validate()

    def test_paper_sources_require_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "paper_id and chunk_id"):
            MemorySource(source_type="paper_chunk", paper_id="p000001").validate()
        with self.assertRaisesRegex(ValueError, "paper_id"):
            MemorySource(source_type="paper_card").validate()
        with self.assertRaises(ValueError):
            MemorySource(
                source_type="paper_chunk",
                paper_id="p000001",
                chunk_id="chunk-1",
                page_number=0,
            ).validate()


if __name__ == "__main__":
    unittest.main()
