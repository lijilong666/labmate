from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import MemoryOperation, MemoryPolicy, MemorySource


class MemoryPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = MemoryPolicy()

    def test_noop_requires_only_a_reason(self) -> None:
        operation = self.policy.validate(self.policy.noop("No explicit memory intent."))

        self.assertEqual(operation.action, "NOOP")

    def test_raw_user_fact_write_requires_explicit_request_and_user_source(self) -> None:
        base = dict(
            action="ADD",
            reason="remember",
            project_id="project-1",
            kind="user_fact",
            canonical_key="answer_language",
            content="Chinese",
        )
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.policy.validate(MemoryOperation(**base))
        with self.assertRaisesRegex(ValueError, "user source"):
            self.policy.validate(
                MemoryOperation(**base, metadata={"explicit_user_request": True})
            )

        validated = self.policy.validate(
            MemoryOperation(
                **base,
                metadata={"explicit_user_request": True},
                sources=(MemorySource(source_type="user"),),
            )
        )
        self.assertEqual(validated.kind, "user_fact")

    def test_update_requires_a_valid_deterministic_reason(self) -> None:
        base = dict(
            action="UPDATE",
            reason="update",
            project_id="project-1",
            target_id="mem-1",
            content="new value",
        )
        with self.assertRaisesRegex(ValueError, "explicit user correction or deterministic"):
            self.policy.validate(MemoryOperation(**base))
        with self.assertRaisesRegex(ValueError, "user source"):
            self.policy.validate(
                MemoryOperation(**base, metadata={"explicit_user_correction": True})
            )

    def test_archive_requires_explicit_user_request(self) -> None:
        operation = MemoryOperation(
            action="ARCHIVE",
            reason="forget",
            project_id="project-1",
            target_id="mem-1",
        )
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.policy.validate(operation)

    def test_research_claim_episode_requires_paper_chunk(self) -> None:
        operation = MemoryOperation(
            action="ADD",
            reason="episode",
            project_id="project-1",
            session_id="session-1",
            kind="episode",
            content="A research conclusion.",
            metadata={
                "query": "question",
                "route": "answer",
                "outcome": "success",
                "contains_research_claims": True,
            },
        )
        with self.assertRaisesRegex(ValueError, "paper_chunk"):
            self.policy.validate(operation)

        validated = self.policy.validate(
            MemoryOperation(
                **{key: value for key, value in operation.__dict__.items() if key != "sources"},
                sources=(
                    MemorySource(
                        source_type="paper_chunk",
                        paper_id="p000001",
                        chunk_id="chunk-1",
                        page_number=2,
                    ),
                ),
            )
        )
        self.assertEqual(validated.action, "ADD")

    def test_successful_answer_summary_cannot_bypass_chunk_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "paper_chunk"):
            self.policy.validate(
                MemoryOperation(
                    action="ADD",
                    reason="episode",
                    project_id="project-1",
                    session_id="session-1",
                    kind="episode",
                    content="Summary without evidence.",
                    metadata={
                        "query": "question",
                        "route": "answer",
                        "outcome": "success",
                        "result_summary": "A research conclusion.",
                        "contains_research_claims": False,
                    },
                )
            )


if __name__ == "__main__":
    unittest.main()
