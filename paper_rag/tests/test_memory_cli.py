from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.memory import MemoryStore
from paper_rag.memory_cli import main


class MemoryCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.directory.name) / "memory.sqlite3"
        MemoryStore(self.db_path).create_session("session-1", "project-1")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def invoke(self, *arguments: str) -> tuple[int, object, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["--db", str(self.db_path), *arguments])
        payload = json.loads(stdout.getvalue()) if stdout.getvalue() else None
        return code, payload, stderr.getvalue()

    def test_add_list_search_show_correct_and_archive(self) -> None:
        code, added, _ = self.invoke(
            "add",
            "--project_id",
            "project-1",
            "--session_id",
            "session-1",
            "--kind",
            "user_fact",
            "--key",
            "answer_language",
            "--content",
            "Prefer Chinese answers",
        )
        self.assertEqual(code, 0)
        memory_id = added["item"]["memory_id"]

        code, listed, _ = self.invoke(
            "list", "--project_id", "project-1", "--session_id", "session-1"
        )
        self.assertEqual(code, 0)
        self.assertEqual([item["memory_id"] for item in listed], [memory_id])

        code, searched, _ = self.invoke(
            "search",
            "Chinese answers",
            "--project_id",
            "project-1",
            "--session_id",
            "session-1",
        )
        self.assertEqual(code, 0)
        self.assertEqual(searched[0]["memory_id"], memory_id)

        code, shown, _ = self.invoke("show", memory_id)
        self.assertEqual(code, 0)
        self.assertEqual(shown["canonical_key"], "answer_language")

        code, corrected, _ = self.invoke(
            "correct",
            memory_id,
            "--project_id",
            "project-1",
            "--session_id",
            "session-1",
            "--content",
            "Prefer English answers",
        )
        self.assertEqual(code, 0)
        corrected_id = corrected["item"]["memory_id"]
        self.assertNotEqual(corrected_id, memory_id)

        code, archived, _ = self.invoke(
            "archive",
            corrected_id,
            "--project_id",
            "project-1",
            "--session_id",
            "session-1",
        )
        self.assertEqual(code, 0)
        self.assertEqual(archived["item"]["status"], "archived")

    def test_task_state_requires_session_and_scope_mismatch_is_rejected(self) -> None:
        code, payload, error = self.invoke(
            "add",
            "--project_id",
            "project-1",
            "--kind",
            "task_state",
            "--key",
            "topic",
            "--content",
            "Memory systems",
        )
        self.assertEqual(code, 2)
        self.assertIsNone(payload)
        self.assertIn("requires --session_id", error)

    def test_consolidate_defaults_to_dry_run(self) -> None:
        code, report, _ = self.invoke(
            "consolidate",
            "--project_id",
            "project-1",
            "--session_id",
            "session-1",
        )
        self.assertEqual(code, 0)
        self.assertFalse(report["applied"])
        self.assertNotIn("memory_consolidation", MemoryStore(self.db_path).get_session("session-1").state)
