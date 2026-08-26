from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from paper_rag.metadata_search import load_paper_cards, search_paper_cards


class MetadataSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cards_path = Path(self.directory.name) / "cards.jsonl"
        cards = [
            {
                "paper_id": "p000001",
                "title": "Frequency Localization",
                "year": 2025,
                "venue": "CVPR",
                "method_keywords": ["frequency", "transformer"],
                "datasets": ["CASIA"],
                "metrics": ["F1", "IoU"],
                "baselines": ["CAT-Net"],
                "summary": "Localizes manipulated regions.",
            },
            {
                "paper_id": "p000002",
                "title": "Diffusion Detection",
                "year": 2024,
                "venue": "ICCV",
                "method_keywords": ["diffusion"],
                "datasets": ["COCO"],
                "metrics": ["AUC"],
                "baselines": ["ResNet"],
                "summary": "Detects generated images.",
            },
        ]
        self.cards_path.write_text(
            "".join(json.dumps(card) + "\n" for card in cards),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_combined_filters_select_expected_card(self) -> None:
        results = search_paper_cards(
            self.cards_path,
            year=2025,
            venue="cvpr",
            keyword="frequency",
            dataset="casia",
            metric="f1",
            baseline="cat-net",
        )

        self.assertEqual([card["paper_id"] for card in results], ["p000001"])

    def test_keyword_searches_title_summary_and_list_fields(self) -> None:
        self.assertEqual(
            search_paper_cards(self.cards_path, keyword="generated")[0]["paper_id"],
            "p000002",
        )
        self.assertEqual(
            search_paper_cards(self.cards_path, keyword="transformer")[0]["paper_id"],
            "p000001",
        )

    def test_invalid_card_json_reports_line_number(self) -> None:
        self.cards_path.write_text('{}\ninvalid\n', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "line 2"):
            load_paper_cards(self.cards_path)


if __name__ == "__main__":
    unittest.main()
