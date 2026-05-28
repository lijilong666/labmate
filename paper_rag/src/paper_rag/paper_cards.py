from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_INVENTORY_PATH = Path("paper_rag/storage/paper_inventory.csv")
DEFAULT_PAPER_CARDS_PATH = Path("paper_rag/storage/paper_cards.jsonl")

VENUE_PATTERNS = [
    ("ACM MM", re.compile(r"\b(?:ACM[-_\s]*)?MM\b", re.IGNORECASE)),
    ("NeurIPS", re.compile(r"\b(?:NeurIPS|NIPS)\b", re.IGNORECASE)),
    ("CVPR", re.compile(r"\bCVPR\b", re.IGNORECASE)),
    ("ICCV", re.compile(r"\bICCV\b", re.IGNORECASE)),
    ("ECCV", re.compile(r"\bECCV\b", re.IGNORECASE)),
    ("AAAI", re.compile(r"\bAAAI\b", re.IGNORECASE)),
    ("ICLR", re.compile(r"\bICLR\b", re.IGNORECASE)),
    ("ICML", re.compile(r"\bICML\b", re.IGNORECASE)),
    ("IJCAI", re.compile(r"\bIJCAI\b", re.IGNORECASE)),
    ("ICASSP", re.compile(r"\bICASSP\b", re.IGNORECASE)),
    ("NAACL", re.compile(r"\bNAACL\b", re.IGNORECASE)),
    ("TIFS", re.compile(r"\bTIFS\b", re.IGNORECASE)),
]


def read_inventory(inventory_path: Path) -> list[dict[str, str]]:
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory file not found: {inventory_path}")

    with inventory_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def clean_title_guess(file_name: str) -> str:
    stem = Path(file_name).stem
    stem = re.sub(r"\barxiv\b", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\b\d{4}\.\d{4,5}v?\d*\b", " ", stem)
    stem = re.sub(r"\b\d{5}-Article Text-\d+-\d+-\d+-\d+\b", " ", stem, flags=re.IGNORECASE)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip(" ._-")
    return stem or Path(file_name).stem


def extract_year(*values: str) -> int | None:
    joined = " ".join(values)
    venue_year_match = re.search(
        r"\b(?:CVPR|ICCV|ECCV|AAAI|ICLR|ICML|IJCAI|ICASSP|NAACL|NeurIPS|NIPS|TIFS|MM)[-_\s]*(20\d{2})\b",
        joined,
        re.IGNORECASE,
    )
    if venue_year_match:
        return int(venue_year_match.group(1))

    year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", joined)
    if year_match:
        return int(year_match.group(1))
    return None


def extract_venue(*values: str) -> str:
    joined = " ".join(values)
    for venue, pattern in VENUE_PATTERNS:
        if pattern.search(joined):
            return venue
    return ""


def generate_paper_card(row: dict[str, str]) -> dict[str, object]:
    source_file = row.get("source_path", "")
    file_name = row.get("file_name", Path(source_file).name)
    parent_dir = row.get("parent_dir", "")
    title_guess = clean_title_guess(file_name)

    return {
        "paper_id": row.get("paper_id", ""),
        "title": title_guess,
        "title_guess": title_guess,
        "year": extract_year(parent_dir, file_name, source_file),
        "venue": extract_venue(parent_dir, file_name, source_file),
        "authors": [],
        "source_file": source_file,
        "file_name": file_name,
        "parent_dir": parent_dir,
        "task": "",
        "method_keywords": [],
        "datasets": [],
        "metrics": [],
        "baselines": [],
        "summary": "",
        "limitations": "",
        "status": "heuristic",
        "extraction_mode": "heuristic",
    }


def generate_paper_cards(
    inventory_path: str | Path = DEFAULT_INVENTORY_PATH,
    output_path: str | Path = DEFAULT_PAPER_CARDS_PATH,
    limit: int | None = None,
) -> int:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0.")

    inventory = read_inventory(Path(inventory_path))
    rows: Iterable[dict[str, str]] = inventory if limit is None else inventory[:limit]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            card = generate_paper_card(row)
            file.write(json.dumps(card, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate heuristic paper cards from an inventory CSV.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH, help="Input inventory CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PAPER_CARDS_PATH, help="Output paper cards JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of papers to process.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        count = generate_paper_cards(args.inventory, args.output, args.limit)
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    print(f"Wrote {count} heuristic paper card(s) to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
