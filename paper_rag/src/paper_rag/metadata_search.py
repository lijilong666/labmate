from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_rag.cli_io import configure_utf8_stdio
from paper_rag.paper_cards import DEFAULT_PAPER_CARDS_PATH


def load_paper_cards(cards_path: str | Path = DEFAULT_PAPER_CARDS_PATH) -> list[dict[str, Any]]:
    path = Path(cards_path)
    if not path.exists():
        raise FileNotFoundError(f"Paper cards file not found: {path}")

    cards: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cards.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid paper card JSON at line {line_number}: {exc}") from exc
    return cards


def text_contains(value: object, needle: str) -> bool:
    return needle.lower() in str(value or "").lower()


def list_contains(values: object, needle: str) -> bool:
    if isinstance(values, list):
        return any(text_contains(value, needle) for value in values)
    return text_contains(values, needle)


def search_paper_cards(
    cards_path: str | Path = DEFAULT_PAPER_CARDS_PATH,
    year: int | str | None = None,
    venue: str | None = None,
    keyword: str | None = None,
    dataset: str | None = None,
    metric: str | None = None,
    baseline: str | None = None,
    paper_id: str | None = None,
) -> list[dict[str, Any]]:
    cards = load_paper_cards(cards_path)
    results: list[dict[str, Any]] = []

    for card in cards:
        if year is not None and str(card.get("year", "")) != str(year):
            continue
        if venue and not text_contains(card.get("venue"), venue):
            continue
        if paper_id and str(card.get("paper_id", "")) != paper_id:
            continue
        if keyword and not (
            list_contains(card.get("method_keywords"), keyword)
            or list_contains(card.get("baselines"), keyword)
            or text_contains(card.get("title"), keyword)
            or text_contains(card.get("title_guess"), keyword)
            or text_contains(card.get("summary"), keyword)
        ):
            continue
        if dataset and not list_contains(card.get("datasets"), dataset):
            continue
        if metric and not list_contains(card.get("metrics"), metric):
            continue
        if baseline and not list_contains(card.get("baselines"), baseline):
            continue

        results.append(card)

    return results


def format_card(card: dict[str, Any]) -> str:
    return (
        f"paper_id={card.get('paper_id', '')} | "
        f"title={card.get('title', '')} | "
        f"year={card.get('year', '')} | "
        f"venue={card.get('venue', '')} | "
        f"status={card.get('status', '')}\n"
        f"source_file={card.get('source_file', '')}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search heuristic paper cards by metadata.")
    parser.add_argument("--cards", type=Path, default=DEFAULT_PAPER_CARDS_PATH, help="Paper cards JSONL path.")
    parser.add_argument("--year", default=None, help="Filter by year, for example 2025.")
    parser.add_argument("--venue", default=None, help="Filter by venue, for example CVPR.")
    parser.add_argument("--keyword", default=None, help="Filter by method keyword or title text.")
    parser.add_argument("--dataset", default=None, help="Filter by dataset name.")
    parser.add_argument("--metric", default=None, help="Filter by metric name.")
    parser.add_argument("--baseline", default=None, help="Filter by baseline name.")
    parser.add_argument("--paper_id", default=None, help="Filter by paper id, for example p000001.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        results = search_paper_cards(
            cards_path=args.cards,
            year=args.year,
            venue=args.venue,
            keyword=args.keyword,
            dataset=args.dataset,
            metric=args.metric,
            baseline=args.baseline,
            paper_id=args.paper_id,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    if not results:
        print("No matching paper cards found.")
        return 0

    for index, card in enumerate(results, start=1):
        print(f"[{index}] {format_card(card)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
