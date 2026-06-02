from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_rag.cli_io import configure_utf8_stdio
from paper_rag.metadata_search import load_paper_cards


COMPARISON_FIELDS = [
    "paper_id",
    "title",
    "year",
    "venue",
    "task",
    "method_keywords",
    "datasets",
    "metrics",
    "baselines",
    "limitations",
    "summary",
]
KEYWORD_FIELDS = [
    "title",
    "task",
    "method_keywords",
    "datasets",
    "metrics",
    "baselines",
    "summary",
    "limitations",
]
COMPACT_MARKDOWN_FIELDS = ["paper_id", "title", "year", "venue", "task", "method_keywords", "datasets", "metrics"]
VERBOSE_MARKDOWN_FIELDS = COMPARISON_FIELDS


def value_contains(value: object, needle: str) -> bool:
    lowered = needle.lower()
    if isinstance(value, list):
        return any(lowered in str(item or "").lower() for item in value)
    return lowered in str(value or "").lower()


def normalize_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or value == "":
        return []
    return [str(value)]


def normalize_card(card: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in COMPARISON_FIELDS:
        value = card.get(field, [] if field in {"method_keywords", "datasets", "metrics", "baselines"} else "")
        if field == "title" and not value:
            value = card.get("title_guess", "")
        if field in {"method_keywords", "datasets", "metrics", "baselines"}:
            normalized[field] = normalize_list(value)
        else:
            normalized[field] = "" if value is None else str(value)
    return normalized


def matches_filters(
    card: dict[str, Any],
    keyword: str | None = None,
    dataset: str | None = None,
    metric: str | None = None,
    year: int | str | None = None,
    venue: str | None = None,
    paper_ids: list[str] | None = None,
) -> bool:
    if paper_ids and str(card.get("paper_id", "")) not in set(paper_ids):
        return False
    if year is not None and str(card.get("year", "")) != str(year):
        return False
    if venue and not value_contains(card.get("venue"), venue):
        return False
    if dataset and not value_contains(card.get("datasets"), dataset):
        return False
    if metric and not value_contains(card.get("metrics"), metric):
        return False
    if keyword and not any(value_contains(card.get(field), keyword) for field in KEYWORD_FIELDS):
        return False
    return True


def compare_papers(
    cards_path: str | Path | None = None,
    keyword: str | None = None,
    dataset: str | None = None,
    metric: str | None = None,
    year: int | str | None = None,
    venue: str | None = None,
    paper_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0.")

    cards = load_paper_cards(cards_path)
    results: list[dict[str, Any]] = []
    for card in cards:
        if not matches_filters(
            card,
            keyword=keyword,
            dataset=dataset,
            metric=metric,
            year=year,
            venue=venue,
            paper_ids=paper_ids,
        ):
            continue

        results.append(normalize_card(card))
        if limit is not None and len(results) >= limit:
            break
    return results


def markdown_escape(value: object) -> str:
    text = format_value(value)
    return text.replace("|", "\\|").replace("\n", " ")


def format_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def truncate_text(text: str, max_length: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def format_markdown_table(papers: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    rows = []
    for paper in papers:
        rows.append(
            "| "
            + " | ".join(markdown_escape(truncate_text(format_value(paper.get(field, "")))) for field in fields)
            + " |"
        )
    return "\n".join([header, separator, *rows])


def format_verbose_details(papers: list[dict[str, Any]]) -> str:
    blocks = []
    for paper in papers:
        paper_id = paper.get("paper_id", "")
        title = paper.get("title", "")
        details = []
        for field in ("baselines", "summary", "limitations"):
            value = format_value(paper.get(field, ""))
            if value:
                details.append(f"- {field}: {value}")
        if details:
            blocks.append("\n".join([f"### {paper_id}: {title}".strip(), *details]))
    return "\n\n".join(blocks)


def format_comparison_markdown(papers: list[dict[str, Any]], verbose: bool = False) -> str:
    if not papers:
        return "No matching papers found."

    fields = VERBOSE_MARKDOWN_FIELDS if verbose else COMPACT_MARKDOWN_FIELDS
    output = [format_markdown_table(papers, fields)]
    if not verbose:
        details = format_verbose_details(papers)
        if details:
            output.extend(["", "## Details", "", details])
    return "\n".join(output)


def format_comparison_json(papers: list[dict[str, Any]]) -> str:
    return json.dumps(papers, ensure_ascii=False, indent=2)


def write_output(content: str, output_path: str | Path | None) -> None:
    if output_path is None:
        print(content)
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote comparison to: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare paper cards using metadata only.")
    parser.add_argument(
        "--cards_path",
        type=Path,
        default=None,
        help=(
            "Paper cards JSONL path. Defaults to the best available file under paper_rag/storage: "
            "paper_cards_cleaned.jsonl, paper_cards_enriched.jsonl, then paper_cards.jsonl."
        ),
    )
    parser.add_argument("--keyword", default=None, help="Case-insensitive keyword filter.")
    parser.add_argument("--dataset", default=None, help="Case-insensitive dataset filter.")
    parser.add_argument("--metric", default=None, help="Case-insensitive metric filter.")
    parser.add_argument("--year", default=None, help="Year filter, for example 2025.")
    parser.add_argument("--venue", default=None, help="Case-insensitive venue filter.")
    parser.add_argument("--paper_id", nargs="+", default=None, help="One or more paper ids.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of matching papers to output.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format.")
    parser.add_argument("--verbose", action="store_true", help="Include all comparison fields in Markdown output.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        papers = compare_papers(
            cards_path=args.cards_path,
            keyword=args.keyword,
            dataset=args.dataset,
            metric=args.metric,
            year=args.year,
            venue=args.venue,
            paper_ids=args.paper_id,
            limit=args.limit,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    if not papers:
        print("No matching papers found.")
        return 0

    content = format_comparison_json(papers) if args.format == "json" else format_comparison_markdown(papers, args.verbose)
    write_output(content, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
