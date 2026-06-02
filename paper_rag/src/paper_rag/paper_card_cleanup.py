from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from paper_rag.cli_io import configure_utf8_stdio
from paper_rag.metadata_search import load_paper_cards
from paper_rag.paper_cards import DEFAULT_PAPER_CARDS_PATH, clean_title_guess


ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}v?\d*$", re.IGNORECASE)
ARXIV_ID_PREFIX_PATTERN = re.compile(r"^\d{4}\.\d{4,5}v?\d*(?:\b|[\s_-]+)", re.IGNORECASE)
ARTICLE_TEXT_PATTERN = re.compile(r"^\d{5}-Article Text-\d+-\d+-\d+-\d+$", re.IGNORECASE)
WEAK_TITLE_FIELDS = {"", "untitled", "unknown", "none", "null"}


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_title_text(value: object) -> str:
    return " ".join(str(value or "").replace("_", " ").split()).strip(" ._-")


def is_filename_like_title(title: object) -> bool:
    normalized = normalize_title_text(title)
    if normalized.lower() in WEAK_TITLE_FIELDS:
        return True
    if ARXIV_ID_PATTERN.fullmatch(normalized):
        return True
    if ARXIV_ID_PREFIX_PATTERN.match(normalized):
        return True
    if ARTICLE_TEXT_PATTERN.fullmatch(normalized):
        return True
    if normalized.lower().endswith(".pdf"):
        return True
    if re.fullmatch(r"[\d\W_]+", normalized):
        return True
    return False


def load_title_overrides(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}

    override_path = Path(path)
    if not override_path.exists():
        raise FileNotFoundError(f"Title overrides file not found: {override_path}")

    with override_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Title overrides JSON must be an object keyed by paper_id.")

    overrides: dict[str, str] = {}
    for paper_id, value in payload.items():
        if isinstance(value, dict):
            title = value.get("title", "")
        else:
            title = value
        title_text = normalize_title_text(title)
        if title_text:
            overrides[str(paper_id)] = title_text
    return overrides


def filename_title_candidate(card: dict[str, Any]) -> str:
    file_name = str(card.get("file_name") or Path(str(card.get("source_file", ""))).name)
    if not file_name:
        return ""
    candidate = clean_title_guess(file_name)
    candidate = ARXIV_ID_PREFIX_PATTERN.sub("", candidate).strip(" ._-")
    return "" if is_filename_like_title(candidate) else normalize_title_text(candidate)


def select_title_candidate(card: dict[str, Any]) -> tuple[str, str]:
    for field in ("title", "title_guess"):
        candidate = normalize_title_text(card.get(field))
        if candidate and not is_filename_like_title(candidate):
            return candidate, field

    candidate = filename_title_candidate(card)
    if candidate:
        return candidate, "file_name"

    return "", ""


def cleanup_one_card(card: dict[str, Any], title_overrides: dict[str, str]) -> dict[str, Any]:
    updated = dict(card)
    paper_id = str(updated.get("paper_id", ""))
    original_title = normalize_title_text(updated.get("title"))

    if paper_id in title_overrides:
        new_title = title_overrides[paper_id]
        if new_title != original_title:
            updated.setdefault("title_original", original_title)
        updated["title"] = new_title
        updated["title_cleanup_status"] = "updated"
        updated["title_cleanup_reason"] = "manual_override"
        return updated

    if not is_filename_like_title(original_title):
        updated["title_cleanup_status"] = "unchanged"
        updated["title_cleanup_reason"] = "title_already_usable"
        return updated

    candidate, source = select_title_candidate(updated)
    if candidate and candidate != original_title:
        updated.setdefault("title_original", original_title)
        updated["title"] = candidate
        if is_filename_like_title(updated.get("title_guess")):
            updated["title_guess"] = candidate
        updated["title_cleanup_status"] = "updated"
        updated["title_cleanup_reason"] = f"replaced_weak_title_from_{source}"
        return updated

    updated.setdefault("title_original", original_title)
    updated["title_cleanup_status"] = "needs_review"
    updated["title_cleanup_reason"] = "weak_title_no_better_candidate_without_pdf_or_llm"
    return updated


def cleanup_paper_cards(
    cards_path: str | Path = DEFAULT_PAPER_CARDS_PATH,
    output_path: str | Path | None = None,
    title_overrides_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0.")

    cards = load_paper_cards(cards_path)
    overrides = load_title_overrides(title_overrides_path)

    cleaned_cards: list[dict[str, Any]] = []
    stats = {"total": len(cards), "processed": 0, "copied": 0, "updated": 0, "unchanged": 0, "needs_review": 0}
    processed = 0
    for card in cards:
        if limit is not None and processed >= limit:
            cleaned = dict(card)
            stats["copied"] += 1
        else:
            cleaned = cleanup_one_card(card, overrides)
            processed += 1
            stats["processed"] += 1
            status = str(cleaned.get("title_cleanup_status", "unchanged"))
            if status in stats:
                stats[status] += 1

        cleaned_cards.append(cleaned)

    destination = output_path or cards_path
    write_jsonl(cleaned_cards, destination)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean weak paper-card title metadata without reading PDFs or calling an LLM.")
    parser.add_argument("--cards", type=Path, default=DEFAULT_PAPER_CARDS_PATH, help="Input paper cards JSONL path.")
    parser.add_argument("--output", type=Path, default=None, help="Output cleaned paper cards JSONL path.")
    parser.add_argument(
        "--title_overrides",
        type=Path,
        default=None,
        help="Optional JSON file mapping paper_id to corrected title.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N cards.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        stats = cleanup_paper_cards(
            cards_path=args.cards,
            output_path=args.output,
            title_overrides_path=args.title_overrides,
            limit=args.limit,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    destination = args.output or args.cards
    print(
        "Processed {processed}/{total} paper card(s): {updated} updated, {unchanged} unchanged, "
        "{needs_review} need review, {copied} copied without changes. Wrote {destination}.".format(
            destination=destination,
            **stats,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
