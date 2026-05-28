from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_rag.llm_client import DEFAULT_LLM_TIMEOUT, OpenAICompatibleClient
from paper_rag.paper_cards import DEFAULT_PAPER_CARDS_PATH


DEFAULT_CHUNKS_PATH = Path("paper_rag/storage/chunks.jsonl")
ENRICHMENT_FIELDS = [
    "task",
    "method_keywords",
    "datasets",
    "metrics",
    "baselines",
    "summary",
    "limitations",
]
LIST_FIELDS = {"method_keywords", "datasets", "metrics", "baselines"}
KEYWORD_HINTS = (
    "dataset",
    "datasets",
    "experiment",
    "experiments",
    "metric",
    "metrics",
    "baseline",
    "baselines",
    "evaluation",
    "result",
    "results",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def normalize_enrichment(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in ENRICHMENT_FIELDS:
        value = payload.get(field, [] if field in LIST_FIELDS else "")
        if field in LIST_FIELDS:
            if isinstance(value, list):
                normalized[field] = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                normalized[field] = [value.strip()]
            else:
                normalized[field] = []
        else:
            normalized[field] = str(value).strip() if value is not None else ""
    return normalized


def select_cards(
    cards: list[dict[str, Any]],
    paper_id: str | None = None,
    limit: int | None = None,
) -> set[str]:
    selected = []
    for card in cards:
        if paper_id and str(card.get("paper_id", "")) != paper_id:
            continue
        selected.append(str(card.get("paper_id", "")))
        if limit is not None and len(selected) >= limit:
            break
    return {item for item in selected if item}


def group_chunks_by_paper(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        paper_id = str(chunk.get("paper_id", ""))
        if paper_id:
            grouped.setdefault(paper_id, []).append(chunk)
    for paper_chunks in grouped.values():
        paper_chunks.sort(key=lambda item: (int(item.get("page_number", 0)), int(item.get("chunk_index", 0))))
    return grouped


def select_context_chunks(
    chunks: list[dict[str, Any]],
    max_context_chars: int = 12000,
    intro_pages: int = 5,
    max_keyword_chunks: int = 8,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk in chunks:
        page_number = int(chunk.get("page_number", 0))
        chunk_id = str(chunk.get("chunk_id", ""))
        if page_number <= intro_pages and chunk_id not in seen:
            selected.append(chunk)
            seen.add(chunk_id)

    keyword_count = 0
    for chunk in chunks:
        if keyword_count >= max_keyword_chunks:
            break
        chunk_id = str(chunk.get("chunk_id", ""))
        text = str(chunk.get("text", "")).lower()
        if chunk_id not in seen and any(keyword in text for keyword in KEYWORD_HINTS):
            selected.append(chunk)
            seen.add(chunk_id)
            keyword_count += 1

    trimmed: list[dict[str, Any]] = []
    total_chars = 0
    for chunk in selected:
        text = " ".join(str(chunk.get("text", "")).split())
        remaining = max_context_chars - total_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[: max(0, remaining - 3)].rstrip() + "..."
        copied = dict(chunk)
        copied["text"] = text
        trimmed.append(copied)
        total_chars += len(text)

    return trimmed


def format_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(
            "chunk_id={chunk_id}; page={page_number}; file={file_name}\n{text}".format(
                chunk_id=chunk.get("chunk_id", ""),
                page_number=chunk.get("page_number", ""),
                file_name=chunk.get("file_name", ""),
                text=chunk.get("text", ""),
            )
        )
    return "\n\n".join(blocks)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(stripped[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object.")
    return payload


def enrich_one_card(
    card: dict[str, Any],
    chunks: list[dict[str, Any]],
    client: OpenAICompatibleClient,
    only_missing: bool = False,
) -> dict[str, Any]:
    if not chunks:
        enriched = dict(card)
        enriched["enrichment_status"] = "failed"
        enriched["enrichment_error"] = "No chunks found for this paper_id."
        return enriched

    context_chunks = select_context_chunks(chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "Extract structured metadata from research paper chunks. "
                "Extract only information supported by the provided paper chunks. "
                "Do not infer unsupported datasets, metrics, or baselines. "
                "If not found, return an empty list or empty string. "
                "Return valid JSON only, with no Markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Paper card seed:\n"
                + json.dumps(
                    {
                        "paper_id": card.get("paper_id"),
                        "title": card.get("title"),
                        "source_file": card.get("source_file"),
                    },
                    ensure_ascii=False,
                )
                + "\n\nRequired JSON schema:\n"
                + '{"task":"","method_keywords":[],"datasets":[],"metrics":[],"baselines":[],"summary":"","limitations":""}'
                + "\n\nPaper chunks:\n"
                + format_context(context_chunks)
            ),
        },
    ]

    enriched = dict(card)
    try:
        raw = client.chat(messages, temperature=0.0)
        extracted = normalize_enrichment(extract_json_object(raw))
        for field, value in extracted.items():
            if only_missing and not is_missing(enriched.get(field)):
                continue
            enriched[field] = value
        enriched["enrichment_status"] = "enriched"
        enriched["enrichment_error"] = ""
    except Exception as exc:  # noqa: BLE001 - one paper failure should not stop batch processing.
        enriched["enrichment_status"] = "failed"
        enriched["enrichment_error"] = str(exc)
    return enriched


def enrich_paper_cards(
    cards_path: str | Path = DEFAULT_PAPER_CARDS_PATH,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    output_path: str | Path | None = None,
    paper_id: str | None = None,
    limit: int | None = None,
    only_missing: bool = False,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_timeout: float = DEFAULT_LLM_TIMEOUT,
) -> tuple[int, int]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0.")

    cards = load_jsonl(cards_path)
    chunks = load_jsonl(chunks_path)
    chunks_by_paper = group_chunks_by_paper(chunks)
    selected_ids = select_cards(cards, paper_id=paper_id, limit=limit)
    client = OpenAICompatibleClient.from_env(model=llm_model, base_url=llm_base_url, timeout=llm_timeout)

    processed = 0
    failed = 0
    output_cards: list[dict[str, Any]] = []
    for card in cards:
        current_paper_id = str(card.get("paper_id", ""))
        if current_paper_id in selected_ids:
            updated = enrich_one_card(
                card=card,
                chunks=chunks_by_paper.get(current_paper_id, []),
                client=client,
                only_missing=only_missing,
            )
            processed += 1
            if updated.get("enrichment_status") == "failed":
                failed += 1
            output_cards.append(updated)
        else:
            output_cards.append(card)

    destination = output_path or cards_path
    write_jsonl(output_cards, destination)
    return processed, failed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich heuristic paper cards with an LLM.")
    parser.add_argument("--cards", type=Path, default=DEFAULT_PAPER_CARDS_PATH, help="Input paper cards JSONL.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH, help="Input chunks JSONL.")
    parser.add_argument("--output", type=Path, default=None, help="Output enriched paper cards JSONL.")
    parser.add_argument("--paper_id", default=None, help="Only enrich one paper id.")
    parser.add_argument("--limit", type=int, default=None, help="Only enrich the first N selected cards.")
    parser.add_argument("--only_missing", action="store_true", help="Do not overwrite non-empty fields.")
    parser.add_argument("--llm_model", default=None, help="Override LABMATE_LLM_MODEL.")
    parser.add_argument("--llm_base_url", default=None, help="Override LABMATE_LLM_BASE_URL.")
    parser.add_argument("--llm_timeout", type=float, default=DEFAULT_LLM_TIMEOUT, help="LLM request timeout in seconds.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        processed, failed = enrich_paper_cards(
            cards_path=args.cards,
            chunks_path=args.chunks,
            output_path=args.output,
            paper_id=args.paper_id,
            limit=args.limit,
            only_missing=args.only_missing,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_timeout=args.llm_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    destination = args.output or args.cards
    print(f"Processed {processed} paper card(s), {failed} failed. Wrote {destination}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
