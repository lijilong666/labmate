from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_rag.cli_io import configure_utf8_stdio
from paper_rag.compare_papers import compare_papers, format_value
from paper_rag.llm_client import DEFAULT_LLM_TIMEOUT, OpenAICompatibleClient
from paper_rag.paper_cards import DEFAULT_PAPER_CARDS_PATH
from paper_rag.qa import resolve_answer_language


DEFAULT_COMPARISON_QUESTION = (
    "Compare these papers in terms of task setting, method design, datasets, "
    "evaluation metrics, baselines, limitations, and practical takeaways."
)
PROMPT_FIELDS = [
    "paper_id",
    "title",
    "year",
    "venue",
    "task",
    "method_keywords",
    "datasets",
    "metrics",
    "baselines",
    "summary",
    "limitations",
]
MAX_TEXT_FIELD_CHARS = 700


def compact_text(value: object, max_chars: int = MAX_TEXT_FIELD_CHARS) -> str:
    text = " ".join(format_value(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def compact_paper_for_prompt(paper: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for field in PROMPT_FIELDS:
        value = paper.get(field, [] if field in {"method_keywords", "datasets", "metrics", "baselines"} else "")
        if isinstance(value, list):
            compact[field] = [compact_text(item, 120) for item in value if compact_text(item, 120)]
        else:
            compact[field] = compact_text(value)
    return compact


def build_prompt_payload(papers: list[dict[str, Any]]) -> str:
    compact_papers = [compact_paper_for_prompt(paper) for paper in papers]
    return json.dumps(compact_papers, ensure_ascii=False, indent=2)


def generate_comparison_summary(
    papers: list[dict[str, Any]],
    question: str,
    answer_language: str,
    client: OpenAICompatibleClient,
) -> str:
    resolved_language = resolve_answer_language(question, answer_language)
    language_instruction = {
        "zh": "Write the summary in Chinese.",
        "en": "Write the summary in English.",
    }[resolved_language]

    messages = [
        {
            "role": "system",
            "content": (
                "You compare research papers using only the provided paper-card metadata. "
                "Do not use outside knowledge. Do not claim that you read the full papers. "
                "Do not invent datasets, metrics, baselines, limitations, page numbers, chunks, or citations. "
                "If a field is missing or empty, say it is not specified in the available paper cards. "
                "Use paper_id references such as M2SFormer (p000060), not chunk-level citations. "
                "Return Markdown with these sections exactly: "
                "# Multi-paper Comparison Summary, ## Selected Papers, ## Task Settings, "
                "## Methodological Differences, ## Dataset Usage, ## Evaluation Metrics, "
                "## Baselines, ## Limitations, ## Practical Takeaways. "
                f"{language_instruction}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Comparison question:\n{question}\n\n"
                "Available paper cards:\n"
                f"{build_prompt_payload(papers)}\n\n"
                "Write a concise comparison summary grounded only in these paper cards."
            ),
        },
    ]
    return client.chat(messages, temperature=0.2)


def compare_papers_with_llm(
    cards_path: str | Path = DEFAULT_PAPER_CARDS_PATH,
    keyword: str | None = None,
    dataset: str | None = None,
    metric: str | None = None,
    year: int | str | None = None,
    venue: str | None = None,
    paper_ids: list[str] | None = None,
    limit: int = 8,
    question: str = DEFAULT_COMPARISON_QUESTION,
    answer_language: str = "en",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_timeout: float = DEFAULT_LLM_TIMEOUT,
) -> dict[str, Any]:
    if limit < 0:
        raise ValueError("limit must be greater than or equal to 0.")

    papers = compare_papers(
        cards_path=cards_path,
        keyword=keyword,
        dataset=dataset,
        metric=metric,
        year=year,
        venue=venue,
        paper_ids=paper_ids,
        limit=limit,
    )
    if not papers:
        return {"papers": [], "summary": "", "llm_called": False}

    clean_question = question.strip() or DEFAULT_COMPARISON_QUESTION
    client = OpenAICompatibleClient.from_env(model=llm_model, base_url=llm_base_url, timeout=llm_timeout)
    summary = generate_comparison_summary(
        papers=papers,
        question=clean_question,
        answer_language=answer_language,
        client=client,
    )
    return {"papers": papers, "summary": summary, "llm_called": True, "model": client.model}


def write_summary(summary: str, output_path: str | Path | None) -> None:
    if output_path is None:
        print(summary)
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote comparison summary to: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an LLM-assisted summary from selected paper cards.")
    parser.add_argument("--cards_path", type=Path, default=DEFAULT_PAPER_CARDS_PATH, help="Paper cards JSONL path.")
    parser.add_argument("--keyword", default=None, help="Case-insensitive keyword filter.")
    parser.add_argument("--dataset", default=None, help="Case-insensitive dataset filter.")
    parser.add_argument("--metric", default=None, help="Case-insensitive metric filter.")
    parser.add_argument("--year", default=None, help="Year filter, for example 2025.")
    parser.add_argument("--venue", default=None, help="Case-insensitive venue filter.")
    parser.add_argument("--paper_id", nargs="+", default=None, help="One or more paper ids.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum number of matching papers to summarize.")
    parser.add_argument("--question", default=DEFAULT_COMPARISON_QUESTION, help="Comparison focus question.")
    parser.add_argument("--answer_language", choices=["auto", "zh", "en"], default="en", help="Summary language.")
    parser.add_argument("--llm_timeout", type=float, default=DEFAULT_LLM_TIMEOUT, help="LLM request timeout in seconds.")
    parser.add_argument("--llm_base_url", default=None, help="Override LABMATE_LLM_BASE_URL.")
    parser.add_argument("--llm_model", default=None, help="Override LABMATE_LLM_MODEL.")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output file path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = compare_papers_with_llm(
            cards_path=args.cards_path,
            keyword=args.keyword,
            dataset=args.dataset,
            metric=args.metric,
            year=args.year,
            venue=args.venue,
            paper_ids=args.paper_id,
            limit=args.limit,
            question=args.question,
            answer_language=args.answer_language,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_timeout=args.llm_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    if not result["papers"]:
        print("No matching papers found.")
        return 0

    write_summary(str(result["summary"]), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
