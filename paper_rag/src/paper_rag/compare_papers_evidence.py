from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from paper_rag.cli_io import configure_utf8_stdio
from paper_rag.compare_papers import compare_papers, format_value
from paper_rag.llm_client import DEFAULT_LLM_TIMEOUT, OpenAICompatibleClient
from paper_rag.paths import resolve_chunk_metadata_path
from paper_rag.qa import resolve_answer_language
from paper_rag.search import load_metadata


DEFAULT_EVIDENCE_QUESTION = (
    "Compare these papers in terms of task setting, method design, datasets, "
    "evaluation metrics, baselines, limitations, and evaluation protocol caveats."
)
EVIDENCE_KEYWORDS = (
    "task",
    "method",
    "dataset",
    "datasets",
    "metric",
    "metrics",
    "baseline",
    "baselines",
    "evaluation",
    "protocol",
    "train",
    "training",
    "test",
    "testing",
    "split",
    "cross-dataset",
    "limitation",
    "limitations",
    "ablation",
    "robustness",
)


def compact_text(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def extract_terms(*values: object) -> list[str]:
    joined = " ".join(format_value(value) for value in values)
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", joined)
    terms = {term.lower() for term in raw_terms}
    terms.update(EVIDENCE_KEYWORDS)
    return sorted(terms)


def load_chunk_metadata(metadata_path: str | Path) -> list[dict[str, Any]]:
    return load_metadata(Path(metadata_path))


def group_chunks_by_paper(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        paper_id = str(chunk.get("paper_id", ""))
        if paper_id:
            grouped.setdefault(paper_id, []).append(chunk)
    for paper_chunks in grouped.values():
        paper_chunks.sort(key=lambda item: (int(item.get("page_number") or 0), str(item.get("chunk_id", ""))))
    return grouped


def score_chunk(chunk: dict[str, Any], terms: list[str]) -> int:
    text = str(chunk.get("text", "")).lower()
    score = 0
    for term in terms:
        if term in text:
            score += 1
    page_number = int(chunk.get("page_number") or 0)
    if page_number <= 3:
        score += 1
    return score


def select_evidence_for_paper(
    paper: dict[str, Any],
    chunks: list[dict[str, Any]],
    question: str,
    chunks_per_paper: int,
    max_chars_per_chunk: int,
) -> list[dict[str, Any]]:
    if chunks_per_paper <= 0:
        raise ValueError("chunks_per_paper must be greater than 0.")

    terms = extract_terms(
        question,
        paper.get("title", ""),
        paper.get("task", ""),
        paper.get("method_keywords", []),
        paper.get("datasets", []),
        paper.get("metrics", []),
        paper.get("baselines", []),
        paper.get("summary", ""),
        paper.get("limitations", ""),
    )
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            score_chunk(chunk, terms),
            -int(chunk.get("page_number") or 0),
            str(chunk.get("chunk_id", "")),
        ),
        reverse=True,
    )

    selected = []
    for chunk in ranked[:chunks_per_paper]:
        copied = {
            "paper_id": chunk.get("paper_id", ""),
            "title": paper.get("title", ""),
            "source_file": chunk.get("source_file") or chunk.get("source_path", ""),
            "file_name": chunk.get("file_name", ""),
            "page_number": chunk.get("page_number", ""),
            "chunk_id": chunk.get("chunk_id", ""),
            "text": compact_text(chunk.get("text", ""), max_chars_per_chunk),
        }
        selected.append(copied)
    return selected


def collect_balanced_evidence(
    papers: list[dict[str, Any]],
    metadata_path: str | Path,
    question: str,
    chunks_per_paper: int,
    max_chars_per_chunk: int,
) -> list[dict[str, Any]]:
    chunks = load_chunk_metadata(metadata_path)
    chunks_by_paper = group_chunks_by_paper(chunks)

    evidence: list[dict[str, Any]] = []
    evidence_id = 1
    for paper in papers:
        paper_id = str(paper.get("paper_id", ""))
        selected = select_evidence_for_paper(
            paper=paper,
            chunks=chunks_by_paper.get(paper_id, []),
            question=question,
            chunks_per_paper=chunks_per_paper,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        for chunk in selected:
            chunk["evidence_id"] = f"E{evidence_id}"
            evidence_id += 1
            evidence.append(chunk)
    return evidence


def build_paper_context(papers: list[dict[str, Any]]) -> str:
    compact = []
    for paper in papers:
        compact.append(
            {
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year", ""),
                "venue": paper.get("venue", ""),
                "task": paper.get("task", ""),
                "method_keywords": paper.get("method_keywords", []),
                "datasets": paper.get("datasets", []),
                "metrics": paper.get("metrics", []),
                "baselines": paper.get("baselines", []),
                "summary": paper.get("summary", ""),
                "limitations": paper.get("limitations", ""),
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def build_evidence_context(evidence: list[dict[str, Any]]) -> str:
    blocks = []
    for item in evidence:
        blocks.append(
            "[{evidence_id}] paper_id={paper_id}; title={title}; page={page_number}; chunk_id={chunk_id}\n{text}".format(
                evidence_id=item.get("evidence_id", ""),
                paper_id=item.get("paper_id", ""),
                title=item.get("title", ""),
                page_number=item.get("page_number", ""),
                chunk_id=item.get("chunk_id", ""),
                text=item.get("text", ""),
            )
        )
    return "\n\n".join(blocks)


def source_lines(evidence: list[dict[str, Any]]) -> list[str]:
    return [
        "[{evidence_id}] {paper_id}, page {page_number}, chunk_id={chunk_id}, file={file_name}".format(
            evidence_id=item.get("evidence_id", ""),
            paper_id=item.get("paper_id", ""),
            page_number=item.get("page_number", ""),
            chunk_id=item.get("chunk_id", ""),
            file_name=item.get("file_name") or item.get("source_file", ""),
        )
        for item in evidence
    ]


def generate_evidence_comparison_summary(
    papers: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
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
                "You compare research papers using only the provided paper-card metadata and evidence chunks. "
                "Do not use outside knowledge. Do not invent datasets, metrics, baselines, results, limitations, "
                "or evaluation protocols. Cite evidence inline using evidence ids such as [E1]. "
                "Do not rank papers or claim direct fairness unless datasets, metrics, splits, baselines, and "
                "protocols are clearly aligned. Always include a section named "
                "'## Comparability and Protocol Caveats'. If protocol information is missing, say it is not "
                "specified in the available evidence. Return Markdown with these sections: "
                "# Evidence-Grounded Multi-Paper Comparison, ## Selected Papers, ## Evidence Coverage, "
                "## Task Settings, ## Method Differences, ## Dataset and Evaluation Protocols, "
                "## Metrics and Baselines, ## Comparability and Protocol Caveats, ## Practical Takeaways, "
                "## Sources. "
                f"{language_instruction}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Comparison question:\n{question}\n\n"
                f"Selected paper cards:\n{build_paper_context(papers)}\n\n"
                f"Evidence chunks:\n{build_evidence_context(evidence)}\n\n"
                "Write a concise evidence-grounded comparison. In the Sources section, list every evidence id used."
            ),
        },
    ]
    return client.chat(messages, temperature=0.2)


def compare_papers_with_evidence(
    cards_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    keyword: str | None = None,
    dataset: str | None = None,
    metric: str | None = None,
    year: int | str | None = None,
    venue: str | None = None,
    paper_ids: list[str] | None = None,
    limit: int = 5,
    chunks_per_paper: int = 3,
    max_chars_per_chunk: int = 1200,
    question: str = DEFAULT_EVIDENCE_QUESTION,
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
        return {"papers": [], "evidence": [], "summary": "", "llm_called": False}

    clean_question = question.strip() or DEFAULT_EVIDENCE_QUESTION
    evidence = collect_balanced_evidence(
        papers=papers,
        metadata_path=resolve_chunk_metadata_path(metadata_path),
        question=clean_question,
        chunks_per_paper=chunks_per_paper,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    if not evidence:
        return {"papers": papers, "evidence": [], "summary": "", "llm_called": False}

    client = OpenAICompatibleClient.from_env(model=llm_model, base_url=llm_base_url, timeout=llm_timeout)
    summary = generate_evidence_comparison_summary(
        papers=papers,
        evidence=evidence,
        question=clean_question,
        answer_language=answer_language,
        client=client,
    )
    if "## Sources" not in summary:
        summary = summary.rstrip() + "\n\n## Sources\n" + "\n".join(source_lines(evidence))
    return {"papers": papers, "evidence": evidence, "summary": summary, "llm_called": True, "model": client.model}


def write_summary(summary: str, output_path: str | Path | None) -> None:
    if output_path is None:
        print(summary)
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote evidence-grounded comparison to: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an evidence-grounded multi-paper comparison summary.")
    parser.add_argument(
        "--cards_path",
        type=Path,
        default=None,
        help=(
            "Paper cards JSONL path. Defaults to the best available file under paper_rag/storage: "
            "paper_cards_cleaned.jsonl, paper_cards_enriched.jsonl, then paper_cards.jsonl."
        ),
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=None,
        help="Chunk metadata JSONL path. Default: paper_rag/storage/vector_store/metadata.jsonl",
    )
    parser.add_argument("--keyword", default=None, help="Case-insensitive paper-card keyword filter.")
    parser.add_argument("--dataset", default=None, help="Case-insensitive dataset filter.")
    parser.add_argument("--metric", default=None, help="Case-insensitive metric filter.")
    parser.add_argument("--year", default=None, help="Year filter, for example 2025.")
    parser.add_argument("--venue", default=None, help="Case-insensitive venue filter.")
    parser.add_argument("--paper_id", nargs="+", default=None, help="One or more paper ids.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of selected papers.")
    parser.add_argument("--chunks_per_paper", type=int, default=3, help="Balanced evidence chunks per selected paper.")
    parser.add_argument("--max_chars_per_chunk", type=int, default=1200, help="Maximum evidence text per chunk.")
    parser.add_argument("--question", default=DEFAULT_EVIDENCE_QUESTION, help="Comparison focus question.")
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
        result = compare_papers_with_evidence(
            cards_path=args.cards_path,
            metadata_path=args.metadata_path,
            keyword=args.keyword,
            dataset=args.dataset,
            metric=args.metric,
            year=args.year,
            venue=args.venue,
            paper_ids=args.paper_id,
            limit=args.limit,
            chunks_per_paper=args.chunks_per_paper,
            max_chars_per_chunk=args.max_chars_per_chunk,
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
    if not result["evidence"]:
        print("No supporting evidence chunks found for the selected papers.")
        return 0

    write_summary(str(result["summary"]), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
