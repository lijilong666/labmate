from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from paper_rag.indexing import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL_CACHE_DIR
from paper_rag.llm_client import DEFAULT_LLM_TIMEOUT
from paper_rag.metadata_search import format_card, search_paper_cards
from paper_rag.paper_cards import DEFAULT_PAPER_CARDS_PATH
from paper_rag.qa import ask_papers, parse_bool
from paper_rag.query_cache import DEFAULT_QUERY_CACHE_PATH, append_query_cache, find_cached_query
from paper_rag.search import DEFAULT_INDEX_DIR, format_result, search_papers


ROUTER_MODES = {"auto", "metadata", "search", "answer"}
METADATA_INTENTS = (
    "which papers",
    "list papers",
    "find papers",
    "哪些论文",
    "哪些文章",
    "有哪些论文",
    "论文列表",
)
SEARCH_INTENTS = ("search", "find chunks", "retrieve", "检索", "查找片段")
ANSWER_INTENTS = (
    "summarize",
    "explain",
    "compare",
    "why",
    "how",
    "总结",
    "解释",
    "比较",
    "为什么",
    "如何",
)
VENUES = ("CVPR", "ICCV", "ECCV", "AAAI", "ACM MM", "NeurIPS", "ICLR", "ICML", "IJCAI", "ICASSP", "NAACL", "TIFS")
METRICS = ("F1", "IoU", "AUC", "AP", "mAP", "bF1", "Accuracy", "Precision", "Recall")


def contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def extract_metadata_filters(query: str) -> dict[str, str]:
    filters: dict[str, str] = {}

    year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", query)
    if year_match:
        filters["year"] = year_match.group(1)

    paper_id_match = re.search(r"\bp\d{6}\b", query, flags=re.IGNORECASE)
    if paper_id_match:
        filters["paper_id"] = paper_id_match.group(0).lower()

    for venue in VENUES:
        if venue.lower() in query.lower():
            filters["venue"] = venue
            break

    for metric in METRICS:
        if re.search(rf"\b{re.escape(metric)}\b", query, flags=re.IGNORECASE):
            filters["metric"] = metric
            break

    dataset_match = re.search(r"\b(?:dataset|datasets|数据集)\s*[:：]?\s*([A-Za-z0-9_\-]+)", query, flags=re.IGNORECASE)
    if dataset_match:
        filters["dataset"] = dataset_match.group(1)

    keyword = extract_keyword(query, filters)
    if keyword:
        filters["keyword"] = keyword

    return filters


def extract_keyword(query: str, filters: dict[str, str]) -> str:
    keyword = query
    replacements = list(METADATA_INTENTS) + ["used", "use", "using", "about", "with", "papers", "paper"]
    replacements += ["用了", "使用了", "使用", "包含", "关于", "的", "吗", "？", "?"]
    for item in replacements:
        keyword = re.sub(re.escape(item), " ", keyword, flags=re.IGNORECASE)

    for value in filters.values():
        keyword = re.sub(re.escape(str(value)), " ", keyword, flags=re.IGNORECASE)

    keyword = re.sub(r"(?<!\d)20\d{2}(?!\d)", " ", keyword)
    keyword = re.sub(r"\b(?:dataset|datasets|metric|metrics|venue|year|数据集|指标)\b", " ", keyword, flags=re.IGNORECASE)
    keyword = re.sub(r"[\s,;:：，。]+", " ", keyword).strip()
    return keyword


def route_query(query: str, mode: str = "auto") -> dict[str, Any]:
    if mode not in ROUTER_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(ROUTER_MODES))}")
    if mode != "auto":
        return {"mode": mode, "filters": extract_metadata_filters(query)}

    if contains_any(query, METADATA_INTENTS):
        filters = extract_metadata_filters(query)
        if filters:
            return {"mode": "metadata", "filters": filters}
        return {"mode": "search", "filters": {}}
    if contains_any(query, SEARCH_INTENTS):
        return {"mode": "search", "filters": {}}
    if contains_any(query, ANSWER_INTENTS):
        return {"mode": "answer", "filters": {}}
    return {"mode": "answer", "filters": {}}


def paper_query(
    query: str,
    mode: str = "auto",
    top_k: int = 5,
    cards: str | Path = DEFAULT_PAPER_CARDS_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: str | Path = DEFAULT_MODEL_CACHE_DIR,
    query_cache: str | Path = DEFAULT_QUERY_CACHE_PATH,
    use_cache: bool = True,
    answer_language: str = "auto",
    rewrite_query: bool = True,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_timeout: float = DEFAULT_LLM_TIMEOUT,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty.")

    if use_cache:
        cached = find_cached_query(query, query_cache)
        if cached is not None:
            return {
                "query": query,
                "selected_mode": cached.get("mode", "cached"),
                "cache_hit": True,
                "search_query": cached.get("search_query", ""),
                "answer": cached.get("answer", ""),
                "results": cached.get("results", []),
                "filters": cached.get("filters", {}),
            }

    route = route_query(query, mode)
    selected_mode = route["mode"]
    filters = route.get("filters", {})

    answer = ""
    results: list[dict[str, Any]] = []
    search_query = query

    if selected_mode == "metadata":
        results = search_paper_cards(cards_path=cards, **filters)
        answer = f"Found {len(results)} matching paper card(s)."
    elif selected_mode == "search":
        results = search_papers(
            query=query,
            top_k=top_k,
            index_dir=index_dir,
            model_name=model_name,
            cache_dir=cache_dir,
        )
        answer = f"Found {len(results)} matching chunk(s)."
    elif selected_mode == "answer":
        qa_result = ask_papers(
            question=query,
            top_k=top_k,
            index_dir=index_dir,
            model_name=model_name,
            cache_dir=cache_dir,
            answer_language=answer_language,
            rewrite_query=rewrite_query,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_timeout=llm_timeout,
        )
        answer = str(qa_result.get("answer", ""))
        results = list(qa_result.get("evidence", []))
        search_query = str(qa_result.get("search_query", query))
    else:
        raise ValueError(f"Unsupported selected mode: {selected_mode}")

    if use_cache:
        append_query_cache(
            query=query,
            mode=selected_mode,
            answer=answer,
            results=results,
            cache_path=query_cache,
            search_query=search_query,
            filters=filters,
        )

    return {
        "query": query,
        "selected_mode": selected_mode,
        "cache_hit": False,
        "search_query": search_query,
        "answer": answer,
        "results": results,
        "filters": filters,
    }


def format_metadata_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No matching paper cards found."
    return "\n\n".join(f"[{index}] {format_card(card)}" for index, card in enumerate(results, start=1))


def format_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No matching chunks found."
    return "\n\n".join(format_result(result, 500) for result in results)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified paper query router for metadata, search, and QA.")
    parser.add_argument("--query", required=True, help="User query.")
    parser.add_argument("--mode", choices=sorted(ROUTER_MODES), default="auto", help="Routing mode.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of search/evidence results.")
    parser.add_argument("--cards", type=Path, default=DEFAULT_PAPER_CARDS_PATH, help="Paper cards JSONL path.")
    parser.add_argument("--index_dir", type=Path, default=DEFAULT_INDEX_DIR, help="FAISS index directory.")
    parser.add_argument("--model_name", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name or local path.")
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR, help="Embedding model cache dir.")
    parser.add_argument("--query_cache", type=Path, default=DEFAULT_QUERY_CACHE_PATH, help="Exact query cache JSONL.")
    parser.add_argument("--use_cache", type=parse_bool, default=True, help="Use exact query cache.")
    parser.add_argument("--answer_language", choices=["auto", "zh", "en"], default="auto", help="Answer language.")
    parser.add_argument("--rewrite_query", type=parse_bool, default=True, help="Rewrite non-English answer queries.")
    parser.add_argument("--llm_base_url", default=None, help="Override LABMATE_LLM_BASE_URL.")
    parser.add_argument("--llm_model", default=None, help="Override LABMATE_LLM_MODEL.")
    parser.add_argument("--llm_timeout", type=float, default=DEFAULT_LLM_TIMEOUT, help="LLM request timeout in seconds.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = paper_query(
            query=args.query,
            mode=args.mode,
            top_k=args.top_k,
            cards=args.cards,
            index_dir=args.index_dir,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            query_cache=args.query_cache,
            use_cache=args.use_cache,
            answer_language=args.answer_language,
            rewrite_query=args.rewrite_query,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_timeout=args.llm_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    selected_mode = result["selected_mode"]
    print(f"Selected mode: {selected_mode}")
    print(f"Cache hit: {result['cache_hit']}")
    if result.get("search_query"):
        print(f"Search query: {result['search_query']}")
    if selected_mode == "metadata" and result.get("filters"):
        print(f"Metadata filters: {result['filters']}")
    print()

    if selected_mode == "metadata":
        print(format_metadata_results(result["results"]))
    elif selected_mode == "search":
        print(format_search_results(result["results"]))
    else:
        print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
