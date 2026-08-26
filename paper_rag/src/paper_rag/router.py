from __future__ import annotations

import argparse
import re
from time import perf_counter
from pathlib import Path
from typing import Any

from paper_rag.cli_io import configure_utf8_stdio
from paper_rag.indexing import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL_CACHE_DIR
from paper_rag.llm_client import DEFAULT_LLM_TIMEOUT
from paper_rag.metadata_search import format_card, search_paper_cards
from paper_rag.memory.integration import (
    memory_result_payload,
    prepare_query_memory,
    record_failed_query_episode,
    record_query_episode,
    resolve_memory_answer_language,
)
from paper_rag.memory.store import DEFAULT_MEMORY_DB_PATH
from paper_rag.paper_cards import DEFAULT_PAPER_CARDS_PATH
from paper_rag.qa import ask_papers, parse_bool
from paper_rag.query_cache import (
    DEFAULT_QUERY_CACHE_PATH,
    append_query_cache,
    build_paper_revision,
    build_request_fingerprint,
    find_cached_query,
)
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
    use_memory: bool = False,
    memory_db: str | Path = DEFAULT_MEMORY_DB_PATH,
    project_id: str = "default-project",
    session_id: str | None = None,
    memory_top_k: int = 6,
    memory_token_budget: int = 800,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty.")

    query_started = perf_counter()
    stage_ms = {
        "memory_prepare_ms": 0.0,
        "cache_lookup_ms": 0.0,
        "rag_ms": 0.0,
        "memory_write_ms": 0.0,
        "cache_write_ms": 0.0,
    }

    memory_preparation = None
    contextualized_query = query
    if use_memory:
        stage_started = perf_counter()
        memory_preparation = prepare_query_memory(
            query,
            project_id=project_id,
            session_id=session_id or "",
            db_path=memory_db,
            top_k=memory_top_k,
            token_budget=memory_token_budget,
        )
        contextualized_query = memory_preparation.contextualized_query
        stage_ms["memory_prepare_ms"] = (perf_counter() - stage_started) * 1000.0

    route = route_query(query, mode)
    selected_mode = route["mode"]
    filters = route.get("filters", {})
    effective_answer_language = resolve_memory_answer_language(
        answer_language,
        memory_preparation,
    )
    paper_revision = build_paper_revision(
        mode=selected_mode,
        cards_path=cards,
        index_dir=index_dir,
    )
    memory_revision = (
        memory_preparation.store.get_session(session_id or "").memory_revision
        if memory_preparation is not None
        else None
    )
    request_fingerprint = build_request_fingerprint(
        {
            "selected_mode": selected_mode,
            "top_k": top_k,
            "cards": str(cards),
            "index_dir": str(index_dir),
            "model_name": model_name,
            "cache_dir": str(cache_dir),
            "answer_language": effective_answer_language,
            "rewrite_query": rewrite_query,
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "memory_top_k": memory_top_k if use_memory else None,
            "memory_token_budget": memory_token_budget if use_memory else None,
            "contextualized_query": contextualized_query,
        }
    )
    cache_key = {
        "project_id": project_id if use_memory else None,
        "session_id": session_id if use_memory else None,
        "memory_revision": memory_revision,
        "paper_revision": paper_revision,
        "request_fingerprint": request_fingerprint,
    }

    cache_enabled = use_cache
    if cache_enabled:
        stage_started = perf_counter()
        cached = find_cached_query(query, query_cache, cache_key=cache_key)
        stage_ms["cache_lookup_ms"] = (perf_counter() - stage_started) * 1000.0
        if cached is not None:
            response = {
                "query": query,
                "selected_mode": cached.get("mode", "cached"),
                "cache_hit": True,
                "search_query": cached.get("search_query", ""),
                "answer": cached.get("answer", ""),
                "results": cached.get("results", []),
                "filters": cached.get("filters", {}),
            }
            if memory_preparation is not None:
                response["memory"] = memory_result_payload(memory_preparation, None)
                response["memory"]["cache_key_memory_revision"] = memory_revision
            response["observability"] = _observability_payload(
                stage_ms,
                query_started=query_started,
                cache_hit=True,
                result_count=len(response["results"]),
                memory_preparation=memory_preparation,
            )
            return response

    answer = ""
    results: list[dict[str, Any]] = []
    search_query = contextualized_query

    stage_started = perf_counter()
    try:
        if selected_mode == "metadata":
            results = search_paper_cards(cards_path=cards, **filters)
            answer = f"Found {len(results)} matching paper card(s)."
        elif selected_mode == "search":
            results = search_papers(
                query=contextualized_query,
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
                answer_language=effective_answer_language,
                rewrite_query=rewrite_query,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_timeout=llm_timeout,
                retrieval_query=contextualized_query,
                memory_context=memory_preparation.prompt_context if memory_preparation else "",
            )
            answer = str(qa_result.get("answer", ""))
            results = list(qa_result.get("evidence", []))
            search_query = str(qa_result.get("search_query", query))
        else:
            raise ValueError(f"Unsupported selected mode: {selected_mode}")
    except Exception as exc:
        stage_ms["rag_ms"] = (perf_counter() - stage_started) * 1000.0
        if memory_preparation is not None:
            try:
                record_failed_query_episode(
                    memory_preparation,
                    original_query=query,
                    selected_mode=selected_mode,
                    error=exc,
                    project_id=project_id,
                    session_id=session_id or "",
                )
            except Exception:
                # Preserve the original RAG failure if memory recording also fails.
                pass
        raise
    stage_ms["rag_ms"] = (perf_counter() - stage_started) * 1000.0

    episode_result = None
    if memory_preparation is not None:
        stage_started = perf_counter()
        episode_result = record_query_episode(
            memory_preparation,
            original_query=query,
            selected_mode=selected_mode,
            answer=answer,
            results=results,
            project_id=project_id,
            session_id=session_id or "",
        )
        stage_ms["memory_write_ms"] = (perf_counter() - stage_started) * 1000.0

    if cache_enabled:
        stage_started = perf_counter()
        if memory_preparation is not None:
            cache_key["memory_revision"] = memory_preparation.store.get_session(
                session_id or ""
            ).memory_revision
        append_query_cache(
            query=query,
            mode=selected_mode,
            answer=answer,
            results=results,
            cache_path=query_cache,
            search_query=search_query,
            filters=filters,
            cache_key=cache_key,
        )
        stage_ms["cache_write_ms"] = (perf_counter() - stage_started) * 1000.0

    response = {
        "query": query,
        "selected_mode": selected_mode,
        "cache_hit": False,
        "search_query": search_query,
        "answer": answer,
        "results": results,
        "filters": filters,
    }
    if memory_preparation is not None and episode_result is not None:
        response["memory"] = memory_result_payload(memory_preparation, episode_result)
        response["memory"]["cache_key_memory_revision"] = cache_key["memory_revision"]
    response["observability"] = _observability_payload(
        stage_ms,
        query_started=query_started,
        cache_hit=False,
        result_count=len(results),
        memory_preparation=memory_preparation,
    )
    return response


def _observability_payload(
    stage_ms: dict[str, float],
    *,
    query_started: float,
    cache_hit: bool,
    result_count: int,
    memory_preparation: Any,
) -> dict[str, Any]:
    recalled_count = 0
    context_tokens = 0
    if memory_preparation is not None:
        recalled_count = len(memory_preparation.packet.entries)
        context_tokens = memory_preparation.packet.estimated_tokens
    return {
        "total_ms": round((perf_counter() - query_started) * 1000.0, 3),
        "stages_ms": {name: round(value, 3) for name, value in stage_ms.items()},
        "cache_hit": cache_hit,
        "result_count": result_count,
        "recalled_memory_count": recalled_count,
        "memory_context_estimated_tokens": context_tokens,
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
    parser.add_argument("--memory", dest="use_memory", type=parse_bool, default=False, help="Enable session memory.")
    parser.add_argument(
        "--memory_db",
        type=Path,
        default=DEFAULT_MEMORY_DB_PATH,
        help="SQLite memory database path.",
    )
    parser.add_argument("--project_id", default="default-project", help="Memory project scope.")
    parser.add_argument("--session_id", default=None, help="Memory session id; required when --memory true.")
    parser.add_argument("--memory_top_k", type=int, default=6, help="Maximum recalled memory items.")
    parser.add_argument(
        "--memory_token_budget",
        type=int,
        default=800,
        help="Approximate token budget for memory context.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
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
            use_memory=args.use_memory,
            memory_db=args.memory_db,
            project_id=args.project_id,
            session_id=args.session_id,
            memory_top_k=args.memory_top_k,
            memory_token_budget=args.memory_token_budget,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    selected_mode = result["selected_mode"]
    print(f"Selected mode: {selected_mode}")
    print(f"Cache hit: {result['cache_hit']}")
    if result.get("search_query"):
        print(f"Search query: {result['search_query']}")
    if result.get("memory"):
        memory = result["memory"]
        print(f"Memory recalled: {len(memory['recalled_memory_ids'])}")
        print(f"Memory episode: {memory['episode_memory_id']}")
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
