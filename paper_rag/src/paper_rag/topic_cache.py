from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_rag.cli_io import configure_utf8_stdio
from paper_rag.indexing import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_CACHE_DIR,
    INDEX_FILE_NAME,
    METADATA_FILE_NAME,
)
from paper_rag.llm_client import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_TIMEOUT, OpenAICompatibleClient
from paper_rag.qa import (
    citation_lines,
    format_answer,
    generate_answer,
    is_probably_english,
    parse_bool,
    resolve_answer_language,
    rewrite_search_query,
)
from paper_rag.search import DEFAULT_INDEX_DIR, search_papers
from paper_rag.topic_cache_store import DEFAULT_TOPIC_CACHE_PATH, find_cached_topic, upsert_topic_cache


def compact_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for item in results:
        sources.append(
            {
                "source_file": item.get("source_file", ""),
                "page_number": item.get("page_number", ""),
                "chunk_id": item.get("chunk_id", ""),
            }
        )
    return sources


def validate_retrieval_files(index_dir: str | Path) -> None:
    index_path = Path(index_dir)
    missing = []
    for file_name in (INDEX_FILE_NAME, METADATA_FILE_NAME):
        path = index_path / file_name
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing retrieval index file(s): "
            + ", ".join(missing)
            + ". Build the index from paper_rag/storage/chunks.jsonl before refreshing the topic cache."
        )


def build_topic_cache_record(
    topic: str,
    query: str,
    answer_language: str,
    answer: str,
    sources: list[dict[str, Any]],
    model: str,
    top_k: int,
    created_at: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "topic": topic,
        "query": query,
        "answer_language": answer_language,
        "answer": answer,
        "sources": sources,
        "created_at": created_at or now,
        "updated_at": now,
        "model": model,
        "top_k": top_k,
    }


def get_topic_summary(
    topic: str,
    query: str,
    force_refresh: bool = False,
    top_k: int = 8,
    answer_language: str = "en",
    rewrite_query: bool = False,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: str | Path = DEFAULT_MODEL_CACHE_DIR,
    cache_path: str | Path = DEFAULT_TOPIC_CACHE_PATH,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_timeout: float = DEFAULT_LLM_TIMEOUT,
) -> dict[str, Any]:
    clean_topic = topic.strip()
    clean_query = query.strip()
    if not clean_topic:
        raise ValueError("topic must not be empty.")
    if not clean_query:
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if llm_provider != DEFAULT_LLM_PROVIDER:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    cached = None if force_refresh else find_cached_topic(clean_topic, cache_path)
    if cached is not None:
        return {"cache_hit": True, "record": cached, "search_query": cached.get("query", ""), "evidence": []}

    validate_retrieval_files(index_dir)
    resolved_language = resolve_answer_language(clean_query, answer_language)
    client = OpenAICompatibleClient.from_env(model=llm_model, base_url=llm_base_url, timeout=llm_timeout)

    search_query = clean_query
    if rewrite_query and not is_probably_english(clean_query):
        search_query = rewrite_search_query(clean_query, client)

    results = search_papers(
        query=search_query,
        top_k=top_k,
        index_dir=index_dir,
        model_name=model_name,
        cache_dir=cache_dir,
    )

    if results:
        answer_body = generate_answer(clean_query, results, client, resolved_language)
        answer = format_answer(answer_body, citation_lines(results))
    else:
        answer = (
            "evidence is insufficient\n\nSources:"
            if resolved_language == "en"
            else "evidence is insufficient: no usable evidence was retrieved.\n\nSources:"
        )

    previous = find_cached_topic(clean_topic, cache_path)
    record = build_topic_cache_record(
        topic=clean_topic,
        query=clean_query,
        answer_language=resolved_language,
        answer=answer,
        sources=compact_sources(results),
        model=client.model,
        top_k=top_k,
        created_at=previous.get("created_at") if previous else None,
    )
    upsert_topic_cache(record, cache_path)

    return {"cache_hit": False, "record": record, "search_query": search_query, "evidence": results}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache reusable topic-level RAG summaries.")
    parser.add_argument("--topic", required=True, help="Stable topic key, such as frequency_domain_features.")
    parser.add_argument("--query", required=True, help="Question used to generate the topic summary on cache miss.")
    parser.add_argument("--force_refresh", action="store_true", help="Refresh the topic summary even if cached.")
    parser.add_argument("--top_k", type=int, default=8, help="Number of evidence chunks to retrieve.")
    parser.add_argument("--answer_language", choices=["auto", "zh", "en"], default="en", help="Answer language.")
    parser.add_argument(
        "--rewrite_query",
        nargs="?",
        const=True,
        type=parse_bool,
        default=False,
        help="Rewrite non-English queries.",
    )
    parser.add_argument("--llm_timeout", type=float, default=DEFAULT_LLM_TIMEOUT, help="LLM request timeout in seconds.")
    parser.add_argument("--index_dir", type=Path, default=DEFAULT_INDEX_DIR, help="FAISS index directory.")
    parser.add_argument("--model_name", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name or local path.")
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR, help="Embedding model cache dir.")
    parser.add_argument("--cache_path", type=Path, default=DEFAULT_TOPIC_CACHE_PATH, help="Topic cache JSONL path.")
    parser.add_argument(
        "--topic_cache",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--llm_provider", default=DEFAULT_LLM_PROVIDER, help="LLM provider name.")
    parser.add_argument("--llm_model", default=None, help="Override LABMATE_LLM_MODEL.")
    parser.add_argument("--llm_base_url", default=None, help="Override LABMATE_LLM_BASE_URL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = get_topic_summary(
            topic=args.topic,
            query=args.query,
            force_refresh=args.force_refresh,
            top_k=args.top_k,
            answer_language=args.answer_language,
            rewrite_query=args.rewrite_query,
            index_dir=args.index_dir,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            cache_path=args.topic_cache or args.cache_path,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_timeout=args.llm_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise actionable failures.
        parser.exit(status=1, message=f"Error: {exc}\n")

    record = result["record"]
    print(f"Cache hit: {result['cache_hit']}")
    if result.get("search_query") and result["search_query"] != record.get("query"):
        print(f"Search query: {result['search_query']}")
    print()
    print(record["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
