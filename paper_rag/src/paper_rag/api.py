from __future__ import annotations

from typing import Any

from paper_rag.paths import resolve_cards_path, resolve_chunk_metadata_path


TOOL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "build_workspace": {
        "stage": "8A",
        "reads": ["data/raw_papers", "paper_rag/storage"],
        "writes": ["paper_rag/storage"],
        "uses_llm": "optional",
        "uses_faiss": True,
        "description": "Build local paper_rag artifacts by orchestrating existing stages.",
    },
    "search_papers": {
        "stage": "3",
        "reads": ["paper_rag/storage/vector_store"],
        "writes": [],
        "uses_llm": False,
        "uses_faiss": True,
        "description": "Retrieve ranked chunks from the local FAISS index.",
    },
    "ask_papers": {
        "stage": "4",
        "reads": ["paper_rag/storage/vector_store"],
        "writes": [],
        "uses_llm": True,
        "uses_faiss": True,
        "description": "Answer a question from retrieved evidence chunks.",
    },
    "cleanup_paper_cards": {
        "stage": "5C",
        "reads": ["paper_rag/storage/paper_cards*.jsonl"],
        "writes": ["paper_rag/storage/paper_cards_cleaned.jsonl"],
        "uses_llm": False,
        "uses_faiss": False,
        "description": "Clean weak paper-card title metadata with lightweight rules.",
    },
    "generate_paper_cards": {
        "stage": "5A",
        "reads": ["paper_rag/storage/paper_inventory.csv"],
        "writes": ["paper_rag/storage/paper_cards.jsonl"],
        "uses_llm": False,
        "uses_faiss": False,
        "description": "Generate heuristic paper cards from the local PDF inventory.",
    },
    "enrich_paper_cards": {
        "stage": "5B",
        "reads": ["paper_rag/storage/paper_cards.jsonl", "paper_rag/storage/chunks.jsonl"],
        "writes": ["paper_rag/storage/paper_cards_enriched.jsonl"],
        "uses_llm": True,
        "uses_faiss": False,
        "description": "Enrich paper cards from limited chunks using an OpenAI-compatible LLM.",
    },
    "search_paper_cards": {
        "stage": "5A",
        "reads": ["paper_rag/storage/paper_cards*.jsonl"],
        "writes": [],
        "uses_llm": False,
        "uses_faiss": False,
        "description": "Search paper-card metadata without vector retrieval or LLM calls.",
    },
    "paper_query": {
        "stage": "6A",
        "reads": ["paper_rag/storage"],
        "writes": ["paper_rag/storage/query_cache.jsonl"],
        "uses_llm": "answer mode only",
        "uses_faiss": "search/answer modes only",
        "description": "Unified metadata/search/answer entry point with exact query cache.",
    },
    "get_topic_summary": {
        "stage": "6B",
        "reads": ["paper_rag/storage/vector_store", "paper_rag/storage/topic_cache.jsonl"],
        "writes": ["paper_rag/storage/topic_cache.jsonl"],
        "uses_llm": "cache miss only",
        "uses_faiss": "cache miss only",
        "description": "Exact topic-level cache for reusable RAG summaries.",
    },
    "compare_papers": {
        "stage": "7A",
        "reads": ["paper_rag/storage/paper_cards*.jsonl"],
        "writes": [],
        "uses_llm": False,
        "uses_faiss": False,
        "description": "Metadata-only structured comparison over selected paper cards.",
    },
    "compare_papers_with_llm": {
        "stage": "7B",
        "reads": ["paper_rag/storage/paper_cards*.jsonl"],
        "writes": [],
        "uses_llm": True,
        "uses_faiss": False,
        "description": "LLM-assisted comparison summary based only on selected paper cards.",
    },
    "compare_papers_with_evidence": {
        "stage": "7C",
        "reads": ["paper_rag/storage/paper_cards*.jsonl", "paper_rag/storage/vector_store/metadata.jsonl"],
        "writes": [],
        "uses_llm": True,
        "uses_faiss": False,
        "description": "Evidence-grounded comparison using balanced chunk metadata snippets.",
    },
}


def __getattr__(name: str):
    if name == "build_workspace":
        from paper_rag.pipeline import build_workspace

        return build_workspace
    if name == "search_papers":
        from paper_rag.search import search_papers

        return search_papers
    if name == "ask_papers":
        from paper_rag.qa import ask_papers

        return ask_papers
    if name == "cleanup_paper_cards":
        from paper_rag.paper_card_cleanup import cleanup_paper_cards

        return cleanup_paper_cards
    if name == "generate_paper_cards":
        from paper_rag.paper_cards import generate_paper_cards

        return generate_paper_cards
    if name == "enrich_paper_cards":
        from paper_rag.paper_card_enricher import enrich_paper_cards

        return enrich_paper_cards
    if name == "search_paper_cards":
        from paper_rag.metadata_search import search_paper_cards

        return search_paper_cards
    if name == "paper_query":
        from paper_rag.router import paper_query

        return paper_query
    if name == "route_query":
        from paper_rag.router import route_query

        return route_query
    if name == "get_topic_summary":
        from paper_rag.topic_cache import get_topic_summary

        return get_topic_summary
    if name == "compare_papers":
        from paper_rag.compare_papers import compare_papers

        return compare_papers
    if name == "compare_papers_with_llm":
        from paper_rag.compare_papers_llm import compare_papers_with_llm

        return compare_papers_with_llm
    if name == "compare_papers_with_evidence":
        from paper_rag.compare_papers_evidence import compare_papers_with_evidence

        return compare_papers_with_evidence
    if name in {"resolve_cards_path", "resolve_chunk_metadata_path", "TOOL_CAPABILITIES"}:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TOOL_CAPABILITIES",
    "ask_papers",
    "build_workspace",
    "cleanup_paper_cards",
    "compare_papers",
    "compare_papers_with_evidence",
    "compare_papers_with_llm",
    "enrich_paper_cards",
    "generate_paper_cards",
    "get_topic_summary",
    "paper_query",
    "resolve_cards_path",
    "resolve_chunk_metadata_path",
    "route_query",
    "search_paper_cards",
    "search_papers",
]
