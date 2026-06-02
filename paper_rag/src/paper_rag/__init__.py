"""Paper RAG utilities for LabMate."""

__all__ = [
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
    "TOOL_CAPABILITIES",
]


def __getattr__(name: str):
    if name == "ask_papers":
        from paper_rag.qa import ask_papers

        return ask_papers
    if name == "build_workspace":
        from paper_rag.pipeline import build_workspace

        return build_workspace
    if name == "cleanup_paper_cards":
        from paper_rag.paper_card_cleanup import cleanup_paper_cards

        return cleanup_paper_cards
    if name == "compare_papers":
        from paper_rag.compare_papers import compare_papers

        return compare_papers
    if name == "compare_papers_with_evidence":
        from paper_rag.compare_papers_evidence import compare_papers_with_evidence

        return compare_papers_with_evidence
    if name == "compare_papers_with_llm":
        from paper_rag.compare_papers_llm import compare_papers_with_llm

        return compare_papers_with_llm
    if name == "enrich_paper_cards":
        from paper_rag.paper_card_enricher import enrich_paper_cards

        return enrich_paper_cards
    if name == "generate_paper_cards":
        from paper_rag.paper_cards import generate_paper_cards

        return generate_paper_cards
    if name == "get_topic_summary":
        from paper_rag.topic_cache import get_topic_summary

        return get_topic_summary
    if name == "paper_query":
        from paper_rag.router import paper_query

        return paper_query
    if name == "resolve_cards_path":
        from paper_rag.paths import resolve_cards_path

        return resolve_cards_path
    if name == "resolve_chunk_metadata_path":
        from paper_rag.paths import resolve_chunk_metadata_path

        return resolve_chunk_metadata_path
    if name == "route_query":
        from paper_rag.router import route_query

        return route_query
    if name == "search_paper_cards":
        from paper_rag.metadata_search import search_paper_cards

        return search_paper_cards
    if name == "search_papers":
        from paper_rag.search import search_papers

        return search_papers
    if name == "TOOL_CAPABILITIES":
        from paper_rag.api import TOOL_CAPABILITIES

        return TOOL_CAPABILITIES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
