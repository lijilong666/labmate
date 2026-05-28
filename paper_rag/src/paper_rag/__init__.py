"""Paper RAG utilities for LabMate."""

__all__ = [
    "ask_papers",
    "compare_papers",
    "enrich_paper_cards",
    "generate_paper_cards",
    "get_topic_summary",
    "paper_query",
    "route_query",
    "search_paper_cards",
    "search_papers",
]


def __getattr__(name: str):
    if name == "ask_papers":
        from paper_rag.qa import ask_papers

        return ask_papers
    if name == "compare_papers":
        from paper_rag.compare_papers import compare_papers

        return compare_papers
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
    if name == "route_query":
        from paper_rag.router import route_query

        return route_query
    if name == "search_paper_cards":
        from paper_rag.metadata_search import search_paper_cards

        return search_paper_cards
    if name == "search_papers":
        from paper_rag.search import search_papers

        return search_papers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
