# TODO

## Phase 0: Repository Foundation

- [x] Create initial project documentation.
- [x] Create module directories.
- [x] Define high-level roadmap and architecture draft.

## Phase 1: `paper_rag` MVP

- [x] Define the MVP input format for papers and metadata.
- [x] Choose a simple local document loader strategy.
- [x] Implement text extraction and chunking.
- [x] Add embedding and local vector index configuration.
- [x] Implement retrieval over a small local paper set.
- [x] Add question answering with source citations.
- [x] Provide a minimal CLI or script entry point.
- [ ] Add basic tests for ingestion, retrieval, and citation formatting.
- [x] Document local setup and storage rules.

## Phase 2: `paper_rag` Refinement

- [x] Improve metadata handling for title, authors, year, venue, and paper URL.
- [ ] Add evaluation examples for retrieval quality.
- [ ] Support incremental indexing.
- [ ] Add clearer error handling and logging.
- [x] Add exact query cache for repeated queries.
- [x] Add topic-level cache for reusable RAG summaries.
- [x] Add structured multi-paper comparison for future experiment-agent use.
- [x] Add LLM-assisted multi-paper comparison summary over paper cards.
- [ ] Add metadata cleanup for weak paper titles and filename-like `title_guess` values.
- [ ] Prefer titles extracted from the PDF first page or enriched LLM card metadata.
- [ ] Optionally support manual title overrides for problematic papers.
- [ ] Re-run cleanup or enrichment for cards whose title looks like an arXiv id or raw PDF filename.
- [ ] Add evidence-grounded multi-paper synthesis with chunk-level citations.

## Phase 3: `experiment_agent` Planning

- [ ] Define experiment metadata schema.
- [ ] Plan result parsing and comparison workflows.
- [ ] Plan report generation outputs.
- [x] Identify initial integration points with `paper_rag`.
