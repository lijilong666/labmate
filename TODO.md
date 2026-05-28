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

## Phase 3: `experiment_agent` Planning

- [ ] Define experiment metadata schema.
- [ ] Plan result parsing and comparison workflows.
- [ ] Plan report generation outputs.
- [x] Identify initial integration points with `paper_rag`.
