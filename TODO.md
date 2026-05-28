# TODO

## Phase 0: Repository Foundation

- [x] Create initial project documentation.
- [x] Create module directories.
- [x] Define high-level roadmap and architecture draft.

## Phase 1: `paper_rag` MVP

- [ ] Define the MVP input format for papers and metadata.
- [ ] Choose a simple local document loader strategy.
- [ ] Implement text extraction and chunking.
- [ ] Add embedding and local vector index configuration.
- [ ] Implement retrieval over a small local paper set.
- [ ] Add question answering with source citations.
- [ ] Provide a minimal CLI or script entry point.
- [ ] Add basic tests for ingestion, retrieval, and citation formatting.
- [ ] Document local setup and storage rules.

## Phase 2: `paper_rag` Refinement

- [ ] Improve metadata handling for title, authors, year, venue, and paper URL.
- [ ] Add evaluation examples for retrieval quality.
- [ ] Support incremental indexing.
- [ ] Add clearer error handling and logging.
- [x] Add topic-level cache for reusable RAG summaries.

## Phase 3: `experiment_agent` Planning

- [ ] Define experiment metadata schema.
- [ ] Plan result parsing and comparison workflows.
- [ ] Plan report generation outputs.
- [ ] Identify integration points with `paper_rag`.
