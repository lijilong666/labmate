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
- [x] Add Chinese interview-preparation notes for the completed `paper_rag` baseline.

## Phase 2: `paper_rag` Refinement

- [x] Improve metadata handling for title, authors, year, venue, and paper URL.
- [ ] Add evaluation examples for retrieval quality.
- [ ] Support incremental indexing.
- [ ] Add clearer error handling and logging.
- [x] Add exact query cache for repeated queries.
- [x] Add topic-level cache for reusable RAG summaries.
- [x] Add structured multi-paper comparison for future experiment-agent use.
- [x] Add LLM-assisted multi-paper comparison summary over paper cards.
- [x] Add rule-based metadata cleanup for weak paper titles and filename-like `title_guess` values.
- [ ] Prefer titles extracted from the PDF first page or enriched LLM card metadata.
- [x] Optionally support manual title overrides for problematic papers.
- [ ] Re-run cleanup or enrichment for cards whose title looks like an arXiv id or raw PDF filename.
- [x] Add lightweight evidence-grounded multi-paper synthesis with per-paper balanced chunks and chunk-level citations.
- [x] Add explicit comparability and protocol caveats for multi-paper comparisons.
- [ ] Defer rigorous fairness/protocol normalization for paper comparisons to a later research-quality evaluation stage.
- [x] Add unified workspace build pipeline that orchestrates existing paper_rag stages.
- [x] Add shared artifact path resolution for downstream paper-card tools.
- [x] Add stable public API and tool capability registry for future experiment-agent integration.
- [x] Add Chinese RAG upgrade roadmap for post-baseline development planning.
- [ ] Add optional dry-run mode for workspace pipeline if needed.

## Phase 3: `experiment_agent` Planning

- [ ] Define experiment metadata schema.
- [ ] Plan result parsing and comparison workflows.
- [ ] Plan report generation outputs.
- [x] Identify initial integration points with `paper_rag`.
- [x] Expose `paper_rag.api` and `TOOL_CAPABILITIES` for future workflow/tool selection.
