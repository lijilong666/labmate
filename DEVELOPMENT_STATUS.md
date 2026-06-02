# Development Status

## Project Overview

LabMate is a lightweight AI lab assistant for deep learning research workflows. It aims to reduce repetitive work around paper retrieval, experiment tracking, result summarization, and report generation.

The current development priority is `paper_rag`. The later module, `experiment_agent`, will focus on experiment management, result analysis, and reporting.

## Completed `paper_rag` Stages

### Stage 1: PDF Inventory and Chunk Ingestion

- Implemented `paper_rag/scripts/ingest_pdfs.py`.
- Generated `paper_rag/storage/paper_inventory.csv` and `paper_rag/storage/chunks.jsonl` locally.
- Tested on 81 PDFs.
- Produced 5394 chunks with 0 failures.

### Stage 2: FAISS Vector Index Building

- Implemented `paper_rag/scripts/build_index.py`.
- Uses `sentence-transformers` and FAISS.
- Tested with local model path: `D:\Work\models\bge-small-en-v1.5`.
- Built an index with 5394 chunks and embedding dimension 384.

### Stage 3: Vector Search

- Implemented `search_papers`.
- Added CLI: `paper_rag/scripts/search_papers.py`.
- Search results include `score`, `chunk_id`, `paper_id`, `source_file`, `page_number`, and `text`.

### Stage 4: Evidence-Based QA

- Implemented `ask_papers`.
- Uses an OpenAI-compatible API.
- Supports DeepSeek, Qwen, OpenAI-style endpoints through environment variables.
- Supports optional query rewrite and citation output.
- Tested once with DeepSeek V4.

### Stage 5A: Heuristic Paper Cards and Metadata Search

- Implemented `paper_rag/scripts/generate_paper_cards.py`.
- Implemented `paper_rag/scripts/metadata_search.py`.
- Supports `--limit` for testing.
- Tested with 5 paper cards and year / paper id search.

### Stage 5B: LLM-Assisted Paper Card Completion

- Implemented `paper_rag/scripts/enrich_paper_cards.py`.
- Added reusable OpenAI-compatible LLM client in `paper_rag/src/paper_rag/llm_client.py`.
- Reused the shared LLM client from `ask_papers`.
- Enriches paper cards from limited chunks instead of sending all chunks to the LLM.
- Supports `--paper_id`, `--limit`, and `--only_missing` to control token cost.
- Writes `enrichment_status` and `enrichment_error` per paper card.
- Metadata search now also supports enriched fields, including baselines.

### Stage 5C: Paper Card Metadata Cleanup

- Implemented `paper_rag/scripts/cleanup_paper_cards.py`.
- Added `cleanup_paper_cards` for rule-based title cleanup and review marking.
- Detects weak `title` / `title_guess` values such as arXiv ids, raw PDF filenames, and article-text filenames.
- Uses better existing card fields or cleaned file names when available.
- Marks cards as `needs_review` when no better title can be recovered without reading PDFs or calling an LLM.
- Supports optional manual title overrides through a JSON file keyed by `paper_id`.
- Does not read PDFs, call an LLM, load embeddings, or load FAISS.

### Stage 6A: Query Router and Exact Query Cache

- Implemented `paper_rag/scripts/paper_query.py`.
- Added `route_query` and `paper_query` as unified metadata / search / answer entry points.
- Added exact-match query cache in `paper_rag/storage/query_cache.jsonl`.
- Auto routing uses simple rules only; it does not call an LLM.
- Cache supports metadata, vector search, and answer results.
- Tested metadata routing and exact cache hit behavior without calling an LLM.

### Stage 6B: Topic Cache

- Implemented `paper_rag/scripts/topic_cache.py` and `get_topic_summary` for topic cache access.
- Added topic-level exact cache storage in `paper_rag/storage/topic_cache.jsonl`.
- Cache lookup is by exact topic key only; semantic cache, automatic topic mining, and LLM routing are intentionally not included.
- On cache miss or `--force_refresh`, the implementation reuses existing vector search and evidence-grounded QA generation.
- On cache hit, it returns the stored topic summary without search or LLM calls.

### Stage 7A: Metadata-Based Multi-Paper Comparison

- Implemented `paper_rag/scripts/compare_papers.py`.
- Added `compare_papers` as a metadata-only comparison API over paper cards.
- Supports filters for keyword, dataset, metric, year, venue, paper id, and limit.
- Supports Markdown and JSON output, plus optional file output.
- Does not call an LLM, load embeddings, load FAISS, use topic cache, or perform semantic matching.

### Stage 7B: LLM-Assisted Multi-Paper Comparison Summary

- Implemented `paper_rag/scripts/compare_papers_llm.py`.
- Added `compare_papers_with_llm` for natural-language comparison summaries over selected paper cards.
- Reuses Stage 7A filtering and normalization logic.
- Uses only compact paper-card fields and the existing OpenAI-compatible LLM client.
- Does not read PDFs, read chunks, load embeddings, load FAISS, use topic cache, perform semantic matching, or provide chunk-level citations.
- Testing confirmed that it can filter paper cards and generate LLM-assisted multi-paper comparison summaries.
- Known metadata quality issue: some paper cards still contain weak `title` / `title_guess` values from arXiv-style filenames or raw PDF filenames, such as `2412.08197v1` for SAFIRE (p000005) and `2504.05224v1` for Re-MTKD (p000006). This affects readability of `compare_papers` and `compare_papers_llm` outputs, but it is a paper-card metadata cleanup issue rather than a Stage 7B comparison-logic issue.

### Stage 7C: Lightweight Evidence-Grounded Multi-Paper Synthesis

- Implemented `paper_rag/scripts/compare_papers_evidence.py`.
- Added `compare_papers_with_evidence` for cited multi-paper comparison summaries.
- Reuses Stage 7A filters to select paper cards.
- Collects a small balanced evidence budget per selected paper from chunk metadata.
- Uses the existing OpenAI-compatible LLM client to generate Markdown summaries with paper/page/chunk evidence ids.
- Includes prompt-level instructions for `Comparability and Protocol Caveats`.
- The first version surfaces protocol caveats but does not perform rigorous fairness judgment, automatic ranking, semantic paper matching, topic-cache integration, or deep protocol normalization.

### Stage 8A: Unified Workspace Build Pipeline

- Implemented `paper_rag/scripts/build_workspace.py`.
- Added `paper_rag.pipeline.build_workspace` as a lightweight orchestration layer.
- Reuses existing stage functions instead of reimplementing ingestion, indexing, paper-card generation, enrichment, or cleanup.
- Supports `--all` for non-LLM stages: ingest, index, paper cards, and cleanup.
- Keeps LLM enrichment explicit through `--run_enrich` so token-consuming work is opt-in.
- Supports `--skip_existing`, `--force`, path overrides, embedding model options, enrichment options, and title overrides.
- Prints per-stage summaries and recommends the downstream paper-card path.

### Stage 8B: Public API and Artifact Defaults

- Added `paper_rag.paths` with shared default path resolution for downstream tools.
- Added `paper_rag.api` as a stable integration surface for future `experiment_agent` workflows.
- Added `TOOL_CAPABILITIES` so callers can inspect which tools use an LLM, use FAISS, or write local storage.
- Updated metadata and comparison tools to prefer downstream paper-card artifacts in this order when no explicit path is passed:
  1. `paper_rag/storage/paper_cards_cleaned.jsonl`
  2. `paper_rag/storage/paper_cards_enriched.jsonl`
  3. `paper_rag/storage/paper_cards.jsonl`
- Preserved explicit `--cards_path` behavior for reproducible runs.
- Kept Stage 8B as integration cleanup only; it does not add new retrieval, LLM, embedding, or UI features.

### CLI Stability Fixes

- Added UTF-8 stdout/stderr configuration for result-printing CLI entry points.
- This avoids Windows console `UnicodeEncodeError` failures when retrieved paper chunks contain symbols or non-GBK characters.

## Encoding Notes

- CSV inventory reads/writes are explicit: `encoding="utf-8-sig"`.
- JSONL reads/writes are explicit: `encoding="utf-8"`.
- JSON output uses `ensure_ascii=False`.
- Existing local generated files were verified with Python, and Chinese path segments read correctly.
- If a terminal or spreadsheet viewer still shows mojibake, regenerate local outputs with the current scripts:
  - `paper_rag/scripts/ingest_pdfs.py`
  - `paper_rag/scripts/generate_paper_cards.py`
  - `paper_rag/scripts/enrich_paper_cards.py` if enriched cards are needed.

## Local Generated Files Not Committed

- `data/raw_papers/`
- `paper_rag/storage/paper_inventory.csv`
- `paper_rag/storage/chunks.jsonl`
- `paper_rag/storage/vector_store/`
- `paper_rag/storage/paper_cards.jsonl`
- `paper_rag/storage/paper_cards_enriched.jsonl`
- `paper_rag/storage/query_cache.jsonl`
- `paper_rag/storage/topic_cache.jsonl`
- `paper_rag/model_cache/`
- `.env`

## Recommended Next Stages

- Stage 7C follow-up.
  - Improve evidence selection quality and coverage diagnostics.
  - Consider protocol-aware extraction after the lightweight cited comparison path is stable.
- Stage 8 follow-up.
  - Add optional dry-run output for the workspace pipeline if needed.
  - Consider a small config file only after CLI parameters become hard to manage.
  - Keep `paper_rag.api.TOOL_CAPABILITIES` updated when new callable tools are added.
- Metadata cleanup follow-up for paper cards.
  - Add first-page title extraction or LLM-assisted title repair for cards still marked `needs_review`.
  - Re-run cleanup or enrichment for cards whose `title_guess` matches filename-like patterns such as `2412.08197v1`, `2504.05224v1`, or raw PDF filenames.
- Stage 8: Public API cleanup and simple CLI / UI.
  - Keep script entry points under `paper_rag/scripts/`.
  - Keep core logic under `paper_rag/src/paper_rag/`.

## Future `experiment_agent` Preparation

- `paper_rag` should remain a lightweight literature knowledge service.
- Future `experiment_agent` workflows should call Python APIs directly instead of shelling out to scripts.
- Stable public APIs to preserve or refine:
  - `build_workspace(...)`
  - `search_papers(...)`
  - `ask_papers(...)`
  - `cleanup_paper_cards(...)`
  - `paper_query(...)`
  - `get_topic_summary(...)`
  - `compare_papers(...)`
  - `compare_papers_with_evidence(...)`
  - `compare_papers_with_llm(...)`
- Return structured records where possible so experiment workflows can reuse paper ids, datasets, metrics, baselines, limitations, citations, and source chunks.
- Do not introduce LangGraph, LangChain, or experiment lifecycle state into `paper_rag`.
- Store experiment analysis records on the `experiment_agent` side, with references back to RAG sources when needed.

## Interview Preparation Notes

- Added `docs/paper_rag_interview_knowledge_base.md` as a personal study note for explaining the current `paper_rag` project in interviews.
- The note is written in Chinese and summarizes the RAG architecture, stage-by-stage development process, key engineering decisions, limitations, and common interview Q&A.
- Updated the interview note through Stage 8B, including the unified pipeline, public API, artifact defaults, and `experiment_agent` integration boundary.
- Added `docs/rag_upgrade_roadmap_zh.md` as a Chinese summary of practical RAG upgrade directions after the baseline framework is complete.
- The upgrade note covers metadata quality, retrieval evaluation, incremental indexing, evidence selection, protocol-aware comparison, cache invalidation, structured outputs, and future `experiment_agent` integration.
- This document is for development/interview preparation, not public-facing product documentation.

## Environment Notes

- Personal Windows conda env path: `D:\Work\conda_envs\labmate`
- Personal Windows Python executable: `D:\Work\conda_envs\labmate\python.exe`
- Recommended Python version: 3.10
- Local embedding model path used during testing: `D:\Work\models\bge-small-en-v1.5`
- Avoid using Anaconda base if possible.
- Before running scripts, check the Python executable.

Example:

```bash
D:\Work\conda_envs\labmate\python.exe -c "import sys; print(sys.executable)"
```

Personal Stage 6B topic cache test command:

```bat
D:\Work\conda_envs\labmate\python.exe paper_rag\scripts\topic_cache.py ^
  --topic frequency_domain_features ^
  --query "Explain what frequency-domain features are used for in image manipulation localization." ^
  --cache_path paper_rag/storage/topic_cache.jsonl ^
  --index_dir paper_rag/storage/vector_store ^
  --model_name D:\Work\models\bge-small-en-v1.5 ^
  --cache_dir paper_rag/model_cache ^
  --top_k 8 ^
  --answer_language en ^
  --rewrite_query false ^
  --llm_timeout 60
```

## Development Workflow

- GitHub is the single source of truth.
- Before starting work on a new device, run `git pull`.
- After finishing a small task, run `git status`, `git add`, `git commit`, and `git push`.
- Do not commit data, PDFs, vector stores, model weights, API keys, or `.env` files.
