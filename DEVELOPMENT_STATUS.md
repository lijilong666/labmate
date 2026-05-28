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

- Stage 7: Multi-paper comparison.
- Stage 8: Public API cleanup and simple CLI / UI.

## Environment Notes

- Windows conda env path: `D:\Work\conda_envs\labmate`
- Recommended Python version: 3.10
- Local embedding model path used during testing: `D:\Work\models\bge-small-en-v1.5`
- Avoid using Anaconda base if possible.
- Before running scripts, check the Python executable.

Example:

```bash
python -c "import sys; print(sys.executable)"
```

## Development Workflow

- GitHub is the single source of truth.
- Before starting work on a new device, run `git pull`.
- After finishing a small task, run `git status`, `git add`, `git commit`, and `git push`.
- Do not commit data, PDFs, vector stores, model weights, API keys, or `.env` files.
