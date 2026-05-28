# `paper_rag`

`paper_rag` is the first core LabMate module. It provides local research paper ingestion, FAISS index building, vector search, heuristic and LLM-assisted paper cards, metadata search, and basic evidence-grounded question answering.

## Goal

Help researchers quickly search, inspect, and ask questions about a local paper collection while keeping source citations visible.

## MVP Scope

The full MVP should support:

- Loading a small local set of paper documents.
- Extracting paper text.
- Splitting text into chunks.
- Building a local retrieval index.
- Retrieving relevant chunks for a user question.
- Generating an answer with source citations.

## Current Implementation

The current implementation includes PDF inventory scanning, text ingestion, local FAISS index building, vector search, heuristic paper cards, LLM-assisted paper card enrichment, metadata search, and basic paper QA through an OpenAI-compatible LLM API.

It can:

- Recursively scan an input directory for PDF files.
- Assign stable sequential paper ids such as `p000001`.
- Write `paper_rag/storage/paper_inventory.csv`.
- Extract PDF text page by page using PyMuPDF, with pypdf as a fallback.
- Split page text into fixed-length chunks.
- Write chunk records to `paper_rag/storage/chunks.jsonl`.
- Build a local FAISS vector index from `chunks.jsonl` using sentence-transformers.
- Save chunk metadata alongside the FAISS index for later retrieval.
- Search indexed chunks with a query embedding and return ranked source chunks.
- Generate heuristic `paper_cards.jsonl` from the paper inventory.
- Enrich paper cards with LLM-extracted fields from limited paper chunks.
- Search paper cards by metadata such as year, venue, keyword, dataset, metric, or paper id.
- Answer questions using retrieved chunks as evidence and append citations.
- Route user queries through metadata search, vector search, or evidence-based QA with exact query caching.

## CLI Usage

### PDF Ingestion

Install PyMuPDF in the runtime environment before ingestion. The script uses PyMuPDF first and falls back to pypdf only when PyMuPDF is unavailable.

```bash
python paper_rag/scripts/ingest_pdfs.py \
  --input_dir data/raw_papers \
  --inventory paper_rag/storage/paper_inventory.csv \
  --output paper_rag/storage/chunks.jsonl
```

Optional chunking arguments:

```bash
python paper_rag/scripts/ingest_pdfs.py \
  --input_dir data/raw_papers \
  --inventory paper_rag/storage/paper_inventory.csv \
  --output paper_rag/storage/chunks.jsonl \
  --chunk_size 1200 \
  --chunk_overlap 150
```

If a PDF fails to parse, ingestion continues and the failure is recorded in the inventory with `status=failed` and an error message.

### Build Vector Index

Install `sentence-transformers` and `faiss-cpu` before building the index.

```bash
python paper_rag/scripts/build_index.py \
  --chunks paper_rag/storage/chunks.jsonl \
  --index_dir paper_rag/storage/vector_store
```

The default embedding model is `BAAI/bge-small-en-v1.5`.

By default, sentence-transformers model files are cached in `paper_rag/model_cache/` instead of the system cache directory. This directory is ignored by Git and should not be committed.

You can pass either a Hugging Face model name or a local model path:

```bash
python paper_rag/scripts/build_index.py \
  --chunks paper_rag/storage/chunks.jsonl \
  --index_dir paper_rag/storage/vector_store \
  --model_name /path/to/local/bge-small-en-v1.5 \
  --cache_dir /path/to/model/cache
```

Use `--cache_dir` to choose a different model cache directory.

If model loading or downloading fails, use a local model path with `--model_name`, set `--cache_dir`, or configure your own Hugging Face / ModelScope mirror source outside the code and retry.

### Search Papers

Search uses the existing FAISS index and the same embedding model used to build it. It returns ranked chunks only; it does not call an LLM or generate a summary.

```bash
python paper_rag/scripts/search_papers.py \
  --query "frequency-domain features" \
  --top_k 5 \
  --index_dir paper_rag/storage/vector_store \
  --model_name /path/to/local/bge-small-en-v1.5
```

On Windows, for example:

```bash
python paper_rag/scripts/search_papers.py \
  --query "frequency-domain features" \
  --top_k 5 \
  --index_dir paper_rag/storage/vector_store \
  --model_name D:\Work\models\bge-small-en-v1.5
```

Python API:

```python
from paper_rag import search_papers

results = search_papers(
    query="frequency-domain features",
    top_k=5,
    index_dir="paper_rag/storage/vector_store",
    model_name="/path/to/local/bge-small-en-v1.5",
)
```

### Generate Paper Cards

Stage 5A paper cards are heuristic and do not call an LLM. Fields such as `authors`, `task`, `method_keywords`, `datasets`, `metrics`, `baselines`, `summary`, and `limitations` may be empty until richer extraction is added.

Generate cards from the inventory:

```bash
python paper_rag/scripts/generate_paper_cards.py \
  --inventory paper_rag/storage/paper_inventory.csv \
  --output paper_rag/storage/paper_cards.jsonl
```

Use `--limit` for quick tests:

```bash
python paper_rag/scripts/generate_paper_cards.py \
  --inventory paper_rag/storage/paper_inventory.csv \
  --output paper_rag/storage/paper_cards.jsonl \
  --limit 5
```

The heuristic generator estimates:

- `paper_id`, `source_file`, `file_name`, and `parent_dir` from the inventory.
- `year` and `venue` from parent directory and file name patterns.
- `title_guess` from a cleaned PDF file name.
- `title` as the same value as `title_guess`.

### Metadata Search

Search paper cards without vector retrieval or LLM calls:

```bash
python paper_rag/scripts/metadata_search.py \
  --cards paper_rag/storage/paper_cards.jsonl \
  --year 2025 \
  --venue CVPR
```

Supported filters:

- `--year 2025`
- `--venue CVPR`
- `--keyword frequency`
- `--dataset CASIA`
- `--metric F1`
- `--baseline SomeMethod`
- `--paper_id p000001`

In Stage 5A, dataset, metric, and method keyword fields are often empty, so those filters may return no results. That is expected for the heuristic version.

### Enrich Paper Cards

Stage 5B enriches existing paper cards with an OpenAI-compatible LLM. It reads limited chunks for each paper and asks the LLM to extract only evidence-supported metadata:

- `task`
- `method_keywords`
- `datasets`
- `metrics`
- `baselines`
- `summary`
- `limitations`

This step calls an LLM API and consumes tokens. Start with one paper or a small limit before running it on the full collection.

Configure the LLM with the same environment variables used by `ask_papers`:

```bash
export LABMATE_LLM_API_KEY="your-api-key"
export LABMATE_LLM_BASE_URL="https://api.example.com/v1"
export LABMATE_LLM_MODEL="your-chat-model"
```

Enrich one paper:

```bash
python paper_rag/scripts/enrich_paper_cards.py \
  --cards paper_rag/storage/paper_cards.jsonl \
  --chunks paper_rag/storage/chunks.jsonl \
  --output paper_rag/storage/paper_cards_enriched.jsonl \
  --paper_id p000001
```

Small batch test:

```bash
python paper_rag/scripts/enrich_paper_cards.py \
  --cards paper_rag/storage/paper_cards.jsonl \
  --chunks paper_rag/storage/chunks.jsonl \
  --output paper_rag/storage/paper_cards_enriched.jsonl \
  --limit 3 \
  --only_missing
```

Optional overrides:

- `--llm_model`: override `LABMATE_LLM_MODEL`.
- `--llm_base_url`: override `LABMATE_LLM_BASE_URL`.
- `--llm_timeout`: LLM request timeout in seconds. Default: `60`.
- `--only_missing`: fill empty fields without overwriting existing non-empty values.

If an LLM response cannot be parsed as JSON, the batch continues and the card is marked with `enrichment_status="failed"` and an `enrichment_error` message.

### Ask Papers

`ask_papers` runs retrieval first, then calls an OpenAI-compatible LLM endpoint to answer from the retrieved evidence. It does not train models or modify the vector index.

Configure the LLM with environment variables:

```bash
export LABMATE_LLM_API_KEY="your-api-key"
export LABMATE_LLM_BASE_URL="https://api.example.com/v1"
export LABMATE_LLM_MODEL="your-chat-model"
```

On Windows Command Prompt:

```bat
set LABMATE_LLM_API_KEY=your-api-key
set LABMATE_LLM_BASE_URL=https://api.example.com/v1
set LABMATE_LLM_MODEL=your-chat-model
```

Run QA:

```bash
python paper_rag/scripts/ask_papers.py \
  --question "What are common frequency-domain methods in image manipulation localization?" \
  --top_k 5 \
  --index_dir paper_rag/storage/vector_store \
  --model_name /path/to/local/bge-small-en-v1.5
```

On Windows, for example:

```bash
python paper_rag/scripts/ask_papers.py \
  --question "What are common frequency-domain methods in image manipulation localization?" \
  --top_k 5 \
  --index_dir paper_rag/storage/vector_store \
  --model_name D:\Work\models\bge-small-en-v1.5
```

Optional arguments:

- `--answer_language auto|zh|en`: `auto` follows the question language.
- `--rewrite_query true|false`: when enabled, non-English questions are rewritten into an English retrieval query before vector search.
- `--llm_model`: override `LABMATE_LLM_MODEL`.
- `--llm_base_url`: override `LABMATE_LLM_BASE_URL`.
- `--llm_timeout`: LLM request timeout in seconds. Default: `60`.

Answers are instructed to use only retrieved chunks. If evidence is insufficient, the answer should say `evidence is insufficient`. The final output always includes citations:

```text
Sources:
[1] paper.pdf, page 4, chunk_id=...
[2] another_paper.pdf, page 7, chunk_id=...
```

Python API:

```python
from paper_rag import ask_papers

result = ask_papers(
    question="哪些论文使用了频域特征？",
    top_k=5,
    index_dir="paper_rag/storage/vector_store",
    model_name="/path/to/local/bge-small-en-v1.5",
    answer_language="auto",
    rewrite_query=True,
)
print(result["answer"])
```

### Unified Paper Query

`paper_query.py` is the unified entry point for paper retrieval workflows. It supports three execution modes:

- `metadata`: search `paper_cards.jsonl`; does not call an LLM.
- `search`: vector-search chunks with FAISS; does not call an LLM.
- `answer`: call `ask_papers`; retrieves evidence and then calls an LLM.

Use `--mode auto` to route by simple rules. The router does not use an LLM:

- Metadata-style requests such as `which papers`, `list papers`, `find papers`, `哪些论文`, or `论文列表` prefer metadata search when filters can be extracted.
- Search-style requests such as `search`, `find chunks`, `retrieve`, `检索`, or `查找片段` use vector search.
- Answer-style requests such as `summarize`, `explain`, `compare`, `why`, `how`, `总结`, `解释`, `比较`, `为什么`, or `如何` use QA.
- If routing is unclear, auto mode defaults to `answer`.

Example:

```bash
python paper_rag/scripts/paper_query.py \
  --query "which papers use LoRA?" \
  --mode auto \
  --cards paper_rag/storage/paper_cards_enriched.jsonl \
  --index_dir paper_rag/storage/vector_store \
  --model_name /path/to/local/bge-small-en-v1.5
```

On Windows, for example:

```bash
python paper_rag/scripts/paper_query.py \
  --query "which papers use LoRA?" \
  --mode auto \
  --cards paper_rag/storage/paper_cards_enriched.jsonl \
  --index_dir paper_rag/storage/vector_store \
  --model_name D:\Work\models\bge-small-en-v1.5
```

Exact query cache:

- Default cache path: `paper_rag/storage/query_cache.jsonl`.
- Only exact query string matches are cached.
- Metadata, search, and answer results can all be cached.
- Use `--use_cache false` to bypass the cache.
- Use `--llm_timeout` to limit LLM calls in `answer` mode. Default: `60` seconds.
- `query_cache.jsonl` is under `paper_rag/storage/` and should not be committed.

### Topic Cache

`topic_cache.py` stores stable topic-level RAG summaries for repeated domain questions. It uses exact topic keys only. It does not implement semantic cache, automatic topic mining, or LLM-based routing.

Check CLI help:

```bat
python paper_rag/scripts/topic_cache.py --help
```

Recommended Windows Command Prompt test command:

```bat
python paper_rag/scripts/topic_cache.py ^
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

Expected behavior:

- First run: prints `Cache hit: False`, retrieves evidence, calls the configured LLM, and writes `paper_rag/storage/topic_cache.jsonl`.
- Second run with the same `--topic`: prints `Cache hit: True` and returns the cached answer without calling the LLM/API.
- Add `--force_refresh` to refresh an existing topic summary.

Force refresh example:

```bat
python paper_rag/scripts/topic_cache.py ^
  --topic frequency_domain_features ^
  --query "Explain what frequency-domain features are used for in image manipulation localization." ^
  --cache_path paper_rag/storage/topic_cache.jsonl ^
  --index_dir paper_rag/storage/vector_store ^
  --model_name D:\Work\models\bge-small-en-v1.5 ^
  --top_k 8 ^
  --answer_language en ^
  --rewrite_query false ^
  --llm_timeout 60 ^
  --force_refresh
```

Supported topic cache arguments:

- `--topic`
- `--query`
- `--cache_path`
- `--index_dir`
- `--model_name`
- `--cache_dir`
- `--top_k`
- `--answer_language`
- `--rewrite_query`
- `--force_refresh`
- `--llm_timeout`
- `--llm_base_url`
- `--llm_model`

`topic_cache.jsonl` is under `paper_rag/storage/` and should not be committed.

## Outputs

`paper_inventory.csv` fields:

- `paper_id`
- `source_path`
- `file_name`
- `parent_dir`
- `file_size`
- `modified_time`
- `status`
- `error`

`chunks.jsonl` fields:

- `chunk_id`
- `paper_id`
- `source_path`
- `file_name`
- `page_number`
- `chunk_index`
- `text`

`vector_store/` files:

- `index.faiss`: local FAISS index.
- `metadata.jsonl`: chunk metadata aligned with FAISS vector ids.
- `manifest.json`: index build metadata.

`paper_cards.jsonl` fields:

- `paper_id`
- `title`
- `title_guess`
- `year`
- `venue`
- `authors`
- `source_file`
- `file_name`
- `parent_dir`
- `task`
- `method_keywords`
- `datasets`
- `metrics`
- `baselines`
- `summary`
- `limitations`
- `status`
- `extraction_mode`

`query_cache.jsonl` fields:

- `query`
- `mode`
- `answer`
- `results`
- `created_at`

`topic_cache.jsonl` fields:

- `topic`
- `query`
- `answer_language`
- `answer`
- `sources`
- `created_at`
- `updated_at`
- `model`
- `top_k`

## Directory Layout

- `src/`: module source code.
- `configs/`: local configuration templates, to be added later.
- `scripts/`: helper scripts and command entry points.
- `storage/`: local runtime storage for indexes and caches.

## Storage Rules

Do not commit downloaded papers, extracted text, vector indexes, API keys, model outputs, or other generated data. Keep `storage/` for local runtime files only.
