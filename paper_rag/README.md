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
- Compare multiple papers using paper-card metadata without calling an LLM.
- Answer questions using retrieved chunks as evidence and append citations.
- Route user queries through metadata search, vector search, or evidence-based QA with exact query caching.

## CLI Usage

### Build Workspace Pipeline

`build_workspace.py` is the unified orchestration entry point for building local `paper_rag` artifacts. It reuses existing stage implementations instead of duplicating their logic.

Run all non-LLM build stages:

```bash
python paper_rag/scripts/build_workspace.py --all --skip_existing
```

This runs:

- PDF ingestion
- FAISS index building
- heuristic paper-card generation
- paper-card metadata cleanup

LLM enrichment is not included in `--all`; enable it explicitly because it calls the configured LLM API and may consume tokens:

```bash
python paper_rag/scripts/build_workspace.py --run_enrich --only_missing
```

Common local example with a custom embedding model:

```bash
python paper_rag/scripts/build_workspace.py \
  --all \
  --skip_existing \
  --input_dir data/raw_papers \
  --storage_dir paper_rag/storage \
  --model_name /path/to/local/bge-small-en-v1.5
```

Use `--force` to rebuild selected stages even if outputs already exist.

### Downstream Artifact Defaults

Stage 8B adds shared path resolution for downstream paper-card tools. When `--cards_path` is not provided, metadata and comparison tools prefer the best available card file under `paper_rag/storage`:

1. `paper_cards_cleaned.jsonl`
2. `paper_cards_enriched.jsonl`
3. `paper_cards.jsonl`

This keeps CLI usage simple after running `build_workspace.py`, while still allowing explicit paths for reproducible tests:

```bash
python paper_rag/scripts/compare_papers.py --cards_path paper_rag/storage/paper_cards.jsonl --keyword LoRA
```

Evidence-grounded comparison uses `paper_rag/storage/vector_store/metadata.jsonl` by default. Pass `--metadata_path` to override it.

For future integrations, `paper_rag.api` exposes stable callable entry points and a `TOOL_CAPABILITIES` registry describing whether each tool uses an LLM, uses FAISS, or writes local storage.

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
  --model_name C:\path\to\bge-small-en-v1.5
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
  --year 2025 \
  --venue CVPR
```

Use `--cards` or `--cards_path` to select a specific JSONL file. If neither is provided, the shared downstream artifact default is used.

Supported filters:

- `--year 2025`
- `--venue CVPR`
- `--keyword frequency`
- `--dataset CASIA`
- `--metric F1`
- `--baseline SomeMethod`
- `--paper_id p000001`

In Stage 5A, dataset, metric, and method keyword fields are often empty, so those filters may return no results. That is expected for the heuristic version.

### Compare Papers

Stage 7A compares paper cards using metadata only. It does not call an LLM, load embeddings, load FAISS, use topic cache, or perform semantic matching.

If `--cards_path` is omitted, the shared downstream artifact default is used.

Check CLI help:

```bash
python paper_rag/scripts/compare_papers.py --help
```

Filter by keyword:

```bash
python paper_rag/scripts/compare_papers.py --keyword LoRA --format markdown --limit 10
```

Filter by year:

```bash
python paper_rag/scripts/compare_papers.py --year 2025 --format markdown --limit 10
```

Filter by dataset:

```bash
python paper_rag/scripts/compare_papers.py --dataset CASIA --format markdown --limit 10
```

Filter by multiple paper ids:

```bash
python paper_rag/scripts/compare_papers.py --paper_id p000001 p000060 --format markdown
```

JSON output:

```bash
python paper_rag/scripts/compare_papers.py --keyword LoRA --format json --limit 5
```

Write Markdown output to a local file:

```bash
python paper_rag/scripts/compare_papers.py --keyword LoRA --format markdown --limit 10 --output paper_rag/storage/comparisons/lora_papers.md
```

Supported filters:

- `--keyword`: searches `title`, `task`, `method_keywords`, `datasets`, `metrics`, `baselines`, `summary`, and `limitations`.
- `--dataset`: searches `datasets`.
- `--metric`: searches `metrics`.
- `--year`: matches `year`.
- `--venue`: searches `venue`.
- `--paper_id`: accepts one or more paper ids.
- `--limit`: limits the number of returned records.

Use `--verbose` with Markdown output to include `baselines`, `summary`, and `limitations` in the table. JSON output always includes the full comparison fields.

### LLM-Assisted Paper Comparison Summary

Stage 7B generates a natural-language comparison summary from selected paper cards. It calls the configured LLM/API, but it still does not load embeddings, load FAISS, read PDFs, read `chunks.jsonl`, use topic cache, or perform semantic paper matching. The first version is based only on available paper-card metadata and does not provide chunk-level citations.

If `--cards_path` is omitted, the shared downstream artifact default is used.

Known limitation: comparison readability depends on paper-card metadata quality. If `title` or `title_guess` still comes from an arXiv-style filename or raw PDF filename, the comparison output may show weak titles. This is a metadata cleanup issue, not a comparison logic issue.

Configure the LLM with the same environment variables used by `ask_papers`:

```bash
export LABMATE_LLM_API_KEY="your-api-key"
export LABMATE_LLM_BASE_URL="https://api.example.com/v1"
export LABMATE_LLM_MODEL="your-chat-model"
```

Check CLI help:

```bash
python paper_rag/scripts/compare_papers_llm.py --help
```

Summarize papers selected by keyword:

```bash
python paper_rag/scripts/compare_papers_llm.py --keyword manipulation --limit 5 --answer_language en --llm_timeout 60
```

Summarize papers selected by year:

```bash
python paper_rag/scripts/compare_papers_llm.py --year 2025 --limit 5 --answer_language en --llm_timeout 60
```

Use a custom comparison question:

```bash
python paper_rag/scripts/compare_papers_llm.py --keyword localization --limit 5 --question "Compare these papers in terms of task setting, method design, datasets, metrics, and limitations." --answer_language en --llm_timeout 60
```

Chinese summary:

```bash
python paper_rag/scripts/compare_papers_llm.py --keyword localization --limit 5 --question "比较这些论文在任务设定、方法设计、数据集、评价指标和局限性上的差异。" --answer_language zh --llm_timeout 60
```

Write the Markdown summary to a local file:

```bash
python paper_rag/scripts/compare_papers_llm.py --year 2025 --limit 5 --answer_language en --output paper_rag/storage/comparisons/compare_2025.md --llm_timeout 60
```

The LLM prompt includes only compact paper-card fields:

- `paper_id`
- `title`
- `year`
- `venue`
- `task`
- `method_keywords`
- `datasets`
- `metrics`
- `baselines`
- `summary`
- `limitations`

The summary should reference papers by `paper_id` only and should explicitly note when information is not specified in the available paper cards.

### Evidence-Grounded Paper Comparison

Stage 7C generates a lightweight cited multi-paper comparison. It reuses Stage 7A filters to select papers, collects a small balanced set of evidence chunks per selected paper from chunk metadata, and calls the configured LLM/API to produce a Markdown comparison with paper/page/chunk evidence ids.

If `--cards_path` is omitted, the shared downstream artifact default is used. If `--metadata_path` is omitted, the default is `paper_rag/storage/vector_store/metadata.jsonl`.

This first version is intentionally conservative: it surfaces comparability and protocol caveats, but it does not perform rigorous fairness judgment, automatic ranking, semantic paper matching, topic-cache integration, or deep protocol normalization.

Check CLI help:

```bash
python paper_rag/scripts/compare_papers_evidence.py --help
```

Compare selected papers with evidence:

```bash
python paper_rag/scripts/compare_papers_evidence.py --year 2025 --limit 3 --chunks_per_paper 2 --answer_language en --llm_timeout 60
```

Use a custom comparison focus:

```bash
python paper_rag/scripts/compare_papers_evidence.py --keyword localization --limit 3 --chunks_per_paper 2 --question "Compare task settings, method designs, datasets, metrics, baselines, limitations, and protocol caveats." --answer_language en --llm_timeout 60
```

Write the Markdown summary to a local file:

```bash
python paper_rag/scripts/compare_papers_evidence.py --paper_id p000005 p000006 --chunks_per_paper 2 --output paper_rag/storage/comparisons/evidence_compare.md --llm_timeout 60
```

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

### Clean Paper Card Metadata

Stage 5C cleans weak paper-card title metadata without reading PDFs, calling an LLM, loading embeddings, or loading FAISS. It detects filename-like titles such as arXiv ids or raw PDF filenames, replaces them with better existing card fields or cleaned file-name titles when possible, and marks unresolved cases as `needs_review`.

Run cleanup:

```bash
python paper_rag/scripts/cleanup_paper_cards.py \
  --cards paper_rag/storage/paper_cards.jsonl \
  --output paper_rag/storage/paper_cards_cleaned.jsonl
```

Use manual title overrides for problematic papers:

```json
{
  "p000005": "SAFIRE: Segment Any Forged Image Region",
  "p000006": "Re-MTKD: Reliable Multi-Teacher Knowledge Distillation for Image Forgery Detection"
}
```

```bash
python paper_rag/scripts/cleanup_paper_cards.py \
  --cards paper_rag/storage/paper_cards.jsonl \
  --output paper_rag/storage/paper_cards_cleaned.jsonl \
  --title_overrides path/to/title_overrides.json
```

Cleanup adds metadata such as:

- `title_original`
- `title_cleanup_status`: `updated`, `unchanged`, or `needs_review`
- `title_cleanup_reason`

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
  --model_name C:\path\to\bge-small-en-v1.5
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
  --model_name C:\path\to\bge-small-en-v1.5
```

Exact query cache:

- Default cache path: `paper_rag/storage/query_cache.jsonl`.
- Only exact query string matches are cached.
- Metadata, search, and answer results can all be cached.
- Use `--use_cache false` to bypass the cache.
- Use `--llm_timeout` to limit LLM calls in `answer` mode. Default: `60` seconds.
- `query_cache.jsonl` is under `paper_rag/storage/` and should not be committed.

Optional memory-aware query:

```bash
python paper_rag/scripts/paper_query.py \
  --query "continue comparing those papers" \
  --mode answer \
  --memory true \
  --project_id default-project \
  --session_id comparison-001 \
  --memory_top_k 6 \
  --memory_token_budget 800 \
  --memory_db paper_rag/storage/memory.sqlite3 \
  --index_dir paper_rag/storage/vector_store \
  --model_name /path/to/local/bge-small-en-v1.5
```

When memory is enabled, the router creates the session if needed, recalls task state and user preferences, builds a
separate non-evidence memory context, contextualizes retrieval with relevant active task state, runs the existing
metadata/search/answer path, and appends an episode with paper-card or paper-chunk source pointers. Failed RAG calls
are also recorded as failed episodes without replacing the original exception.

Memory-aware cache entries use schema version 2 and include project, session, post-episode `memory_revision`, paper
artifact revision, and a request-options fingerprint. A cache hit does not append another episode, so the cache does
not invalidate itself. Changes to task/user memory, session, paper index/cards, Top-K, model, language, rewrite
settings, contextualized query, or memory budget cause a cache miss. Legacy query-only JSONL records remain readable
through the low-level cache API but are intentionally not reused by `paper_query`'s versioned lookup.

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
  --model_name C:\path\to\bge-small-en-v1.5 ^
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
  --model_name C:\path\to\bge-small-en-v1.5 ^
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

## Lightweight Memory Storage

`paper_rag.memory` provides the storage and policy layer used by the optional memory-aware `paper_query` flow.
Memory remains opt-in, so existing retrieval and QA behavior is unchanged when `--memory false`.

The default database is `paper_rag/storage/memory.sqlite3`. SQLite is the source of truth and FTS5 supplies the
initial lexical retrieval baseline. The storage layer supports:

- session state and a monotonic `memory_revision`;
- `task_state`, `user_fact`, and `episode` memories;
- user, paper-card, paper-chunk, and episode source references;
- active, superseded, and archived lifecycle states;
- append-and-supersede history instead of destructive overwrite;
- project/session/type/status filters and FTS5 search;
- schema migrations and transactional writes.

Python API example:

```python
from paper_rag.memory import MemorySource, MemoryStore

store = MemoryStore()
store.create_session("rag-session-001", "default-project")

memory = store.add_memory(
    kind="user_fact",
    canonical_key="answer_language",
    content="The user prefers Chinese answers.",
    project_id="default-project",
    session_id="rag-session-001",
    sources=[MemorySource(source_type="user")],
)

results = store.search_memories(
    "Chinese answers",
    project_id="default-project",
    statuses=["active"],
)
```

Paper-derived memories can retain evidence pointers with `MemorySource(source_type="paper_chunk", paper_id=...,
page_number=..., chunk_id=..., source_path=...)`. Evidence-write policy and automatic episode recording belong to
the deterministic writer layer. This layer does not use an LLM; `paper_query` invokes it after memory-enabled tasks.

```python
from paper_rag.memory import MemoryWriter

writer = MemoryWriter(store)

writer.remember_user_fact(
    canonical_key="answer_language",
    content="The user prefers Chinese answers.",
    project_id="default-project",
    session_id="rag-session-001",
    explicit_user_request=True,
)

writer.record_rag_episode(
    query="Explain the method.",
    route="answer",
    outcome="success",
    result_summary="The paper uses frequency features.",
    project_id="default-project",
    session_id="rag-session-001",
    retrieved_sources=[
        MemorySource(
            source_type="paper_chunk",
            paper_id="p000001",
            page_number=4,
            chunk_id="p000001-c0003",
        )
    ],
    contains_research_claims=True,
)
```

Deterministic write rules currently enforce the following boundaries:

- unclassified interactions and implicit preferences produce `NOOP`;
- stable user facts require an explicit remember request and a user source;
- conflicting values require an explicit correction and preserve the superseded record;
- repeated facts, unchanged task state, identical corrections, and repeated archives produce `NOOP`;
- task state changes use append-and-supersede versioning;
- completed RAG tasks are stored as episodes, not silently promoted to stable facts;
- successful answer/compare summaries and explicit research claims require paper-chunk evidence;
- forgetting archives records instead of physically deleting them.

Memory retrieval is also available as a standalone layer:

```python
from paper_rag.memory import (
    MemoryContextBuilder,
    MemoryContextConfig,
    MemoryRetrievalConfig,
    MemoryRetriever,
)

retriever = MemoryRetriever(store)
builder = MemoryContextBuilder(retriever)

packet = builder.build(
    "continue comparing the selected papers",
    project_id="default-project",
    session_id="rag-session-001",
    config=MemoryContextConfig(
        token_budget=800,
        retrieval=MemoryRetrievalConfig(
            top_k=6,
            candidate_k=50,
            kinds=("task_state", "user_fact", "episode"),
        ),
    ),
)
print(packet.text)
```

The retriever uses FTS5 with a dependency-free CJK substring fallback, then filters by project, session, type,
lifecycle, observation time, and validity interval. Session recall can see the current session plus project-global
memory, while recall without a session sees project-global memory only. Final ranking combines lexical rank, scope,
confidence, importance, source quality, and recency. Every result exposes its score components and recall reasons.
The context builder enforces a fixed approximate token budget and marks truncated entries. Its header explicitly
states that memory may guide query interpretation and preferences but is not citable paper evidence.

### Memory management CLI

Use `memory_cli.py` to inspect and explicitly manage the SQLite memory source of truth. Every command emits JSON.

```bat
python paper_rag/scripts/memory_cli.py list ^
  --project_id default-project ^
  --session_id rag-session-001

python paper_rag/scripts/memory_cli.py search "frequency features" ^
  --project_id default-project ^
  --session_id rag-session-001 ^
  --include_global

python paper_rag/scripts/memory_cli.py show MEMORY_ID

python paper_rag/scripts/memory_cli.py add ^
  --project_id default-project ^
  --session_id rag-session-001 ^
  --kind user_fact ^
  --key answer_language ^
  --content "The user prefers Chinese answers."

python paper_rag/scripts/memory_cli.py correct MEMORY_ID ^
  --project_id default-project ^
  --session_id rag-session-001 ^
  --content "The user prefers English answers."

python paper_rag/scripts/memory_cli.py archive MEMORY_ID ^
  --project_id default-project ^
  --session_id rag-session-001
```

`update` is an alias for `correct`. Corrections are limited to active `user_fact` records and create a new version;
archives retain the record and its provenance. Scope checks prevent a command from changing memory in another
project or session. Repeat `--kind` or `--status` to select multiple values.

Offline consolidation is deterministic and dry-run by default:

```bat
python paper_rag/scripts/memory_cli.py consolidate ^
  --project_id default-project ^
  --session_id rag-session-001

python paper_rag/scripts/memory_cli.py consolidate ^
  --project_id default-project ^
  --session_id rag-session-001 ^
  --apply
```

The dry run reports exact duplicate episode groups and the proposed session summary without changing SQLite.
`--apply` keeps the newest episode in each exact structured duplicate group, archives the older copies, and writes
route/outcome counts, evidence coverage, and recent queries to `session.state.memory_consolidation`. Different paper
chunk evidence is never treated as a duplicate. This version intentionally performs zero automatic fact promotions:
an answer summary cannot silently become a stable research fact.

### Memory evaluation and observability

`memory_eval.py` runs offline and never calls an LLM. Audit the current memory database with:

```bat
python paper_rag/scripts/memory_eval.py audit ^
  --db paper_rag/storage/memory.sqlite3 ^
  --project_id default-project
```

The audit reports lifecycle/provenance violations, multiple active versions of one canonical key, exact active
duplicates, the memory redundancy ratio, and whether the scan reached its configured limit.

Retrieval benchmarks are JSONL. Each case must provide exactly one relevance representation: memory IDs or
canonical keys. `forbidden_memory_ids` identifies stale or otherwise harmful memories that must not be recalled.

```json
{"case_id":"language","query":"Which language should you answer in?","project_id":"default-project","session_id":"rag-session-001","relevant_canonical_keys":["answer_language"],"forbidden_memory_ids":[]}
```

```bat
python paper_rag/scripts/memory_eval.py retrieval ^
  --db paper_rag/storage/memory.sqlite3 ^
  --benchmark memory_benchmark.jsonl ^
  --top_k 6 ^
  --candidate_k 50
```

The report includes Memory Recall@K, Precision@K, MRR, nDCG@K, stale-memory error rate, stale-case rate, and
per-case returned IDs. Use fixed database snapshots when comparing retrieval configurations.

For memory on/off comparisons, save one JSONL file per variant. Records are paired by `case_id`; supported numeric
fields are `task_success`, `latency_ms`, `token_count`, `citation_accuracy`, and `answer_faithfulness`.

```json
{"case_id":"q001","task_success":true,"latency_ms":820.4,"token_count":640,"citation_accuracy":1.0,"answer_faithfulness":0.9}
```

```bat
python paper_rag/scripts/memory_eval.py compare ^
  --baseline results_memory_off.jsonl ^
  --candidate results_memory_on.jsonl
```

Only paired cases with numeric values contribute to each metric, and the report exposes unpaired case IDs. Positive
deltas mean the candidate value is larger; whether that is desirable depends on the metric (higher success is good,
higher latency is not).

Every `paper_query` response also contains an `observability` object with total latency, memory preparation, cache
lookup, RAG execution, memory write, and cache write timings, plus result count, recalled-memory count, and estimated
memory-context tokens. These measurements are diagnostic wall-clock timings, not a substitute for repeated benchmark
runs with warm-up and fixed inputs.

### Deterministic scale testing

`memory_scale_test.py` builds a disposable synthetic database and exercises multi-session writes, corrections,
archives, duplicate consolidation, retrieval evaluation, scope isolation, and the final audit. With no `--db`, the
database is created under the system temporary directory and automatically removed. A supplied `--db` must not
already exist, so the command cannot overwrite an existing memory database.

```bat
python paper_rag/scripts/memory_scale_test.py ^
  --sessions 100 ^
  --facts_per_session 50 ^
  --episodes_per_session 15 ^
  --duplicate_pairs_per_session 3 ^
  --global_facts 100 ^
  --queries 1000 ^
  --top_k 6 ^
  --seed 20260826
```

The JSON report includes configuration, database size, memory/session/query counts, retrieval metrics, invariant
checks, stage timings, audit health, and approximate operations per second. The workload uses public memory APIs and
real SQLite/FTS5 operations; it does not sleep, mock storage, or call an LLM.

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

Downstream tools may also read:

- `paper_cards_enriched.jsonl`: LLM-enriched paper cards.
- `paper_cards_cleaned.jsonl`: metadata-cleaned paper cards, preferred by default when present.

`query_cache.jsonl` fields:

- `query`
- `mode`
- `answer`
- `results`
- `created_at`
- `cache_schema_version` (`2` for versioned entries)
- `project_id`
- `session_id`
- `memory_revision`
- `paper_revision`
- `request_fingerprint`

Runtime `paper_query` responses additionally contain `observability.total_ms`, `observability.stages_ms`,
`observability.result_count`, `observability.recalled_memory_count`, and
`observability.memory_context_estimated_tokens`.

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
