# `paper_rag`

`paper_rag` is the first core LabMate module. It provides local research paper ingestion and FAISS index building now, and will later add retrieval and question answering.

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

The current implementation includes PDF inventory scanning, text ingestion, and local FAISS index building. It does not call an LLM.

It can:

- Recursively scan an input directory for PDF files.
- Assign stable sequential paper ids such as `p000001`.
- Write `paper_rag/storage/paper_inventory.csv`.
- Extract PDF text page by page using PyMuPDF, with pypdf as a fallback.
- Split page text into fixed-length chunks.
- Write chunk records to `paper_rag/storage/chunks.jsonl`.
- Build a local FAISS vector index from `chunks.jsonl` using sentence-transformers.
- Save chunk metadata alongside the FAISS index for later retrieval.

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

## Directory Layout

- `src/`: module source code.
- `configs/`: local configuration templates, to be added later.
- `scripts/`: helper scripts and command entry points.
- `storage/`: local runtime storage for indexes and caches.

## Storage Rules

Do not commit downloaded papers, extracted text, vector indexes, API keys, model outputs, or other generated data. Keep `storage/` for local runtime files only.
