# `paper_rag`

`paper_rag` is the first core LabMate module. It provides local research paper ingestion now and will later add retrieval and question answering.

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

The current implementation includes PDF inventory scanning and text ingestion only. It does not call an LLM and does not build a vector index.

It can:

- Recursively scan an input directory for PDF files.
- Assign stable sequential paper ids such as `p000001`.
- Write `paper_rag/storage/paper_inventory.csv`.
- Extract PDF text page by page using PyMuPDF, with pypdf as a fallback.
- Split page text into fixed-length chunks.
- Write chunk records to `paper_rag/storage/chunks.jsonl`.

## CLI Usage

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

## Directory Layout

- `src/`: module source code.
- `configs/`: local configuration templates, to be added later.
- `scripts/`: helper scripts and command entry points.
- `storage/`: local runtime storage for indexes and caches.

## Storage Rules

Do not commit downloaded papers, extracted text, vector indexes, API keys, model outputs, or other generated data. Keep `storage/` for local runtime files only.
