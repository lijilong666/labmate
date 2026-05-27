# `paper_rag`

`paper_rag` is the first core LabMate module. It will provide retrieval and question answering over research papers.

## Goal

Help researchers quickly search, inspect, and ask questions about a local paper collection while keeping source citations visible.

## MVP Scope

The first MVP should support:

- Loading a small local set of paper documents.
- Extracting paper text.
- Splitting text into chunks.
- Building a local retrieval index.
- Retrieving relevant chunks for a user question.
- Generating an answer with source citations.

## Directory Layout

- `src/`: module source code, to be added later.
- `configs/`: local configuration templates, to be added later.
- `scripts/`: helper scripts and command entry points, to be added later.
- `storage/`: local runtime storage for indexes and caches.

## Storage Rules

Do not commit downloaded papers, extracted text, vector indexes, API keys, model outputs, or other generated data. Keep `storage/` for local runtime files only.
