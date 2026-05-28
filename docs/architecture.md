# Architecture Draft

LabMate is organized as a lightweight research assistant with two planned modules: `paper_rag` and `experiment_agent`.

## Module Overview

### `paper_rag`

`paper_rag` is the first development target. It will handle research paper ingestion, text processing, retrieval, and question answering with citations.

Planned responsibilities:

- Load paper documents and metadata.
- Extract and normalize text.
- Split paper text into retrievable chunks.
- Build or update a local retrieval index.
- Answer user questions using retrieved source context.
- Return citations that point back to source papers or chunks.
- Provide structured metadata search, exact query cache, and topic cache.
- Expose small Python APIs that future experiment workflows can call directly.

### `experiment_agent`

`experiment_agent` is a future module for experiment management, result analysis, and reporting.

Planned responsibilities:

- Track experiment metadata and run summaries.
- Parse result files or logs.
- Compare runs and highlight changes.
- Generate concise reports for research notes or meetings.

## Relationship Between Modules

The two modules should remain loosely coupled.

`paper_rag` helps researchers understand prior work and retrieve supporting evidence. `experiment_agent` helps researchers manage their own experiments and summarize outcomes. In future versions, `experiment_agent` may call `paper_rag` to connect experiment findings with related papers, but the initial implementation should keep each module independently usable.

The boundary should stay explicit: `paper_rag` stores literature knowledge and source evidence, while `experiment_agent` stores experiment records, logs, parsed metrics, and generated reports. LangGraph orchestration, if used, belongs in `experiment_agent`, not in `paper_rag`.

## Storage Policy

Runtime data should stay local and out of version control. This includes downloaded papers, extracted text caches, vector indexes, API keys, model outputs, and experiment artifacts.

The `paper_rag/storage/` directory is reserved for local runtime storage only and should not contain committed data.

## Initial Design Direction

The first architecture should be simple:

- Local files for small paper collections.
- A clear ingestion pipeline.
- A local retrieval index.
- A minimal question answering interface.
- Explicit configuration files for paths and model settings.

Shared abstractions should be added only after both modules need them.

Current `paper_rag` public functions that should remain easy to call from other modules include `search_papers(...)`, `ask_papers(...)`, `paper_query(...)`, `get_topic_summary(...)`, and `compare_papers(...)`.
