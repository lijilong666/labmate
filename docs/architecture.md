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

Current `paper_rag` public functions that should remain easy to call from other modules include `search_papers(...)`, `ask_papers(...)`, `cleanup_paper_cards(...)`, `paper_query(...)`, `get_topic_summary(...)`, `compare_papers(...)`, `compare_papers_with_llm(...)`, and `compare_papers_with_evidence(...)`.

Stage 8B introduces `paper_rag.api` as the preferred integration surface. It exposes the callable tools above plus `TOOL_CAPABILITIES`, a lightweight registry that marks whether a tool calls an LLM, depends on FAISS, or writes local storage. This is meant to help future `experiment_agent` workflows select tools intentionally instead of guessing from script names.

The workspace pipeline entry point `build_workspace(...)` orchestrates these existing components. It should remain a thin coordination layer and should not duplicate low-level stage logic.

Downstream paper-card tools use shared artifact resolution: prefer `paper_cards_cleaned.jsonl`, then `paper_cards_enriched.jsonl`, then `paper_cards.jsonl` when no explicit card path is provided.

## Planned Stage 7C Direction

Stage 7C adds a lightweight evidence-grounded multi-paper synthesis path. The initial design:

- Reuse metadata filters from Stage 7A to select papers.
- Collect a small balanced set of supporting chunks for each selected paper.
- Ask the LLM to compare task settings, methods, datasets, metrics, baselines, limitations, and evaluation protocols.
- Cite evidence with paper id, page, and chunk id.
- Include a `Comparability and Protocol Caveats` section.
- Avoid automatic ranking or claims of direct fairness when protocols differ or are not specified.

The fairness issue is important in research practice: papers may use different datasets, train/test splits, metrics, baselines, preprocessing, or cross-dataset protocols. The first Stage 7C version should surface these caveats rather than trying to solve protocol normalization deeply.
