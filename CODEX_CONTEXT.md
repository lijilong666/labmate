# Codex Context

This file is long-term development context for future Codex sessions working on LabMate. Keep it updated when project scope, module boundaries, or storage rules change.

## Project Name

LabMate

## Project Goal

LabMate is a lightweight AI lab assistant for deep learning researchers. It aims to reduce repetitive research work such as paper retrieval, experiment tracking, result summarization, and report generation.

## Overall Modules

### `paper_rag`

`paper_rag` is the first module to implement. It is a general research paper RAG tool.

The default demo domain is image forgery detection, image manipulation localization, and AIGC or diffusion image localization. This domain must not be hard-coded into the core framework. Domain-specific examples should live in configs, examples, or documentation.

In later stages, `paper_rag` should serve as a literature search tool for `experiment_agent`.

### `experiment_agent`

`experiment_agent` will be implemented later. It focuses on experiment management, testing, result parsing, result analysis, and reporting.

It should not automatically design core model architectures. It may provide experiment-level suggestions based on recent results and RAG evidence, such as adding seeds, checking unstable metrics, adding cross-dataset evaluation, changing thresholds, checking logs, or adding ablations.

## `paper_rag` Goals

The `paper_rag` module should eventually support:

- Batch ingestion of research paper PDFs.
- PDF text parsing while preserving file name, paper id, page number, and preferably section information.
- Text chunking.
- Local vector index construction.
- Vector retrieval.
- Structured paper card generation.
- Metadata-first search.
- Topic-level knowledge cache for frequently queried knowledge.
- Basic paper question answering with citations.
- Multi-paper comparison.
- Simple Python APIs for future use by `experiment_agent`.

## `paper_rag` Storage

The `paper_rag/storage/` directory is reserved for local runtime artifacts. Generated data should generally stay out of Git.

Expected local artifacts:

- `vector_store/`: local vector index, not committed to Git.
- `chunks.jsonl`: parsed chunks, not committed if large.
- `paper_cards.jsonl`: structured paper metadata; may be generated locally.
- `topic_cache.jsonl`: reusable topic knowledge cache.
- `query_cache.jsonl`: optional exact or semantic query cache.

## Important RAG Design

The system should not call the LLM for every query. Use this priority order:

1. `topic_cache`
2. `paper_cards` and metadata search
3. vector retrieval
4. LLM answer generation

The goal is to reduce token cost and improve answer stability.

## Suggested `paper_card` Schema

Each structured paper card should prefer these fields:

- `paper_id`
- `title`
- `year`
- `authors`
- `venue`
- `task`
- `method_keywords`
- `datasets`
- `metrics`
- `baselines`
- `summary`
- `limitations`
- `source_file`

## `topic_cache` Examples

Useful topic cache entries for the default demo domain may include:

- Common datasets in image manipulation localization.
- Common metrics such as F1, IoU, AUC, and bF1.
- Common baselines.
- Cross-dataset evaluation protocols.
- Frequency-domain feature usage.
- Edge supervision and boundary metrics.

These are examples only. The core cache design should remain domain-neutral.

## Experiment Agent Boundary

`experiment_agent` should manage the experiment lifecycle, not invent model architectures.

It may suggest experiment-level actions such as:

- Add or vary random seeds.
- Check unstable metrics.
- Add cross-dataset evaluation.
- Change thresholds.
- Inspect logs.
- Add ablations.

Experiment analysis records should be stored on the `experiment_agent` side, not inside `paper_rag`.

`paper_rag` should store stable literature knowledge. Experiment analysis should store references to RAG sources when RAG evidence is used.

## Deployment Assumptions

- The main runtime environment is a remote Linux server.
- A Windows laptop and Mac mini may be used as local development machines.
- GitHub is the single source of truth for code synchronization.
- Local Codex may be used for development, then changes are pushed to GitHub.
- The remote Linux server pulls code from GitHub.
- Data, PDFs, vector stores, checkpoints, logs, and API keys must not be committed.

## Development Priorities

### Stage 1: `paper_rag` MVP

- Create project structure.
- Implement PDF loader.
- Implement chunker.
- Save chunks with metadata.
- Build local embedding index.
- Implement vector search.
- Implement paper card schema.
- Implement metadata search.
- Implement basic QA with citations.
- Expose `search_papers()` and `ask_papers()`.

### Stage 2: `paper_rag` Enhancement

- Add `topic_cache`.
- Add `query_cache`.
- Add multi-paper comparison.
- Add a simple CLI or Streamlit UI.

### Stage 3: `experiment_agent`

- Define experiment schema.
- Implement queue manager.
- Implement result parser.
- Implement report generator.
- Connect `paper_rag` as a tool for analysis suggestions.

## Coding Constraints

- Keep code modular.
- Do not hard-code image forgery-specific logic in core components.
- Put domain-specific examples in configs or examples.
- Prefer local vector storage such as FAISS or Chroma.
- Keep secrets in `.env` and never commit them.
- Keep large generated files out of Git.
- Do not commit API keys, credentials, tokens, private configuration, PDFs, vector databases, checkpoints, logs, or experiment artifacts.
- Prefer simple local-first designs before adding infrastructure.
- Keep implementation scoped to the current milestone.
