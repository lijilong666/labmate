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
- Exact query cache.
- Topic-level knowledge cache for frequently queried knowledge.
- Basic paper question answering with citations.
- Multi-paper comparison.
- Simple Python APIs for future use by `experiment_agent`.

Current implemented `paper_rag` capabilities include ingestion, FAISS index building, vector search, evidence-based QA, heuristic paper cards, LLM-assisted paper card enrichment, rule-based paper-card metadata cleanup, metadata search, rule-based query routing, exact query cache, topic cache, metadata-based multi-paper comparison, LLM-assisted comparison summaries over paper cards, lightweight evidence-grounded multi-paper synthesis, a unified workspace build pipeline, shared artifact path resolution, and a public API/capability registry for future integrations.

Stage 8B adds shared downstream artifact defaults and a public integration module:

- `paper_rag.paths.resolve_cards_path(...)` prefers cleaned cards, then enriched cards, then raw cards when no explicit path is passed.
- `paper_rag.paths.resolve_chunk_metadata_path(...)` resolves the default chunk metadata file for evidence-grounded tools.
- `paper_rag.api` exposes stable callable entry points and `TOOL_CAPABILITIES` for future `experiment_agent` tool selection.

Personal Chinese study notes exist under `docs/`:

- `docs/paper_rag_interview_knowledge_base.md`: interview-oriented explanation of the full `paper_rag` development process through Stage 8B.
- `docs/rag_upgrade_roadmap_zh.md`: post-baseline RAG upgrade roadmap covering quality, evaluation, evidence, protocol comparison, cache, and experiment-agent integration.

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

Downstream paper-card tools should use the shared artifact default unless the user or caller passes an explicit path. Preferred order:

1. `paper_cards_cleaned.jsonl`
2. `paper_cards_enriched.jsonl`
3. `paper_cards.jsonl`

## Workspace Build Pipeline

Stage 8A adds a lightweight orchestration layer for building a local `paper_rag` workspace. It connects existing stage functions rather than reimplementing them. `--all` runs non-LLM stages only: ingestion, FAISS indexing, heuristic paper-card generation, and metadata cleanup. LLM enrichment remains explicit through `--run_enrich` to avoid accidental token usage.

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

For future integration, keep `paper_rag` callable through small Python functions with structured inputs and outputs. Scripts under `paper_rag/scripts/` are for human CLI usage; the future LangGraph-based experiment workflow should call core APIs from `paper_rag/src/paper_rag/`.

The preferred integration module is `paper_rag.api`; use `TOOL_CAPABILITIES` to avoid accidentally calling token-consuming or FAISS-dependent tools.

## Multi-Paper Comparison Fairness Caveat

Multi-paper comparison in scientific workflows is sensitive to evaluation protocol differences. A comparison can be misleading if papers use different datasets, train/test splits, metrics, baselines, preprocessing, robustness tests, or cross-dataset protocols.

Stage 7C implements a lightweight evidence-grounded comparison that collects balanced evidence chunks per selected paper and cites them. It includes explicit caveats about comparability and protocol differences, but it should not rank papers or claim fairness unless protocols are clearly aligned. Rigorous protocol normalization is a later research-quality evaluation task, not part of the first Stage 7C implementation.

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

- Add `topic_cache`. Done.
- Add `query_cache`. Done.
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
