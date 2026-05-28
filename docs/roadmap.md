# Roadmap

## Phase 0: Foundation

Set up the repository structure, documentation, and module boundaries.

Deliverables:

- Project README.
- Codex context notes.
- Architecture draft.
- Roadmap.
- Module README files.

## Phase 1: `paper_rag` MVP

Build a minimal paper retrieval and question answering workflow.

Target capabilities:

- Ingest a small local set of papers.
- Extract and chunk text.
- Build a local retrieval index.
- Ask questions over the indexed papers.
- Return answers with source citations.

## Phase 2: `paper_rag` Usability

Improve reliability and day-to-day usefulness.

Target capabilities:

- Better metadata handling. Initial implementation is available through paper cards and LLM-assisted enrichment.
- Exact query cache and topic cache for repeated questions.
- Structured multi-paper comparison for experiment planning and reporting context.
- Incremental indexing.
- Retrieval quality checks.
- Clear setup and troubleshooting documentation.

## Phase 3: `experiment_agent` Design

Define the future experiment workflow without overbuilding it too early.

Target planning areas:

- Experiment metadata schema.
- Result and log parsing.
- Run comparison.
- Report generation.
- Direct calls into `paper_rag` APIs for related methods, datasets, metrics, baselines, limitations, and source citations.

## Phase 4: Integrated Research Assistant

Connect paper understanding and experiment analysis into a coherent research workflow.

Target capabilities:

- Link experiment results to related literature.
- Generate research summaries that combine paper evidence and local results.
- Support repeatable project notes and lightweight reporting.
