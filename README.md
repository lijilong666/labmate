# LabMate

LabMate is a lightweight AI lab assistant for deep learning research workflows. Its goal is to reduce repetitive research labor around reading papers, organizing experiments, analyzing results, and preparing summaries.

## Modules

LabMate is planned around two core modules:

- `paper_rag`: Paper retrieval and question answering for research literature. This is the first module to build.
- `experiment_agent`: Experiment management, result analysis, and reporting. This module is planned for later development.

## Current Stage

The project is in the foundation stage. The repository currently contains the initial documentation and directory layout only. No production feature code, data files, API keys, vector stores, or experiment artifacts should be committed at this stage.

## Initial Focus

The first development milestone is a minimal `paper_rag` MVP that can ingest a small set of paper documents, build a local retrieval index, and answer questions with cited source context.

See:

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [TODO](TODO.md)
