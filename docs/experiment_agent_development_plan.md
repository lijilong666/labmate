# `experiment_agent` Development Plan

This document defines the first implementation plan for LabMate's `experiment_agent` module.
The module should manage experiment metadata, runs, results, log parsing, analysis, and reports.
LangGraph orchestration belongs here, not in `paper_rag`.

`paper_rag` remains a literature knowledge service. `experiment_agent` may call
`paper_rag.api` for paper metadata, retrieval, evidence-grounded QA, topic summaries, and
multi-paper comparison, but it must store experiment state on the `experiment_agent` side.

## Design Goals

- Keep experiment tracking local-first and file-based for the first version.
- Make run records reproducible, diffable, and easy to inspect by humans.
- Separate deterministic parsing/comparison from LLM-assisted narrative analysis.
- Avoid token-consuming calls unless the user explicitly requests literature support or report synthesis.
- Keep experiment lifecycle state, logs, metrics, and reports out of `paper_rag`.
- Keep generated experiment artifacts out of Git by default.

## Module Boundary

`experiment_agent` owns:

- Experiment project metadata.
- Experiment configuration snapshots.
- Run lifecycle records.
- Raw and parsed logs.
- Parsed metrics and artifacts.
- Run comparison records.
- Analysis records and report drafts.
- LangGraph state and workflow orchestration.

`paper_rag` owns:

- Paper ingestion and chunks.
- FAISS index and retrieval.
- Paper cards and metadata search.
- Query/topic cache for literature knowledge.
- Evidence-grounded paper QA.
- Multi-paper comparison and citations.

Do not write experiment lifecycle state into `paper_rag/storage`.
When literature evidence is used, store references to paper ids, chunk ids, citations, and tool call metadata inside `experiment_agent` analysis/report records.

## Proposed Directory Layout

Source code:

```text
experiment_agent/
  README.md
  configs/
    example_experiment.yaml
  scripts/
    init_experiment.py
    register_run.py
    parse_results.py
    compare_runs.py
    generate_report.py
  src/
    experiment_agent/
      __init__.py
      api.py
      schemas.py
      storage.py
      parsers.py
      comparison.py
      reporting.py
      literature.py
      graph.py
      cli_io.py
  storage/
    .gitignore
    .gitkeep
```

Runtime workspace:

```text
experiment_agent/storage/
  experiments/
    exp_000001/
      experiment.yaml
      notes.md
      runs/
        run_000001/
          run.yaml
          config_snapshot.yaml
          logs/
            stdout.log
            stderr.log
            train.log
          results/
            metrics.json
            metrics_history.jsonl
            parsed_summary.json
          artifacts/
            README.md
          analysis/
            comparison.json
            literature_context.json
            report.md
```

Storage rules:

- Commit source code, templates, and small examples only.
- Do not commit real logs, checkpoints, generated reports, experiment outputs, `.env`, API keys, model weights, or large artifacts.
- Use `experiment_agent/storage/.gitignore` to ignore runtime files while keeping the directory available.

## Experiment Metadata Schema

The first version should support YAML input and JSON-compatible Python dictionaries.
Use dataclasses or Pydantic only if the dependency burden is acceptable; otherwise start with typed dictionaries plus validation helpers.

Experiment-level schema:

```yaml
experiment_id: exp_000001
name: "baseline_resnet50_casia"
project: "image_manipulation_localization"
task: "binary image forgery localization"
description: "Baseline training run for initial protocol validation."
created_at: "2026-06-02T00:00:00+08:00"
updated_at: "2026-06-02T00:00:00+08:00"
owner: "local_user"
tags:
  - baseline
  - casia
status: planned

research_context:
  paper_ids: []
  literature_topics: []
  notes: ""

dataset:
  train:
    name: "CASIA"
    split: "train"
    path_hint: "data/datasets/casia/train"
  validation:
    name: "CASIA"
    split: "val"
    path_hint: "data/datasets/casia/val"
  test:
    name: "CASIA"
    split: "test"
    path_hint: "data/datasets/casia/test"

method:
  model_name: "resnet50_unet"
  code_version: ""
  checkpoint_init: ""
  method_keywords:
    - baseline

training:
  command: "python train.py --config configs/baseline.yaml"
  seeds: [0]
  hardware: ""
  expected_runtime: ""

evaluation:
  metrics:
    - F1
    - IoU
    - AUC
  primary_metric: F1
  protocol_notes: ""
```

Run-level schema:

```yaml
run_id: run_000001
experiment_id: exp_000001
status: completed
created_at: "2026-06-02T00:00:00+08:00"
started_at: "2026-06-02T00:10:00+08:00"
ended_at: "2026-06-02T02:30:00+08:00"

command:
  argv:
    - python
    - train.py
    - --config
    - configs/baseline.yaml
  cwd: ""
  env_keys:
    - CUDA_VISIBLE_DEVICES

git:
  repo: ""
  branch: ""
  commit: ""
  dirty: null

config:
  config_path: "config_snapshot.yaml"
  seed: 0
  overrides: {}

outputs:
  log_files:
    - logs/train.log
  metrics_file: results/metrics.json
  metrics_history_file: results/metrics_history.jsonl
  artifact_paths: []

summary:
  primary_metric_value: null
  best_epoch: null
  duration_seconds: null
  failure_reason: ""
```

Parsed metrics schema:

```json
{
  "run_id": "run_000001",
  "metrics": {
    "F1": 0.812,
    "IoU": 0.734,
    "AUC": 0.941
  },
  "best": {
    "metric": "F1",
    "value": 0.812,
    "epoch": 42,
    "mode": "max"
  },
  "history": [
    {
      "epoch": 1,
      "split": "val",
      "F1": 0.501,
      "IoU": 0.390
    }
  ],
  "parser": {
    "name": "json_metrics",
    "version": "0.1",
    "source_files": ["results/metrics.json"]
  }
}
```

## Result And Log Organization

Each run should preserve both raw and parsed information:

- `run.yaml`: lifecycle, command, config, source control, output paths.
- `config_snapshot.yaml`: immutable snapshot of the config used for the run.
- `logs/*.log`: raw stdout/stderr/training logs.
- `results/metrics.json`: final parsed metrics.
- `results/metrics_history.jsonl`: optional epoch/step-level metric history.
- `results/parsed_summary.json`: parser diagnostics, warnings, and extracted highlights.
- `analysis/*.json`: deterministic and LLM-assisted analysis records.
- `analysis/report.md`: generated human-readable report.

The parser should not overwrite raw logs. If parsing is repeated, it should update parsed outputs and record parser version/time.

## LangGraph State

The first graph state should be small and explicit:

```python
class ExperimentGraphState(TypedDict, total=False):
    experiment_id: str
    run_ids: list[str]
    user_request: str
    experiment: dict
    runs: list[dict]
    parsed_results: list[dict]
    comparison: dict
    literature_request: dict
    literature_context: dict
    analysis: dict
    report: str
    warnings: list[str]
    errors: list[str]
```

Keep LangGraph orchestration in `experiment_agent.graph`.
The nodes should call plain Python functions from `storage`, `parsers`, `comparison`, `literature`, and `reporting`.
This keeps the workflow testable without requiring LangGraph for every unit test.

## LangGraph Node Design

Initial graph nodes:

1. `load_experiment`
   - Reads `experiment.yaml` and selected `run.yaml` files.
   - Does not call an LLM.

2. `validate_metadata`
   - Checks required fields, missing config snapshots, missing metrics, and inconsistent dataset/metric names.
   - Does not call an LLM.

3. `parse_logs_and_results`
   - Parses JSON metrics, JSONL histories, CSV metrics, and simple regex log patterns.
   - Does not call an LLM.

4. `compare_runs`
   - Computes metric deltas, best run, regressions, missing seeds, and instability warnings.
   - Does not call an LLM.

5. `decide_literature_need`
   - Rule-based decision using user request, experiment tags, datasets, metrics, and analysis warnings.
   - Does not call an LLM in the first version.

6. `retrieve_literature_context`
   - Calls `paper_rag.api` only when needed.
   - Checks `TOOL_CAPABILITIES` before selecting tools.
   - May call an LLM only through `paper_rag` tools whose capability metadata says so.

7. `analyze_results`
   - Produces deterministic observations first.
   - Optional LLM-assisted explanation can be enabled explicitly.

8. `generate_report`
   - Creates Markdown and JSON report outputs.
   - Can run without an LLM using templates.
   - Optional LLM rewrite/synthesis can be enabled explicitly.

9. `save_outputs`
   - Writes analysis and report files under `experiment_agent/storage`.
   - Does not write to `paper_rag/storage`.

## Calling `paper_rag.api`

All literature calls should go through `paper_rag.api`, not scripts.
Before invoking a tool, inspect `TOOL_CAPABILITIES`.

Recommended import pattern:

```python
from paper_rag.api import TOOL_CAPABILITIES

def can_call(tool_name: str, allow_llm: bool, allow_faiss: bool) -> bool:
    caps = TOOL_CAPABILITIES[tool_name]
    uses_llm = caps["uses_llm"] is True
    uses_faiss = caps["uses_faiss"] is True
    return (allow_llm or not uses_llm) and (allow_faiss or not uses_faiss)
```

Recommended tool selection:

- Use `search_paper_cards` for structured dataset, metric, method, baseline, year, or venue queries.
- Use `compare_papers` for deterministic metadata-only paper comparison.
- Use `get_topic_summary` for stable topic summaries, with awareness that cache misses may call FAISS and an LLM.
- Use `search_papers` when chunk-level evidence is needed without generation.
- Use `ask_papers` only when a natural-language evidence-grounded answer is explicitly needed.
- Use `compare_papers_with_evidence` when a report needs cited multi-paper comparison and protocol caveats.
- Avoid `build_workspace`, `generate_paper_cards`, `enrich_paper_cards`, and `cleanup_paper_cards` from routine experiment analysis unless the user is explicitly maintaining the literature workspace.

Literature context record:

```json
{
  "request": "Find papers using cross-dataset evaluation for this task.",
  "tools": [
    {
      "name": "compare_papers",
      "uses_llm": false,
      "uses_faiss": false
    }
  ],
  "results": {},
  "sources": [
    {
      "paper_id": "p000001",
      "chunk_id": "p000001_page4_chunk2",
      "page_number": 4,
      "source_file": "paper.pdf"
    }
  ],
  "created_at": "2026-06-02T00:00:00+08:00"
}
```

## Initial CLI

Keep CLI entry points thin wrappers around `experiment_agent.api`.

Planned commands:

```bash
python experiment_agent/scripts/init_experiment.py \
  --name baseline_resnet50_casia \
  --project image_manipulation_localization \
  --output experiment_agent/storage/experiments/exp_000001
```

```bash
python experiment_agent/scripts/register_run.py \
  --experiment experiment_agent/storage/experiments/exp_000001 \
  --command "python train.py --config configs/baseline.yaml" \
  --config configs/baseline.yaml \
  --run_id run_000001
```

```bash
python experiment_agent/scripts/parse_results.py \
  --experiment experiment_agent/storage/experiments/exp_000001 \
  --run_id run_000001
```

```bash
python experiment_agent/scripts/compare_runs.py \
  --experiment experiment_agent/storage/experiments/exp_000001 \
  --metric F1
```

```bash
python experiment_agent/scripts/generate_report.py \
  --experiment experiment_agent/storage/experiments/exp_000001 \
  --include_literature false \
  --allow_llm false
```

## Initial Python API

Preferred public functions:

```python
from experiment_agent.api import (
    init_experiment,
    register_run,
    parse_run_results,
    compare_runs,
    analyze_experiment,
    generate_report,
)
```

API responsibilities:

- `init_experiment(...)`: create an experiment folder and `experiment.yaml`.
- `register_run(...)`: create `run.yaml`, copy config snapshot, and register output paths.
- `parse_run_results(...)`: parse metrics/logs into normalized result records.
- `compare_runs(...)`: compute deterministic metric deltas and warnings.
- `analyze_experiment(...)`: run the graph or selected analysis nodes.
- `generate_report(...)`: write Markdown/JSON report outputs.

## LLM Usage Policy

No LLM by default:

- Metadata validation.
- Config snapshotting.
- Log discovery.
- JSON/CSV/JSONL metric parsing.
- Regex metric extraction.
- Run comparison and metric deltas.
- Best-run selection.
- Missing seed checks.
- Missing metric checks.
- Storage writes.

Optional LLM through `paper_rag` only:

- Evidence-grounded literature QA via `ask_papers`.
- Cited multi-paper comparison via `compare_papers_with_evidence`.
- Topic summary refresh via `get_topic_summary` on cache miss.
- LLM-assisted paper-card comparison via `compare_papers_with_llm`.

Optional LLM inside `experiment_agent`:

- Natural-language interpretation of deterministic findings.
- Report polishing or meeting-summary generation.
- Suggesting next experiment-level actions from parsed results.

The first version should require explicit flags such as `allow_llm=True` or `--allow_llm true`.
LLM calls should record model name, timestamp, prompt purpose, input summary, and output file path.

## Non-LLM Analysis Rules

Initial deterministic checks:

- Rank runs by primary metric.
- Compute absolute and relative metric deltas.
- Flag metric regressions.
- Flag missing expected metrics.
- Flag missing or inconsistent seeds.
- Flag unstable metrics across seeds when multiple seeds exist.
- Flag incomplete logs or failed runs.
- Flag dataset/protocol mismatches across compared runs.
- Flag comparisons where primary metrics differ.

Experiment-level suggestions should stay conservative:

- Add more seeds when only one seed is present.
- Add cross-dataset evaluation when the experiment claims generalization but only has in-dataset results.
- Inspect logs when metrics are missing or training stops early.
- Compare against baseline only when dataset and metric protocols match.
- Avoid claiming a model is better when protocols differ.

## Report Outputs

The first Markdown report should include:

- Experiment overview.
- Run table.
- Primary metric comparison.
- Important warnings.
- Best run and caveats.
- Parsed log highlights.
- Literature context, if requested.
- Suggested next experiment-level actions.

The companion JSON report should preserve structured fields:

```json
{
  "experiment_id": "exp_000001",
  "run_ids": ["run_000001"],
  "comparison": {},
  "warnings": [],
  "literature_context": {},
  "recommendations": [],
  "report_markdown_path": "analysis/report.md"
}
```

## First Implementation Milestones

1. Add schemas, storage helpers, and ignored runtime storage.
2. Implement experiment initialization and run registration.
3. Implement JSON/JSONL metrics parsing.
4. Implement deterministic run comparison.
5. Implement Markdown/JSON report generation without LLM.
6. Add `literature.py` wrapper around `paper_rag.api` with capability checks.
7. Add LangGraph orchestration around the existing plain functions.
8. Add optional LLM-assisted report synthesis.

## Open Questions

- Should the first implementation use Pydantic, dataclasses, or lightweight validation helpers?
- Should run ids be sequential per experiment or timestamp-based?
- Which metric file formats should be prioritized first: JSON, JSONL, CSV, TensorBoard event files, or framework-specific logs?
- Should `experiment_agent` copy config snapshots automatically, or only store paths for remote-server workflows?
- Should LangGraph be a hard dependency immediately, or optional until the graph implementation lands?
