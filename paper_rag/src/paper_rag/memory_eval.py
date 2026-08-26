from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from paper_rag.memory.evaluation import (
    audit_memory_store,
    compare_result_sets,
    evaluate_retrieval,
    load_jsonl,
)
from paper_rag.memory.store import DEFAULT_MEMORY_DB_PATH, MemoryStore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and evaluate paper_rag memory without LLM calls.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit lifecycle, provenance, and redundancy invariants.")
    audit.add_argument("--db", type=Path, default=DEFAULT_MEMORY_DB_PATH)
    audit.add_argument("--project_id", required=True)
    audit.add_argument("--as_of")
    audit.add_argument("--limit", type=int, default=100_000)

    retrieval = subparsers.add_parser("retrieval", help="Evaluate Recall@K, MRR, nDCG, and stale hits.")
    retrieval.add_argument("--db", type=Path, default=DEFAULT_MEMORY_DB_PATH)
    retrieval.add_argument("--benchmark", type=Path, required=True)
    retrieval.add_argument("--top_k", type=int, default=6)
    retrieval.add_argument("--candidate_k", type=int, default=50)

    compare = subparsers.add_parser("compare", help="Compare recorded memory-off and memory-on result JSONL.")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "audit":
        return audit_memory_store(
            MemoryStore(args.db),
            project_id=args.project_id,
            as_of=args.as_of,
            limit=args.limit,
        )
    if args.command == "retrieval":
        return evaluate_retrieval(
            MemoryStore(args.db),
            load_jsonl(args.benchmark),
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
    if args.command == "compare":
        return compare_result_sets(load_jsonl(args.baseline), load_jsonl(args.candidate))
    raise RuntimeError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(build_arg_parser().parse_args(argv))
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except (OSError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
