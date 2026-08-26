from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from paper_rag.memory.scale_test import MemoryScaleConfig, run_memory_scale_test


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic multi-session memory scale test.")
    parser.add_argument("--db", type=Path, help="New database path to retain; must not already exist.")
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--facts_per_session", type=int, default=30)
    parser.add_argument("--episodes_per_session", type=int, default=10)
    parser.add_argument("--duplicate_pairs_per_session", type=int, default=2)
    parser.add_argument("--global_facts", type=int, default=20)
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def _config(args: argparse.Namespace) -> MemoryScaleConfig:
    return MemoryScaleConfig(
        session_count=args.sessions,
        facts_per_session=args.facts_per_session,
        episodes_per_session=args.episodes_per_session,
        duplicate_episode_pairs_per_session=args.duplicate_pairs_per_session,
        global_fact_count=args.global_facts,
        query_count=args.queries,
        top_k=args.top_k,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.db is not None:
            report = run_memory_scale_test(args.db, _config(args))
        else:
            with tempfile.TemporaryDirectory(prefix="paper-rag-memory-scale-") as directory:
                report = run_memory_scale_test(Path(directory) / "memory.sqlite3", _config(args))
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
