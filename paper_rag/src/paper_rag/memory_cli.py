from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from paper_rag.memory.consolidator import MemoryConsolidator
from paper_rag.memory.models import MEMORY_KINDS, MEMORY_STATUSES, MemoryItem
from paper_rag.memory.store import DEFAULT_MEMORY_DB_PATH, MemoryStore
from paper_rag.memory.writer import MemoryWriter


def _add_scope_arguments(parser: argparse.ArgumentParser, *, require_session: bool = False) -> None:
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--session_id", required=require_session)


def _add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    _add_scope_arguments(parser)
    parser.add_argument("--kind", action="append", choices=sorted(MEMORY_KINDS), dest="kinds")
    parser.add_argument("--status", action="append", choices=sorted(MEMORY_STATUSES), dest="statuses")
    parser.add_argument("--include_global", action="store_true")
    parser.add_argument("--limit", type=int, default=100)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and manage paper_rag lightweight memory.")
    parser.add_argument("--db", type=Path, default=DEFAULT_MEMORY_DB_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List memory records in a project/session scope.")
    _add_filter_arguments(list_parser)

    search_parser = subparsers.add_parser("search", help="Lexically search memory records.")
    search_parser.add_argument("query")
    _add_filter_arguments(search_parser)

    show_parser = subparsers.add_parser("show", help="Show one complete memory record.")
    show_parser.add_argument("memory_id")

    add_parser = subparsers.add_parser("add", help="Explicitly add a user fact or task state.")
    _add_scope_arguments(add_parser)
    add_parser.add_argument("--kind", required=True, choices=["user_fact", "task_state"])
    add_parser.add_argument("--key", required=True, dest="canonical_key")
    add_parser.add_argument("--content", required=True)
    add_parser.add_argument("--importance", type=float)

    for name in ("correct", "update"):
        correct_parser = subparsers.add_parser(name, help="Explicitly correct an active user fact.")
        correct_parser.add_argument("memory_id")
        _add_scope_arguments(correct_parser)
        correct_parser.add_argument("--content", required=True)
        correct_parser.add_argument("--importance", type=float, default=0.8)

    archive_parser = subparsers.add_parser("archive", help="Archive a memory without deleting its history.")
    archive_parser.add_argument("memory_id")
    _add_scope_arguments(archive_parser)

    consolidate_parser = subparsers.add_parser(
        "consolidate",
        help="Plan or apply deterministic session episode consolidation.",
    )
    _add_scope_arguments(consolidate_parser, require_session=True)
    consolidate_parser.add_argument("--limit", type=int, default=10_000)
    consolidate_parser.add_argument("--recent_query_limit", type=int, default=10)
    consolidate_parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive exact duplicate episodes and persist the session summary.",
    )
    return parser


def _item_payload(item: MemoryItem) -> dict[str, Any]:
    return asdict(item)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _filtered_items(store: MemoryStore, args: argparse.Namespace, *, search: bool) -> list[MemoryItem]:
    statuses: Sequence[str] = args.statuses or ("active",)
    kwargs = {
        "project_id": args.project_id,
        "session_id": args.session_id,
        "kinds": args.kinds,
        "statuses": statuses,
        "include_global": args.include_global,
    }
    if search:
        return store.search_memories(args.query, top_k=args.limit, **kwargs)
    return store.list_memories(limit=args.limit, **kwargs)


def run(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    store = MemoryStore(args.db)
    writer = MemoryWriter(store)

    if args.command == "list":
        return [_item_payload(item) for item in _filtered_items(store, args, search=False)]
    if args.command == "search":
        return [_item_payload(item) for item in _filtered_items(store, args, search=True)]
    if args.command == "show":
        return _item_payload(store.get_memory(args.memory_id))
    if args.command == "add":
        if args.kind == "user_fact":
            result = writer.remember_user_fact(
                canonical_key=args.canonical_key,
                content=args.content,
                project_id=args.project_id,
                session_id=args.session_id,
                explicit_user_request=True,
                importance=0.7 if args.importance is None else args.importance,
            )
        else:
            if not args.session_id:
                raise ValueError("task_state requires --session_id.")
            result = writer.set_task_state(
                canonical_key=args.canonical_key,
                content=args.content,
                project_id=args.project_id,
                session_id=args.session_id,
                importance=0.6 if args.importance is None else args.importance,
            )
        return asdict(result)
    if args.command in {"correct", "update"}:
        target = store.get_memory(args.memory_id)
        if target.session_id != args.session_id:
            raise ValueError("Target memory does not belong to the requested session scope.")
        return asdict(
            writer.correct_user_fact(
                target_id=args.memory_id,
                content=args.content,
                project_id=args.project_id,
                session_id=args.session_id,
                explicit_user_correction=True,
                importance=args.importance,
            )
        )
    if args.command == "archive":
        target = store.get_memory(args.memory_id)
        if target.session_id != args.session_id:
            raise ValueError("Target memory does not belong to the requested session scope.")
        return asdict(
            writer.archive(
                target_id=args.memory_id,
                project_id=args.project_id,
                session_id=args.session_id,
                explicit_user_request=True,
            )
        )
    if args.command == "consolidate":
        return asdict(
            MemoryConsolidator(store).consolidate_session(
                project_id=args.project_id,
                session_id=args.session_id,
                apply=args.apply,
                limit=args.limit,
                recent_query_limit=args.recent_query_limit,
            )
        )
    raise RuntimeError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        _print_json(run(parser.parse_args(argv)))
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
