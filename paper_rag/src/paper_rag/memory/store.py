from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from paper_rag.memory.models import (
    MEMORY_KINDS,
    MEMORY_STATUSES,
    MemoryItem,
    MemorySearchHit,
    MemorySession,
    MemorySource,
    utc_now_iso,
    validate_json_object,
)


DEFAULT_MEMORY_DB_PATH = Path("paper_rag/storage/memory.sqlite3")
LATEST_SCHEMA_VERSION = 1


SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    state_json TEXT NOT NULL DEFAULT '{}',
    memory_revision INTEGER NOT NULL DEFAULT 0 CHECK (memory_revision >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);

CREATE TABLE IF NOT EXISTS memory_items (
    memory_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('task_state', 'user_fact', 'episode')),
    canonical_key TEXT,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'archived')),
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    importance REAL NOT NULL DEFAULT 0.5 CHECK (importance >= 0.0 AND importance <= 1.0),
    observed_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    supersedes_id TEXT REFERENCES memory_items(memory_id),
    session_id TEXT REFERENCES sessions(session_id),
    project_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_items(project_id, session_id, status);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory_items(kind, status);
CREATE INDEX IF NOT EXISTS idx_memory_key ON memory_items(project_id, canonical_key, status);
CREATE INDEX IF NOT EXISTS idx_memory_observed ON memory_items(observed_at);

CREATE TABLE IF NOT EXISTS memory_sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memory_items(memory_id) ON DELETE CASCADE,
    source_type TEXT NOT NULL CHECK (source_type IN ('user', 'paper_chunk', 'paper_card', 'episode')),
    paper_id TEXT,
    page_number INTEGER CHECK (page_number IS NULL OR page_number > 0),
    chunk_id TEXT,
    source_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_sources_memory ON memory_sources(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_sources_chunk ON memory_sources(paper_id, chunk_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
    memory_id UNINDEXED,
    canonical_key,
    content,
    tokenize = 'unicode61'
);

CREATE TRIGGER IF NOT EXISTS memory_items_fts_insert
AFTER INSERT ON memory_items BEGIN
    INSERT INTO memory_items_fts(memory_id, canonical_key, content)
    VALUES (new.memory_id, COALESCE(new.canonical_key, ''), new.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_items_fts_delete
AFTER DELETE ON memory_items BEGIN
    DELETE FROM memory_items_fts WHERE memory_id = old.memory_id;
END;

CREATE TRIGGER IF NOT EXISTS memory_items_fts_update
AFTER UPDATE OF canonical_key, content ON memory_items BEGIN
    DELETE FROM memory_items_fts WHERE memory_id = old.memory_id;
    INSERT INTO memory_items_fts(memory_id, canonical_key, content)
    VALUES (new.memory_id, COALESCE(new.canonical_key, ''), new.content);
END;
"""


class MemoryStore:
    """SQLite-backed source of truth for lightweight RAG memory."""

    def __init__(self, db_path: str | Path = DEFAULT_MEMORY_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            applied = {
                int(row["version"])
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            if applied and max(applied) > LATEST_SCHEMA_VERSION:
                raise RuntimeError(
                    "Memory database schema is newer than this code supports: "
                    f"database={max(applied)}, supported={LATEST_SCHEMA_VERSION}."
                )
            if 1 not in applied:
                connection.executescript(SCHEMA_V1)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, utc_now_iso()),
                )

    def schema_version(self) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
        return int(row["version"] or 0)

    def create_session(
        self,
        session_id: str,
        project_id: str,
        state: dict[str, Any] | None = None,
    ) -> MemorySession:
        resolved_state = {} if state is None else state
        session = MemorySession(session_id=session_id, project_id=project_id, state=resolved_state)
        session.validate()
        now = utc_now_iso()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO sessions(
                        session_id, project_id, state_json, memory_revision, created_at, updated_at
                    ) VALUES (?, ?, ?, 0, ?, ?)
                    """,
                    (session_id, project_id, self._json(resolved_state), now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Session already exists: {session_id}") from exc
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> MemorySession:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Session not found: {session_id}")
        return self._session_from_row(row)

    def update_session_state(self, session_id: str, state: dict[str, Any]) -> MemorySession:
        validate_json_object(state, "state")
        now = utc_now_iso()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET state_json = ?, memory_revision = memory_revision + 1, updated_at = ?
                WHERE session_id = ?
                """,
                (self._json(state), now, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")
        return self.get_session(session_id)

    def add_memory(
        self,
        *,
        kind: str,
        content: str,
        project_id: str,
        session_id: str | None = None,
        canonical_key: str | None = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        observed_at: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        metadata: dict[str, Any] | None = None,
        sources: Iterable[MemorySource] = (),
        memory_id: str | None = None,
    ) -> MemoryItem:
        item = self._new_item(
            memory_id=memory_id,
            kind=kind,
            content=content,
            project_id=project_id,
            session_id=session_id,
            canonical_key=canonical_key,
            confidence=confidence,
            importance=importance,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
            sources=sources,
        )
        with self._connection() as connection:
            self._validate_session_scope(connection, item.session_id, item.project_id)
            self._insert_item(connection, item)
            self._increment_revision(connection, item.session_id)
        return self.get_memory(item.memory_id)

    def get_memory(self, memory_id: str) -> MemoryItem:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM memory_items WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Memory not found: {memory_id}")
            sources = self._load_sources(connection, memory_id)
        return self._item_from_row(row, sources)

    def find_active_by_key(
        self,
        *,
        project_id: str,
        canonical_key: str,
        kind: str | None = None,
        session_id: str | None = None,
    ) -> MemoryItem | None:
        if not project_id.strip():
            raise ValueError("project_id must not be empty.")
        if not canonical_key.strip():
            raise ValueError("canonical_key must not be empty.")
        if kind is not None and kind not in MEMORY_KINDS:
            raise ValueError(f"Unsupported memory kind: {kind}")
        clauses = ["project_id = ?", "canonical_key = ?", "status = 'active'"]
        parameters: list[Any] = [project_id, canonical_key]
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(kind)
        if session_id is None:
            clauses.append("session_id IS NULL")
        else:
            clauses.append("session_id = ?")
            parameters.append(session_id)
        sql = "SELECT * FROM memory_items WHERE " + " AND ".join(clauses)
        sql += " ORDER BY observed_at DESC, created_at DESC LIMIT 1"
        with self._connection() as connection:
            row = connection.execute(sql, parameters).fetchone()
            if row is None:
                return None
            sources = self._load_sources(connection, row["memory_id"])
        return self._item_from_row(row, sources)

    def list_memories(
        self,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[str] | None = ("active",),
        limit: int = 100,
        include_global: bool = False,
        global_only: bool = False,
    ) -> list[MemoryItem]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")
        clauses, parameters = self._filter_clauses(
            project_id,
            session_id,
            kinds,
            statuses,
            include_global=include_global,
            global_only=global_only,
        )
        sql = "SELECT * FROM memory_items"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY observed_at DESC, created_at DESC LIMIT ?"
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
            return [self._item_from_row(row, self._load_sources(connection, row["memory_id"])) for row in rows]

    def search_memories(
        self,
        query: str,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[str] | None = ("active",),
        top_k: int = 10,
        include_global: bool = False,
        global_only: bool = False,
    ) -> list[MemoryItem]:
        return [
            hit.item
            for hit in self.search_memory_hits(
                query,
                project_id=project_id,
                session_id=session_id,
                kinds=kinds,
                statuses=statuses,
                top_k=top_k,
                include_global=include_global,
                global_only=global_only,
            )
        ]

    def search_memory_hits(
        self,
        query: str,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[str] | None = ("active",),
        top_k: int = 10,
        include_global: bool = False,
        global_only: bool = False,
    ) -> list[MemorySearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        fts_query = self._fts_query(query)
        clauses, parameters = self._filter_clauses(
            project_id,
            session_id,
            kinds,
            statuses,
            prefix="m",
            include_global=include_global,
            global_only=global_only,
        )
        clauses.insert(0, "memory_items_fts MATCH ?")
        parameters.insert(0, fts_query)
        sql = """
            SELECT m.*, bm25(memory_items_fts) AS lexical_score
            FROM memory_items_fts
            JOIN memory_items AS m ON m.memory_id = memory_items_fts.memory_id
            WHERE {where}
            ORDER BY lexical_score ASC, m.importance DESC, m.observed_at DESC
            LIMIT ?
        """.format(where=" AND ".join(clauses))
        parameters.append(top_k)

        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
            hits = [
                MemorySearchHit(
                    item=self._item_from_row(row, self._load_sources(connection, row["memory_id"])),
                    lexical_rank=rank,
                    match_source="fts5",
                    raw_lexical_score=float(row["lexical_score"]),
                )
                for rank, row in enumerate(rows, start=1)
            ]

            # unicode61 does not segment every CJK phrase usefully. Exact substring
            # fallback keeps Chinese memory usable without adding a tokenizer dependency.
            if len(hits) < top_k:
                seen = {hit.item.memory_id for hit in hits}
                fallback_clauses, fallback_parameters = self._filter_clauses(
                    project_id,
                    session_id,
                    kinds,
                    statuses,
                    include_global=include_global,
                    global_only=global_only,
                )
                fallback_clauses.insert(
                    0,
                    "(content LIKE ? ESCAPE '\\' OR COALESCE(canonical_key, '') LIKE ? ESCAPE '\\')",
                )
                escaped_query = (
                    query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                )
                like_query = f"%{escaped_query}%"
                fallback_parameters[0:0] = [like_query, like_query]
                fallback_sql = "SELECT * FROM memory_items WHERE " + " AND ".join(fallback_clauses)
                fallback_sql += " ORDER BY importance DESC, observed_at DESC LIMIT ?"
                fallback_parameters.append(top_k)
                for row in connection.execute(fallback_sql, fallback_parameters):
                    if row["memory_id"] in seen:
                        continue
                    hits.append(
                        MemorySearchHit(
                            item=self._item_from_row(
                                row,
                                self._load_sources(connection, row["memory_id"]),
                            ),
                            lexical_rank=len(hits) + 1,
                            match_source="substring",
                        )
                    )
                    seen.add(row["memory_id"])
                    if len(hits) >= top_k:
                        break
        return hits

    def supersede_memory(
        self,
        memory_id: str,
        *,
        content: str,
        confidence: float | None = None,
        importance: float | None = None,
        observed_at: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        metadata: dict[str, Any] | None = None,
        sources: Iterable[MemorySource] = (),
        new_memory_id: str | None = None,
    ) -> MemoryItem:
        old = self.get_memory(memory_id)
        if old.status != "active":
            raise ValueError("Only active memory can be superseded.")
        new_item = self._new_item(
            memory_id=new_memory_id,
            kind=old.kind,
            content=content,
            project_id=old.project_id,
            session_id=old.session_id,
            canonical_key=old.canonical_key,
            confidence=old.confidence if confidence is None else confidence,
            importance=old.importance if importance is None else importance,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=old.metadata if metadata is None else metadata,
            sources=sources,
            supersedes_id=old.memory_id,
        )
        now = utc_now_iso()
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE memory_items SET status = 'superseded', valid_to = COALESCE(valid_to, ?), updated_at = ? "
                "WHERE memory_id = ? AND status = 'active'",
                (new_item.observed_at, now, memory_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Memory changed before it could be superseded.")
            self._insert_item(connection, new_item)
            self._increment_revision(connection, old.session_id)
        return self.get_memory(new_item.memory_id)

    def archive_memory(self, memory_id: str) -> MemoryItem:
        now = utc_now_iso()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT session_id FROM memory_items WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Memory not found: {memory_id}")
            cursor = connection.execute(
                "UPDATE memory_items SET status = 'archived', updated_at = ? "
                "WHERE memory_id = ? AND status != 'archived'",
                (now, memory_id),
            )
            if cursor.rowcount:
                self._increment_revision(connection, row["session_id"])
        return self.get_memory(memory_id)

    def _new_item(
        self,
        *,
        memory_id: str | None,
        kind: str,
        content: str,
        project_id: str,
        session_id: str | None,
        canonical_key: str | None,
        confidence: float,
        importance: float,
        observed_at: str | None,
        valid_from: str | None,
        valid_to: str | None,
        metadata: dict[str, Any] | None,
        sources: Iterable[MemorySource],
        supersedes_id: str | None = None,
    ) -> MemoryItem:
        now = utc_now_iso()
        item = MemoryItem(
            memory_id=memory_id or f"mem_{uuid.uuid4().hex}",
            kind=kind,
            content=content.strip(),
            project_id=project_id.strip(),
            session_id=session_id.strip() if session_id else None,
            canonical_key=canonical_key.strip() if canonical_key else None,
            confidence=float(confidence),
            importance=float(importance),
            observed_at=observed_at or now,
            valid_from=valid_from,
            valid_to=valid_to,
            supersedes_id=supersedes_id,
            metadata=dict({} if metadata is None else metadata),
            created_at=now,
            updated_at=now,
            sources=tuple(sources),
        )
        item.validate()
        return item

    def _insert_item(self, connection: sqlite3.Connection, item: MemoryItem) -> None:
        connection.execute(
            """
            INSERT INTO memory_items(
                memory_id, kind, canonical_key, content, status, confidence, importance,
                observed_at, valid_from, valid_to, supersedes_id, session_id, project_id,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.memory_id,
                item.kind,
                item.canonical_key,
                item.content,
                item.status,
                item.confidence,
                item.importance,
                item.observed_at,
                item.valid_from,
                item.valid_to,
                item.supersedes_id,
                item.session_id,
                item.project_id,
                self._json(item.metadata),
                item.created_at,
                item.updated_at,
            ),
        )
        for source in item.sources:
            connection.execute(
                """
                INSERT INTO memory_sources(
                    memory_id, source_type, paper_id, page_number, chunk_id, source_path
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.memory_id,
                    source.source_type,
                    source.paper_id,
                    source.page_number,
                    source.chunk_id,
                    source.source_path,
                ),
            )

    @staticmethod
    def _validate_session_scope(
        connection: sqlite3.Connection,
        session_id: str | None,
        project_id: str,
    ) -> None:
        if session_id is None:
            return
        row = connection.execute(
            "SELECT project_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Session not found: {session_id}")
        if row["project_id"] != project_id:
            raise ValueError("Memory project_id must match its session project_id.")

    @staticmethod
    def _increment_revision(connection: sqlite3.Connection, session_id: str | None) -> None:
        if session_id is None:
            return
        connection.execute(
            "UPDATE sessions SET memory_revision = memory_revision + 1, updated_at = ? WHERE session_id = ?",
            (utc_now_iso(), session_id),
        )

    @staticmethod
    def _load_sources(connection: sqlite3.Connection, memory_id: str) -> tuple[MemorySource, ...]:
        rows = connection.execute(
            "SELECT source_type, paper_id, page_number, chunk_id, source_path "
            "FROM memory_sources WHERE memory_id = ? ORDER BY source_id",
            (memory_id,),
        ).fetchall()
        return tuple(
            MemorySource(
                source_type=row["source_type"],
                paper_id=row["paper_id"],
                page_number=row["page_number"],
                chunk_id=row["chunk_id"],
                source_path=row["source_path"],
            )
            for row in rows
        )

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> MemorySession:
        return MemorySession(
            session_id=row["session_id"],
            project_id=row["project_id"],
            state=json.loads(row["state_json"]),
            memory_revision=int(row["memory_revision"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _item_from_row(row: sqlite3.Row, sources: tuple[MemorySource, ...]) -> MemoryItem:
        return MemoryItem(
            memory_id=row["memory_id"],
            kind=row["kind"],
            canonical_key=row["canonical_key"],
            content=row["content"],
            status=row["status"],
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            observed_at=row["observed_at"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            supersedes_id=row["supersedes_id"],
            session_id=row["session_id"],
            project_id=row["project_id"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            sources=sources,
        )

    @staticmethod
    def _filter_clauses(
        project_id: str | None,
        session_id: str | None,
        kinds: Sequence[str] | None,
        statuses: Sequence[str] | None,
        *,
        prefix: str = "",
        include_global: bool = False,
        global_only: bool = False,
    ) -> tuple[list[str], list[Any]]:
        column = f"{prefix}." if prefix else ""
        clauses: list[str] = []
        parameters: list[Any] = []
        if project_id is not None:
            clauses.append(f"{column}project_id = ?")
            parameters.append(project_id)
        if global_only:
            if session_id is not None:
                raise ValueError("global_only cannot be combined with session_id.")
            clauses.append(f"{column}session_id IS NULL")
        elif session_id is not None:
            if include_global:
                clauses.append(f"({column}session_id = ? OR {column}session_id IS NULL)")
            else:
                clauses.append(f"{column}session_id = ?")
            parameters.append(session_id)
        if kinds is not None:
            invalid = set(kinds) - MEMORY_KINDS
            if invalid:
                raise ValueError(f"Unsupported memory kinds: {', '.join(sorted(invalid))}")
            if not kinds:
                clauses.append("1 = 0")
            else:
                clauses.append(f"{column}kind IN ({','.join('?' for _ in kinds)})")
                parameters.extend(kinds)
        if statuses is not None:
            invalid = set(statuses) - MEMORY_STATUSES
            if invalid:
                raise ValueError(f"Unsupported memory statuses: {', '.join(sorted(invalid))}")
            if not statuses:
                clauses.append("1 = 0")
            else:
                clauses.append(f"{column}status IN ({','.join('?' for _ in statuses)})")
                parameters.extend(statuses)
        return clauses, parameters

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            raise ValueError("query must contain searchable characters.")
        return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)

    @staticmethod
    def _json(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
