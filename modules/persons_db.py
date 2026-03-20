"""
modules/persons_db.py
Simple person registry in SQL (PostgreSQL preferred, SQLite fallback).
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import uuid
from typing import Optional

from config import SQL_BACKEND, SQL_DATABASE_URL, SQL_SQLITE_PATH

_NAME_RE = re.compile(r"^[a-zA-Z0-9 '_-]+$")
_MAX_NAME_LEN = 64
_init_lock = threading.Lock()
_initialized = False


def _normalize_name(name: str) -> tuple[str, str]:
    display_name = (name or "").strip()[:_MAX_NAME_LEN]
    if not display_name or not _NAME_RE.fullmatch(display_name):
        raise ValueError("Invalid name")
    return display_name, display_name.lower()


def _is_postgres() -> bool:
    if SQL_BACKEND.strip().lower() == "postgres":
        return True
    return SQL_DATABASE_URL.lower().startswith("postgres")


# Computed once at module load — avoids re-evaluating config strings on every DB call
_POSTGRES: bool = _is_postgres()

# SQLite singleton connection (WAL mode, thread-safe).
# Avoids opening/closing a new connection on every DB call.
# Only used when _POSTGRES is False.
_sqlite_conn: Optional[sqlite3.Connection] = None
_sqlite_conn_lock = threading.Lock()


def _get_sqlite() -> sqlite3.Connection:
    """Return the module-level SQLite connection, creating it if needed."""
    global _sqlite_conn
    if _sqlite_conn is not None:
        return _sqlite_conn
    with _sqlite_conn_lock:
        if _sqlite_conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(SQL_SQLITE_PATH)), exist_ok=True)
            conn = sqlite3.connect(SQL_SQLITE_PATH, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            _sqlite_conn = conn
    return _sqlite_conn


# ── Schema ────────────────────────────────────────────────────────────────

def _ensure_schema() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        if _POSTGRES:
            _ensure_schema_postgres()
        else:
            _ensure_schema_sqlite()
        _initialized = True


def _ensure_schema_postgres() -> None:
    conn = _pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id UUID PRIMARY KEY,
                    name_key TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def _ensure_schema_sqlite() -> None:
    conn = _get_sqlite()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            name_key TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def _pg_connect():
    if not SQL_DATABASE_URL:
        raise RuntimeError(
            "SQL_DATABASE_URL is empty. Set sql.database_url (or env SQL_DATABASE_URL) for PostgreSQL."
        )
    try:
        import psycopg2
    except Exception as exc:
        raise RuntimeError(
            "psycopg2 is required for PostgreSQL backend. Install psycopg2-binary."
        ) from exc
    return psycopg2.connect(SQL_DATABASE_URL)


# ── Public API ────────────────────────────────────────────────────────────

def create_person(display_name: str) -> Optional[dict]:
    _ensure_schema()
    display_name, name_key = _normalize_name(display_name)

    existing = get_person_by_name(display_name)
    if existing:
        return existing

    person_id = str(uuid.uuid4())

    if _POSTGRES:
        conn = _pg_connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO persons (person_id, name_key, display_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name_key) DO NOTHING
                    """,
                    (person_id, name_key, display_name),
                )
            conn.commit()
        finally:
            conn.close()
    else:
        conn = _get_sqlite()
        conn.execute(
            """
            INSERT OR IGNORE INTO persons (person_id, name_key, display_name)
            VALUES (?, ?, ?)
            """,
            (person_id, name_key, display_name),
        )
        conn.commit()

    return get_person_by_name(display_name)


def get_person_by_name(name: str) -> Optional[dict]:
    _ensure_schema()
    _, name_key = _normalize_name(name)

    if _POSTGRES:
        conn = _pg_connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT person_id::text, name_key, display_name, created_at::text"
                    " FROM persons WHERE name_key=%s",
                    (name_key,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
    else:
        conn = _get_sqlite()
        row = conn.execute(
            "SELECT person_id, name_key, display_name, created_at FROM persons WHERE name_key=?",
            (name_key,),
        ).fetchone()

    if not row:
        return None
    return {
        "person_id":    row[0],
        "name_key":     row[1],
        "display_name": row[2],
        "created_at":   row[3],
    }


def get_person_by_id(person_id: str) -> Optional[dict]:
    _ensure_schema()
    pid = (person_id or "").strip()
    if not pid:
        return None
    try:
        uuid.UUID(pid)
    except ValueError:
        return None

    if _POSTGRES:
        conn = _pg_connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT person_id::text, name_key, display_name, created_at::text"
                    " FROM persons WHERE person_id=%s::uuid",
                    (pid,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
    else:
        conn = _get_sqlite()
        row = conn.execute(
            "SELECT person_id, name_key, display_name, created_at FROM persons WHERE person_id=?",
            (pid,),
        ).fetchone()

    if not row:
        return None
    return {
        "person_id":    row[0],
        "name_key":     row[1],
        "display_name": row[2],
        "created_at":   row[3],
    }


def delete_person_by_name(name: str) -> bool:
    _ensure_schema()
    _, name_key = _normalize_name(name)

    if _POSTGRES:
        conn = _pg_connect()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM persons WHERE name_key=%s", (name_key,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted
        finally:
            conn.close()
    else:
        conn = _get_sqlite()
        cur = conn.execute("DELETE FROM persons WHERE name_key=?", (name_key,))
        conn.commit()
        return cur.rowcount > 0


def upsert_person(person_id: str, display_name: str) -> Optional[dict]:
    """
    Ensure a person row exists with a specific person_id and display_name.
    If an equivalent row already exists by name or id, it is reused.
    """
    _ensure_schema()
    display_name, name_key = _normalize_name(display_name)
    pid = (person_id or "").strip()
    try:
        uuid.UUID(pid)
    except ValueError as exc:
        raise ValueError("Invalid person_id UUID") from exc

    by_name = get_person_by_name(display_name)
    if by_name:
        return by_name

    by_id = get_person_by_id(pid)
    if by_id:
        return by_id

    if _POSTGRES:
        conn = _pg_connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO persons (person_id, name_key, display_name)
                    VALUES (%s::uuid, %s, %s)
                    ON CONFLICT (person_id) DO UPDATE
                    SET display_name = EXCLUDED.display_name
                    """,
                    (pid, name_key, display_name),
                )
            conn.commit()
        finally:
            conn.close()
    else:
        conn = _get_sqlite()
        conn.execute(
            """
            INSERT OR IGNORE INTO persons (person_id, name_key, display_name)
            VALUES (?, ?, ?)
            """,
            (pid, name_key, display_name),
        )
        conn.commit()

    return get_person_by_id(pid) or get_person_by_name(display_name)