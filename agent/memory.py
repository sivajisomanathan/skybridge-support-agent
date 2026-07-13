"""
Persistent, SQLite-backed replacement for Phase 6/7's in-memory SESSIONS and
FEEDBACK_STORE dicts. Same retention/reset semantics as before (last
MAX_HISTORY_TURNS pairs per thread; a new thread_id starts empty), but now
survives a process restart within the same deployed instance -- which an
in-memory dict cannot do.

Scoping note (documented in README, not hidden): this is a single SQLite file
on local disk. It is NOT a shared/multi-instance store -- if the API is ever
scaled to more than one running instance, each instance would have its own
database file and its own view of session/feedback state. That would need a
real shared database (Postgres, etc.) to fix, which is out of scope here.
"""
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from agent.config import DB_PATH, MAX_HISTORY_TURNS

_local = threading.local()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_thread ON turns(thread_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                thread_id TEXT NOT NULL,
                intent_category TEXT NOT NULL,
                negative_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (thread_id, intent_category)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def touch_conversation(thread_id: str, title_hint: Optional[str] = None) -> None:
    """Registers a thread in the conversations table (for the UI's sidebar
    list) and refreshes its updated_at timestamp. title_hint, if provided,
    only sets the title the first time -- later calls just bump updated_at."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        row = conn.execute(
            "SELECT thread_id FROM conversations WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            title = (title_hint or "New conversation")[:60]
            conn.execute(
                "INSERT INTO conversations (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, title, now, now),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE thread_id = ?", (now, thread_id)
            )


def list_conversations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_history(thread_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE thread_id = ? ORDER BY turn_index ASC",
            (thread_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]


def append_turn(thread_id: str, user_input: str, assistant_response: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        last = conn.execute(
            "SELECT MAX(turn_index) AS m FROM turns WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        next_index = (last["m"] + 1) if last["m"] is not None else 0

        conn.execute(
            "INSERT INTO turns (thread_id, turn_index, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (thread_id, next_index, "user", user_input, now),
        )
        conn.execute(
            "INSERT INTO turns (thread_id, turn_index, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (thread_id, next_index + 1, "assistant", assistant_response, now),
        )

        # Retention rule: keep only the most recent MAX_HISTORY_TURNS pairs for
        # this thread. Delete anything older than that, same cap as Phase 6/7.
        max_messages = MAX_HISTORY_TURNS * 2
        ids = conn.execute(
            "SELECT id FROM turns WHERE thread_id = ? ORDER BY turn_index DESC LIMIT -1 OFFSET ?",
            (thread_id, max_messages),
        ).fetchall()
        if ids:
            id_list = [row["id"] for row in ids]
            conn.executemany("DELETE FROM turns WHERE id = ?", [(i,) for i in id_list])


def submit_feedback(thread_id: str, intent_category: str, rating: str) -> int:
    """rating: 'up' or 'down'. Returns the resulting negative_count for this
    (thread_id, intent_category) pair. 'up' resets it to zero."""
    assert rating in ("up", "down"), "rating must be 'up' or 'down'"
    with _connect() as conn:
        row = conn.execute(
            "SELECT negative_count FROM feedback WHERE thread_id = ? AND intent_category = ?",
            (thread_id, intent_category),
        ).fetchone()
        current = row["negative_count"] if row else 0
        new_count = (current + 1) if rating == "down" else 0
        conn.execute(
            """INSERT INTO feedback (thread_id, intent_category, negative_count)
               VALUES (?, ?, ?)
               ON CONFLICT(thread_id, intent_category)
               DO UPDATE SET negative_count = excluded.negative_count""",
            (thread_id, intent_category, new_count),
        )
        return new_count


def has_prior_negative_feedback(thread_id: str, intent_category: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT negative_count FROM feedback WHERE thread_id = ? AND intent_category = ?",
            (thread_id, intent_category),
        ).fetchone()
        return bool(row and row["negative_count"] > 0)
