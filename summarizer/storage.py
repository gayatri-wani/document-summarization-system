import os
import sqlite3
from contextlib import closing
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "history.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(_connect()) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                title TEXT,
                method TEXT,
                query TEXT,
                original_words INTEGER,
                summary_words INTEGER,
                summary TEXT
            )"""
        )
        conn.commit()


def save_summary(title, method, query, original_words, summary_words, summary):
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO history (created_at, title, method, query, original_words, "
            "summary_words, summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), title, method, query or "",
             original_words, summary_words, summary),
        )
        conn.commit()


def list_history(limit=50):
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_history(row_id):
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM history WHERE id = ?", (row_id,))
        conn.commit()


def clear_history():
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM history")
        conn.commit()