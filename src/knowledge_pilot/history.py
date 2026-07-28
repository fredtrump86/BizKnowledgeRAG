from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HistoryStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    route TEXT,
                    confidence REAL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    trace_json TEXT NOT NULL DEFAULT '[]',
                    warning TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );
                """
            )
            self._ensure_column(
                connection,
                "messages",
                "sections_json",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            self._ensure_column(
                connection,
                "messages",
                "intent_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def create_conversation(self, title: str = "新会话") -> str:
        conversation_id = uuid.uuid4().hex
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations(id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, title[:80], now, now),
            )
        return conversation_id

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        route: str = "",
        confidence: float = 0.0,
        sources: list[dict[str, Any]] | None = None,
        sections: list[dict[str, Any]] | None = None,
        intent: dict[str, Any] | None = None,
        trace: list[str] | None = None,
        warning: str = "",
    ) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages(
                    id, conversation_id, role, content, route, confidence,
                    sources_json, sections_json, intent_json, trace_json,
                    warning, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    conversation_id,
                    role,
                    content,
                    route,
                    confidence,
                    json.dumps(sources or [], ensure_ascii=False),
                    json.dumps(sections or [], ensure_ascii=False),
                    json.dumps(intent or {}, ensure_ascii=False),
                    json.dumps(trace or [], ensure_ascii=False),
                    warning,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE conversations SET updated_at = ? WHERE id = ?
                """,
                (now, conversation_id),
            )
            if role == "user":
                connection.execute(
                    """
                    UPDATE conversations SET title = ?
                    WHERE id = ? AND title = '新会话'
                    """,
                    (content[:40], conversation_id),
                )

    def load_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, route, confidence, sources_json,
                       sections_json, intent_json, trace_json, warning,
                       created_at
                FROM messages WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json"))
            item["sections"] = json.loads(item.pop("sections_json"))
            item["intent"] = json.loads(item.pop("intent_json"))
            item["trace"] = json.loads(item.pop("trace_json"))
            messages.append(item)
        return messages

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
