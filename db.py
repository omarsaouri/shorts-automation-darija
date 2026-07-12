"""SQLite state store shared by all pipeline stages.

Owns schema creation for `state.db`. Each stage imports `get_connection`
rather than opening its own connection, so schema stays in one place.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "state.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_videos (
    video_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    discovered_at TEXT NOT NULL,
    downloaded_at TEXT
);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the state.db connection with schema applied.

    Inputs: db_path, path to the SQLite file (defaults to state.db at repo root).
    Outputs: sqlite3.Connection with source_videos table present.
    Tables touched: creates `source_videos` if missing.
    """
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn
