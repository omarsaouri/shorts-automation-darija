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

CREATE TABLE IF NOT EXISTS clips (
    clip_id TEXT PRIMARY KEY,
    source_video_id TEXT NOT NULL REFERENCES source_videos(video_id),
    title TEXT,
    score REAL,
    status TEXT NOT NULL DEFAULT 'pending_qc',
    clip_path TEXT,
    posted_video_id TEXT,
    created_at TEXT NOT NULL
);
"""

# Columns added after the initial clips table shipped. Additive-only so a
# populated state.db is migrated in place rather than wiped.
_CLIPS_MIGRATIONS = {
    "fingerprint": "TEXT",  # content hash for qc_gate.py dedup
    "qc_reason": "TEXT",  # human-readable reject/hold reason
}


def _ensure_clip_columns(conn: sqlite3.Connection) -> None:
    """Add any missing clips columns from _CLIPS_MIGRATIONS.

    Tables touched: clips (ALTER TABLE ADD COLUMN for missing columns only).
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(clips)")}
    for column, sql_type in _CLIPS_MIGRATIONS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE clips ADD COLUMN {column} {sql_type}")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the state.db connection with schema applied.

    Inputs: db_path, path to the SQLite file (defaults to state.db at repo root).
    Outputs: sqlite3.Connection with source_videos and clips tables present.
    Tables touched: creates `source_videos` and `clips` if missing; adds any
        pending additive columns to `clips`.
    """
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    _ensure_clip_columns(conn)
    conn.commit()
    return conn
