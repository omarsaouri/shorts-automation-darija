"""Channel Watcher (architecture doc §3.1).

Polls each source channel's public RSS feed, diffs new video IDs against
`state.db`, and enqueues them for download. No YouTube Data API quota spent
here — RSS only.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree

import yaml

from db import get_connection

logger = logging.getLogger(__name__)

RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_YT_NS = "{http://www.youtube.com/xml/schemas/2015}"


def fetch_feed(channel_id: str) -> bytes:
    """Download the raw Atom RSS feed for a channel.

    Inputs: channel_id, YouTube channel ID.
    Outputs: raw XML bytes.
    Tables touched: none.
    """
    with urlopen(RSS_URL.format(channel_id=channel_id), timeout=10) as resp:
        return resp.read()


def parse_feed(xml_bytes: bytes) -> list[dict[str, str]]:
    """Parse an Atom feed into video entries.

    Inputs: xml_bytes, raw feed content (from fetch_feed or a test fixture).
    Outputs: list of dicts with video_id, channel_id, title, published.
    Tables touched: none.
    """
    root = ElementTree.fromstring(xml_bytes)
    entries = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        video_id = entry.findtext(f"{_YT_NS}videoId")
        channel_id = entry.findtext(f"{_YT_NS}channelId")
        title = entry.findtext(f"{_ATOM_NS}title")
        published = entry.findtext(f"{_ATOM_NS}published")
        if video_id and channel_id:
            entries.append(
                {
                    "video_id": video_id,
                    "channel_id": channel_id,
                    "title": title or "",
                    "published": published or "",
                }
            )
    return entries


def enqueue_new_videos(
    conn: sqlite3.Connection, entries: list[dict[str, str]]
) -> list[str]:
    """Insert entries not already in source_videos, with status='queued'.

    Inputs: conn, open state.db connection. entries, parsed feed entries.
    Outputs: list of video_ids newly inserted.
    Tables touched: source_videos (read for dedup, insert new rows).
    """
    newly_queued = []
    now = datetime.now(timezone.utc).isoformat()
    for entry in entries:
        existing = conn.execute(
            "SELECT 1 FROM source_videos WHERE video_id = ?", (entry["video_id"],)
        ).fetchone()
        if existing:
            continue
        conn.execute(
            "INSERT INTO source_videos (video_id, channel_id, title, status, discovered_at) "
            "VALUES (?, ?, ?, 'queued', ?)",
            (entry["video_id"], entry["channel_id"], entry["title"], now),
        )
        newly_queued.append(entry["video_id"])
        logger.info("queued new video %s (%s)", entry["video_id"], entry["title"])
    conn.commit()
    return newly_queued


def load_channel_ids(config_path: Path) -> list[str]:
    """Read channel IDs out of config/channels.yaml.

    Inputs: config_path, path to the YAML config.
    Outputs: list of channel_id strings.
    Tables touched: none.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return [c["channel_id"] for c in config.get("channels", [])]


def run(
    config_path: Path = Path(__file__).parent / "config" / "channels.yaml",
    db_path: Path = Path(__file__).parent / "state.db",
) -> list[str]:
    """Poll all configured channels and enqueue newly discovered videos.

    Inputs: config_path, channels.yaml location. db_path, state.db location.
    Outputs: list of video_ids newly queued across all channels.
    Tables touched: source_videos (via enqueue_new_videos).
    """
    conn = get_connection(db_path)
    channel_ids = load_channel_ids(config_path)
    all_new: list[str] = []
    for channel_id in channel_ids:
        try:
            xml_bytes = fetch_feed(channel_id)
        except Exception:
            logger.exception("failed to fetch RSS feed for channel %s", channel_id)
            continue
        entries = parse_feed(xml_bytes)
        all_new.extend(enqueue_new_videos(conn, entries))
    conn.close()
    return all_new


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
