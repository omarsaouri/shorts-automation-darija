from datetime import datetime, timezone

from db import get_connection
from watcher import enqueue_new_videos, parse_feed

FIXTURE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:yt="http://www.youtube.com/xml/schemas/2015">
  <entry>
    <yt:videoId>vid_new</yt:videoId>
    <yt:channelId>chan_1</yt:channelId>
    <title>Brand new video</title>
    <published>2026-07-10T00:00:00+00:00</published>
  </entry>
  <entry>
    <yt:videoId>vid_seen</yt:videoId>
    <yt:channelId>chan_1</yt:channelId>
    <title>Already downloaded</title>
    <published>2026-07-09T00:00:00+00:00</published>
  </entry>
</feed>
"""


def test_parse_feed_extracts_all_entries():
    entries = parse_feed(FIXTURE_FEED)
    assert [e["video_id"] for e in entries] == ["vid_new", "vid_seen"]
    assert entries[0]["title"] == "Brand new video"
    assert entries[0]["channel_id"] == "chan_1"


def test_enqueue_new_videos_skips_existing_and_inserts_new(tmp_path):
    conn = get_connection(tmp_path / "state.db")
    conn.execute(
        "INSERT INTO source_videos (video_id, channel_id, title, status, discovered_at) "
        "VALUES ('vid_seen', 'chan_1', 'Already downloaded', 'downloaded', ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()

    entries = parse_feed(FIXTURE_FEED)
    newly_queued = enqueue_new_videos(conn, entries)

    assert newly_queued == ["vid_new"]
    rows = conn.execute("SELECT video_id, status FROM source_videos").fetchall()
    assert ("vid_new", "queued") in rows
    assert ("vid_seen", "downloaded") in rows  # untouched, not re-queued
