from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import publisher
from db import get_connection


@pytest.fixture(autouse=True)
def _no_real_file_upload():
    """clip_path values in these tests are fixtures, not real files —
    MediaFileUpload's constructor stats/opens the path immediately, so it
    must be mocked out to keep these tests filesystem- and API-free.
    """
    with patch("publisher.MediaFileUpload", return_value=MagicMock()):
        yield


def _seed_source_video(conn, video_id="v1"):
    conn.execute(
        "INSERT OR IGNORE INTO source_videos (video_id, channel_id, status, discovered_at) "
        "VALUES (?, 'chan1', 'captioned', '2026-01-01T00:00:00+00:00')",
        (video_id,),
    )


def _seed_clip(
    conn,
    clip_id,
    source_video_id="v1",
    score=90.0,
    status="queued",
    clip_path="/clips/v1/short_01.captioned.mp4",
    posted_at=None,
):
    _seed_source_video(conn, source_video_id)
    conn.execute(
        "INSERT INTO clips "
        "(clip_id, source_video_id, title, score, status, clip_path, posted_at, created_at) "
        "VALUES (?, ?, 'Some title', ?, ?, ?, ?, '2026-01-01T00:00:00+00:00')",
        (clip_id, source_video_id, score, status, clip_path, posted_at),
    )
    conn.commit()


def _status(conn, clip_id):
    row = conn.execute(
        "SELECT status, posted_video_id, qc_reason FROM clips WHERE clip_id = ?",
        (clip_id,),
    ).fetchone()
    return row


def _mock_youtube(video_id="yt123"):
    youtube = MagicMock()
    youtube.videos.return_value.insert.return_value.execute.return_value = {
        "id": video_id
    }
    return youtube


def _failing_youtube():
    youtube = MagicMock()
    youtube.videos.return_value.insert.return_value.execute.side_effect = RuntimeError(
        "quota exceeded"
    )
    return youtube


# --- happy path ---


def test_publish_next_uploads_highest_score_queued_clip():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "low", score=50.0)
    _seed_clip(conn, "high", score=90.0)
    youtube = _mock_youtube("yt-high")

    result = publisher.publish_next(conn=conn, youtube=youtube)

    assert result == {
        "status": "published",
        "clip_id": "high",
        "posted_video_id": "yt-high",
    }
    status, posted_id, _ = _status(conn, "high")
    assert status == "posted"
    assert posted_id == "yt-high"
    # untouched — only the single highest-score clip is published per call
    assert _status(conn, "low")[0] == "queued"


def test_publish_next_returns_no_clips_when_queue_empty():
    conn = get_connection(Path(":memory:"))
    youtube = _mock_youtube()

    result = publisher.publish_next(conn=conn, youtube=youtube)

    assert result == {"status": "no_clips"}
    youtube.videos.return_value.insert.assert_not_called()


# --- quota accounting ---


def test_publish_next_skips_when_daily_quota_budget_exhausted():
    conn = get_connection(Path(":memory:"))
    _seed_clip(
        conn, "already_posted", status="posted", posted_at="2026-07-26T00:00:00+00:00"
    )
    _seed_clip(conn, "next_up", score=99.0)
    youtube = _mock_youtube()

    result = publisher.publish_next(conn=conn, youtube=youtube, daily_quota_budget=1600)

    assert result["status"] == "quota_exhausted"
    assert result["used"] == 1600
    youtube.videos.return_value.insert.assert_not_called()
    assert _status(conn, "next_up")[0] == "queued"


def test_publish_next_ignores_clips_posted_on_other_days_for_quota():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "old_post", status="posted", posted_at="2020-01-01T00:00:00+00:00")
    _seed_clip(conn, "next_up", score=99.0)
    youtube = _mock_youtube("yt-next")

    result = publisher.publish_next(conn=conn, youtube=youtube, daily_quota_budget=1600)

    assert result["status"] == "published"


# --- failure + halt behavior ---


def test_publish_next_marks_clip_failed_and_increments_failure_counter():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1")
    youtube = _failing_youtube()

    result = publisher.publish_next(conn=conn, youtube=youtube)

    assert result["status"] == "failed"
    status, posted_id, reason = _status(conn, "c1")
    assert status == "publish_failed"
    assert posted_id is None
    assert "quota exceeded" in reason
    (failures,) = conn.execute(
        "SELECT consecutive_failures FROM publisher_state WHERE id = 1"
    ).fetchone()
    assert failures == 1


def test_publish_next_halts_after_max_consecutive_failures():
    conn = get_connection(Path(":memory:"))
    for i in range(publisher.MAX_CONSECUTIVE_FAILURES):
        _seed_clip(conn, f"fail{i}")
        result = publisher.publish_next(conn=conn, youtube=_failing_youtube())
        assert result["status"] == "failed"

    # one more clip queued up, but the publisher should now refuse to try
    _seed_clip(conn, "should_not_be_attempted")
    youtube = _mock_youtube()

    result = publisher.publish_next(conn=conn, youtube=youtube)

    assert result["status"] == "halted"
    youtube.videos.return_value.insert.assert_not_called()
    assert _status(conn, "should_not_be_attempted")[0] == "queued"


def test_publish_next_resets_failure_counter_on_success():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "fails_first")
    publisher.publish_next(conn=conn, youtube=_failing_youtube())

    _seed_clip(conn, "succeeds", score=99.0)
    publisher.publish_next(conn=conn, youtube=_mock_youtube())

    (failures,) = conn.execute(
        "SELECT consecutive_failures FROM publisher_state WHERE id = 1"
    ).fetchone()
    assert failures == 0
