from pathlib import Path
from unittest.mock import MagicMock, patch

import reporter
from db import get_connection


def _seed_source_video(conn, video_id, title="Source Title"):
    conn.execute(
        "INSERT OR IGNORE INTO source_videos (video_id, channel_id, title, status, discovered_at) "
        "VALUES (?, 'chan1', ?, 'captioned', '2026-01-01T00:00:00+00:00')",
        (video_id, title),
    )


def _seed_clip(
    conn,
    clip_id,
    source_video_id="v1",
    title="Clip Title",
    score=90.0,
    status="posted",
    posted_video_id="yt123",
    posted_at="2026-08-02T12:00:00+00:00",
    qc_reason=None,
    qc_checked_at=None,
):
    _seed_source_video(conn, source_video_id)
    conn.execute(
        "INSERT INTO clips "
        "(clip_id, source_video_id, title, score, status, posted_video_id, posted_at, "
        "qc_reason, qc_checked_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-08-02T00:00:00+00:00')",
        (
            clip_id,
            source_video_id,
            title,
            score,
            status,
            posted_video_id,
            posted_at,
            qc_reason,
            qc_checked_at,
        ),
    )
    conn.commit()


def _mock_analytics(views=100, likes=10, retention=55.5):
    client = MagicMock()
    client.reports.return_value.query.return_value.execute.return_value = {
        "rows": [[views, likes, retention]]
    }
    return client


# --- fetch_clip_stats ---


def test_fetch_clip_stats_parses_response_row():
    client = _mock_analytics(views=200, likes=15, retention=61.2)
    stats = reporter.fetch_clip_stats(client, "yt123", "2026-08-02")
    assert stats == {"views": 200, "likes": 15, "retention": 61.2}


def test_fetch_clip_stats_no_data_returns_zeros():
    client = MagicMock()
    client.reports.return_value.query.return_value.execute.return_value = {"rows": []}
    stats = reporter.fetch_clip_stats(client, "yt123", "2026-08-02")
    assert stats == {"views": 0, "likes": 0, "retention": 0.0}


# --- record_daily_stats ---


def test_record_daily_stats_upserts_on_rerun():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1")

    reporter.record_daily_stats(
        conn, "2026-08-02", "c1", {"views": 10, "likes": 1, "retention": 20.0}
    )
    reporter.record_daily_stats(
        conn, "2026-08-02", "c1", {"views": 50, "likes": 5, "retention": 40.0}
    )
    conn.commit()

    rows = conn.execute("SELECT views, likes, retention FROM daily_stats").fetchall()
    assert rows == [(50, 5, 40.0)]


# --- fetch_stats_for_posted_clips ---


def test_fetch_stats_for_posted_clips_only_touches_posted_that_day():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "posted_today", posted_at="2026-08-02T09:00:00+00:00")
    _seed_clip(conn, "posted_yesterday", posted_at="2026-08-01T09:00:00+00:00")
    _seed_clip(
        conn, "not_posted", status="queued", posted_video_id=None, posted_at=None
    )

    client = _mock_analytics(views=5, likes=1, retention=30.0)
    count = reporter.fetch_stats_for_posted_clips(conn, client, "2026-08-02")

    assert count == 1
    rows = conn.execute("SELECT clip_id FROM daily_stats").fetchall()
    assert rows == [("posted_today",)]


# --- generate_report_markdown (fixture comparison) ---


def test_generate_report_markdown_matches_fixture():
    conn = get_connection(Path(":memory:"))
    _seed_clip(
        conn,
        "c1",
        source_video_id="v1",
        title="Best Moment",
        posted_at="2026-08-02T10:00:00+00:00",
    )
    reporter.record_daily_stats(
        conn, "2026-08-02", "c1", {"views": 1200, "likes": 80, "retention": 62.5}
    )
    _seed_clip(
        conn,
        "c2_rejected",
        source_video_id="v2",
        status="rejected_score",
        posted_video_id=None,
        posted_at=None,
        qc_reason="score 40.0 < threshold 60",
        qc_checked_at="2026-08-02T08:00:00+00:00",
    )
    conn.commit()

    content = reporter.generate_report_markdown(conn, "2026-08-02")
    fixture = (Path(__file__).parent / "fixtures" / "report_2026-08-02.md").read_text(
        encoding="utf-8"
    )
    assert content == fixture


def test_generate_report_markdown_empty_day():
    conn = get_connection(Path(":memory:"))
    content = reporter.generate_report_markdown(conn, "2026-08-02")
    assert "No clips posted today." in content
    assert "No QC rejections today." in content
    assert "Used: 0 / Budget:" in content


# --- write_report ---


def test_write_report_writes_to_date_named_file(tmp_path):
    out = reporter.write_report("hello", "2026-08-02", reports_dir=tmp_path)
    assert out == tmp_path / "2026-08-02.md"
    assert out.read_text(encoding="utf-8") == "hello"


# --- run_daily_report ---


def test_run_daily_report_skips_analytics_client_when_nothing_posted():
    conn = get_connection(Path(":memory:"))
    with (
        patch("reporter.get_analytics_credentials") as mock_creds,
        patch("reporter.write_report") as mock_write,
    ):
        mock_write.return_value = Path("reports/2026-08-02.md")
        reporter.run_daily_report(date="2026-08-02", conn=conn)

    mock_creds.assert_not_called()
    mock_write.assert_called_once()


def test_run_daily_report_fetches_stats_when_clips_posted():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1", posted_at="2026-08-02T10:00:00+00:00")
    client = _mock_analytics(views=99, likes=9, retention=50.0)

    with patch("reporter.write_report") as mock_write:
        mock_write.return_value = Path("reports/2026-08-02.md")
        reporter.run_daily_report(
            date="2026-08-02", conn=conn, youtube_analytics=client
        )

    rows = conn.execute("SELECT views FROM daily_stats WHERE clip_id = 'c1'").fetchall()
    assert rows == [(99,)]
    mock_write.assert_called_once()
