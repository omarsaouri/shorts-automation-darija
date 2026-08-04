from pathlib import Path
from unittest.mock import patch

import processor
from db import get_connection


def _seed_source_video(conn, video_id="abc123"):
    conn.execute(
        "INSERT INTO source_videos (video_id, channel_id, title, status, discovered_at) "
        "VALUES (?, 'chan1', 'Some title', 'queued', '2026-01-01T00:00:00+00:00')",
        (video_id,),
    )
    conn.commit()


def _fake_generate_shorts_result():
    return {
        "source_video_url": "/raw/abc123.mp4",
        "transcript": {
            "duration": 120.0,
            "segments": [{"start": 0.0, "end": 5.0, "text": "hello"}],
        },
        "highlights": [],
        "shorts": [
            {
                "title": "Clip 1",
                "start_time": 10.0,
                "end_time": 70.0,
                "score": 90,
                "clip_url": "/clips/abc123/short_01.mp4",
            },
            {
                "title": "Clip 2 (crop failed)",
                "start_time": 80.0,
                "end_time": 140.0,
                "score": 70,
                "clip_url": None,
                "error": "ffmpeg failed",
            },
        ],
    }


def test_process_video_captions_successful_clips_and_skips_failed_ones():
    conn = get_connection(Path(":memory:"))
    _seed_source_video(conn)

    with (
        patch(
            "processor.generate_shorts", return_value=_fake_generate_shorts_result()
        ) as mock_generate,
        patch(
            "processor.captioner.burn_captions",
            return_value="/clips/abc123/short_01.captioned.mp4",
        ) as mock_burn,
    ):
        result = processor.process_video("abc123", num_clips=2, conn=conn)

    mock_generate.assert_called_once()
    assert mock_generate.call_args.args[0] == "https://www.youtube.com/watch?v=abc123"
    kwargs = mock_generate.call_args.kwargs
    assert kwargs["num_clips"] == 2
    assert kwargs["aspect_ratio"] == "9:16"

    # only the successfully-cropped clip gets captioned
    mock_burn.assert_called_once_with(
        "/clips/abc123/short_01.mp4",
        [{"start": 0.0, "end": 5.0, "text": "hello"}],
        window_start=10.0,
        window_end=70.0,
    )

    assert result[0]["captioned_path"] == "/clips/abc123/short_01.captioned.mp4"
    assert result[0]["end_time"] - result[0]["start_time"] == 60.0
    assert result[1]["captioned_path"] is None


def test_process_video_writes_clips_rows_with_correct_status():
    conn = get_connection(Path(":memory:"))
    _seed_source_video(conn)

    with (
        patch("processor.generate_shorts", return_value=_fake_generate_shorts_result()),
        patch(
            "processor.captioner.burn_captions",
            return_value="/clips/abc123/short_01.captioned.mp4",
        ),
    ):
        processor.process_video("abc123", conn=conn)

    rows = conn.execute(
        "SELECT clip_id, status, clip_path, score, fingerprint FROM clips ORDER BY clip_id"
    ).fetchall()
    assert rows[0][:4] == (
        "abc123_01",
        "pending_qc",
        "/clips/abc123/short_01.captioned.mp4",
        90.0,
    )
    assert rows[0][4]  # fingerprint written, non-empty
    assert rows[1][1] == "failed"
    assert rows[1][2] is None


def test_process_video_updates_source_video_status_to_captioned():
    conn = get_connection(Path(":memory:"))
    _seed_source_video(conn)

    with (
        patch("processor.generate_shorts", return_value=_fake_generate_shorts_result()),
        patch("processor.captioner.burn_captions", return_value="/out.mp4"),
    ):
        processor.process_video("abc123", conn=conn)

    row = conn.execute(
        "SELECT status, downloaded_at FROM source_videos WHERE video_id = 'abc123'"
    ).fetchone()
    assert row[0] == "captioned"
    assert row[1] is not None


def test_process_video_marks_failed_on_generate_shorts_error():
    conn = get_connection(Path(":memory:"))
    _seed_source_video(conn)

    with patch(
        "processor.generate_shorts", side_effect=RuntimeError("download failed")
    ):
        try:
            processor.process_video("abc123", conn=conn)
            assert False, "expected RuntimeError to propagate"
        except RuntimeError:
            pass

    row = conn.execute(
        "SELECT status FROM source_videos WHERE video_id = 'abc123'"
    ).fetchone()
    assert row[0] == "failed"


def test_process_video_skips_captioning_when_clip_url_missing():
    conn = get_connection(Path(":memory:"))
    _seed_source_video(conn)
    result = _fake_generate_shorts_result()
    result["shorts"] = [result["shorts"][1]]  # only the crop-failed one

    with (
        patch("processor.generate_shorts", return_value=result),
        patch("processor.captioner.burn_captions") as mock_burn,
    ):
        processor.process_video("abc123", conn=conn)

    mock_burn.assert_not_called()
