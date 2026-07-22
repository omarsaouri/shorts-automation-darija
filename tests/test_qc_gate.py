from pathlib import Path
from unittest.mock import patch

import qc_gate
from db import get_connection


def _seed_source_video(conn, video_id):
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
    status="pending_qc",
    clip_path="/clips/v1/short_01.captioned.mp4",
    fingerprint=None,
):
    _seed_source_video(conn, source_video_id)
    conn.execute(
        "INSERT INTO clips "
        "(clip_id, source_video_id, title, score, status, clip_path, fingerprint, created_at) "
        "VALUES (?, ?, 'Some title', ?, ?, ?, ?, '2026-01-01T00:00:00+00:00')",
        (clip_id, source_video_id, score, status, clip_path, fingerprint),
    )
    conn.commit()


def _status(conn, clip_id):
    row = conn.execute(
        "SELECT status, qc_reason FROM clips WHERE clip_id = ?", (clip_id,)
    ).fetchone()
    return row


def _valid_probe(*args, **kwargs):
    return (720, 1280, 30.0)  # 9:16, 30s


# --- dedup ---


def test_dedup_rejects_clip_matching_a_posted_fingerprint():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "posted_1", status="posted", fingerprint="hash-a")
    _seed_clip(conn, "candidate_1", fingerprint="hash-a")

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        qc_gate.run_qc_gate(conn=conn)

    status, reason = _status(conn, "candidate_1")
    assert status == "rejected_duplicate"
    assert "posted_1" in reason


def test_dedup_passes_clip_with_different_fingerprint():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "posted_1", status="posted", fingerprint="hash-a")
    _seed_clip(conn, "candidate_1", fingerprint="hash-b")

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        qc_gate.run_qc_gate(conn=conn)

    status, _ = _status(conn, "candidate_1")
    assert status == "queued"


# --- score threshold ---


def test_score_above_threshold_passes():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1", score=65.0)

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        qc_gate.run_qc_gate(conn=conn)

    assert _status(conn, "c1")[0] == "queued"


def test_score_below_threshold_rejected():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1", score=40.0)

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        qc_gate.run_qc_gate(conn=conn)

    status, reason = _status(conn, "c1")
    assert status == "rejected_score"
    assert "40" in reason


# --- format validation ---


def test_format_passes_when_captioned_correct_duration_and_aspect():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1", clip_path="/clips/v1/short_01.captioned.mp4")

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        qc_gate.run_qc_gate(conn=conn)

    assert _status(conn, "c1")[0] == "queued"


def test_format_rejects_missing_captioned_suffix():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1", clip_path="/clips/v1/short_01.mp4")

    with patch("qc_gate._probe_clip", side_effect=_valid_probe) as mock_probe:
        qc_gate.run_qc_gate(conn=conn)

    status, reason = _status(conn, "c1")
    assert status == "rejected_format"
    assert "captions not present" in reason
    mock_probe.assert_not_called()


def test_format_rejects_duration_at_or_over_60s():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1")

    with patch("qc_gate._probe_clip", return_value=(720, 1280, 60.0)):
        qc_gate.run_qc_gate(conn=conn)

    status, reason = _status(conn, "c1")
    assert status == "rejected_format"
    assert "duration" in reason


def test_format_rejects_wrong_aspect_ratio():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1")

    with patch("qc_gate._probe_clip", return_value=(1280, 720, 30.0)):  # 16:9
        qc_gate.run_qc_gate(conn=conn)

    status, reason = _status(conn, "c1")
    assert status == "rejected_format"
    assert "aspect ratio" in reason


# --- source-diversity throttle ---


def test_source_diversity_throttle_holds_beyond_cap_per_source_video():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "v1_01", source_video_id="v1", score=90.0)
    _seed_clip(conn, "v1_02", source_video_id="v1", score=80.0)
    _seed_clip(conn, "v1_03", source_video_id="v1", score=70.0)

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        qc_gate.run_qc_gate(conn=conn)

    assert _status(conn, "v1_01")[0] == "queued"
    assert _status(conn, "v1_02")[0] == "queued"
    status, reason = _status(conn, "v1_03")
    assert status == "held"
    assert "diversity" in reason


# --- daily cap ---


def test_daily_cap_holds_beyond_top_ten_across_sources():
    conn = get_connection(Path(":memory:"))
    for i in range(12):
        _seed_clip(
            conn,
            f"c{i:02d}",
            source_video_id=f"v{i:02d}",
            score=100.0 - i,
        )

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        qc_gate.run_qc_gate(conn=conn)

    for i in range(10):
        assert _status(conn, f"c{i:02d}")[0] == "queued"
    for i in range(10, 12):
        status, reason = _status(conn, f"c{i:02d}")
        assert status == "held"
        assert "daily cap" in reason


# --- held clips are reconsidered ---


def test_held_clip_from_a_previous_run_is_reconsidered():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1", status="held", score=90.0)

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        qc_gate.run_qc_gate(conn=conn)

    assert _status(conn, "c1")[0] == "queued"


def test_rejected_clip_is_not_reconsidered():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "c1", status="rejected_score", score=90.0)

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        qc_gate.run_qc_gate(conn=conn)

    # untouched — still terminal
    assert _status(conn, "c1")[0] == "rejected_score"


# --- summary ---


def test_run_qc_gate_returns_outcome_counts():
    conn = get_connection(Path(":memory:"))
    _seed_clip(conn, "pass1", score=90.0)
    _seed_clip(conn, "fail_score", source_video_id="v2", score=10.0)

    with patch("qc_gate._probe_clip", side_effect=_valid_probe):
        summary = qc_gate.run_qc_gate(conn=conn)

    assert summary["queued"] == 1
    assert summary["rejected_score"] == 1
