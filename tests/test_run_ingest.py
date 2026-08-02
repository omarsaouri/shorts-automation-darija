from pathlib import Path
from unittest.mock import patch

import run_ingest
from db import get_connection


def _seed_queued_video(conn, video_id):
    conn.execute(
        "INSERT INTO source_videos (video_id, channel_id, status, discovered_at) "
        "VALUES (?, 'chan1', 'queued', '2026-08-02T00:00:00+00:00')",
        (video_id,),
    )
    conn.commit()


def test_processes_every_queued_video_and_runs_qc_gate():
    conn = get_connection(Path(":memory:"))
    _seed_queued_video(conn, "v1")
    _seed_queued_video(conn, "v2")

    with (
        patch("run_ingest.watcher.run", return_value=["v3"]),
        patch("run_ingest.processor.process_video") as mock_process,
        patch("run_ingest.qc_gate.run_qc_gate", return_value={"queued": 0}) as mock_qc,
    ):
        result = run_ingest.run_ingest_cycle(conn=conn)

    assert mock_process.call_count == 2
    processed_ids = {call.args[0] for call in mock_process.call_args_list}
    assert processed_ids == {"v1", "v2"}
    mock_qc.assert_called_once()
    assert result["new_videos"] == 1
    assert set(result["processed"]) == {"v1", "v2"}
    assert result["failed"] == []
    assert result["qc_summary"] == {"queued": 0}


def test_one_video_failing_does_not_stop_the_others():
    conn = get_connection(Path(":memory:"))
    _seed_queued_video(conn, "good")
    _seed_queued_video(conn, "bad")

    def _fake_process(video_id, num_clips=3, conn=None):
        if video_id == "bad":
            raise RuntimeError("highlight generator exploded")

    with (
        patch("run_ingest.watcher.run", return_value=[]),
        patch("run_ingest.processor.process_video", side_effect=_fake_process),
        patch("run_ingest.qc_gate.run_qc_gate", return_value={}) as mock_qc,
    ):
        result = run_ingest.run_ingest_cycle(conn=conn)

    assert result["processed"] == ["good"]
    assert result["failed"] == ["bad"]
    mock_qc.assert_called_once()  # still runs QC even after a failure


def test_qc_gate_runs_even_with_no_queued_videos():
    conn = get_connection(Path(":memory:"))

    with (
        patch("run_ingest.watcher.run", return_value=[]),
        patch("run_ingest.processor.process_video") as mock_process,
        patch("run_ingest.qc_gate.run_qc_gate", return_value={"held": 1}) as mock_qc,
    ):
        result = run_ingest.run_ingest_cycle(conn=conn)

    mock_process.assert_not_called()
    mock_qc.assert_called_once()
    assert result["qc_summary"] == {"held": 1}


def test_watcher_failure_does_not_abort_the_cycle():
    conn = get_connection(Path(":memory:"))
    _seed_queued_video(conn, "v1")

    with (
        patch("run_ingest.watcher.run", side_effect=RuntimeError("bad channels.yaml")),
        patch("run_ingest.processor.process_video") as mock_process,
        patch("run_ingest.qc_gate.run_qc_gate", return_value={}),
    ):
        result = run_ingest.run_ingest_cycle(conn=conn)

    assert result["new_videos"] == 0
    mock_process.assert_called_once()
