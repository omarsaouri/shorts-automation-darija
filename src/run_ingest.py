"""Ingest cycle (architecture doc §6's watcher -> processor -> QC gate chain).

The sequence diagram in §6 shows the watcher, downloader, processor, and QC
gate running as one chain every time the scheduler wakes up (every 2-4h) —
not four independently-scheduled cron/launchd jobs. This is that chain,
driving watcher.py + processor.py + qc_gate.py (each already independently
runnable and already tested on their own) across every currently-queued
video in one invocation. No new pipeline logic lives here — this is
orchestration only, the way a human running the three scripts by hand in
sequence would.

A single video's processing failure doesn't abort the cycle — processor.py
already marks that video 'failed' and re-raises (per its own contract);
this just catches that, logs it, and moves on to the next queued video, same
"one bad thing doesn't take down the batch" pattern as
shorts_generator.highlights.get_highlights's per-chunk resilience.

qc_gate.run_qc_gate() always runs, even with zero newly-processed videos —
it also reconsiders 'held' clips from a previous cycle (source-diversity/
daily-cap holds), which needs to happen every cycle regardless of whether
this cycle found anything new.

Reads/writes state.db: source_videos (via watcher.run() + processor.process_video()),
clips (via processor.process_video() + qc_gate.run_qc_gate()).
"""

import argparse
import logging
from typing import Dict, List

import processor
import qc_gate
import watcher
from db import get_connection

logger = logging.getLogger(__name__)


def run_ingest_cycle(conn=None, num_clips: int = 3) -> Dict:
    """Poll for new videos, process every currently-queued one, then run QC gate.

    Inputs: conn, optional open state.db connection (mainly for tests; a
        fresh one via get_connection() is opened and closed otherwise).
        num_clips, forwarded to processor.process_video per video.
    Outputs: {"new_videos": int, "processed": [video_id, ...],
        "failed": [video_id, ...], "qc_summary": dict}.
    Tables touched: see module docstring.
    """
    own_conn = conn is None
    conn = conn or get_connection()

    try:
        new_video_ids: List[str] = watcher.run()
    except Exception:
        logger.exception("watcher.run() failed — continuing with 0 new videos")
        new_video_ids = []

    queued_ids = [
        row[0]
        for row in conn.execute(
            "SELECT video_id FROM source_videos WHERE status = 'queued'"
        ).fetchall()
    ]

    processed: List[str] = []
    failed: List[str] = []
    for video_id in queued_ids:
        try:
            processor.process_video(video_id, num_clips=num_clips, conn=conn)
            processed.append(video_id)
        except Exception:
            logger.exception("processing failed for video %s", video_id)
            failed.append(video_id)
            continue

    qc_summary = qc_gate.run_qc_gate(conn=conn)

    if own_conn:
        conn.close()

    return {
        "new_videos": len(new_video_ids),
        "processed": processed,
        "failed": failed,
        "qc_summary": qc_summary,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clips", type=int, default=3)
    args = parser.parse_args()
    summary = run_ingest_cycle(num_clips=args.num_clips)
    logger.info("ingest cycle summary: %s", summary)
