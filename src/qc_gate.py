"""QC gate (architecture doc §3.8/§5).

The last checkpoint before a clip is eligible for `publisher.py`. Reads
`clips` rows at status 'pending_qc' or 'held' and, per clip, checks (in
order) dedup vs. previously posted clips, score threshold, and format
validity (captions present, duration, aspect ratio). Clips passing all three
then go through a source-diversity throttle and a daily-clip cap, both of
which prefer higher-scoring clips. No bypass flag exists anywhere in this
file, per CLAUDE.md's non-negotiable QC gate constraint.

Reads/writes state.db: `clips` (status + qc_reason + qc_checked_at transitions only).
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from db import get_connection

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 60
SOURCE_DIVERSITY_CAP = 2
DAILY_CLIP_CAP = 10
MAX_DURATION_SECONDS = 60.0
TARGET_ASPECT_RATIO = 9 / 16
ASPECT_RATIO_TOLERANCE = 0.02

_CANDIDATE_STATUSES = ("pending_qc", "held")


def _probe_clip(clip_path: str) -> Tuple[int, int, float]:
    """ffprobe a clip for (width, height, duration_seconds).

    Inputs: clip_path, path to a clip video file.
    Outputs: (width, height, duration_seconds) from the first video stream.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            clip_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return (
        int(stream["width"]),
        int(stream["height"]),
        float(data["format"]["duration"]),
    )


def _check_format(clip_path: Optional[str]) -> Optional[str]:
    """Format validation: captions present, duration, aspect ratio.

    Inputs: clip_path, the clips.clip_path value for one clip.
    Outputs: None if the clip passes every format check, else a reason string.
    """
    if not clip_path:
        return "no clip_path recorded"
    if not clip_path.endswith(".captioned.mp4"):
        return f"captions not present: {clip_path} is not a .captioned.mp4 file"

    try:
        width, height, duration = _probe_clip(clip_path)
    except Exception as e:
        return f"ffprobe failed on {clip_path}: {e}"

    if duration >= MAX_DURATION_SECONDS:
        return f"duration {duration:.1f}s >= {MAX_DURATION_SECONDS:.0f}s limit"

    ratio = width / height
    if abs(ratio - TARGET_ASPECT_RATIO) > ASPECT_RATIO_TOLERANCE:
        return (
            f"aspect ratio {width}x{height} ({ratio:.3f}) not within tolerance of 9:16"
        )

    return None


def _find_duplicate(conn, fingerprint: Optional[str]) -> Optional[str]:
    """Look up whether `fingerprint` matches an already-posted clip.

    Inputs: conn, open state.db connection. fingerprint, clips.fingerprint
        value for the candidate clip.
    Outputs: clip_id of the matching posted clip, or None.
    Tables touched: clips (read-only).
    """
    if not fingerprint:
        return None
    row = conn.execute(
        "SELECT clip_id FROM clips WHERE status = 'posted' AND fingerprint = ? LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return row[0] if row else None


def _transition(conn, clip_id: str, status: str, reason: str) -> None:
    """Update a clip's status + qc_reason, stamping qc_checked_at (UTC ISO)
    so reporter.py can scope "QC rejections today" to a real timestamp
    instead of guessing from created_at.
    """
    logger.info("clip %s -> %s (%s)", clip_id, status, reason)
    conn.execute(
        "UPDATE clips SET status = ?, qc_reason = ?, qc_checked_at = ? WHERE clip_id = ?",
        (status, reason, datetime.now(timezone.utc).isoformat(), clip_id),
    )


def run_qc_gate(conn=None) -> Dict[str, int]:
    """Run every pending_qc/held clip through the QC gate.

    Inputs: conn, optional open state.db connection (mainly for tests; a
        fresh one via get_connection() is opened and closed otherwise).
    Outputs: dict of outcome -> count (rejected_duplicate, rejected_score,
        rejected_format, held, queued).
    Tables touched: clips (status + qc_reason updated for every candidate row).
    """
    own_conn = conn is None
    conn = conn or get_connection()

    placeholders = ",".join("?" for _ in _CANDIDATE_STATUSES)
    candidates = conn.execute(
        f"SELECT * FROM clips WHERE status IN ({placeholders}) ORDER BY score DESC",
        _CANDIDATE_STATUSES,
    ).fetchall()
    columns = [d[0] for d in conn.execute("SELECT * FROM clips LIMIT 0").description]
    candidates = [dict(zip(columns, row)) for row in candidates]

    counts = {
        "rejected_duplicate": 0,
        "rejected_score": 0,
        "rejected_format": 0,
        "held": 0,
        "queued": 0,
    }

    eligible: List[Dict] = []
    for clip in candidates:
        clip_id = clip["clip_id"]

        dup_of = _find_duplicate(conn, clip["fingerprint"])
        if dup_of:
            _transition(
                conn, clip_id, "rejected_duplicate", f"duplicate of clip {dup_of}"
            )
            counts["rejected_duplicate"] += 1
            continue

        if clip["score"] is None or clip["score"] < SCORE_THRESHOLD:
            _transition(
                conn,
                clip_id,
                "rejected_score",
                f"score {clip['score']} < threshold {SCORE_THRESHOLD}",
            )
            counts["rejected_score"] += 1
            continue

        format_reason = _check_format(clip["clip_path"])
        if format_reason:
            _transition(conn, clip_id, "rejected_format", format_reason)
            counts["rejected_format"] += 1
            continue

        eligible.append(clip)

    # Source-diversity throttle: keep only the top SOURCE_DIVERSITY_CAP
    # clips per source video (eligible is still score-desc from the query).
    per_source_count: Dict[str, int] = {}
    diversity_survivors: List[Dict] = []
    for clip in eligible:
        video_id = clip["source_video_id"]
        seen = per_source_count.get(video_id, 0)
        if seen >= SOURCE_DIVERSITY_CAP:
            _transition(
                conn,
                clip["clip_id"],
                "held",
                f"source diversity cap ({SOURCE_DIVERSITY_CAP}) reached for {video_id}",
            )
            counts["held"] += 1
            continue
        per_source_count[video_id] = seen + 1
        diversity_survivors.append(clip)

    # Daily cap: of the diversity survivors (still score-desc), only the top
    # DAILY_CLIP_CAP get queued for publish; the rest are held for later.
    for i, clip in enumerate(diversity_survivors):
        if i < DAILY_CLIP_CAP:
            _transition(
                conn, clip["clip_id"], "queued", "passed QC, queued for publish"
            )
            counts["queued"] += 1
        else:
            _transition(
                conn,
                clip["clip_id"],
                "held",
                f"daily cap ({DAILY_CLIP_CAP}) reached",
            )
            counts["held"] += 1

    conn.commit()
    if own_conn:
        conn.close()

    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = run_qc_gate()
    logger.info("qc_gate summary: %s", summary)
