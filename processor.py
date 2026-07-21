"""Processor (architecture doc §4 pipeline detail).

Wraps vendor's `generate_shorts(mode="local")` — download, transcription,
highlight scoring, and scene-cut-snapped vertical crop all happen inside
that one call, per CLAUDE.md's "call generate_shorts(...) rather than
reimplementing its flow" — and adds the one net-new step generate_shorts
doesn't do: caption burn-in on each rendered clip. Scene detection is not
called directly here either; `darija_overrides.scene_snap_crop` intersects
it into generate_shorts's internal crop step by monkeypatching
`crop_highlights_local` (see that module's docstring for why that's the
only way to do it without reimplementing pipeline.py's flow).

Reads/writes state.db: `source_videos` (status transitions) and `clips`
(one row per rendered clip, ready for `qc_gate.py` to pick up next).
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_VENDOR_PATH = Path(__file__).parent / "vendor" / "ai-youtube-shorts-generator"
if str(_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PATH))

import captioner  # noqa: E402
from db import get_connection  # noqa: E402
from darija_overrides import (  # noqa: E402
    clipper_stable,
    highlights_chunking,
    highlights_duration_filter,
    llm_ollama,
    scene_snap_crop,
    transcriber_darija,
)
from shorts_generator.pipeline import generate_shorts  # noqa: E402


def install_overrides() -> None:
    """Install every darija_overrides patch generate_shorts(mode="local")
    needs. Idempotent — each module's own install() only captures the
    original vendor function once, so calling this per-video is cheap and
    safe.
    """
    llm_ollama.install()
    transcriber_darija.install()
    clipper_stable.install()
    highlights_chunking.install()
    highlights_duration_filter.install()
    scene_snap_crop.install()


def _caption_short(short: Dict, transcript_segments: List[Dict]) -> Dict:
    """Burn captions onto one generate_shorts()-rendered clip.

    Inputs: short, one entry from generate_shorts()'s "shorts" list.
        transcript_segments, the full source video's transcript segments.
    Outputs: `short` plus `captioned_path` (None if the crop itself already
        failed, or if burn-in raises — `caption_error` is set in that case).
    """
    clip_url = short.get("clip_url")
    if not clip_url:
        return {**short, "captioned_path": None}

    try:
        captioned_path = captioner.burn_captions(
            clip_url,
            transcript_segments,
            window_start=float(short["start_time"]),
            window_end=float(short["end_time"]),
        )
        return {**short, "captioned_path": captioned_path}
    except Exception as e:
        logger.exception("caption burn-in failed for %s", clip_url)
        return {**short, "captioned_path": None, "caption_error": str(e)}


def _record_clip(
    conn, source_video_id: str, index: int, short: Dict, created_at: str
) -> None:
    """Insert/replace one clips row. clip_id is deterministic
    (`{video_id}_{index}`) so re-processing a video overwrites its old rows
    rather than accumulating duplicates.

    Tables touched: clips (insert or replace).
    """
    clip_id = f"{source_video_id}_{index:02d}"
    conn.execute(
        "INSERT OR REPLACE INTO clips "
        "(clip_id, source_video_id, title, score, status, clip_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            clip_id,
            source_video_id,
            short.get("title"),
            float(short.get("score", 0)),
            "pending_qc" if short.get("captioned_path") else "failed",
            short.get("captioned_path") or short.get("clip_url"),
            created_at,
        ),
    )


def process_video(
    video_id: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    language: Optional[str] = None,
    conn=None,
) -> List[Dict]:
    """Run one source video through generate_shorts(mode="local") + caption
    burn-in.

    Inputs: video_id, YouTube video ID (source_videos.video_id). num_clips,
        aspect_ratio, language — forwarded to generate_shorts. conn,
        optional open state.db connection (mainly for tests; a fresh one
        via get_connection() is opened and closed otherwise).
    Outputs: list of short dicts (highlight fields + clip_url +
        captioned_path), one per candidate generate_shorts rendered.
    Tables touched: source_videos (status: processing -> captioned/failed),
        clips (one row per rendered short).
    """
    install_overrides()

    own_conn = conn is None
    conn = conn or get_connection()
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "UPDATE source_videos SET status = 'processing' WHERE video_id = ?",
        (video_id,),
    )
    conn.commit()

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        result = generate_shorts(
            youtube_url,
            num_clips=num_clips,
            aspect_ratio=aspect_ratio,
            language=language,
            mode="local",
        )
    except Exception:
        conn.execute(
            "UPDATE source_videos SET status = 'failed' WHERE video_id = ?",
            (video_id,),
        )
        conn.commit()
        if own_conn:
            conn.close()
        raise

    transcript_segments = result["transcript"]["segments"]
    captioned_shorts = [
        _caption_short(short, transcript_segments) for short in result["shorts"]
    ]

    for i, short in enumerate(captioned_shorts, 1):
        _record_clip(conn, video_id, i, short, now)

    # generate_shorts(mode="local") downloads, transcribes, scores, crops,
    # and (via the scene-snap override) scene-detects in one call, so
    # per-step timestamps aren't available — downloaded_at is set here,
    # once the whole run (including caption burn-in) has succeeded.
    conn.execute(
        "UPDATE source_videos SET status = 'captioned', downloaded_at = ? WHERE video_id = ?",
        (now, video_id),
    )
    conn.commit()
    if own_conn:
        conn.close()

    return captioned_shorts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="YouTube video ID to process")
    parser.add_argument("--num-clips", type=int, default=3)
    parser.add_argument("--aspect-ratio", default="9:16")
    parser.add_argument("--language", default=None)
    args = parser.parse_args()
    process_video(
        args.video_id,
        num_clips=args.num_clips,
        aspect_ratio=args.aspect_ratio,
        language=args.language,
    )
