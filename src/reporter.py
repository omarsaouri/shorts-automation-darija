"""Reporter (architecture doc §3.10).

Runs at end of day: pulls per-clip stats via the YouTube Analytics API for
clips posted that day, records them, and compiles a markdown digest at
`reports/{date}.md` — clips posted (with views/likes/retention), source
videos, QC rejections and why, and quota used/remaining. "Just track the
results" layer — no manual dashboard checking needed to know how the day's
batch performed.

Reads/writes state.db: `clips` (read-only), `daily_stats` (writes one row
per clip per day, idempotent on re-run), `publisher_state` (read-only, via
publisher.QUOTA_COST_PER_UPLOAD/DAILY_QUOTA_BUDGET).
"""

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build

from db import get_connection
from publisher import CLIENT_SECRETS_PATH, DAILY_QUOTA_BUDGET, QUOTA_COST_PER_UPLOAD

logger = logging.getLogger(__name__)

# Separate scope/token from publisher.py's upload token (youtube.upload) —
# a token minted for one scope can't be silently reused for another, and
# analytics-read is a different permission than upload.
ANALYTICS_SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly"]
ANALYTICS_TOKEN_PATH = Path(
    os.environ.get("YOUTUBE_ANALYTICS_TOKEN_PATH")
    or Path(__file__).parent.parent / "config" / "youtube_analytics_token.json"
)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def get_analytics_credentials(
    client_secrets_path: Path = CLIENT_SECRETS_PATH,
    token_path: Path = ANALYTICS_TOKEN_PATH,
) -> Credentials:
    """Load cached OAuth credentials for the Analytics API, same refresh/
    consent-flow shape as publisher.get_credentials but a different scope
    and token file.
    """
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), ANALYTICS_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secrets_path), ANALYTICS_SCOPES
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
    return creds


def build_analytics_client(credentials: Credentials) -> Resource:
    """Build the YouTube Analytics API v2 client."""
    return build("youtubeAnalytics", "v2", credentials=credentials)


def fetch_clip_stats(youtube_analytics: Resource, video_id: str, date: str) -> Dict:
    """Query views/likes/retention for one video on one day.

    Inputs: youtube_analytics, an authenticated Analytics API client.
        video_id, the posted YouTube video ID. date, "YYYY-MM-DD".
    Outputs: {"views": int, "likes": int, "retention": float}. A video with
        no data yet for that day (e.g. posted minutes ago) returns zeros
        rather than raising — a quiet first day of stats is normal, not an
        error.
    """
    response = (
        youtube_analytics.reports()
        .query(
            ids="channel==MINE",
            startDate=date,
            endDate=date,
            metrics="views,likes,averageViewPercentage",
            filters=f"video=={video_id}",
        )
        .execute()
    )
    rows = response.get("rows") or []
    if not rows:
        return {"views": 0, "likes": 0, "retention": 0.0}
    views, likes, retention = rows[0]
    return {"views": int(views), "likes": int(likes), "retention": float(retention)}


def record_daily_stats(conn, date: str, clip_id: str, stats: Dict) -> None:
    """Upsert one clip's stats for one day.

    Tables touched: daily_stats (insert or replace, keyed on (date, clip_id)
    — safe to call more than once for the same day/clip).
    """
    conn.execute(
        "INSERT INTO daily_stats (date, clip_id, views, likes, retention) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(date, clip_id) DO UPDATE SET "
        "views = excluded.views, likes = excluded.likes, retention = excluded.retention",
        (date, clip_id, stats["views"], stats["likes"], stats["retention"]),
    )


def fetch_stats_for_posted_clips(conn, youtube_analytics: Resource, date: str) -> int:
    """Fetch + record Analytics stats for every clip posted on `date`.

    Inputs: conn, open state.db connection. youtube_analytics, an
        authenticated Analytics API client. date, "YYYY-MM-DD".
    Outputs: number of clips fetched.
    Tables touched: clips (read-only), daily_stats (one row written per
        posted clip that day).
    """
    rows = conn.execute(
        "SELECT clip_id, posted_video_id FROM clips "
        "WHERE status = 'posted' AND date(posted_at) = ?",
        (date,),
    ).fetchall()
    for clip_id, posted_video_id in rows:
        stats = fetch_clip_stats(youtube_analytics, posted_video_id, date)
        record_daily_stats(conn, date, clip_id, stats)
    conn.commit()
    return len(rows)


def _posted_clips_with_stats(conn, date: str) -> List[Dict]:
    rows = conn.execute(
        """
        SELECT c.clip_id, c.title, c.source_video_id, sv.title AS source_title,
               COALESCE(ds.views, 0), COALESCE(ds.likes, 0), ds.retention
        FROM clips c
        JOIN source_videos sv ON sv.video_id = c.source_video_id
        LEFT JOIN daily_stats ds ON ds.clip_id = c.clip_id AND ds.date = ?
        WHERE c.status = 'posted' AND date(c.posted_at) = ?
        ORDER BY COALESCE(ds.views, 0) DESC
        """,
        (date, date),
    ).fetchall()
    return [
        {
            "clip_id": r[0],
            "title": r[1] or "(untitled)",
            "source_video_id": r[2],
            "source_title": r[3] or "(untitled)",
            "views": r[4],
            "likes": r[5],
            "retention": r[6],
        }
        for r in rows
    ]


def _qc_rejections(conn, date: str) -> List[Dict]:
    rows = conn.execute(
        """
        SELECT clip_id, source_video_id, status, qc_reason
        FROM clips
        WHERE status LIKE 'rejected_%' AND date(qc_checked_at) = ?
        ORDER BY clip_id
        """,
        (date,),
    ).fetchall()
    return [
        {
            "clip_id": r[0],
            "source_video_id": r[1],
            "status": r[2],
            "reason": r[3] or "",
        }
        for r in rows
    ]


def _quota_summary(conn, date: str) -> Dict:
    (posted_count,) = conn.execute(
        "SELECT COUNT(*) FROM clips WHERE status = 'posted' AND date(posted_at) = ?",
        (date,),
    ).fetchone()
    used = posted_count * QUOTA_COST_PER_UPLOAD
    return {
        "used": used,
        "budget": DAILY_QUOTA_BUDGET,
        "remaining": max(0, DAILY_QUOTA_BUDGET - used),
    }


def generate_report_markdown(conn, date: str) -> str:
    """Build the day's markdown report from whatever's in state.db.

    Inputs: conn, open state.db connection. date, "YYYY-MM-DD".
    Outputs: full markdown text for reports/{date}.md.
    Tables touched: clips, source_videos, daily_stats (all read-only).
    """
    posted = _posted_clips_with_stats(conn, date)
    rejections = _qc_rejections(conn, date)
    quota = _quota_summary(conn, date)

    lines = [f"# Daily Report — {date}", ""]

    lines.append("## Clips posted")
    lines.append("")
    if posted:
        lines.append("| Clip | Source video | Views | Likes | Retention |")
        lines.append("|---|---|---|---|---|")
        for c in posted:
            retention = f"{c['retention']:.1f}%" if c["retention"] is not None else "—"
            lines.append(
                f"| {c['title']} ({c['clip_id']}) | {c['source_title']} "
                f"({c['source_video_id']}) | {c['views']} | {c['likes']} | {retention} |"
            )
    else:
        lines.append("No clips posted today.")
    lines.append("")

    lines.append("## QC rejections")
    lines.append("")
    if rejections:
        lines.append("| Clip | Source video | Status | Reason |")
        lines.append("|---|---|---|---|")
        for r in rejections:
            lines.append(
                f"| {r['clip_id']} | {r['source_video_id']} | {r['status']} | {r['reason']} |"
            )
    else:
        lines.append("No QC rejections today.")
    lines.append("")

    lines.append("## Quota")
    lines.append("")
    lines.append(
        f"Used: {quota['used']} / Budget: {quota['budget']} "
        f"({quota['remaining']} remaining)"
    )
    lines.append("")

    return "\n".join(lines)


def write_report(content: str, date: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Write the report markdown to reports/{date}.md, creating the dir if needed."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{date}.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def run_daily_report(
    date: Optional[str] = None,
    conn=None,
    youtube_analytics: Optional[Resource] = None,
) -> Path:
    """Fetch stats for today's posted clips, then compile + write the report.

    Inputs: date, "YYYY-MM-DD" (defaults to today, UTC). conn, optional open
        state.db connection (tests mainly; a fresh one is opened/closed
        otherwise). youtube_analytics, optional authenticated Analytics API
        client (tests mainly; built via get_analytics_credentials()
        otherwise) — only built/called if there's at least one posted clip
        to fetch stats for, so a quiet day never needs real API access.
    Outputs: path to the written reports/{date}.md.
    Tables touched: see fetch_stats_for_posted_clips and
        generate_report_markdown.
    """
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    own_conn = conn is None
    conn = conn or get_connection()

    try:
        (has_posted,) = conn.execute(
            "SELECT EXISTS(SELECT 1 FROM clips WHERE status = 'posted' AND date(posted_at) = ?)",
            (date,),
        ).fetchone()
        if has_posted:
            client = youtube_analytics or build_analytics_client(
                get_analytics_credentials()
            )
            fetch_stats_for_posted_clips(conn, client, date)

        content = generate_report_markdown(conn, date)
        return write_report(content, date)
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date", default=None, help="YYYY-MM-DD, defaults to today (UTC)"
    )
    args = parser.parse_args()
    path = run_daily_report(date=args.date)
    logger.info("report written: %s", path)
