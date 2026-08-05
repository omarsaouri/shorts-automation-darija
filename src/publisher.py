"""Publisher (architecture doc §3.9).

Takes the day's QC-passed queue (`clips` rows at status 'queued') and
uploads the single highest-score one via `videos.insert`
(OAuth, `youtube.upload` scope). Publishes one clip per invocation — the
"spaced out through the day" pacing from the architecture doc's daily
schedule (§6) is the scheduler's job (cron/launchd calling this script
every ~90 min), not an internal loop here.

Manages the two hard constraints from CLAUDE.md: never spend more than the
configured daily quota budget (1,600 units/upload), and never retry a
failed upload in a loop — a failed upload is logged and the clip marked
`publish_failed`; after MAX_CONSECUTIVE_FAILURES in a row, the publisher
halts entirely (further calls are no-ops) until `publisher_state` is reset.

Reads/writes state.db: `clips` (status/posted_video_id/posted_at on the
published clip, or status/qc_reason on a failed one) and `publisher_state`
(consecutive failure counter + halted flag).
"""

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import yaml
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaFileUpload

from db import get_connection

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRETS_PATH = Path(
    os.environ.get("YOUTUBE_CLIENT_SECRETS_PATH")
    or Path(__file__).parent.parent / "config" / "youtube_client_secret.json"
)
TOKEN_PATH = Path(
    os.environ.get("YOUTUBE_TOKEN_PATH")
    or Path(__file__).parent.parent / "config" / "youtube_token.json"
)

QUOTA_COST_PER_UPLOAD = 1600
DAILY_QUOTA_BUDGET = int(os.environ.get("YOUTUBE_DAILY_QUOTA_BUDGET", 10000))
MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_CATEGORY_ID = "24"  # Entertainment
CHANNEL_PROFILE_PATH = Path(
    os.environ.get("CHANNEL_PROFILE_PATH")
    or Path(__file__).parent.parent / "config" / "channel_profile.yaml"
)


def load_channel_profile(path: Path = CHANNEL_PROFILE_PATH) -> Dict:
    """Load branding/tags config (TRK-60/61) used to build upload metadata.

    Reads no state.db tables. Missing file returns {} so upload_clip falls
    back to a bare title + "#Shorts", same behavior as before this config
    existed — a missing/unfilled-in channel profile is never a hard failure.
    """
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_credentials(
    client_secrets_path: Path = CLIENT_SECRETS_PATH, token_path: Path = TOKEN_PATH
) -> Credentials:
    """Load cached OAuth credentials, refreshing or running the one-time
    local-server consent flow as needed.

    Inputs: client_secrets_path, the OAuth client JSON downloaded from
        Google Cloud Console. token_path, where the refreshable user token
        is cached between runs.
    Outputs: valid google.oauth2.credentials.Credentials.
    Tables touched: none. Interactive (opens a browser) only on first run
        or once the cached refresh token is revoked.
    """
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secrets_path), SCOPES
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
    return creds


def build_youtube_client(credentials: Credentials) -> Resource:
    """Build the YouTube Data API v3 client.

    Inputs: credentials, valid OAuth credentials from get_credentials().
    Outputs: googleapiclient Resource for the youtube v3 API.
    """
    return build("youtube", "v3", credentials=credentials)


def _row_to_clip(conn, row) -> Dict:
    columns = [d[0] for d in conn.execute("SELECT * FROM clips LIMIT 0").description]
    return dict(zip(columns, row))


def _quota_used_today(conn) -> int:
    """Units already spent on uploads posted today.

    Tables touched: clips (read-only).
    """
    (count,) = conn.execute(
        "SELECT COUNT(*) FROM clips WHERE status = 'posted' AND date(posted_at) = date('now')"
    ).fetchone()
    return count * QUOTA_COST_PER_UPLOAD


def _halted_reason(conn) -> Optional[str]:
    row = conn.execute(
        "SELECT halted_reason FROM publisher_state WHERE id = 1 AND halted = 1"
    ).fetchone()
    return row[0] if row else None


def _record_failure(conn) -> None:
    conn.execute(
        "INSERT INTO publisher_state (id, consecutive_failures) VALUES (1, 1) "
        "ON CONFLICT(id) DO UPDATE SET consecutive_failures = consecutive_failures + 1"
    )
    (failures,) = conn.execute(
        "SELECT consecutive_failures FROM publisher_state WHERE id = 1"
    ).fetchone()
    if failures >= MAX_CONSECUTIVE_FAILURES:
        conn.execute(
            "UPDATE publisher_state SET halted = 1, halted_reason = ? WHERE id = 1",
            (f"{failures} consecutive publish failures — halted, not retrying",),
        )
    conn.commit()


def _record_success(conn) -> None:
    conn.execute(
        "INSERT INTO publisher_state (id, consecutive_failures) VALUES (1, 0) "
        "ON CONFLICT(id) DO UPDATE SET consecutive_failures = 0"
    )
    conn.commit()


def upload_clip(
    youtube: Resource,
    clip: Dict,
    privacy_status: str,
    channel_profile: Optional[Dict] = None,
) -> str:
    """Upload one clip via videos.insert.

    Inputs: youtube, an authenticated API client. clip, a clips row dict
        (needs title, clip_path). privacy_status, "public"/"unlisted"/"private".
        channel_profile, parsed config/channel_profile.yaml (loaded via
        load_channel_profile() if not passed) — supplies the tagline,
        hashtags, and channel keywords appended to every upload's
        description/tags (TRK-60/61); a missing/empty profile falls back to
        the original bare "#Shorts" behavior.
    Outputs: the newly published YouTube video ID.
    """
    channel_profile = (
        channel_profile if channel_profile is not None else load_channel_profile()
    )
    title = (clip.get("title") or "Darija Short")[:100]

    hashtags = channel_profile.get("default_hashtags") or ["#Shorts"]
    description_parts = [clip.get("title") or ""]
    if channel_profile.get("tagline"):
        description_parts.append(channel_profile["tagline"])
    description_parts.append(" ".join(hashtags))
    description = "\n\n".join(p for p in description_parts if p).strip()

    tags = [h.lstrip("#") for h in hashtags] + list(
        channel_profile.get("channel_keywords") or []
    )

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": channel_profile.get("category_id", DEFAULT_CATEGORY_ID),
            "tags": tags,
        },
        "status": {"privacyStatus": privacy_status},
    }
    media = MediaFileUpload(clip["clip_path"], chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status", body=body, media_body=media
    )
    response = request.execute()
    return response["id"]


def publish_next(
    conn=None,
    youtube: Optional[Resource] = None,
    privacy_status: str = "public",
    daily_quota_budget: int = DAILY_QUOTA_BUDGET,
) -> Dict:
    """Publish the single highest-score 'queued' clip, if quota and halt
    state allow it.

    Inputs: conn, optional open state.db connection (mainly for tests).
        youtube, optional pre-built API client (mainly for tests — real
        runs build one from get_credentials()/build_youtube_client()).
        privacy_status, daily_quota_budget — see module constants.
    Outputs: dict with at least a "status" key: "halted", "quota_exhausted",
        "no_clips", "failed", or "published" (plus clip_id/posted_video_id
        as applicable).
    Tables touched: clips (status transition on the published/failed clip),
        publisher_state (failure counter).
    """
    own_conn = conn is None
    conn = conn or get_connection()

    halted_reason = _halted_reason(conn)
    if halted_reason:
        logger.error("publisher halted: %s", halted_reason)
        if own_conn:
            conn.close()
        return {"status": "halted", "reason": halted_reason}

    used = _quota_used_today(conn)
    if used + QUOTA_COST_PER_UPLOAD > daily_quota_budget:
        logger.info(
            "daily quota budget reached (%d/%d units used today), skipping",
            used,
            daily_quota_budget,
        )
        if own_conn:
            conn.close()
        return {"status": "quota_exhausted", "used": used}

    row = conn.execute(
        "SELECT * FROM clips WHERE status = 'queued' ORDER BY score DESC LIMIT 1"
    ).fetchone()
    if not row:
        if own_conn:
            conn.close()
        return {"status": "no_clips"}
    clip = _row_to_clip(conn, row)

    if youtube is None:
        youtube = build_youtube_client(get_credentials())

    try:
        posted_id = upload_clip(youtube, clip, privacy_status)
    except Exception as e:
        logger.exception("publish failed for clip %s", clip["clip_id"])
        conn.execute(
            "UPDATE clips SET status = 'publish_failed', qc_reason = ? WHERE clip_id = ?",
            (f"upload failed: {e}", clip["clip_id"]),
        )
        conn.commit()
        _record_failure(conn)
        if own_conn:
            conn.close()
        return {"status": "failed", "clip_id": clip["clip_id"], "error": str(e)}

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE clips SET status = 'posted', posted_video_id = ?, posted_at = ? WHERE clip_id = ?",
        (posted_id, now, clip["clip_id"]),
    )
    conn.commit()
    _record_success(conn)
    if own_conn:
        conn.close()
    return {
        "status": "published",
        "clip_id": clip["clip_id"],
        "posted_video_id": posted_id,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--privacy-status",
        default=os.environ.get("YOUTUBE_PRIVACY_STATUS", "public"),
        choices=["public", "unlisted", "private"],
    )
    parser.add_argument("--daily-quota-budget", type=int, default=DAILY_QUOTA_BUDGET)
    args = parser.parse_args()
    result = publish_next(
        privacy_status=args.privacy_status,
        daily_quota_budget=args.daily_quota_budget,
    )
    logger.info("publisher result: %s", result)
