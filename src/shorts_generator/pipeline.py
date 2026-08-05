"""End-to-end orchestrator.

Local only — yt-dlp + faster-whisper + local LLM + ffmpeg/opencv. The
upstream base repo's other mode (MuAPI-backed, paid, `mode="api"`) has been
removed entirely; see CLAUDE.md's no-paid-APIs constraint.
"""
from typing import Dict, List, Optional

from .highlights import get_highlights


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.

    Returns:
        {
          "source_video_url": str,   # local path to the downloaded source
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with local clip path
        }
    """
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm
    from .local.transcriber import transcribe_local

    source_path = download_youtube_local(youtube_url, fmt=download_format)

    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_local_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights_local(source_path, top, aspect_ratio=aspect_ratio)

    return {
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }
