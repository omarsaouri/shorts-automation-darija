"""Caption burn-in — new pipeline stage, not in the vendored base repo.

Takes the Whisper transcript segments already produced by the transcriber
stage and a rendered clip (clipper.py's output), builds an .ass subtitle
file, and burns it into the clip via ffmpeg's libass-backed `ass` filter.

Reads/writes no state.db tables — pure file-in, file-out transform.

Darija/RTL note (deliberately left open per project direction): Arabic
script is right-to-left and this is flagged in the architecture doc as a
common silent-breakage point (reversed/misaligned text). ffmpeg here has
libass + libharfbuzz built in, which gives it a real shot at correct
shaping, but this module does not verify RTL rendering correctness — only
that valid captions of some form are burned in. Revisit once real Darija
clips are flowing.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_FONT = "Arial Unicode MS"
DEFAULT_FONT_SIZE_RATIO = 1 / 18  # fontsize = video height * this ratio
DEFAULT_OUTLINE = 3
DEFAULT_MARGIN_V_RATIO = 1 / 12  # bottom margin = video height * this ratio


def _probe_dimensions(video_path: str) -> Tuple[int, int]:
    """Return (width, height) of a video's first video stream via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _format_ass_timestamp(seconds: float) -> str:
    """ASS uses H:MM:SS.CC (centiseconds), unlike SRT's HH:MM:SS,MMM."""
    seconds = max(0.0, seconds)
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape characters with special meaning in ASS dialogue text."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\N")
    )


def _segments_for_window(
    segments: List[Dict], window_start: float, window_end: float
) -> List[Dict]:
    """Return segments overlapping [window_start, window_end], rebased so
    window_start becomes 0 — matches the rendered clip's own timeline.
    """
    result = []
    for s in segments:
        start = max(float(s["start"]), window_start)
        end = min(float(s["end"]), window_end)
        if end <= start:
            continue
        result.append(
            {
                "start": start - window_start,
                "end": end - window_start,
                "text": s["text"],
            }
        )
    return result


def build_ass(
    segments: List[Dict],
    width: int,
    height: int,
    font: str = DEFAULT_FONT,
) -> str:
    """Build .ass subtitle file content for the given segments and clip size."""
    font_size = max(12, round(height * DEFAULT_FONT_SIZE_RATIO))
    margin_v = max(10, round(height * DEFAULT_MARGIN_V_RATIO))

    header = f"""[Script Info]
Title: Auto-generated captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,{DEFAULT_OUTLINE},0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for s in segments:
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        start = _format_ass_timestamp(float(s["start"]))
        end = _format_ass_timestamp(float(s["end"]))
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{_escape_ass_text(text)}"
        )

    return header + "\n".join(lines) + "\n"


def burn_captions(
    clip_path: str,
    segments: List[Dict],
    window_start: float,
    window_end: float,
    out_path: Optional[str] = None,
    font: str = DEFAULT_FONT,
) -> str:
    """Burn captions for [window_start, window_end] of the source transcript
    into `clip_path` (already cropped to its final window, so its own
    timeline starts at 0). Returns the captioned clip's path.
    """
    width, height = _probe_dimensions(clip_path)
    clip_segments = _segments_for_window(segments, window_start, window_end)
    ass_content = build_ass(clip_segments, width, height, font=font)

    out_path = out_path or str(
        Path(clip_path).with_name(Path(clip_path).stem + ".captioned.mp4")
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ass", delete=False, encoding="utf-8"
    ) as f:
        f.write(ass_content)
        ass_path = f.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                clip_path,
                "-vf",
                f"ass={ass_path}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                out_path,
            ],
            check=True,
        )
    finally:
        Path(ass_path).unlink(missing_ok=True)

    return out_path


if __name__ == "__main__":
    # ponytail: smallest runnable check — burns real captions into a real
    # clip via argv, since the whole point is confirming ffmpeg/libass
    # actually renders text, not mocking subprocess calls. Not part of the
    # pytest suite (that covers the pure ASS-building logic only).
    import sys

    if len(sys.argv) < 4:
        print(
            "usage: python captioner.py <clip.mp4> <window_start> <window_end> [text]"
        )
        sys.exit(1)

    clip, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    demo_text = sys.argv[4] if len(sys.argv) > 4 else "Hello, this is a caption test."
    demo_segments = [{"start": start, "end": end, "text": demo_text}]

    result = burn_captions(clip, demo_segments, start, end)
    assert Path(result).exists()
    print("OK:", result)
