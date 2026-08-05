"""Caption burn-in — new pipeline stage, not in the vendored base repo.

Takes the Whisper transcript segments already produced by the transcriber
stage and a rendered clip (clipper.py's output), builds an .ass subtitle
file, and burns it into the clip via ffmpeg's libass-backed `ass` filter.

Reads/writes no state.db tables — pure file-in, file-out transform.

Style (architecture-doc deviation, deliberate — see project session notes):
short-form "modern" caption cards, word-count-capped at two lines, with one
keyword per line statically highlighted in yellow (the highlighted word
itself is chosen by a deterministic stopword filter, not per-word timing —
see _highlight_word). The architecture doc's §3.4 describes per-word
karaoke-sync highlighting instead; that's not what this does. What this
*does* do is time each caption card to real speech: cards are built from
segment["words"] (per-word start/end spans) when the transcript carries
them — see shorts_generator.local.transcriber's
_group_words_into_segments — so a card appears and disappears in sync with
the words it shows, not on a proportional guess. Segments without "words"
(currently only the generic-Whisper fallback transcription path) fall back
to proportional interpolation across the segment's own duration — see
_split_into_cards.

Darija/RTL: Arabic script is right-to-left and this is flagged in the
architecture doc as a common silent-breakage point (reversed/misaligned
text). Line breaks here are inserted at word boundaries in transcript
(logical reading) order — libass's own fribidi-based bidi reordering
handles the visual (right-to-left) layout, so this module never reorders
characters itself. Verified by burning real Darija text and inspecting the
rendered frame, not just the raw .ass text.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_FONT = "Al Nile"  # Apple's Arabic system font; has a real Bold face and
# is a public font family — ".SF Arabic" looked right in `fc-list`/`fc-match`
# but is a private CoreText-only name: ffmpeg silently substituted Times New
# Roman for it on a real render, caught only by inspecting the burned frame.
DEFAULT_FONT_SIZE_RATIO = 1 / 14  # fontsize = video height * this ratio
DEFAULT_OUTLINE = 12  # BorderStyle 3: this is the box's padding, not a text outline
DEFAULT_MARGIN_V_RATIO = 1 / 10  # bottom margin = video height * this ratio
DEFAULT_BOX_COLOR = "&H60000000"  # semi-transparent black card background

# Brand overlay (TRK-60/61): "3la Rassi" logo pill + accent border, burned in
# alongside captions in the same ffmpeg call. Assets + their generator live
# in assets/brand/ (generate.py — one-off design script, not part of the
# pipeline, only needed again if the branding changes).
BRAND_LOGO_PATH = Path(__file__).parent.parent / "assets" / "brand" / "overlay_ribbon.png"
BRAND_BORDER_COLOR = "0xFF6B35"  # brand orange
BRAND_BORDER_THICKNESS_RATIO = 1 / 180  # of video height
BRAND_LOGO_WIDTH_RATIO = 0.42  # of video width
BRAND_LOGO_MARGIN_RATIO = 0.04  # of video width, from the top-left corner

# Inline \c override tags take &Hbbggrr& (no alpha byte) — distinct from the
# 8-hex &Haabbggrr Style-line format used above and in build_ass's header.
HIGHLIGHT_COLOR_INLINE = "&H00FFFF&"  # yellow
WHITE_COLOR_INLINE = "&HFFFFFF&"  # matches the style's PrimaryColour

WORDS_PER_LINE = 3
MAX_LINES = 2
MAX_WORDS_PER_CARD = WORDS_PER_LINE * MAX_LINES

# Common Darija/MSA function words — filtered out when picking each line's
# highlight word so emphasis lands on content words, not particles/pronouns.
STOPWORDS = {
    "و",
    "أو",
    "او",
    "ف",
    "فـ",
    "ثم",
    "لكن",
    "ولكن",
    "في",
    "علي",
    "على",
    "من",
    "إلى",
    "الى",
    "عن",
    "مع",
    "ديال",
    "ديالي",
    "ديالك",
    "ديالو",
    "ديالها",
    "ديالنا",
    "ديالكم",
    "ديالهم",
    "بحال",
    "هذا",
    "هاذ",
    "هاد",
    "هادا",
    "هادي",
    "هادو",
    "ذلك",
    "هذه",
    "دا",
    "ديك",
    "ديك",
    "أنا",
    "انا",
    "نتا",
    "نتي",
    "نتوما",
    "هو",
    "هي",
    "حنا",
    "احنا",
    "هوما",
    "أنت",
    "انت",
    "أنتم",
    "انتم",
    "لي",
    "لك",
    "لو",
    "لها",
    "لنا",
    "لكم",
    "لهم",
    "ما",
    "لا",
    "ماشي",
    "لاباس",
    "واخا",
    "أما",
    "اما",
    "كان",
    "كانت",
    "غادي",
    "راه",
    "راها",
    "راهم",
    "هل",
    "أش",
    "اش",
    "شنو",
    "علاش",
    "فين",
    "كيفاش",
    "شحال",
    "امتى",
    "إمتى",
}


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


def _split_into_cards(segment: Dict) -> List[Dict]:
    """Split one transcript segment's words into ≤MAX_WORDS_PER_CARD-word
    caption cards, so each card fits MAX_LINES lines of WORDS_PER_LINE words.

    If the segment carries real per-word timestamps (segment["words"], from
    shorts_generator.local.transcriber's darija-primary path), each card's
    [start, end] is the actual span of its first/last word, so cards change
    in sync with speech. Otherwise (segment has no "words" — currently only
    the generic-Whisper fallback transcription path, which has no per-word
    timestamps) a card's [start, end] is interpolated by dividing the
    segment's own duration proportionally to word count — an approximation,
    not real per-word timing. ponytail: fallback-only ceiling; revisit if
    the fallback path starts firing often enough to matter.
    """
    words_meta = segment.get("words")
    if words_meta:
        words = [str(w["text"]).strip() for w in words_meta]
        starts = [float(w["start"]) for w in words_meta]
        ends = [float(w["end"]) for w in words_meta]
    else:
        words = str(segment.get("text", "")).split()
        starts = ends = None

    if not words:
        return []

    seg_start, seg_end = float(segment["start"]), float(segment["end"])
    duration = seg_end - seg_start
    total = len(words)

    cards = []
    for i in range(0, total, MAX_WORDS_PER_CARD):
        chunk = words[i : i + MAX_WORDS_PER_CARD]
        if starts is not None:
            card_start = starts[i]
            card_end = ends[i + len(chunk) - 1]
        else:
            card_start = seg_start + duration * (i / total)
            card_end = seg_start + duration * (min(i + len(chunk), total) / total)
        cards.append({"start": card_start, "end": card_end, "words": chunk})
    return cards


def _highlight_word(words: List[str]) -> Optional[str]:
    """Pick the longest non-stopword in `words` to highlight, or None if
    every word is a stopword (e.g. a line of pure filler/particles).
    """
    candidates = [w for w in words if w.strip(".,!?؟،") not in STOPWORDS]
    if not candidates:
        return None
    return max(candidates, key=len)


def _render_line(words: List[str]) -> str:
    """Render one line of a caption card, wrapping its one highlight word
    (if any) in ASS inline color-override tags.
    """
    highlight = _highlight_word(words)
    rendered = []
    for w in words:
        escaped = _escape_ass_text(w)
        if w == highlight:
            rendered.append(
                f"{{\\c{HIGHLIGHT_COLOR_INLINE}}}{escaped}{{\\c{WHITE_COLOR_INLINE}}}"
            )
            highlight = None  # only the first matching occurrence gets highlighted
        else:
            rendered.append(escaped)
    return " ".join(rendered)


def _card_text(words: List[str]) -> str:
    """Render a caption card's words as ≤MAX_LINES ASS lines (joined with
    the literal ASS line-break token), each with its own highlight word.
    """
    line_chunks = [
        words[i : i + WORDS_PER_LINE] for i in range(0, len(words), WORDS_PER_LINE)
    ]
    return "\\N".join(_render_line(line) for line in line_chunks)


def _rebase_words_for_window(
    words: List[Dict], window_start: float, window_end: float
) -> List[Dict]:
    """Same clip/rebase as _segments_for_window, applied to one segment's
    per-word spans.
    """
    result = []
    for w in words:
        start = max(float(w["start"]), window_start)
        end = min(float(w["end"]), window_end)
        if end <= start:
            continue
        result.append(
            {
                "start": start - window_start,
                "end": end - window_start,
                "text": w["text"],
            }
        )
    return result


def _segments_for_window(
    segments: List[Dict], window_start: float, window_end: float
) -> List[Dict]:
    """Return segments overlapping [window_start, window_end], rebased so
    window_start becomes 0 — matches the rendered clip's own timeline. Each
    segment's "words" (see _split_into_cards), if present, are clipped and
    rebased the same way so per-word timing survives into the clip's window.
    """
    result = []
    for s in segments:
        start = max(float(s["start"]), window_start)
        end = min(float(s["end"]), window_end)
        if end <= start:
            continue
        rebased = {
            "start": start - window_start,
            "end": end - window_start,
            "text": s["text"],
        }
        words = s.get("words")
        if words:
            rebased_words = _rebase_words_for_window(words, window_start, window_end)
            if rebased_words:
                rebased["words"] = rebased_words
        result.append(rebased)
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
Style: Default,{font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,{DEFAULT_BOX_COLOR},-1,0,0,0,100,100,0,0,3,{DEFAULT_OUTLINE},0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for s in segments:
        if not str(s.get("text", "")).strip():
            continue
        for card in _split_into_cards(s):
            start = _format_ass_timestamp(card["start"])
            end = _format_ass_timestamp(card["end"])
            text = _card_text(card["words"])
            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    return header + "\n".join(lines) + "\n"


def burn_captions(
    clip_path: str,
    segments: List[Dict],
    window_start: float,
    window_end: float,
    out_path: Optional[str] = None,
    font: str = DEFAULT_FONT,
    logo_path: Optional[Path] = BRAND_LOGO_PATH,
) -> str:
    """Burn captions for [window_start, window_end] of the source transcript
    into `clip_path` (already cropped to its final window, so its own
    timeline starts at 0). Also burns in the brand accent border and, if
    `logo_path` exists, the logo pill overlay (top-left) — same ffmpeg call
    as the caption burn, so the clip is only re-encoded once. Pass
    `logo_path=None` to skip the logo (border still burns in). Returns the
    captioned clip's path.
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

    border_t = max(2, round(height * BRAND_BORDER_THICKNESS_RATIO))
    have_logo = logo_path is not None and Path(logo_path).exists()

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", clip_path]
    if have_logo:
        logo_w = max(1, round(width * BRAND_LOGO_WIDTH_RATIO))
        margin = max(1, round(width * BRAND_LOGO_MARGIN_RATIO))
        cmd += ["-i", str(logo_path)]
        filter_complex = (
            f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:t={border_t}:color={BRAND_BORDER_COLOR}@1.0[framed];"
            f"[1:v]scale={logo_w}:-1[logo];"
            f"[framed][logo]overlay=x={margin}:y={margin}[branded];"
            f"[branded]ass={ass_path}[vout]"
        )
        cmd += ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "0:a"]
    else:
        cmd += [
            "-vf",
            f"drawbox=x=0:y=0:w=iw:h=ih:t={border_t}:color={BRAND_BORDER_COLOR}@1.0,ass={ass_path}",
        ]

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        out_path,
    ]

    try:
        subprocess.run(cmd, check=True)
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
