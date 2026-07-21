"""Fix vendored `shorts_generator.highlights`' giant-highlight collapse bug.

Bug seen in real-world testing (7-min `SkxfKZgy9kw` video, `progress.md`):
`get_highlights` sometimes returns a single highlight spanning nearly the
whole video instead of several short clips. Root cause confirmed by
capturing Atlas-Chat-9B's raw responses directly: the model does not
reliably respect `HIGHLIGHT_SYSTEM_PROMPT`'s own stated duration bounds
("20-180 seconds") — it sometimes answers with one or more highlights
spanning hundreds of seconds. Nothing in `_sanitize_highlights` enforces
those bounds; it only checks `start >= 0`, `end > start`, and clamps to the
video's total duration, so an oversized highlight sails through untouched.
`dedupe_highlights` then keeps whichever overlapping highlight has the
highest score — since a video-spanning highlight overlaps essentially
everything, if it scores highest (or ties), every well-formed short
highlight gets dropped as "overlapping", collapsing the result down to one
giant clip.

This isn't a deterministic code bug like the chunking one
(`highlights_chunking.py`) — a local 9B model won't reliably follow a
prose duration instruction — so the fix is to stop trusting the model on
duration and enforce the prompt's own stated bounds in code instead.
"""

from typing import Callable, Dict, List, Optional

import shorts_generator.highlights as _vendor

# Mirrors HIGHLIGHT_SYSTEM_PROMPT's own stated floor/ceiling ("Go shorter
# (20-44s) only for a perfect standalone one-liner... Go longer (91-180s)
# only when a story arc needs full context to land") — if that prompt text
# changes, update these to match.
MIN_CLIP_SECONDS = 20.0
MAX_CLIP_SECONDS = 180.0

_original_sanitize_highlights: Optional[Callable[[object, float], List[Dict]]] = None


def _sanitize_highlights_bounded(raw_highlights: object, duration: float) -> List[Dict]:
    """Vendor's own sanitization, plus a hard duration-bounds filter.

    Dropping (rather than clamping) an out-of-bounds highlight here — before
    `dedupe_highlights` ever sees it — means it can no longer out-score and
    evict every well-formed highlight it overlaps with. If every candidate
    gets dropped, `call_highlight_api` already treats an empty list the same
    as "no valid highlights in response" and retries, so no extra plumbing
    is needed for that case.
    """
    cleaned = _original_sanitize_highlights(raw_highlights, duration)
    return [
        h
        for h in cleaned
        if MIN_CLIP_SECONDS <= (h["end_time"] - h["start_time"]) <= MAX_CLIP_SECONDS
    ]


def install() -> None:
    """Monkeypatch shorts_generator.highlights._sanitize_highlights.

    `call_highlight_api` looks up `_sanitize_highlights` in its own module's
    globals at call time (same reason `clipper_stable.py`/
    `highlights_chunking.py` can each patch a single vendor function), so
    patching that one attribute is enough — `dedupe_highlights` and
    `get_highlights` itself stay unmodified vendor logic.
    """
    global _original_sanitize_highlights
    if _original_sanitize_highlights is None:
        _original_sanitize_highlights = _vendor._sanitize_highlights
    _vendor._sanitize_highlights = _sanitize_highlights_bounded
