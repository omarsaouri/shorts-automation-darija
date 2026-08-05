"""Fix vendored `shorts_generator.highlights`' long-video chunking bug.

Bug seen in real-world testing (42-min video, `progress.md`): `chunk_transcript`
builds each chunk's `segments` list from the transcript's *absolute* (whole-
video) timestamps — it never rebases them to the chunk's own start. So
`build_transcript_text` shows the model absolute time labels (e.g. chunk 2 of
a 42-min video shows `[1146.2s]` through `[2377.9s]`), and the model naturally
answers with highlights in that same absolute range.

But `call_highlight_api` receives `chunk["duration"]` (the chunk's *relative*
span, e.g. `1200`) as the `max_end` clamp in `_sanitize_highlights` — every
highlight past `1200` gets clamped to `1200` on both ends and dropped as
zero-length. `get_highlights` then adds `+offset` on top of whatever
(nearly nothing) survives. Net effect: every chunk after the first loses
almost all its highlights, and `call_highlight_api` raises after 3 failed
attempts, aborting the whole multi-chunk loop.

This patches only `chunk_transcript` — it delegates to the vendored function
for all the actual chunk-boundary math (chunk size, overlap, offset), then
rebases each returned chunk's segment timestamps to start at 0. That makes
the prompt's time labels match the relative `duration` the rest of the
pipeline already assumes, without touching `call_highlight_api` or
`get_highlights` at all.
"""

from typing import Callable, Dict, List, Optional

import shorts_generator.highlights as _vendor

_original_chunk_transcript: Optional[Callable[[Dict], List[Dict]]] = None


def chunk_transcript_rebased(transcript: Dict) -> List[Dict]:
    """Same chunks as the vendored `chunk_transcript`, with each chunk's
    segment timestamps rebased relative to that chunk's own start (`_offset`)
    instead of the whole video — see module docstring for why that matters.
    """
    chunks = _original_chunk_transcript(transcript)
    for chunk in chunks:
        offset = chunk.get("_offset", 0)
        chunk["segments"] = [
            {**s, "start": s["start"] - offset, "end": s["end"] - offset}
            for s in chunk["segments"]
        ]
    return chunks


def install() -> None:
    """Monkeypatch shorts_generator.highlights.chunk_transcript.

    `get_highlights` looks up `chunk_transcript` in its own module's globals
    at call time (same reason `clipper_stable.py` can patch a single vendor
    function rather than shadowing the whole module), so patching that one
    attribute is enough — `call_highlight_api`, `dedupe_highlights`, and
    `get_highlights` itself stay unmodified vendor logic.
    """
    global _original_chunk_transcript
    if _original_chunk_transcript is None:
        _original_chunk_transcript = _vendor.chunk_transcript
    _vendor.chunk_transcript = chunk_transcript_rebased
