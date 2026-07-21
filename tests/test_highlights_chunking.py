import json
import re

import pytest

from darija_overrides import highlights_chunking as hc


def _build_long_transcript(duration: float = 2500.0, segment_every: float = 15.0):
    """A transcript long enough to trigger vendor's chunking (>= 1800s)."""
    segments = []
    t = 0.0
    while t < duration:
        segments.append({"start": t, "end": t + 10.0, "text": f"word at {t:.0f}"})
        t += segment_every
    return {"segments": segments, "duration": duration}


def _fake_llm_fn(prompt: str) -> str:
    """Mimics a real model: answers with whatever time labels it was shown.

    For the content-type detection call, returns a fixed classification. For
    a highlight call, picks the timestamp label in the *middle* of what it
    was shown as a highlight's start — same coordinate system (absolute or
    chunk-relative) the caller's prompt used.
    """
    if (
        "content_type" in prompt
        and "density" in prompt
        and "Transcript sample:" in prompt
    ):
        return '{"content_type": "podcast", "density": "medium"}'

    timestamps = [float(m) for m in re.findall(r"\[(\d+\.\d+)s\]", prompt)]
    mid = timestamps[len(timestamps) // 2]
    highlight = {
        "title": "Highlight",
        "start_time": mid,
        "end_time": mid + 45.0,
        "score": 80,
        "hook_sentence": "hook",
        "virality_reason": "reason",
    }
    return json.dumps({"highlights": [highlight]})


def test_unpatched_vendor_chunking_loses_highlights_on_later_chunks():
    """Reproduces the real bug: absolute timestamps in the prompt vs. a
    relative `max_end` clamp drop every highlight past the first chunk,
    which exhausts all retries and raises.
    """
    import shorts_generator.highlights as vendor_highlights

    transcript = _build_long_transcript()
    with pytest.raises(RuntimeError):
        vendor_highlights.get_highlights(transcript, num_clips=3, llm_fn=_fake_llm_fn)


def test_install_fixes_chunking_so_later_chunk_highlights_survive():
    import shorts_generator.highlights as vendor_highlights

    original = vendor_highlights.chunk_transcript
    try:
        hc.install()
        transcript = _build_long_transcript()
        result = vendor_highlights.get_highlights(
            transcript, num_clips=3, llm_fn=_fake_llm_fn
        )
    finally:
        vendor_highlights.chunk_transcript = original

    assert result["highlights"]
    for h in result["highlights"]:
        assert 0 <= h["start_time"] < h["end_time"] <= 2500.0 + 60


def test_install_patches_vendor_chunk_transcript():
    import shorts_generator.highlights as vendor_highlights

    original = vendor_highlights.chunk_transcript
    try:
        hc.install()
        assert vendor_highlights.chunk_transcript is hc.chunk_transcript_rebased
    finally:
        vendor_highlights.chunk_transcript = original


def test_chunk_transcript_rebased_starts_each_chunk_near_zero():
    import shorts_generator.highlights as vendor_highlights

    original = vendor_highlights.chunk_transcript
    try:
        hc.install()
        chunks = hc.chunk_transcript_rebased(_build_long_transcript())
    finally:
        vendor_highlights.chunk_transcript = original

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["segments"][0]["start"] < 5.0
