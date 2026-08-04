import json
import re

from shorts_generator import highlights as vendor_highlights


def _build_long_transcript(duration: float = 2500.0, segment_every: float = 15.0):
    """A transcript long enough to trigger chunking (>= 1800s)."""
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
    was shown as a highlight's start — same coordinate system (chunk-relative)
    the caller's prompt used.
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


def test_later_chunk_highlights_survive_because_timestamps_are_rebased():
    """Each chunk's segments are rebased to start at 0, so the prompt's time
    labels match the relative `duration` call_highlight_api clamps against —
    without that, every chunk after the first would lose its highlights to
    the clamp and exhaust its retries.
    """
    transcript = _build_long_transcript()
    result = vendor_highlights.get_highlights(
        transcript, num_clips=3, llm_fn=_fake_llm_fn
    )

    assert result["highlights"]
    for h in result["highlights"]:
        assert 0 <= h["start_time"] < h["end_time"] <= 2500.0 + 60


def test_chunk_transcript_starts_each_chunk_near_zero():
    chunks = vendor_highlights.chunk_transcript(_build_long_transcript())

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["segments"][0]["start"] < 5.0
