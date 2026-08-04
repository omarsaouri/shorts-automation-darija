import json
import re

import pytest

from shorts_generator import highlights as vendor_highlights


def _build_long_transcript(duration: float = 2500.0, segment_every: float = 15.0):
    """A transcript long enough to trigger chunking (>= 1800s)."""
    segments = []
    t = 0.0
    while t < duration:
        segments.append({"start": t, "end": t + 10.0, "text": f"word at {t:.0f}"})
        t += segment_every
    return {"segments": segments, "duration": duration}


def _make_llm_fn(bad_offset_threshold: float):
    """Valid JSON for chunks starting below the threshold, garbage above it.

    Mimics the real finding: Atlas-Chat-9B can consistently fail on one
    specific chunk's content while succeeding on others. Detects "badness"
    from the segment text's own embedded absolute time ("word at N") rather
    than the prompt's `[N.Ns]` labels, since chunk_transcript rebases those
    to start at 0 for every chunk — the absolute offset only survives in
    the transcript text itself.
    """

    def _fn(prompt: str) -> str:
        if (
            "content_type" in prompt
            and "density" in prompt
            and "Transcript sample:" in prompt
        ):
            return '{"content_type": "podcast", "density": "medium"}'

        absolute_offsets = [float(m) for m in re.findall(r"word at (\d+)", prompt)]
        if absolute_offsets and min(absolute_offsets) >= bad_offset_threshold:
            return "سمح ليا، ما فهمتش شنو باغي تقول."
        timestamps = [float(m) for m in re.findall(r"\[(\d+\.\d+)s\]", prompt)]
        mid = timestamps[len(timestamps) // 2] if timestamps else 0.0
        highlight = {
            "title": "Highlight",
            "start_time": mid,
            "end_time": mid + 45.0,
            "score": 80,
            "hook_sentence": "hook",
            "virality_reason": "reason",
        }
        return json.dumps({"highlights": [highlight]})

    return _fn


def test_one_bad_chunk_is_skipped_but_others_still_contribute():
    transcript = _build_long_transcript()
    llm_fn = _make_llm_fn(bad_offset_threshold=500.0)
    result = vendor_highlights.get_highlights(transcript, num_clips=3, llm_fn=llm_fn)

    # Only chunk 1 (absolute offset 0, timestamps up to ~1140s) is below the
    # threshold — chunks 2/3 (offsets 1140s/2280s) both exhaust retries.
    assert result["highlights"]
    for h in result["highlights"]:
        assert h["start_time"] < 1200.0


def test_all_chunks_failing_raises():
    transcript = _build_long_transcript()
    llm_fn = _make_llm_fn(bad_offset_threshold=0.0)
    with pytest.raises(RuntimeError):
        vendor_highlights.get_highlights(transcript, num_clips=3, llm_fn=llm_fn)


def test_short_video_is_unaffected_by_chunk_resilience():
    short_transcript = {
        "segments": [{"start": 0.0, "end": 10.0, "text": "hello world"}],
        "duration": 60.0,
    }
    llm_fn = _make_llm_fn(bad_offset_threshold=999999.0)
    result = vendor_highlights.get_highlights(
        short_transcript, num_clips=1, llm_fn=llm_fn
    )

    assert result["highlights"]


def test_max_highlight_api_attempts_is_bumped_above_vendor_default():
    assert vendor_highlights.MAX_HIGHLIGHT_API_ATTEMPTS == 5
