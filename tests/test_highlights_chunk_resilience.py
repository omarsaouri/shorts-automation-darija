import json
import re

import pytest

from darija_overrides import highlights_chunk_resilience as hcr


def _build_long_transcript(duration: float = 2500.0, segment_every: float = 15.0):
    """A transcript long enough to trigger vendor's chunking (>= 1800s)."""
    segments = []
    t = 0.0
    while t < duration:
        segments.append({"start": t, "end": t + 10.0, "text": f"word at {t:.0f}"})
        t += segment_every
    return {"segments": segments, "duration": duration}


def _make_llm_fn(bad_offset_threshold: float):
    """Valid JSON for chunks starting below the threshold, garbage above it.

    Mimics the real finding: Atlas-Chat-9B can consistently fail on one
    specific chunk's content while succeeding on others.
    """

    def _fn(prompt: str) -> str:
        if (
            "content_type" in prompt
            and "density" in prompt
            and "Transcript sample:" in prompt
        ):
            return '{"content_type": "podcast", "density": "medium"}'

        timestamps = [float(m) for m in re.findall(r"\[(\d+\.\d+)s\]", prompt)]
        if min(timestamps) >= bad_offset_threshold:
            return "سمح ليا، ما فهمتش شنو باغي تقول."
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

    return _fn


def test_one_bad_chunk_is_skipped_but_others_still_contribute():
    import shorts_generator.pipeline as vendor_pipeline

    original = vendor_pipeline.get_highlights
    try:
        hcr.install()
        transcript = _build_long_transcript()
        llm_fn = _make_llm_fn(bad_offset_threshold=500.0)
        result = vendor_pipeline.get_highlights(transcript, num_clips=3, llm_fn=llm_fn)
    finally:
        vendor_pipeline.get_highlights = original

    # Only chunk 1 (absolute offset 0, timestamps up to ~1140s) is below the
    # threshold — chunks 2/3 (offsets 1140s/2280s) both exhaust retries.
    assert result["highlights"]
    for h in result["highlights"]:
        assert h["start_time"] < 1200.0


def test_all_chunks_failing_raises():
    import shorts_generator.pipeline as vendor_pipeline

    original = vendor_pipeline.get_highlights
    try:
        hcr.install()
        transcript = _build_long_transcript()
        llm_fn = _make_llm_fn(bad_offset_threshold=0.0)
        with pytest.raises(RuntimeError):
            vendor_pipeline.get_highlights(transcript, num_clips=3, llm_fn=llm_fn)
    finally:
        vendor_pipeline.get_highlights = original


def test_install_patches_pipeline_get_highlights_not_highlights_module():
    import shorts_generator.highlights as vendor_highlights
    import shorts_generator.pipeline as vendor_pipeline

    original_pipeline = vendor_pipeline.get_highlights
    original_highlights = vendor_highlights.get_highlights
    try:
        hcr.install()
        assert vendor_pipeline.get_highlights is hcr.get_highlights_resilient
        assert vendor_highlights.get_highlights is original_highlights
    finally:
        vendor_pipeline.get_highlights = original_pipeline


def test_short_video_delegates_to_original_get_highlights():
    import shorts_generator.pipeline as vendor_pipeline

    original = vendor_pipeline.get_highlights
    try:
        hcr.install()
        short_transcript = {
            "segments": [{"start": 0.0, "end": 10.0, "text": "hello world"}],
            "duration": 60.0,
        }
        llm_fn = _make_llm_fn(bad_offset_threshold=999999.0)
        result = vendor_pipeline.get_highlights(
            short_transcript, num_clips=1, llm_fn=llm_fn
        )
    finally:
        vendor_pipeline.get_highlights = original

    assert result["highlights"]
