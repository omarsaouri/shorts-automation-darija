import json

import pytest

from shorts_generator import highlights as vendor_highlights


def _short_transcript(duration: float = 419.4):
    """A short (non-chunked) transcript, matching the real 7-min
    `SkxfKZgy9kw` test video that exposed this bug."""
    segments = [{"start": 0.0, "end": duration, "text": "some transcript text"}]
    return {"segments": segments, "duration": duration}


def _fake_llm_fn_with_one_giant_highlight(prompt: str) -> str:
    """Reproduces a real Atlas-Chat-9B response captured against the
    SkxfKZgy9kw video: one highlight spans nearly the whole video and
    out-scores three otherwise well-formed short highlights.
    """
    if (
        "content_type" in prompt
        and "density" in prompt
        and "Transcript sample:" in prompt
    ):
        return '{"content_type": "lecture", "density": "high"}'

    highlights = [
        {
            "title": "giant",
            "start_time": 0.0,
            "end_time": 419.4,
            "score": 95,
            "hook_sentence": "hook",
            "virality_reason": "reason",
        },
        {
            "title": "short 1",
            "start_time": 47.5,
            "end_time": 100.0,
            "score": 90,
            "hook_sentence": "hook",
            "virality_reason": "reason",
        },
        {
            "title": "short 2",
            "start_time": 150.0,
            "end_time": 220.0,
            "score": 85,
            "hook_sentence": "hook",
            "virality_reason": "reason",
        },
        {
            "title": "short 3",
            "start_time": 300.0,
            "end_time": 370.0,
            "score": 80,
            "hook_sentence": "hook",
            "virality_reason": "reason",
        },
    ]
    return json.dumps({"highlights": highlights})


def test_giant_highlight_is_filtered_so_short_ones_survive():
    """Without the duration bound, the giant highlight would overlap and
    out-score every well-formed short one, and dedupe_highlights would
    collapse the result down to just the giant clip.
    """
    transcript = _short_transcript()
    result = vendor_highlights.get_highlights(
        transcript, num_clips=3, llm_fn=_fake_llm_fn_with_one_giant_highlight
    )

    assert len(result["highlights"]) == 3
    for h in result["highlights"]:
        duration = h["end_time"] - h["start_time"]
        assert (
            vendor_highlights.MIN_CLIP_SECONDS
            <= duration
            <= vendor_highlights.MAX_CLIP_SECONDS
        )


@pytest.mark.parametrize(
    "start,end,should_survive",
    [
        (0.0, 10.0, False),  # too short
        (0.0, 45.0, True),  # in bounds
        (0.0, 180.0, True),  # exactly the ceiling
        (0.0, 200.0, False),  # too long
        (0.0, 419.4, False),  # the giant/whole-video case
    ],
)
def test_sanitize_highlights_drops_out_of_range_durations(start, end, should_survive):
    raw = [
        {
            "title": "t",
            "start_time": start,
            "end_time": end,
            "score": 50,
            "hook_sentence": "h",
            "virality_reason": "r",
        }
    ]
    cleaned = vendor_highlights._sanitize_highlights(raw, duration=500.0)
    assert bool(cleaned) == should_survive
