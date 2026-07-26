from unittest.mock import patch

import pytest

import captioner


def test_format_ass_timestamp_under_a_minute():
    assert captioner._format_ass_timestamp(1.5) == "0:00:01.50"


def test_format_ass_timestamp_over_an_hour():
    assert captioner._format_ass_timestamp(3725.25) == "1:02:05.25"


def test_format_ass_timestamp_never_negative():
    assert captioner._format_ass_timestamp(-5.0) == "0:00:00.00"


def test_escape_ass_text_braces_and_newlines():
    assert captioner._escape_ass_text("hi {there}\nfriend") == "hi \\{there\\}\\Nfriend"


def test_segments_for_window_filters_and_rebases():
    segments = [
        {"start": 0.0, "end": 5.0, "text": "before window"},
        {"start": 8.0, "end": 12.0, "text": "inside window"},
        {"start": 20.0, "end": 25.0, "text": "after window"},
    ]
    result = captioner._segments_for_window(segments, window_start=6.0, window_end=15.0)
    assert result == [{"start": 2.0, "end": 6.0, "text": "inside window"}]


def test_segments_for_window_clips_partial_overlap():
    segments = [{"start": 4.0, "end": 10.0, "text": "straddles start"}]
    result = captioner._segments_for_window(segments, window_start=6.0, window_end=15.0)
    assert result == [{"start": 0.0, "end": 4.0, "text": "straddles start"}]


def test_segments_for_window_rebases_words_when_present():
    segments = [
        {
            "start": 8.0,
            "end": 12.0,
            "text": "inside window",
            "words": [
                {"start": 8.0, "end": 9.0, "text": "inside"},
                {"start": 9.0, "end": 12.0, "text": "window"},
            ],
        }
    ]
    result = captioner._segments_for_window(segments, window_start=6.0, window_end=15.0)
    assert result == [
        {
            "start": 2.0,
            "end": 6.0,
            "text": "inside window",
            "words": [
                {"start": 2.0, "end": 3.0, "text": "inside"},
                {"start": 3.0, "end": 6.0, "text": "window"},
            ],
        }
    ]


def test_segments_for_window_drops_words_outside_the_window():
    segments = [
        {
            "start": 4.0,
            "end": 10.0,
            "text": "straddles start",
            "words": [
                {
                    "start": 4.0,
                    "end": 5.5,
                    "text": "straddles",
                },  # entirely before window
                {"start": 6.0, "end": 10.0, "text": "start"},
            ],
        }
    ]
    result = captioner._segments_for_window(segments, window_start=6.0, window_end=15.0)
    assert result == [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "straddles start",
            "words": [{"start": 0.0, "end": 4.0, "text": "start"}],
        }
    ]


def test_build_ass_contains_header_and_dialogue():
    segments = [{"start": 0.0, "end": 2.0, "text": "سلام"}]
    ass = captioner.build_ass(segments, width=1080, height=1920)
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert (
        "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,"
        "{\\c&H00FFFF&}سلام{\\c&HFFFFFF&}" in ass
    )


def test_build_ass_skips_empty_text_segments():
    segments = [{"start": 0.0, "end": 1.0, "text": "  "}]
    ass = captioner.build_ass(segments, width=1080, height=1920)
    assert "Dialogue:" not in ass


# --- caption cards: word-count-capped splitting ---


def test_split_into_cards_single_short_segment_is_one_card():
    segment = {"start": 0.0, "end": 3.0, "text": "هادا نهار زوين بزاف"}
    cards = captioner._split_into_cards(segment)
    assert len(cards) == 1
    assert cards[0]["words"] == ["هادا", "نهار", "زوين", "بزاف"]
    assert cards[0]["start"] == 0.0
    assert cards[0]["end"] == 3.0


def test_split_into_cards_splits_long_segment_by_word_count():
    words = [f"كلمة{i}" for i in range(9)]  # 9 words, cap is 6 -> 2 cards
    segment = {"start": 0.0, "end": 9.0, "text": " ".join(words)}
    cards = captioner._split_into_cards(segment)
    assert len(cards) == 2
    assert cards[0]["words"] == words[:6]
    assert cards[1]["words"] == words[6:]
    # proportional-to-word-count timing: 6/9 of the duration, then the rest
    assert cards[0]["start"] == pytest.approx(0.0)
    assert cards[0]["end"] == pytest.approx(6.0)
    assert cards[1]["start"] == pytest.approx(6.0)
    assert cards[1]["end"] == pytest.approx(9.0)


def test_split_into_cards_empty_text_yields_no_cards():
    assert captioner._split_into_cards({"start": 0.0, "end": 1.0, "text": "  "}) == []


def test_split_into_cards_uses_real_word_timestamps_when_present():
    # Deliberately uneven word durations — proportional interpolation would
    # NOT reproduce these boundaries, so this proves real timing is used.
    words = [f"كلمة{i}" for i in range(7)]  # 7 words -> cards of 6 and 1
    segment = {
        "start": 0.0,
        "end": 100.0,  # if interpolated, card0 would span ~0 -> ~85.7
        "text": " ".join(words),
        "words": [
            {"start": float(i), "end": float(i) + 0.3, "text": w}
            for i, w in enumerate(words)
        ],
    }
    cards = captioner._split_into_cards(segment)
    assert len(cards) == 2
    assert cards[0]["words"] == words[:6]
    assert cards[0]["start"] == 0.0  # first word's own start
    assert cards[0]["end"] == 5.3  # last word (index 5) end: 5.0 + 0.3
    assert cards[1]["words"] == words[6:]
    assert cards[1]["start"] == 6.0
    assert cards[1]["end"] == 6.3


def test_split_into_cards_falls_back_to_interpolation_without_word_timestamps():
    # Same shape as above but no "words" key (e.g. generic-Whisper fallback
    # transcription) — must use the old proportional-interpolation timing,
    # not crash or silently invent per-word timing.
    words = [f"كلمة{i}" for i in range(7)]
    segment = {"start": 0.0, "end": 7.0, "text": " ".join(words)}
    cards = captioner._split_into_cards(segment)
    assert cards[0]["start"] == pytest.approx(0.0)
    assert cards[0]["end"] == pytest.approx(6.0)
    assert cards[1]["start"] == pytest.approx(6.0)
    assert cards[1]["end"] == pytest.approx(7.0)


# --- highlight word selection ---


def test_highlight_word_picks_longest_non_stopword():
    assert captioner._highlight_word(["و", "المدرسة", "ديال"]) == "المدرسة"


def test_highlight_word_none_when_all_stopwords():
    assert captioner._highlight_word(["و", "ديال", "هاد"]) is None


def test_render_line_wraps_only_the_highlight_word():
    line = captioner._render_line(["و", "المدرسة", "كبيرة"])
    assert line == "و {\\c&H00FFFF&}المدرسة{\\c&HFFFFFF&} كبيرة"


def test_card_text_caps_at_two_lines():
    words = [f"كلمة{i}" for i in range(6)]
    text = captioner._card_text(words)
    assert text.count("\\N") == 1  # exactly 2 lines for a full 6-word card


def test_burn_captions_invokes_ffmpeg_with_ass_filter(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")

    with (
        patch.object(captioner, "_probe_dimensions", return_value=(1080, 1920)),
        patch("captioner.subprocess.run") as mock_run,
    ):
        out = captioner.burn_captions(
            str(clip), [{"start": 0.0, "end": 1.0, "text": "hi"}], 0.0, 1.0
        )

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    vf_index = cmd.index("-vf")
    assert cmd[vf_index + 1].startswith("ass=")
    assert out == str(clip.with_name("clip.captioned.mp4"))
