from unittest.mock import patch

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


def test_build_ass_contains_header_and_dialogue():
    segments = [{"start": 0.0, "end": 2.0, "text": "سلام"}]
    ass = captioner.build_ass(segments, width=1080, height=1920)
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,سلام" in ass


def test_build_ass_skips_empty_text_segments():
    segments = [{"start": 0.0, "end": 1.0, "text": "  "}]
    ass = captioner.build_ass(segments, width=1080, height=1920)
    assert "Dialogue:" not in ass


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
