import json
import os
import wave
from unittest.mock import patch

from shorts_generator.local import transcriber as td


def _segments(*texts):
    return {
        "duration": float(len(texts)),
        "segments": [
            {"start": float(i), "end": float(i + 1), "text": t}
            for i, t in enumerate(texts)
        ],
    }


def test_looks_garbled_empty_segments():
    assert td._looks_garbled({"duration": 0.0, "segments": []})


def test_looks_garbled_all_blank_text():
    assert td._looks_garbled(_segments("", "  ", ""))


def test_looks_garbled_repetition_loop():
    assert td._looks_garbled(_segments("a", "a", "a", "a", "a"))


def test_not_garbled_normal_output():
    assert not td._looks_garbled(_segments("hello", "how are you", "goodbye"))


def test_looks_garbled_too_few_segments_for_duration():
    # 3 segments over a 10-minute (600s) transcript — segmentation collapsed
    # into a few giant blobs, the actual bug a real 40-min video surfaced.
    transcript = {
        "duration": 600.0,
        "segments": [
            {"start": 0.0, "end": 200.0, "text": "a huge wall of text"},
            {"start": 200.0, "end": 400.0, "text": "another huge wall"},
            {"start": 400.0, "end": 600.0, "text": "yet another"},
        ],
    }
    assert td._looks_garbled(transcript)


def test_not_garbled_normal_density_over_long_duration():
    # ~1 segment every 4s over 10 minutes — plenty of segments per minute.
    segments = [
        {"start": float(i * 4), "end": float(i * 4 + 3), "text": f"word{i}"}
        for i in range(150)
    ]
    transcript = {"duration": 600.0, "segments": segments}
    assert not td._looks_garbled(transcript)


def test_looks_garbled_repeated_character_run():
    assert td._looks_garbled(_segments("hello", "wooooooooooow there", "bye"))


def test_group_words_into_segments_breaks_on_pause():
    words = [
        {"timestamp": (0.0, 0.5), "text": "hello"},
        {"timestamp": (0.5, 1.0), "text": " world"},
        {"timestamp": (3.0, 3.5), "text": " goodbye"},
    ]
    result = td._group_words_into_segments(words)
    assert result == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello world",
            "words": [
                {"start": 0.0, "end": 0.5, "text": "hello"},
                {"start": 0.5, "end": 1.0, "text": "world"},
            ],
        },
        {
            "start": 3.0,
            "end": 3.5,
            "text": "goodbye",
            "words": [{"start": 3.0, "end": 3.5, "text": "goodbye"}],
        },
    ]


def test_group_words_into_segments_breaks_on_max_duration():
    words = [
        {"timestamp": (float(i), float(i) + 0.4), "text": f" w{i}"} for i in range(20)
    ]
    result = td._group_words_into_segments(words)
    assert len(result) > 1
    assert all(seg["end"] - seg["start"] <= td.MAX_SEGMENT_SECONDS for seg in result)


def test_group_words_into_segments_skips_empty_and_none_start():
    words = [
        {"timestamp": (None, 1.0), "text": "skip me"},
        {"timestamp": (0.0, 0.5), "text": "  "},
        {"timestamp": (1.0, 1.5), "text": "kept"},
    ]
    assert td._group_words_into_segments(words) == [
        {
            "start": 1.0,
            "end": 1.5,
            "text": "kept",
            "words": [{"start": 1.0, "end": 1.5, "text": "kept"}],
        }
    ]


def test_extract_audio_wav_invokes_ffmpeg():
    with patch("shorts_generator.local.transcriber.subprocess.run") as mock_run:
        wav_path = td._extract_audio_wav("media.mp4")

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "media.mp4" in cmd
    assert wav_path.endswith(".wav")
    os.remove(wav_path)


def test_run_darija_transcription_windows_wav_groups_words_and_cleans_up():
    fake_words = [
        {"timestamp": (0.0, 0.5), "text": "hi"},
        {"timestamp": (0.5, 1.0), "text": " there"},
    ]
    with (
        patch.object(td, "_load_pipeline", return_value="fake-pipe"),
        patch.object(
            td, "_extract_audio_wav", return_value="/tmp/fake.wav"
        ) as mock_extract,
        patch.object(
            td, "_transcribe_wav_in_windows", return_value=fake_words
        ) as mock_windows,
        patch("os.remove") as mock_remove,
    ):
        result = td._run_darija_transcription("media.mp4", None)

    mock_extract.assert_called_once_with("media.mp4")
    mock_windows.assert_called_once_with("fake-pipe", "/tmp/fake.wav", {})
    mock_remove.assert_called_once_with("/tmp/fake.wav")
    assert result == {
        "duration": 1.0,
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hi there",
                "words": [
                    {"start": 0.0, "end": 0.5, "text": "hi"},
                    {"start": 0.5, "end": 1.0, "text": "there"},
                ],
            }
        ],
    }


def _make_silent_wav(path: str, seconds: float, framerate: int = 8000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * int(seconds * framerate))


def test_iter_wav_windows_covers_whole_file_with_padding(tmp_path):
    wav_path = str(tmp_path / "src.wav")
    _make_silent_wav(wav_path, seconds=130.0)

    windows = list(td._iter_wav_windows(wav_path, window_seconds=60.0, pad_seconds=3.0))
    try:
        assert len(windows) == 3
        (p0, c0s, c0e, last0, _), (p1, c1s, c1e, last1, _), (p2, c2s, c2e, last2, _) = (
            windows
        )

        assert (c0s, c0e, last0) == (0.0, 60.0, False)
        assert (c1s, c1e, last1) == (60.0, 120.0, False)
        assert (c2s, c2e, last2) == (120.0, 130.0, True)

        assert p0 == 0.0  # no padding available before t=0
        assert p1 == 57.0
        assert p2 == 117.0
    finally:
        for *_rest, window_path in windows:
            os.remove(window_path)


def test_keep_word_in_window_core_span_only():
    assert td._keep_word_in_window(10.0, core_start=0.0, core_end=60.0, is_last=False)
    # boundary word belongs to the next window's core span, not this one
    assert not td._keep_word_in_window(
        60.0, core_start=0.0, core_end=60.0, is_last=False
    )
    assert not td._keep_word_in_window(
        -1.0, core_start=0.0, core_end=60.0, is_last=False
    )


def test_keep_word_in_window_last_window_includes_its_end():
    assert td._keep_word_in_window(60.0, core_start=0.0, core_end=60.0, is_last=True)


def test_extract_window_words_rebases_and_drops_padding():
    chunks = [
        {"timestamp": (0.0, 1.0), "text": "pad-before"},  # in padding, dropped
        {"timestamp": (3.5, 4.0), "text": "kept"},
        {"timestamp": (None, 1.0), "text": "no-start"},  # dropped
    ]
    # core span is [3, 63); 3s of left padding means padded_start=0
    words = td._extract_window_words(
        chunks, padded_start=0.0, core_start=3.0, core_end=63.0, is_last=False
    )
    assert words == [{"timestamp": (3.5, 4.0), "text": "kept"}]


def test_transcribe_wav_in_windows_merges_across_windows_and_flushes_cache():
    fake_windows = [
        (0.0, 0.0, 60.0, False, "/tmp/w0.wav"),
        (57.0, 60.0, 120.0, True, "/tmp/w1.wav"),
    ]

    def fake_pipe(path, return_timestamps, generate_kwargs):
        assert return_timestamps == "word"
        if path == "/tmp/w0.wav":
            return {"chunks": [{"timestamp": (1.0, 1.5), "text": "one"}]}
        return {"chunks": [{"timestamp": (3.0, 3.5), "text": "two"}]}

    with (
        patch.object(td, "_iter_wav_windows", return_value=iter(fake_windows)),
        patch.object(td, "_flush_accelerator_cache") as mock_flush,
        patch("os.remove") as mock_remove,
    ):
        words = td._transcribe_wav_in_windows(fake_pipe, "/tmp/full.wav", {})

    assert words == [
        {"timestamp": (1.0, 1.5), "text": "one"},
        {"timestamp": (60.0, 60.5), "text": "two"},
    ]
    assert mock_flush.call_count == 2
    assert mock_remove.call_count == 2


def test_flush_accelerator_cache_runs_without_error():
    td._flush_accelerator_cache()


def test_transcribe_local_reuses_cache_without_running_model(tmp_path):
    media_path = tmp_path / "media.mp4"
    media_path.write_bytes(b"fake")
    cache_path = tmp_path / "media.json"
    cache_path.write_text(json.dumps(_segments("cached line")), encoding="utf-8")

    with patch.object(
        td, "_transcript_cache_path", return_value=tmp_path / "media.srt"
    ):
        with patch.object(td, "_run_darija_transcription") as mock_run:
            result = td.transcribe_local(str(media_path))

    mock_run.assert_not_called()
    assert result == _segments("cached line")


def test_transcribe_local_falls_back_when_garbled(tmp_path):
    media_path = tmp_path / "media.mp4"
    media_path.write_bytes(b"fake")

    with patch.object(
        td, "_transcript_cache_path", return_value=tmp_path / "media.srt"
    ):
        with (
            patch.object(
                td,
                "_run_darija_transcription",
                return_value=_segments("a", "a", "a", "a"),
            ),
            patch.object(
                td, "_fallback_transcribe", return_value=_segments("fallback result")
            ) as mock_fallback,
        ):
            result = td.transcribe_local(str(media_path), language="ar")

    mock_fallback.assert_called_once_with(str(media_path), "ar")
    assert result == _segments("fallback result")
    cache_path = tmp_path / "media.json"
    assert json.loads(cache_path.read_text(encoding="utf-8")) == result


def test_transcribe_local_keeps_darija_output_when_not_garbled(tmp_path):
    media_path = tmp_path / "media.mp4"
    media_path.write_bytes(b"fake")
    good = _segments("hello", "how are you")

    with patch.object(
        td, "_transcript_cache_path", return_value=tmp_path / "media.srt"
    ):
        with (
            patch.object(td, "_run_darija_transcription", return_value=good),
            patch.object(td, "_fallback_transcribe") as mock_fallback,
        ):
            result = td.transcribe_local(str(media_path))

    mock_fallback.assert_not_called()
    assert result == good


def test_fallback_transcribe_forces_large_model_and_restores_default():
    td.LOCAL_WHISPER_MODEL = "base"

    def check_model_during_call(media_path, language=None):
        assert td.LOCAL_WHISPER_MODEL == td.FALLBACK_WHISPER_MODEL
        return _segments("ok")

    with patch.object(
        td, "_transcribe_local_whisper", side_effect=check_model_during_call
    ):
        td._fallback_transcribe("media.mp4", None)

    assert td.LOCAL_WHISPER_MODEL == "base"
