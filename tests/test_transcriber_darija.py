import sys
from unittest.mock import patch

from darija_overrides import transcriber_darija as td


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
        {"start": 0.0, "end": 1.0, "text": "hello world"},
        {"start": 3.0, "end": 3.5, "text": "goodbye"},
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
        {"start": 1.0, "end": 1.5, "text": "kept"}
    ]


def test_extract_audio_wav_invokes_ffmpeg():
    with patch("darija_overrides.transcriber_darija.subprocess.run") as mock_run:
        wav_path = td._extract_audio_wav("media.mp4")

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "media.mp4" in cmd
    assert wav_path.endswith(".wav")
    import os as _os

    _os.remove(wav_path)


def test_run_darija_transcription_extracts_wav_groups_words_and_cleans_up():
    def fake_pipe(wav_path, return_timestamps, generate_kwargs):
        return {
            "chunks": [
                {"timestamp": (0.0, 0.5), "text": "hi"},
                {"timestamp": (0.5, 1.0), "text": " there"},
            ]
        }

    with (
        patch.object(td, "_load_pipeline", return_value=fake_pipe),
        patch.object(
            td, "_extract_audio_wav", return_value="/tmp/fake.wav"
        ) as mock_extract,
        patch("os.remove") as mock_remove,
    ):
        result = td._run_darija_transcription("media.mp4", None)

    mock_extract.assert_called_once_with("media.mp4")
    mock_remove.assert_called_once_with("/tmp/fake.wav")
    assert result == {
        "duration": 1.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "hi there"}],
    }


def test_transcribe_local_reuses_cache_without_running_model():
    with (
        patch.object(td, "_vendor") as mock_vendor,
        patch("os.path.getmtime", return_value=1.0),
    ):
        mock_vendor._transcript_cache_path.return_value.exists.return_value = True
        mock_vendor._transcript_cache_path.return_value.stat.return_value.st_mtime = 2.0
        mock_vendor._load_srt_cache.return_value = _segments("cached line")

        with patch.object(td, "_run_darija_transcription") as mock_run:
            result = td.transcribe_local("media.mp4")

        mock_run.assert_not_called()
        assert result == _segments("cached line")


def test_transcribe_local_falls_back_when_garbled():
    with patch.object(td, "_vendor") as mock_vendor:
        mock_vendor._transcript_cache_path.return_value.exists.return_value = False

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
            result = td.transcribe_local("media.mp4", language="ar")

        mock_fallback.assert_called_once_with("media.mp4", "ar")
        assert result == _segments("fallback result")
        mock_vendor._write_srt_cache.assert_called_once_with("media.mp4", result)


def test_transcribe_local_keeps_darija_output_when_not_garbled():
    with patch.object(td, "_vendor") as mock_vendor:
        mock_vendor._transcript_cache_path.return_value.exists.return_value = False
        good = _segments("hello", "how are you")

        with (
            patch.object(td, "_run_darija_transcription", return_value=good),
            patch.object(td, "_fallback_transcribe") as mock_fallback,
        ):
            result = td.transcribe_local("media.mp4")

        mock_fallback.assert_not_called()
        assert result == good


def test_fallback_transcribe_forces_large_model_and_restores_default():
    with patch.object(td, "_vendor") as mock_vendor:
        mock_vendor.LOCAL_WHISPER_MODEL = "base"

        def check_model_during_call(media_path, language=None):
            assert mock_vendor.LOCAL_WHISPER_MODEL == td.FALLBACK_WHISPER_MODEL
            return _segments("ok")

        mock_vendor.transcribe_local.side_effect = check_model_during_call

        td._fallback_transcribe("media.mp4", None)

        assert mock_vendor.LOCAL_WHISPER_MODEL == "base"


def test_install_shadows_vendor_transcriber_module():
    sys.modules.pop("shorts_generator.local.transcriber", None)
    try:
        td.install()
        assert sys.modules["shorts_generator.local.transcriber"] is td
        assert (
            sys.modules["shorts_generator.local.transcriber"].transcribe_local
            is td.transcribe_local
        )
    finally:
        sys.modules.pop("shorts_generator.local.transcriber", None)
