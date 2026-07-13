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
