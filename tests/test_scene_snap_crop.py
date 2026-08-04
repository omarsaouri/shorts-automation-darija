from unittest.mock import patch

from shorts_generator.local import clipper


def test_snap_to_nearest_cut_within_tolerance():
    cuts = [10.0, 42.3, 100.0]
    assert clipper._snap_to_nearest_cut(41.0, cuts) == 42.3


def test_snap_to_nearest_cut_outside_tolerance_is_unchanged():
    cuts = [10.0, 100.0]
    assert clipper._snap_to_nearest_cut(50.0, cuts) == 50.0


def test_snap_to_nearest_cut_with_no_cuts_is_unchanged():
    assert clipper._snap_to_nearest_cut(50.0, []) == 50.0


def test_video_out_dir_derives_from_source_filename():
    assert clipper._video_out_dir("output/source_abc123.mp4") == "output/abc123"


def test_video_out_dir_falls_back_when_filename_doesnt_match_pattern():
    assert clipper._video_out_dir("output/some_other_file.mp4") == "output"


def _fake_crop_clip_local(captured):
    def _fn(source_path, start_time, end_time, aspect_ratio, out_path):
        captured["start"] = start_time
        captured["end"] = end_time
        captured["out_path"] = out_path
        return out_path

    return _fn


def test_crop_highlights_local_moves_boundaries_onto_cuts(tmp_path):
    source_path = str(tmp_path / "source_abc123.mp4")
    highlights = [{"title": "h1", "start_time": 41.0, "end_time": 89.5, "score": 80}]
    captured = {}

    with (
        patch.object(clipper, "_detect_cut_seconds", return_value=[42.3, 90.0]),
        patch.object(
            clipper, "crop_clip_local", side_effect=_fake_crop_clip_local(captured)
        ),
    ):
        result = clipper.crop_highlights_local(source_path, highlights)

    assert captured["start"] == 42.3
    assert captured["end"] == 90.0
    assert result[0]["start_time"] == 42.3
    assert result[0]["end_time"] == 90.0
    assert result[0]["clip_url"] == captured["out_path"]


def test_crop_highlights_local_keeps_original_window_if_snap_collapses_it(tmp_path):
    # both boundaries would snap onto the same nearby cut
    source_path = str(tmp_path / "source_abc123.mp4")
    highlights = [{"title": "h1", "start_time": 41.0, "end_time": 43.0, "score": 80}]
    captured = {}

    with (
        patch.object(clipper, "_detect_cut_seconds", return_value=[42.0]),
        patch.object(
            clipper, "crop_clip_local", side_effect=_fake_crop_clip_local(captured)
        ),
    ):
        clipper.crop_highlights_local(source_path, highlights)

    assert captured["start"] == 41.0
    assert captured["end"] == 43.0


def test_crop_highlights_local_falls_back_when_scene_detection_fails(tmp_path):
    source_path = str(tmp_path / "source_abc123.mp4")
    highlights = [{"title": "h1", "start_time": 41.0, "end_time": 89.5, "score": 80}]
    captured = {}

    with (
        patch.object(clipper, "_detect_cut_seconds", side_effect=RuntimeError("boom")),
        patch.object(
            clipper, "crop_clip_local", side_effect=_fake_crop_clip_local(captured)
        ),
    ):
        clipper.crop_highlights_local(source_path, highlights)

    assert captured["start"] == 41.0
    assert captured["end"] == 89.5


def test_crop_highlights_local_defaults_out_dir_per_video(tmp_path):
    source_path = str(tmp_path / "source_abc123.mp4")
    highlights = [{"title": "h1", "start_time": 41.0, "end_time": 89.5, "score": 80}]
    captured = {}

    with (
        patch.object(clipper, "_detect_cut_seconds", return_value=[]),
        patch.object(
            clipper, "crop_clip_local", side_effect=_fake_crop_clip_local(captured)
        ),
    ):
        clipper.crop_highlights_local(source_path, highlights)

    assert captured["out_path"] == str(tmp_path / "abc123" / "short_01.mp4")


def test_crop_highlights_local_respects_explicit_out_dir(tmp_path):
    source_path = str(tmp_path / "source_abc123.mp4")
    highlights = [{"title": "h1", "start_time": 41.0, "end_time": 89.5, "score": 80}]
    custom_dir = str(tmp_path / "custom")
    captured = {}

    with (
        patch.object(clipper, "_detect_cut_seconds", return_value=[]),
        patch.object(
            clipper, "crop_clip_local", side_effect=_fake_crop_clip_local(captured)
        ),
    ):
        clipper.crop_highlights_local(source_path, highlights, out_dir=custom_dir)

    assert captured["out_path"] == str(tmp_path / "custom" / "short_01.mp4")
