from unittest.mock import patch

from darija_overrides import scene_snap_crop as ssc


def test_snap_to_nearest_cut_within_tolerance():
    cuts = [10.0, 42.3, 100.0]
    assert ssc._snap_to_nearest_cut(41.0, cuts) == 42.3


def test_snap_to_nearest_cut_outside_tolerance_is_unchanged():
    cuts = [10.0, 100.0]
    assert ssc._snap_to_nearest_cut(50.0, cuts) == 50.0


def test_snap_to_nearest_cut_with_no_cuts_is_unchanged():
    assert ssc._snap_to_nearest_cut(50.0, []) == 50.0


def test_crop_highlights_local_snapped_moves_boundaries_onto_cuts():
    highlights = [{"title": "h1", "start_time": 41.0, "end_time": 89.5, "score": 80}]
    captured = {}

    def fake_original(source_path, highlights, aspect_ratio="9:16", out_dir=None):
        captured["highlights"] = highlights
        return [{**highlights[0], "clip_url": "/tmp/out.mp4"}]

    with patch.object(ssc, "_detect_cut_seconds", return_value=[42.3, 90.0]):
        ssc._original_crop_highlights_local = fake_original
        result = ssc.crop_highlights_local_snapped("source.mp4", highlights)

    assert captured["highlights"][0]["start_time"] == 42.3
    assert captured["highlights"][0]["end_time"] == 90.0
    assert result[0]["clip_url"] == "/tmp/out.mp4"


def test_crop_highlights_local_snapped_keeps_original_window_if_snap_collapses_it():
    # both boundaries would snap onto the same nearby cut
    highlights = [{"title": "h1", "start_time": 41.0, "end_time": 43.0, "score": 80}]
    captured = {}

    def fake_original(source_path, highlights, aspect_ratio="9:16", out_dir=None):
        captured["highlights"] = highlights
        return []

    with patch.object(ssc, "_detect_cut_seconds", return_value=[42.0]):
        ssc._original_crop_highlights_local = fake_original
        ssc.crop_highlights_local_snapped("source.mp4", highlights)

    assert captured["highlights"][0]["start_time"] == 41.0
    assert captured["highlights"][0]["end_time"] == 43.0


def test_crop_highlights_local_snapped_falls_back_when_scene_detection_fails():
    highlights = [{"title": "h1", "start_time": 41.0, "end_time": 89.5, "score": 80}]
    captured = {}

    def fake_original(source_path, highlights, aspect_ratio="9:16", out_dir=None):
        captured["highlights"] = highlights
        return []

    with patch.object(ssc, "_detect_cut_seconds", side_effect=RuntimeError("boom")):
        ssc._original_crop_highlights_local = fake_original
        ssc.crop_highlights_local_snapped("source.mp4", highlights)

    assert captured["highlights"][0]["start_time"] == 41.0
    assert captured["highlights"][0]["end_time"] == 89.5


def test_install_patches_vendor_crop_highlights_local():
    import shorts_generator.local.clipper as vendor_clipper

    original = vendor_clipper.crop_highlights_local
    try:
        ssc.install()
        assert vendor_clipper.crop_highlights_local is ssc.crop_highlights_local_snapped
    finally:
        vendor_clipper.crop_highlights_local = original
