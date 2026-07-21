"""Snap highlight boundaries onto real scene cuts before cropping
(architecture doc §3.5 / §4: scene detection is "intersected with the LLM's
candidate time ranges to snap clip start/end points to the nearest real
cut" so a clip doesn't start or end mid-shot).

CLAUDE.md requires calling `pipeline.generate_shorts(...)` rather than
reimplementing its flow, and vendor's local-mode crop step runs inside that
call with no hook to inject snapped boundaries. `pipeline._run_local`
re-resolves `crop_highlights_local` via a fresh `from .local.clipper import
crop_highlights_local` on every call, though (a plain module-attribute
lookup, not a bound closure) — so monkeypatching that one attribute is
enough to intersect scene detection into the flow without touching
`generate_shorts` itself, the same single-function-patch pattern as
`clipper_stable.py`/`highlights_chunking.py`/`highlights_duration_filter.py`.
"""

from typing import Callable, Dict, List, Optional

import shorts_generator.local.clipper as _vendor

# A detected cut within this many seconds of the LLM's chosen boundary is
# trusted as "the same edit, just imprecise" and snapped to. Anything
# farther is left alone — the highlight's own timing, not a distant cut,
# is still the better signal for where the clip should start/end.
MAX_SNAP_DISTANCE_SECONDS = 2.0

_original_crop_highlights_local: Optional[Callable[..., List[Dict]]] = None


def _detect_cut_seconds(source_path: str) -> List[float]:
    """All scene-boundary timestamps in `source_path`, via PySceneDetect's
    default content-based detector. Every scene's start and end are cut
    points (adjacent scenes share a boundary, so this naturally dedupes).
    """
    from scenedetect import ContentDetector, detect

    scenes = detect(source_path, ContentDetector())
    cuts = set()
    for start, end in scenes:
        cuts.add(start.get_seconds())
        cuts.add(end.get_seconds())
    return sorted(cuts)


def _snap_to_nearest_cut(time: float, cuts: List[float]) -> float:
    if not cuts:
        return time
    nearest = min(cuts, key=lambda c: abs(c - time))
    if abs(nearest - time) <= MAX_SNAP_DISTANCE_SECONDS:
        return nearest
    return time


def crop_highlights_local_snapped(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
) -> List[Dict]:
    """Same as vendor's `crop_highlights_local`, after snapping each
    highlight's start/end onto the nearest real scene cut (see module
    docstring). Falls back to unsnapped boundaries if scene detection
    itself fails — a crop on the LLM's original timing beats no clip at
    all.
    """
    try:
        cuts = _detect_cut_seconds(source_path)
    except Exception as e:
        print(
            f"[clip/scene-snap] scene detection failed, using unsnapped boundaries: {e}",
            flush=True,
        )
        cuts = []

    snapped: List[Dict] = []
    for h in highlights:
        start = float(h["start_time"])
        end = float(h["end_time"])
        new_start = _snap_to_nearest_cut(start, cuts)
        new_end = _snap_to_nearest_cut(end, cuts)
        if new_end <= new_start:
            # Both boundaries snapped onto (or past) the same cut — keep
            # the LLM's original window rather than ship a collapsed clip.
            new_start, new_end = start, end
        snapped.append({**h, "start_time": new_start, "end_time": new_end})

    return _original_crop_highlights_local(
        source_path, snapped, aspect_ratio=aspect_ratio, out_dir=out_dir
    )


def install() -> None:
    """Monkeypatch shorts_generator.local.clipper.crop_highlights_local.

    `pipeline._run_local` looks this up via a fresh `from .local.clipper
    import crop_highlights_local` at call time, so patching this one
    attribute is enough — `crop_clip_local`, `_cut_subclip`, and
    `_reframe_vertical` (already patched by `clipper_stable.py`) stay
    unmodified/independently-patched vendor logic.
    """
    global _original_crop_highlights_local
    if _original_crop_highlights_local is None:
        _original_crop_highlights_local = _vendor.crop_highlights_local
    _vendor.crop_highlights_local = crop_highlights_local_snapped
