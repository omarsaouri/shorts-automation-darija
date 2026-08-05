"""Stabilize shorts_generator.local.clipper's face-tracking crop.

Two bugs fixed here, both in the vendored `_reframe_vertical`:

1. It re-targets the crop window to *whichever* detection is largest, every
   single frame, smoothing 15% of the way toward it immediately. A single-
   frame false positive (background object, a face in the crowd, a graphic)
   gets treated exactly like a real, sustained subject — so the crop
   visibly snaps away from the speaker and back the instant the false
   positive appears and disappears. Fixed with the debounced `_FaceTracker`
   below.
2. It uses OpenCV's Haar cascade for detection, which has a high false-
   positive/false-negative rate on real footage. Swapped for MediaPipe Face
   Detection, run every FACE_DETECT_INTERVAL frames rather than every frame
   (MediaPipe is heavier per-call than Haar; sampling is enough since the
   tracker only needs a fresh fix a few times a second).

This patches only `_reframe_vertical` — `crop_clip_local`,
`crop_highlights_local`, and `_cut_subclip` are unmodified vendor logic
(ffmpeg subclip cut + per-highlight orchestration), reused as-is via
monkeypatching the one function that has the bug rather than shadowing
the whole module.
"""

import os
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import shorts_generator.local.clipper as _vendor

# A detected face within this fraction of the frame's larger dimension from
# the currently tracked center is trusted immediately (real subject motion,
# camera pans). Anything further needs REQUIRED_CONSECUTIVE_FRAMES of
# consistent detections before the tracker follows it — filters out
# single-frame false positives without needing a full object tracker.
MAX_TRUSTED_JUMP_FRACTION = 0.25
REQUIRED_CONSECUTIVE_FRAMES = 3
PENDING_MATCH_TOLERANCE_PX = 20
SMOOTHING = 0.15  # how aggressively to chase a trusted new position

# Run MediaPipe detection every Nth frame rather than every frame.
FACE_DETECT_INTERVAL = 10
MEDIAPIPE_MIN_CONFIDENCE = 0.5

# Short-range model: faces within ~2m, fits talking-head/interview framing.
# The mediapipe pip package (>=1.0) only ships the Tasks API loader, not the
# model weights — this is downloaded once and cached on first use.
_FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
_FACE_MODEL_PATH = (
    Path(__file__).parent.parent.parent
    / ".cache"
    / "mediapipe"
    / "blaze_face_short_range.tflite"
)


def _ensure_face_model() -> str:
    """Download the MediaPipe face detector model on first use, then reuse
    the cached copy on every call after. Returns the local file path.
    """
    if not _FACE_MODEL_PATH.exists():
        _FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
    return str(_FACE_MODEL_PATH)


class _FaceTracker:
    """Debounced face-center tracker — the actual fix, kept separate from
    cv2/video I/O so it's unit-testable without a real video file.
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.max_trusted_jump = MAX_TRUSTED_JUMP_FRACTION * max(frame_w, frame_h)
        self.default_center = (frame_w // 2, frame_h // 2)
        self.last_center: Optional[Tuple[int, int]] = None
        self.pending_candidate: Optional[Tuple[int, int]] = None
        self.pending_count = 0

    def update(self, detected: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if self.last_center is None:
            self.last_center = detected or self.default_center
            return self.last_center

        if detected is None:
            # No detection this frame — hold position steady rather than
            # snapping back to center.
            return self.last_center

        cx, cy = detected
        lx, ly = self.last_center
        jump = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5

        if jump <= self.max_trusted_jump:
            self.last_center = (
                int(lx + (cx - lx) * SMOOTHING),
                int(ly + (cy - ly) * SMOOTHING),
            )
            self.pending_candidate, self.pending_count = None, 0
            return self.last_center

        if (
            self.pending_candidate is not None
            and abs(self.pending_candidate[0] - cx) < PENDING_MATCH_TOLERANCE_PX
            and abs(self.pending_candidate[1] - cy) < PENDING_MATCH_TOLERANCE_PX
        ):
            self.pending_count += 1
        else:
            self.pending_candidate, self.pending_count = (cx, cy), 1

        if self.pending_count >= REQUIRED_CONSECUTIVE_FRAMES:
            self.last_center = (
                int(lx + (cx - lx) * SMOOTHING),
                int(ly + (cy - ly) * SMOOTHING),
            )
            self.pending_candidate, self.pending_count = None, 0
        # else: likely a false positive — hold position steady.

        return self.last_center


def _select_closest_face(
    faces: List[Tuple[int, int, int]], last_center: Optional[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """Pick which face to track when a frame has more than one detection.

    Each face is (cx, cy, area). Picks whichever is spatially closest to
    the last tracked center, so the crop follows the established subject
    instead of jumping to whichever face is momentarily biggest. On the
    first detection (no established center yet) falls back to the largest
    face, same heuristic as picking the presumed speaker.
    """
    if not faces:
        return None
    if last_center is None:
        cx, cy, _area = max(faces, key=lambda f: f[2])
        return (cx, cy)
    lx, ly = last_center
    cx, cy, _area = min(faces, key=lambda f: (f[0] - lx) ** 2 + (f[1] - ly) ** 2)
    return (cx, cy)


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision

    target_ratio = _vendor._ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    tracker = _FaceTracker(src_w, src_h)
    cx, cy = tracker.default_center

    base_options = mp.tasks.BaseOptions(model_asset_path=_ensure_face_model())
    detector_options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=MEDIAPIPE_MIN_CONFIDENCE,
    )
    with mp_vision.FaceDetector.create_from_options(detector_options) as detector:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FACE_DETECT_INTERVAL == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                faces: List[Tuple[int, int, int]] = []
                for det in result.detections:
                    box = det.bounding_box
                    faces.append(
                        (
                            box.origin_x + box.width // 2,
                            box.origin_y + box.height // 2,
                            box.width * box.height,
                        )
                    )

                detected = _select_closest_face(faces, tracker.last_center)
                cx, cy = tracker.update(detected)

            x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
            y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
            cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
            writer.write(cropped)
            frame_idx += 1

    cap.release()
    writer.release()

    # Mux audio from the cut clip back onto the silent reframed video —
    # same ffmpeg invocation as the vendored function.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            silent_path,
            "-i",
            in_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-shortest",
            out_path,
        ],
        check=True,
    )
    os.remove(silent_path)
    return out_path


def install() -> None:
    """Monkeypatch shorts_generator.local.clipper._reframe_vertical.

    crop_clip_local looks up `_reframe_vertical` in its own module's
    globals at call time, so patching that one attribute (rather than
    shadowing the whole module like the LLM/transcriber overrides) is
    enough to redirect it here — crop_clip_local/crop_highlights_local/
    _cut_subclip stay the unmodified vendored versions.
    """
    _vendor._reframe_vertical = _reframe_vertical
