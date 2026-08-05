"""Stabilize shorts_generator.local.clipper's face-tracking crop.

Bug seen in real-world testing: the vendored `_reframe_vertical` re-targets
the crop window to *whichever* Haar cascade detection is largest, every
single frame, smoothing 15% of the way toward it immediately. A single-
frame false positive (background object, a face in the crowd, a graphic)
gets treated exactly like a real, sustained subject — so the crop visibly
snaps away from the speaker and back the instant the false positive
appears and disappears.

This patches only `_reframe_vertical` — `crop_clip_local`,
`crop_highlights_local`, and `_cut_subclip` are unmodified vendor logic
(ffmpeg subclip cut + per-highlight orchestration), reused as-is via
monkeypatching the one function that has the bug rather than shadowing
the whole module.
"""

import os
import subprocess
from typing import Optional, Tuple

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


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    import cv2  # type: ignore

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

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    tracker = _FaceTracker(src_w, src_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        detected: Optional[Tuple[int, int]] = None
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            detected = (x + w // 2, y + h // 2)

        cx, cy = tracker.update(detected)
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        writer.write(cropped)

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
