"""Swap shorts_generator.local.transcriber's faster-whisper model for
anaszil/whisper-large-v3-turbo-darija — a LoRA fine-tune of
openai/whisper-large-v3-turbo for Moroccan Darija.

This adapter has no CTranslate2/GGUF conversion (checked its HF repo:
only adapter_config.json + adapter_model.safetensors), so it can't be
loaded through faster-whisper like the vendored transcriber. Instead this
uses the standard transformers + PEFT loading path the model card itself
documents, running on MPS/CUDA/CPU (faster-whisper's engine doesn't
support Metal, but plain transformers does via torch's MPS backend).

Requests word-level timestamps rather than the ASR pipeline's
`chunk_length_s`-based windowing: a real 40-min video test showed
`chunk_length_s` (transformers itself flags this as "very experimental
with seq2seq models") collapsing the whole transcript into 1-2 giant
segments instead of per-sentence ones. Word-level timestamps are reliable;
`_group_words_into_segments` re-groups them into phrase-sized segments at
natural pauses, which is what both highlight scoring and caption burn-in
actually need. That non-chunked code path also needs a real audio file
(WAV) rather than an mp4 container — `_extract_audio_wav` handles that.

Per CLAUDE.md's hard constraint, generic Whisper is a fallback only, never
silently primary: if the Darija model's output looks garbled (empty, a
repetition loop, a repeated-character hallucination run, or too few
segments for the audio's length — all failure modes seen in testing, more
common on heavy Darija/French code-switching per the architecture doc),
this falls back to the vendored faster-whisper transcriber forced to
large-v3, and logs the fallback with the media path so it's visible in
pipeline logs.
"""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional

import shorts_generator.local.transcriber as _vendor

BASE_MODEL_ID = "openai/whisper-large-v3-turbo"
DARIJA_ADAPTER_ID = "anaszil/whisper-large-v3-turbo-darija"
FALLBACK_WHISPER_MODEL = os.environ.get("FALLBACK_WHISPER_MODEL", "large-v3")

# Word-grouping heuristic: break a new segment at a pause longer than
# MAX_WORD_GAP_SECONDS, or once a segment would exceed MAX_SEGMENT_SECONDS
# — mirrors faster-whisper's own natural-pause segmentation so highlight
# scoring and caption burn-in get similarly-sized segments either way.
MAX_WORD_GAP_SECONDS = 0.6
MAX_SEGMENT_SECONDS = 12.0

_VENDOR_TRANSCRIBER_MODULE = "shorts_generator.local.transcriber"

_pipe = None  # lazy singleton — loading the model is expensive, do it once per process


def _resolve_device() -> str:
    import torch  # type: ignore

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    try:
        from peft import PeftModel  # type: ignore
        from transformers import (  # type: ignore
            WhisperForConditionalGeneration,
            WhisperProcessor,
            pipeline,
        )
    except ImportError as e:
        raise RuntimeError(
            "transformers/peft are required for the Darija transcriber. Install with:\n"
            "    pip install torch transformers peft accelerate"
        ) from e

    device = _resolve_device()
    print(f"[transcribe/darija] loading {DARIJA_ADAPTER_ID} on {device}", flush=True)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base_model, DARIJA_ADAPTER_ID)
    model = model.merge_and_unload()
    model.to(device)

    # No chunk_length_s: relies on the model's own long-form generate()
    # chunking instead of the pipeline's experimental external windowing.
    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )
    return _pipe


def _extract_audio_wav(media_path: str) -> str:
    """ffmpeg-extract mono 16kHz WAV. The ASR pipeline's non-chunked code
    path (needed for word-level timestamps) fails to container-sniff .mp4
    inputs directly — a plain WAV sidesteps that.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            media_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            wav_path,
        ],
        check=True,
    )
    return wav_path


def _group_words_into_segments(word_chunks: List[Dict]) -> List[Dict]:
    """Re-group word-level ASR chunks into phrase-sized segments, breaking
    at pauses > MAX_WORD_GAP_SECONDS or once a segment would run past
    MAX_SEGMENT_SECONDS.
    """
    segments: List[Dict] = []
    current_words: List[str] = []
    current_start: Optional[float] = None
    prev_end: Optional[float] = None

    for chunk in word_chunks:
        start, end = chunk["timestamp"]
        text = (chunk["text"] or "").strip()
        if not text or start is None:
            continue
        if end is None:
            end = start

        if current_start is None:
            current_start, current_words, prev_end = start, [text], end
            continue

        gap = start - prev_end
        would_be_duration = end - current_start
        if gap > MAX_WORD_GAP_SECONDS or would_be_duration > MAX_SEGMENT_SECONDS:
            segments.append(
                {
                    "start": current_start,
                    "end": prev_end,
                    "text": " ".join(current_words),
                }
            )
            current_start, current_words = start, [text]
        else:
            current_words.append(text)
        prev_end = end

    if current_words:
        segments.append(
            {"start": current_start, "end": prev_end, "text": " ".join(current_words)}
        )
    return segments


def _run_darija_transcription(media_path: str, language: Optional[str]) -> Dict:
    pipe = _load_pipeline()
    wav_path = _extract_audio_wav(media_path)
    try:
        generate_kwargs = {"language": language} if language else {}
        result = pipe(
            wav_path, return_timestamps="word", generate_kwargs=generate_kwargs
        )
    finally:
        os.remove(wav_path)

    segments = _group_words_into_segments(result.get("chunks", []))
    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


_REPEATED_CHAR_RUN = re.compile(r"(.)\1{9,}")


def _looks_garbled(transcript: Dict) -> bool:
    """Flags known Whisper failure modes seen in testing: empty output, a
    back-to-back segment repetition loop, a 10+ repeated-character run
    within one segment (both classic hallucination artifacts), or too few
    segments for the audio's length (segmentation collapsed into a few
    giant blobs instead of per-sentence chunks — the actual bug a real
    40-min video test surfaced).
    # ponytail: heuristics, not a full quality classifier — good enough to
    # catch what's been seen; revisit if real Darija runs show a different
    # garbling pattern.
    """
    segments = transcript.get("segments", [])
    non_empty = [s for s in segments if s["text"].strip()]
    if not non_empty:
        return True

    repeats = sum(1 for a, b in zip(non_empty, non_empty[1:]) if a["text"] == b["text"])
    if len(non_empty) >= 4 and repeats / len(non_empty) > 0.3:
        return True

    duration = transcript.get("duration", 0.0)
    if duration > 60 and len(non_empty) / (duration / 60.0) < 2:
        return True

    return any(_REPEATED_CHAR_RUN.search(s["text"]) for s in non_empty)


def _fallback_transcribe(media_path: str, language: Optional[str]) -> Dict:
    original_model = _vendor.LOCAL_WHISPER_MODEL
    _vendor.LOCAL_WHISPER_MODEL = FALLBACK_WHISPER_MODEL
    try:
        return _vendor.transcribe_local(media_path, language=language)
    finally:
        _vendor.LOCAL_WHISPER_MODEL = original_model


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Darija-first transcription; same (media_path, language) -> {duration,
    segments[start,end,text]} contract as the vendored transcriber.

    Reads/writes no state.db tables. Writes the same .srt cache format as
    shorts_generator.local.transcriber, reusing its cache helpers directly.
    """
    cache_path = _vendor._transcript_cache_path(media_path)
    if cache_path.exists() and cache_path.stat().st_mtime >= os.path.getmtime(
        media_path
    ):
        cached = _vendor._load_srt_cache(cache_path)
        if cached["segments"] and cached["duration"] > 0.0:
            print(
                f"[transcribe/darija] reusing cached transcript: {cache_path}",
                flush=True,
            )
            return cached
        cache_path.unlink(missing_ok=True)

    transcript = _run_darija_transcription(media_path, language)
    if _looks_garbled(transcript):
        print(
            f"[transcribe/darija] FALLBACK: {DARIJA_ADAPTER_ID} output looks garbled for "
            f"{media_path!r} — falling back to generic faster-whisper ({FALLBACK_WHISPER_MODEL})",
            flush=True,
        )
        transcript = _fallback_transcribe(media_path, language)
    else:
        print(
            f"[transcribe/darija] {len(transcript['segments'])} segments, "
            f"{transcript['duration']:.0f}s of audio",
            flush=True,
        )

    _vendor._write_srt_cache(media_path, transcript)
    return transcript


def install() -> None:
    """Shadow-import this module over shorts_generator.local.transcriber.

    Call before shorts_generator.pipeline.generate_shorts(mode="local"):
    pipeline._run_local does `from .local.transcriber import
    transcribe_local` lazily inside the function body, so patching
    sys.modules before that call routes it here instead of the vendored
    faster-whisper-only transcriber.
    """
    import sys

    sys.modules[_VENDOR_TRANSCRIBER_MODULE] = sys.modules[__name__]


if __name__ == "__main__":
    # ponytail: smallest runnable check — needs a real media file, passed
    # as argv[1], since the whole point is confirming the model loads and
    # produces real segments. Not part of the pytest suite (that mocks
    # the heavy model load per CLAUDE.md's testing rules).
    import sys

    media = sys.argv[1] if len(sys.argv) > 1 else None
    assert media, "usage: python transcriber_darija.py <media_path>"
    out = transcribe_local(media)
    assert "segments" in out and "duration" in out
    print("OK:", out)
