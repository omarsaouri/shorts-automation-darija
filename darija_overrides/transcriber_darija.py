"""Swap shorts_generator.local.transcriber's faster-whisper model for
anaszil/whisper-large-v3-turbo-darija — a LoRA fine-tune of
openai/whisper-large-v3-turbo for Moroccan Darija.

This adapter has no CTranslate2/GGUF conversion (checked its HF repo:
only adapter_config.json + adapter_model.safetensors), so it can't be
loaded through faster-whisper like the vendored transcriber. Instead this
uses the standard transformers + PEFT loading path the model card itself
documents, running on MPS/CUDA/CPU (faster-whisper's engine doesn't
support Metal, but plain transformers does via torch's MPS backend).

Per CLAUDE.md's hard constraint, generic Whisper is a fallback only, never
silently primary: if the Darija model's output looks garbled (empty, or a
repetition loop — a known Whisper hallucination failure mode, more common
on heavy Darija/French code-switching per the architecture doc), this
falls back to the vendored faster-whisper transcriber forced to
large-v3, and logs the fallback with the media path so it's visible in
pipeline logs.
"""

import os
from typing import Dict, Optional

import shorts_generator.local.transcriber as _vendor

BASE_MODEL_ID = "openai/whisper-large-v3-turbo"
DARIJA_ADAPTER_ID = "anaszil/whisper-large-v3-turbo-darija"
FALLBACK_WHISPER_MODEL = os.environ.get("FALLBACK_WHISPER_MODEL", "large-v3")

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

    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        chunk_length_s=30,
    )
    return _pipe


def _run_darija_transcription(media_path: str, language: Optional[str]) -> Dict:
    pipe = _load_pipeline()
    generate_kwargs = {"language": language} if language else {}
    result = pipe(media_path, return_timestamps=True, generate_kwargs=generate_kwargs)

    segments = []
    for chunk in result.get("chunks", []):
        start, end = chunk["timestamp"]
        if start is None or end is None:
            continue
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "text": (chunk["text"] or "").strip(),
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


def _looks_garbled(transcript: Dict) -> bool:
    """Empty output, or a back-to-back repetition loop (Whisper's classic
    hallucination artifact) counts as garbled.
    # ponytail: two heuristics, not a full quality classifier — good enough
    # to catch the failure mode the architecture doc calls out; revisit if
    # real Darija runs show a different garbling pattern.
    """
    segments = transcript.get("segments", [])
    non_empty = [s for s in segments if s["text"].strip()]
    if not non_empty:
        return True
    repeats = sum(1 for a, b in zip(non_empty, non_empty[1:]) if a["text"] == b["text"])
    return len(non_empty) >= 4 and repeats / len(non_empty) > 0.3


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
