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

Audio is transcribed in short windows (`_transcribe_wav_in_windows`), not
one long-form call over the whole file. Real-video testing found that
`return_timestamps="word"` forces the model into eager attention and runs a
per-segment DTW cross-attention capture (`transformers`'
`_extract_token_timestamps`) inside its own internal ~30s-window generate
loop, and that memory keeps growing the more windows a *single* generate()
call processes: a 21-minute video took system swap from ~1GB to ~44GB and
force-restarted the machine, while the identical file transcribed with a
flat memory profile once verified with non-word timestamps (no
cross-attention capture) at the same window count. So the audio is chunked
into `TRANSCRIBE_WINDOW_SECONDS`-long windows *before* calling the ASR
pipeline, bounding each call's internal loop to a couple of iterations —
well under the ~14 iterations a known-safe 7-minute video used in one call
— and the accelerator cache is explicitly flushed between windows (the same
`gc.collect()` + `torch.mps.empty_cache()` sequence `_unload_pipeline`
already used successfully), so nothing accumulates across a long video
regardless of its length.
"""

import json
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
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

# Per-window transcription span (see module docstring for why): a single
# ~30s internal generate() iteration per window — real testing showed 60s
# windows (2 internal iterations) still let memory climb late in a long
# video, so this trades a bit more per-window call overhead for more
# headroom under any one demanding window. Padding gives the model context
# on both sides of a window so a word isn't cut off mid-utterance at the
# boundary; padding-only audio is dropped from the merged result by
# _keep_word_in_window.
TRANSCRIBE_WINDOW_SECONDS = 30.0
TRANSCRIBE_WINDOW_PAD_SECONDS = 3.0

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

    import torch  # type: ignore

    device = _resolve_device()
    print(f"[transcribe/darija] loading {DARIJA_ADAPTER_ID} on {device}", flush=True)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    # fp16 + low_cpu_mem_usage: halves the ~3-4GB fp32 footprint and avoids
    # holding a second full copy in RAM while from_pretrained builds it.
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
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


def _flush_accelerator_cache() -> None:
    """Release cached (but unused) accelerator memory back to the OS.

    Shared by _unload_pipeline (once per video) and
    _transcribe_wav_in_windows (once per audio window) — see module
    docstring for why per-window flushing during transcription matters.
    Synchronizes MPS before emptying its cache: Metal work can still be
    in-flight asynchronously right after a generate() call returns, and
    empty_cache() can't reclaim buffers that haven't actually finished
    being used yet.
    """
    import gc

    import torch  # type: ignore

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_pipeline() -> None:
    """Free the Whisper model after use. The vendored pipeline runs
    transcribe -> highlight-scoring -> crop in one process, and Ollama's
    Atlas-Chat-9B loads for the scoring stage right after this returns —
    two multi-GB models resident at once is what was overloading RAM on
    the M1 Pro. Reload cost next video (~10-20s) beats the crash.
    # ponytail: unload-every-call, not a keep-alive toggle; revisit only
    # if reload time actually becomes the bottleneck.
    """
    global _pipe
    if _pipe is None:
        return
    model = _pipe.model
    _pipe = None
    del model
    _flush_accelerator_cache()


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


def _iter_wav_windows(wav_path: str, window_seconds: float, pad_seconds: float):
    """Yield (padded_start, core_start, core_end, is_last, window_wav_path)
    for consecutive, non-overlapping [core_start, core_end) windows
    covering the whole file, each padded with `pad_seconds` of extra
    context audio on either side (clipped at file boundaries). Caller must
    remove `window_wav_path` after use.
    """
    with wave.open(wav_path, "rb") as src:
        n_channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        total_seconds = src.getnframes() / float(framerate)

        core_start = 0.0
        while core_start < total_seconds:
            core_end = min(core_start + window_seconds, total_seconds)
            is_last = core_end >= total_seconds - 1e-6
            padded_start = max(0.0, core_start - pad_seconds)
            padded_end = min(total_seconds, core_end + pad_seconds)

            start_frame = int(padded_start * framerate)
            end_frame = int(padded_end * framerate)
            src.setpos(start_frame)
            frames = src.readframes(end_frame - start_frame)

            fd, window_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(window_path, "wb") as dst:
                dst.setnchannels(n_channels)
                dst.setsampwidth(sampwidth)
                dst.setframerate(framerate)
                dst.writeframes(frames)

            yield padded_start, core_start, core_end, is_last, window_path
            core_start += window_seconds


def _keep_word_in_window(
    word_start: float, core_start: float, core_end: float, is_last: bool
) -> bool:
    """True if `word_start` falls in this window's own core span rather
    than its context padding (which a neighboring window already owns, or
    will own) — keeps a word from being counted twice across windows.
    """
    if is_last:
        return core_start <= word_start <= core_end + 1e-6
    return core_start <= word_start < core_end


def _extract_window_words(
    chunks: List[Dict],
    padded_start: float,
    core_start: float,
    core_end: float,
    is_last: bool,
) -> List[Dict]:
    """Rebase one window's word-level ASR chunks to absolute (whole-file)
    time and drop any that fall in this window's context padding.
    """
    words: List[Dict] = []
    for chunk in chunks:
        start, end = chunk.get("timestamp", (None, None))
        if start is None:
            continue
        abs_start = padded_start + start
        if not _keep_word_in_window(abs_start, core_start, core_end, is_last):
            continue
        abs_end = padded_start + end if end is not None else None
        words.append({"timestamp": (abs_start, abs_end), "text": chunk.get("text", "")})
    return words


def _transcribe_wav_in_windows(
    pipe, wav_path: str, generate_kwargs: Dict
) -> List[Dict]:
    """Transcribe `wav_path` in short windows rather than one long-form
    call, so the model's internal ~30s-window generate loop never runs long
    enough in a single call to accumulate the memory growth word-level
    timestamps trigger (see module docstring). Flushes the accelerator
    cache between windows so nothing carries over to the next one.
    """
    all_words: List[Dict] = []
    for padded_start, core_start, core_end, is_last, window_path in _iter_wav_windows(
        wav_path, TRANSCRIBE_WINDOW_SECONDS, TRANSCRIBE_WINDOW_PAD_SECONDS
    ):
        try:
            result = pipe(
                window_path, return_timestamps="word", generate_kwargs=generate_kwargs
            )
        finally:
            os.remove(window_path)
        all_words.extend(
            _extract_window_words(
                result.get("chunks", []), padded_start, core_start, core_end, is_last
            )
        )
        _flush_accelerator_cache()
    return all_words


def _group_words_into_segments(word_chunks: List[Dict]) -> List[Dict]:
    """Re-group word-level ASR chunks into phrase-sized segments, breaking
    at pauses > MAX_WORD_GAP_SECONDS or once a segment would run past
    MAX_SEGMENT_SECONDS.

    Each segment keeps the original per-word (start, end, text) spans in a
    "words" list alongside the joined "text" — captioner.py uses these to
    time caption cards to actual speech instead of interpolating.
    """
    segments: List[Dict] = []
    current_words: List[str] = []
    current_word_spans: List[Dict] = []
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
            current_start, current_words = start, [text]
            current_word_spans = [{"start": start, "end": end, "text": text}]
            prev_end = end
            continue

        gap = start - prev_end
        would_be_duration = end - current_start
        if gap > MAX_WORD_GAP_SECONDS or would_be_duration > MAX_SEGMENT_SECONDS:
            segments.append(
                {
                    "start": current_start,
                    "end": prev_end,
                    "text": " ".join(current_words),
                    "words": current_word_spans,
                }
            )
            current_start, current_words = start, [text]
            current_word_spans = [{"start": start, "end": end, "text": text}]
        else:
            current_words.append(text)
            current_word_spans.append({"start": start, "end": end, "text": text})
        prev_end = end

    if current_words:
        segments.append(
            {
                "start": current_start,
                "end": prev_end,
                "text": " ".join(current_words),
                "words": current_word_spans,
            }
        )
    return segments


def _run_darija_transcription(media_path: str, language: Optional[str]) -> Dict:
    pipe = _load_pipeline()
    wav_path = _extract_audio_wav(media_path)
    try:
        generate_kwargs = {"language": language} if language else {}
        word_chunks = _transcribe_wav_in_windows(pipe, wav_path, generate_kwargs)
    finally:
        os.remove(wav_path)

    segments = _group_words_into_segments(word_chunks)
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


def _cache_path(media_path: str) -> Path:
    """JSON cache path for the word-level transcript.

    Deliberately not the vendored .srt cache format: SRT can only carry one
    (start, end, text) span per line, which can't round-trip each segment's
    per-word "words" timing — and nothing else in this repo reads this
    file, so there's no compatibility reason to keep it lossy.
    """
    return Path(_vendor._transcript_cache_path(media_path)).with_suffix(".json")


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Darija-first transcription; same (media_path, language) -> {duration,
    segments[start,end,text,words]} contract as the vendored transcriber,
    plus a per-segment "words" list (see _group_words_into_segments) when
    the darija-primary path produced the result. The generic-Whisper
    fallback path has no per-word timestamps, so its segments only carry
    start/end/text — captioner.py handles both shapes.

    Reads/writes no state.db tables. Caches the full transcript as JSON
    (see _cache_path for why, not the vendored .srt format).
    """
    cache_path = _cache_path(media_path)
    if cache_path.exists() and cache_path.stat().st_mtime >= os.path.getmtime(
        media_path
    ):
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("segments") and cached.get("duration", 0.0) > 0.0:
            print(
                f"[transcribe/darija] reusing cached transcript: {cache_path}",
                flush=True,
            )
            return cached
        cache_path.unlink(missing_ok=True)

    transcript = _run_darija_transcription(media_path, language)
    _unload_pipeline()
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

    cache_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
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
