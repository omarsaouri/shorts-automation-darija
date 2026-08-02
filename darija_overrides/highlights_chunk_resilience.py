"""Don't let one bad chunk abort highlights for an entire long video, and
give each chunk more shots at succeeding in the first place.

Real-world finding (42-min video `Zhj07EXj4HY`, see `progress.md`): on one
20-min chunk, Atlas-Chat-9B went fully conversational instead of returning
JSON on 5/5 direct attempts — verified with Ollama's `system`/`prompt` fields
properly separated, so this isn't the usual malformed-but-JSON-ish flakiness
`llm_ollama.py`'s fixes already cover. It's a genuine 9B-model capability
limit on that chunk's content, nothing to parse-fix. A follow-up sweep across
chunk sizes (1200s/600s) and temperatures (0.7/0.2) found neither lever
raises the per-attempt success rate (~1-in-3 across the board, small sample)
— see the `MAX_HIGHLIGHT_API_ATTEMPTS` comment below for the mitigation that
was actually applied instead.

Vendored `get_highlights`' long-video branch calls `call_highlight_api` per
chunk with no try/except, so one chunk exhausting its 3 retries raises
`RuntimeError` straight out of the `for` loop and kills highlights for the
*whole* video — even chunks that succeeded are thrown away.

This patches `get_highlights` at the call site (`shorts_generator.pipeline`,
not `shorts_generator.highlights`): pipeline.py does
`from .highlights import get_highlights` at module import time, an early-
bound reference, so patching the attribute on `highlights` itself (the trick
`highlights_chunking.py` uses for `chunk_transcript`, referenced by bare name
*within* that same module) wouldn't be seen by pipeline.py's already-bound
name. Patching `pipeline.get_highlights` is the correct target, same
single-attribute pattern as everywhere else in `darija_overrides/`.

Short-video path (< LONG_VIDEO_THRESHOLD) is untouched vendor logic, reused
as-is via `_original_get_highlights`. `chunk_transcript`, `call_highlight_api`,
`dedupe_highlights`, `detect_content_type` are looked up on the live
`shorts_generator.highlights` module at call time, so this stays compatible
with `highlights_chunking.py` / `highlights_duration_filter.py` regardless of
install order.
"""

from typing import Dict, List, Optional

import shorts_generator.highlights as _highlights
import shorts_generator.pipeline as _pipeline

# Real-world measurement (2026-08-02, see progress.md): single-attempt
# success rate against a real 42-min video's transcript was ~1-in-3,
# consistent across chunk sizes (1200s/600s) and temperatures (0.7/0.2) — a
# small sweep (13-15 calls), but neither lever moved the needle, so this
# isn't fixable by re-chunking or cooling the model down. Raising retries
# from vendor's default 3 to 5 takes per-chunk success from ~70% to ~87% at
# that observed rate — the cheapest lever left, at the cost of more Ollama
# time on chunks that are already failing.
MAX_HIGHLIGHT_API_ATTEMPTS = 5

_original_get_highlights = None


def get_highlights_resilient(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[_highlights.LLMFn] = None,
) -> Dict:
    """Same contract as vendored `get_highlights`, chunk failures don't abort the run.

    Reads no `state.db` tables. Writes none.
    """
    llm_fn = llm_fn or _highlights.call_muapi_llm
    duration = transcript.get("duration", 0)

    if duration < _highlights.LONG_VIDEO_THRESHOLD:
        return _original_get_highlights(transcript, num_clips=num_clips, llm_fn=llm_fn)

    content_info = _highlights.detect_content_type(transcript, llm_fn=llm_fn)
    print(
        f"[highlights] content={content_info.get('content_type')} "
        f"density={content_info.get('density')} duration={duration:.0f}s",
        flush=True,
    )

    chunks = _highlights.chunk_transcript(transcript)
    print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
    all_highlights: List[Dict] = []
    skipped = 0
    for i, chunk in enumerate(chunks):
        offset = chunk.get("_offset", 0)
        text = _highlights.build_transcript_text(chunk)
        print(
            f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)",
            flush=True,
        )
        try:
            result = _highlights.call_highlight_api(
                text,
                content_info,
                chunk["duration"],
                num_clips=num_clips,
                is_chunk=True,
                llm_fn=llm_fn,
            )
        except RuntimeError as e:
            skipped += 1
            print(
                f"[highlights] chunk {i + 1}/{len(chunks)} skipped after exhausting retries: {e}",
                flush=True,
            )
            continue
        for h in result.get("highlights", []):
            h["start_time"] = float(h["start_time"]) + offset
            h["end_time"] = float(h["end_time"]) + offset
            all_highlights.append(h)

    if skipped == len(chunks):
        raise RuntimeError(f"All {len(chunks)} chunks failed to produce highlights")

    return {"highlights": _highlights.dedupe_highlights(all_highlights)}


def install() -> None:
    """Monkeypatch shorts_generator.pipeline.get_highlights (see module docstring for why this target)
    and bump shorts_generator.highlights.MAX_HIGHLIGHT_API_ATTEMPTS (see the
    constant's comment above) — call_highlight_api reads it as a bare name
    from its own module's globals at call time, same reason chunk_transcript
    is patchable as a single attribute in highlights_chunking.py.
    """
    global _original_get_highlights
    if _original_get_highlights is None:
        _original_get_highlights = _highlights.get_highlights
    _pipeline.get_highlights = get_highlights_resilient
    _highlights.MAX_HIGHLIGHT_API_ATTEMPTS = MAX_HIGHLIGHT_API_ATTEMPTS
