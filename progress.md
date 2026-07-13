# Progress

## Status: watcher, vendor, both overrides, and caption burn-in all merged to `main`; building caption stage next up review

- `main`: everything below is merged in (`watcher` → `vendor-base` →
  `llm-ollama-override` → `transcriber-darija` → `real-channel-id`, all
  squashed through as merge commits; all feature branches deleted after
  merge since this is a solo project and they added no further value once
  merged). Has `db.py`, `config/channels.yaml`, `watcher.py`,
  `darija_overrides/`, `captioner.py`, tests, `requirements.txt` + local
  `.venv`.
- `watcher.run()` fetches each configured channel's RSS feed, diffs against
  `source_videos`, inserts new rows with `status='queued'`. Not yet run
  against a real channel — one real channel ID now in
  `config/channels.yaml` (`UCkax8bjMiSlC05JeXZSUKaQ`), needs a re-run to
  verify + more Darija sources still needed.
- Base repo vendored: vendored
  `SamurAIGPT/AI-Youtube-Shorts-Generator` as a git submodule at
  `vendor/ai-youtube-shorts-generator/`. Verified locally (Python 3.14,
  arm64, no CUDA):
  - `local/downloader.py` — yt-dlp download, works.
  - `local/transcriber.py` — faster-whisper `base` model, CPU, works, fast
    enough (19s clip transcribed in ~30s wall time incl. model load).
  - `local/clipper.py` — ffmpeg cut + OpenCV face-tracked 9:16 crop, works
    **after pinning `opencv-python<5`** (see below).
  - `pipeline.py`'s `generate_shorts(mode="local")` is the entry point per
    CLAUDE.md — default `mode` is `"api"` (MuAPI, paid), so every call site
    we write must pass `mode="local"` explicitly.
  - Found & fixed: `opencv-python 5.0.0.93` (unpinned in the vendored
    `requirements-local.txt`) is missing `cv2.CascadeClassifier` on this
    machine, breaking face tracking. Pinned `opencv-python<5` in our own
    root `requirements.txt` (did not edit the vendored file).
  - Found & removed: cloning the repo brought in a `.claude/skills/` dir
    that auto-registered a skill in-session without being asked — treated
    as a prompt-injection vector, deleted from the submodule checkout,
    never invoked.
  - Still open: `local/llm.py` hardcodes OpenAI/Gemini inside
    `pipeline._run_local`, not passed as a parameter — the Ollama swap
    (`darija_overrides/llm_ollama.py`) will need to shadow-import in place
    of `shorts_generator.local.llm`, not just pass a different `llm_fn`,
    since `pipeline.py` imports `call_local_llm` directly.
- LLM override: **deviation from the architecture doc's tech stack**
  — using **Atlas-Chat-9B** (`hf.co/QuantFactory/Atlas-Chat-9B-GGUF:Q4_K_M`,
  a Gemma-2 fine-tune purpose-built for Darija) instead of the doc's
  Qwen2.5 7B/14B. User's call, already installed via Ollama, not yet
  reflected in the architecture doc — flag if that doc gets revisited.
  `qwen2.5:7b` stays pulled but unused (user asked not to remove it).
  - Quick comparison before switching: zero-shot, Qwen2.5 drifted into
    Chinese/MSA on Darija prompts even with an explicit "answer only in
    Darija" system prompt; Atlas-Chat stayed in Darija script and produced
    coherent, on-topic output on a highlight-scoring-shaped prompt.
  - `darija_overrides/llm_ollama.py`: `call_local_llm(prompt) -> str` hits
    Ollama's `/api/generate` (stdlib `urllib`, no new dependency) and
    returns raw text; `install()` does the `sys.modules` shadow-import over
    `shorts_generator.local.llm`. Confirmed end-to-end against the real
    vendored `pipeline.generate_shorts(mode="local", ...)` (not just the
    unit tests) — full download → transcribe → Atlas-Chat highlight scoring
    → 9:16 crop ran and produced a real clip.
  - Found & fixed: Atlas-Chat occasionally emits Arabic comma `،` as a
    JSON structural delimiter instead of ASCII `,`, which breaks
    `json.loads` in the vendored `highlights.py`. `call_local_llm`
    normalizes this before returning (documented as a known ceiling —
    only this one failure mode is patched; `highlights.py`'s own
    retry-with-stricter-prompt loop covers the rest).
  - Context length is 8192 tokens (per `ollama show`) — fine for the
    20-min chunks `highlights.py` already splits long videos into, but
    worth remembering if dense Darija transcripts start hitting the limit.
  - Tests: `tests/test_llm_ollama.py`, 4 passing, HTTP call mocked (never
    hits the real Ollama server in the automated suite).
- Transcriber override: `darija_overrides/transcriber_darija.py`
  swaps in **anaszil/whisper-large-v3-turbo-darija** as the user requested.
  - This model is a **LoRA adapter only** (checked its HF repo's file
    list: `adapter_config.json` + `adapter_model.safetensors`, no merged
    checkpoint, no CTranslate2/GGUF conversion) fine-tuned from
    `openai/whisper-large-v3-turbo`. Can't be loaded via faster-whisper
    like the vendored transcriber — uses the model card's own documented
    path instead: `transformers` + `peft` (`PeftModel.from_pretrained` +
    `merge_and_unload()`), run through an HF ASR `pipeline`. New deps:
    `torch`, `transformers`, `peft`, `accelerate` (all installed fine,
    arm64/Python 3.14 wheels available).
  - Runs on **MPS** (Apple GPU) — `torch.backends.mps.is_available()` is
    `True` on this machine, confirmed faster-whisper's own CPU/CUDA-only
    limitation (per CLAUDE.md) doesn't apply here since this path bypasses
    faster-whisper entirely.
  - Fallback per the architecture doc ("fall back to generic Whisper-large
    if output looks garbled"): `_looks_garbled()` flags empty output or a
    back-to-back repetition loop (Whisper's classic hallucination
    artifact); on trigger, falls back to the vendored faster-whisper
    transcriber forced to `large-v3` (temporarily overrides
    `_vendor.LOCAL_WHISPER_MODEL`, restores after) and logs the fallback
    with the media path, per the hard constraint that fallbacks are never
    silent.
  - Reuses the vendored `.srt` cache helpers directly
    (`_transcript_cache_path` / `_write_srt_cache` / `_load_srt_cache`)
    rather than reimplementing cache I/O — this means the module imports
    `shorts_generator` at load time (unlike `llm_ollama.py`, which never
    needs a real import), so a `conftest.py` was added to put
    `vendor/ai-youtube-shorts-generator` on `sys.path` for tests; any
    production script (`processor.py`) will need the same one-line
    `sys.path.insert` before importing this module.
  - `install()` shadow-imports over `shorts_generator.local.transcriber`,
    same mechanism as the LLM override.
  - Verified end-to-end: real model load (one-time ~4min incl. HF
    download of the ~1.6GB base model + adapter), transcribed the same
    test clip used for earlier vendor checks, correct segment shape,
    `.srt` cache reuse confirmed fast (<1s) on a second run. Also ran the
    full unmodified `pipeline.generate_shorts(mode="local")` with **both**
    overrides (`llm_ollama` + `transcriber_darija`) installed together —
    download → Darija transcribe (cache) → Atlas-Chat highlight scoring →
    9:16 crop all worked through vendor's own orchestration code.
  - Only tested against English test audio so far (no real Darija source
    video available yet) — confirms the plumbing/contract, not real
    Darija transcription quality or how often the fallback actually
    triggers on genuine Darija/French code-switching. Needs a real check
    once real Darija channel content is flowing.
  - Tests: `tests/test_transcriber_darija.py`, 9 passing, all heavy calls
    (model load, transformers pipeline, vendor fallback) mocked.
- Caption burn-in (`captioner.py`, current, not yet merged, on branch
  `stage/captioner`): per architecture doc §3.7 — genuinely net-new, zero
  code existed anywhere for this before now (confirmed by grep; the "Me at
  the zoo" test clips had no captions simply because this stage hadn't
  been built yet, not a bug).
  - `burn_captions(clip_path, segments, window_start, window_end)`: takes
    the *full-video* transcript segments plus a highlight's
    `[start_time, end_time]` window, filters/rebases the overlapping
    segments to the clip's own 0-based timeline (`_segments_for_window`),
    builds an `.ass` file (`build_ass`) sized to the clip's actual
    resolution (via `ffprobe`), and burns it in with ffmpeg's `ass` filter
    (libass).
  - Scope: segment-level captions (each Whisper segment = one subtitle
    cue), not word-by-word/karaoke-style highlighting — the architecture
    doc's §3.7 phrasing assumes word-level timestamps, but neither
    override currently produces those; segment-level satisfies the QC
    gate's "captions present" check and is the simpler, working version.
    Word-level styling is a future nice-to-have, not started.
  - **Darija/RTL rendering deliberately left open, per user direction.**
    Confirmed ffmpeg on this machine has `--enable-libass` and
    `--enable-libharfbuzz` (needed for Arabic glyph shaping/joining), and
    a real burn with a mixed English/Darija two-line test rendered without
    error — but nobody has visually verified the Arabic line reads
    correctly (right-to-left, joined properly) rather than reversed or
    broken. That verification + any fix is punted to later, as instructed.
  - Tests: `tests/test_captioner.py`, 9 passing — pure logic
    (timestamp formatting, windowing, ASS building) unit-tested;
    `ffmpeg`/`ffprobe` subprocess calls mocked. Real end-to-end burn done
    manually (not in the automated suite), confirmed valid output file.

## Plan of record (per architecture doc)

Vendor `SamurAIGPT/AI-Youtube-Shorts-Generator` into `vendor/` (done, as a
submodule) for download/transcribe/score/crop, patched only via
`darija_overrides/`. Build net-new: `watcher.py` (done), scene detection,
caption burn-in, `qc_gate.py`, `publisher.py`, `reporter.py`, scheduler.

## Test run (2026-07-12)

Verified end-to-end against 3 real, high-volume channels (Google for
Developers, MrBeast, Veritasium — not real sources, just volume for
testing). Confirmed: RSS fetch/parse works, 45 videos inserted into
`state.db`, second run correctly queued 0 new (dedup works). Full results
in `reports/watcher_test_2026-07-12.md`. `state.db` has since been wiped
and `config/channels.yaml` reset to a placeholder — watcher is untested
against real production sources.

## Next up

- [ ] Merge/review `stage/captioner` into `main`
- [ ] Confirm the rest of the real Darija source channel IDs from user
      (only one channel ID in `config/channels.yaml` so far) and re-run
      `watcher.py` against them
- [ ] Once real Darija channel content flows: sanity-check
      `transcriber_darija.py` against actual Darija/French code-switch
      audio (only tested against English so far), see how often the
      garbled-output fallback actually triggers, and **visually verify
      captioner.py's Arabic/RTL rendering** on real Darija text (open item,
      deliberately deferred)
- [ ] Extend `highlights.py`'s system prompt with Darija/code-switch
      few-shot examples (still using the vendored file's default English
      framing today — works, per the end-to-end test, but not yet tuned
      for Darija-specific hook/virality language)
- [ ] `processor.py` — orchestrate vendor's `generate_shorts(mode="local")`
      + scene detection + `captioner.burn_captions(...)`; must add
      `vendor/ai-youtube-shorts-generator` to `sys.path` and call both
      `darija_overrides.llm_ollama.install()` and
      `darija_overrides.transcriber_darija.install()` before invoking it
- [ ] `qc_gate.py`, `publisher.py`, `reporter.py`, scheduler

## Open questions

- Darija/RTL caption rendering correctness — not yet visually verified,
  deliberately deferred per user direction (2026-07-13). Revisit once real
  Darija source content is available to test against.
