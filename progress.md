# Progress

## Status: watcher, vendor, all overrides, and caption burn-in merged to `main`; two real-video bugs found and fixed, on branch `fix/real-video-test-findings`

- `main`: everything below is merged in (`watcher` → `vendor-base` →
  `llm-ollama-override` → `transcriber-darija` → `real-channel-id` →
  `captioner`, all as merge commits; all feature branches deleted after
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

## Real Darija video test (2026-07-13) — found 2 real bugs, both fixed

First test against real Darija content: `montakhab fans`
(`UCkax8bjMiSlC05JeXZSUKaQ`), a Moroccan football commentary channel —
`watcher.run()` queued 15 real videos into `state.db` successfully. Ran
the full manual pipeline (download → Darija transcribe → Atlas-Chat
scoring → crop → caption) on a 40.5-minute video
(`KazZdpoVvio`) via an ad hoc script (not a committed `processor.py` yet).

**Bug 1 — transcriber segmentation collapsed into 2 giant blobs.**
`transcriber_darija.py`'s `chunk_length_s=30` ASR pipeline setting
(transformers itself warns this is "very experimental with seq2seq
models") produced only 2 segments for the entire 39-minute audio instead
of per-sentence ones, including a visible hallucination artifact (a
50+-character run of repeated و). This cascaded into broken captions —
`captioner.py` dumped an ~2800-character wall of text onto a 15-second
clip, since it just renders whatever segment(s) overlap a clip's window.
Fixed: switched to word-level timestamps (reliable) + WAV extraction
(the non-chunked pipeline path can't reliably read .mp4 directly) +
`_group_words_into_segments` re-groups words into phrase-sized segments
at natural pauses. Verified on a 3-min real sample: 2 blobs → 14 clean
segments. Also strengthened `_looks_garbled` (which should have caught
this and didn't) with a segments-per-minute density check and a
repeated-character-run check. See `fix/real-video-test-findings` commit
`3ba074b`.

**Bug 2 — face-tracking crop snapping to false positives.** User-observed
on `short_02.mp4`: the vertical crop's focus visibly jerked away from the
speaker and back, "like it's catching other objects and returning."
Root cause in vendor's `local/clipper.py::_reframe_vertical`: it re-targets
the crop to whichever Haar cascade detection is largest, every frame,
immediately smoothing toward it — so a single-frame false positive gets
chased exactly like a real subject. Fixed via
`darija_overrides/clipper_stable.py`, which patches only
`_reframe_vertical` (monkeypatches the one vendor function attribute,
`crop_clip_local`/`crop_highlights_local`/`_cut_subclip` stay unmodified
vendor code) with a debounced `_FaceTracker`: small movements are trusted
immediately, but a big jump needs `REQUIRED_CONSECUTIVE_FRAMES` (3) of
consistent detections before the crop follows it. This is the first
"reuse as-is" vendor component (per the architecture doc's reuse table)
that turned out to need a fix — flagging the deviation from plan, same as
the Qwen2.5→Atlas-Chat swap. See commit `0b0baba`.

**Both fixes verified in isolation** (3-min sample for the transcriber;
re-cropping the exact buggy window for the clipper) before re-running the
full 40-minute video end-to-end with all three overrides
(`llm_ollama` + `transcriber_darija` + `clipper_stable`) installed
together, output written to `clips/KazZdpoVvio/`.

Also added `output/` to `.gitignore` — the vendored pipeline's default
local-mode scratch dir (source videos, crops, `.srt` caches) wasn't
excluded yet.

## Memory-overload investigation & fix (2026-07-18) — two real machine crashes, root-caused and fixed; uncommitted on `main`

User reported the real pipeline overloading memory on the M1 Pro (16GB).
Investigation found **two independent causes**, both real, both now fixed
in the working tree (not yet committed/branched — see below).

**Cause 1 (small): Ollama's default keep-alive.** `darija_overrides/llm_ollama.py`'s
`call_local_llm` sent no `keep_alive` to Ollama's `/api/generate`, so
Atlas-Chat-9B (~5.5-6GB) stayed resident for Ollama's default 5-minute
keep-alive — overlapping with the *next* video's Whisper reload if a batch
run processed videos back-to-back, since Ollama (Metal) and torch (MPS)
share the same unified memory pool on Apple Silicon. Fixed: payload now
sends `"keep_alive": 0` so the model evicts immediately after each
highlight-scoring call, mirroring the Whisper-side unload
(`transcriber_darija._unload_pipeline`) that already existed. Verified with
a mocked test (`tests/test_llm_ollama.py`) and confirmed no crash across
one real end-to-end run (7-min `SkxfKZgy9kw` video, full pipeline —
download → Darija transcribe → Ollama scoring → crop → caption — completed
clean; output at `clips/SkxfKZgy9kw/short_01.captioned.mp4`).

**Cause 2 (the real crash driver): word-level timestamps blow up memory
with audio length.** Two actual hard crashes happened during testing —
confirmed via `uptime` resetting to ~1-2 min and the `/tmp` scratchpad
being wiped, not just an app-level exception:
- 42-min video (`Zhj07EXj4HY`) — crashed with no time to react (watchdog's
  5s polling wasn't fast enough); logs lost with the reboot.
- 21-min video (`CadW5Vyh-hg`) — logged to a persistent path this time
  (`output/debug_logs/`, gitignored) before it crashed: swap grew from
  ~1GB to ~44GB in under 2 minutes, all during the Darija transcribe step,
  before Ollama was ever called.

Root cause, confirmed by reading `transformers`' `generation_whisper.py`
(v5.13.1): `return_timestamps="word"` forces the model into eager
attention and runs a per-~30s-window DTW cross-attention capture
(`_extract_token_timestamps`) inside Whisper's own internal long-form
`generate()` loop, and that memory keeps growing the more windows a
*single* `generate()` call processes — i.e. it scales with audio duration.
Confirmed directly: re-running the exact 21-min file with
`return_timestamps=True` (segment-level, model's own timestamp tokens, no
cross-attention capture) instead of `"word"` stayed completely flat
(66-67% free, zero swap) for the full file, at the same internal window
count — isolating the cross-attention capture as the driver, not duration
or iteration count alone.

**Fix:** `darija_overrides/transcriber_darija.py` now chunks audio into
`TRANSCRIBE_WINDOW_SECONDS`-long windows *before* calling the ASR pipeline,
instead of one long-form call over the whole file — bounds each call's
internal loop to a couple of iterations regardless of total video length.
New functions: `_iter_wav_windows` (stdlib `wave`, no new ffmpeg calls,
padded windows so words aren't cut at boundaries), `_keep_word_in_window` /
`_extract_window_words` (rebase + dedupe words across window padding),
`_transcribe_wav_in_windows` (orchestrates + flushes the accelerator cache
between every window via a new shared `_flush_accelerator_cache`, which
`_unload_pipeline` now also calls). Keeps word-level timestamp quality
(still feeds the existing `_group_words_into_segments`) rather than
falling back to coarser segment-level timestamps.

Iterated on window size against the real 21-min crash file:
- 60s windows: swap plateaued at 3-4GB for ~10 min (huge improvement over
  unbounded 44GB) but spiked to 5GB/5% free right at the very end; a
  memory watchdog script killed the process — though the transcript had
  actually already finished and written a complete, valid `.srt` cache
  moments before/during that kill. Likely fine, but too close to trust.
- Tightened to 30s windows + added `torch.mps.synchronize()` before
  `torch.mps.empty_cache()` in `_flush_accelerator_cache` (Metal work can
  still be in-flight when `empty_cache()` runs, so it wasn't necessarily
  reclaiming everything it could). Re-ran on the same file with a
  (bug-fixed) watchdog: **943s, 116 segments, full 1273/1280s coverage,
  `py_rss` bounded 620-850MB the whole run, swap capped at 3-4GB, free%
  never below ~28%.** Real margin this time, not a near-miss.

Debugging note worth keeping: the ad hoc memory-watchdog shell script had
its own real bug — on this machine `python3` re-execs into
`.../Python.app/Contents/MacOS/Python` on startup (a macOS framework-Python
quirk), so pattern-matching the literal string `"python3"` in `ps`/`pgrep`
output silently finds nothing. Early "successful" watchdog readings were
accidentally matching the test harness's own wrapper shell (whose command
text happened to contain `"python3 ... script.py"` as literal source, not
the real re-exec'd process) — which is why RSS readings were bogus
(~0-3MB) even though the system-wide free%/swap numbers were accurate.
Fixed by matching the script's bare filename and excluding the watchdog's
own PID directly, no `"python3"` anchor needed.

**Not yet done:** the 42-min video that originally crashed (`Zhj07EXj4HY`)
has not been re-tested against the fix — only the 7-min and 21-min videos
have. Only `transcribe_local` in isolation was tested at the 21-min length
(no download/Ollama/crop) to avoid unnecessary risk while iterating on
window size; the full pipeline hasn't been re-run end-to-end at that
length yet. All of the above is **uncommitted on `main`** — per CLAUDE.md
this should go through a branch (e.g. `fix/transcriber-memory-overload`)
before merging, not yet done. Tests: `tests/test_transcriber_darija.py`
extended with coverage for the new windowing/merging logic;
`tests/test_llm_ollama.py` extended for the `keep_alive` assertion; full
suite (46 tests) passing, `black`/`ruff` clean.

**Separate, unfixed issue noticed along the way:** on the 7-min
`SkxfKZgy9kw` test, `get_highlights` returned only 1 highlight spanning 396
of the video's 419 seconds — essentially the whole video, not a short
clip. Likely Atlas-Chat-9B not reliably following `highlights.py`'s
"45-90s sweet spot, no >50% overlap" prompt instructions, or
`dedupe_highlights` collapsing several heavily-overlapping candidates down
to one. Not investigated this session.

## Memory-overload fix committed + re-tested against real videos (2026-07-20/21)

Memory-overload fix from the section above is now committed on
`fix/transcriber-memory-overload` (`e60d52d`), branched off `main` per
CLAUDE.md's workflow — was sitting uncommitted on `main` before this. Not
yet merged.

**42-min re-test (`Zhj07EXj4HY`, the video that originally crashed the
machine) — transcribe-only, passed clean.** Ran under a memory watchdog
(2s-interval `vm_stat`/`sysctl vm.swapusage` polling, kills the process on
swap > 8GB or free% < 8% for 3 consecutive samples — both well short of the
~44GB swap spike that caused the original crash). Result: 221 segments,
full 2532s/42.2min coverage, wall time 29m22s, peak swap 6.68GB, peak
py_rss 1.1GB, no watchdog intervention. Output cached at
`output/source_Zhj07EXj4HY.srt`. Closes the "only 7-min/21-min verified"
gap from the section above.

**Full pipeline (transcribe → Ollama scoring → crop → caption) re-tested,
found and fixed two more real bugs in `llm_ollama.py` along the way** (both
only reachable with a long/large transcript, which is why they weren't seen
in earlier shorter-video tests):

1. **Unhandled network/timeout exception crashed the whole pipeline.**
   `highlights.py`'s `call_highlight_api` retry loop only wraps *JSON
   parsing* in try/except, not the `llm_fn(prompt)` call itself — so when a
   large 20-min transcript chunk (per `highlights.py`'s own
   `CHUNK_SIZE_SECONDS`) made Atlas-Chat-9B's response take longer than the
   300s Ollama call timeout, the `TimeoutError` propagated straight out and
   killed the run instead of being retried. Fixed in `llm_ollama.py`:
   catches `URLError`/`socket.timeout`/`TimeoutError` and returns `""`
   instead of raising, which routes it through the vendor's *existing*
   retry loop the same way a malformed-JSON response already is. Also
   bumped `OLLAMA_TIMEOUT_SECONDS` 300 → 480 for headroom. Root-cause fix
   stayed entirely inside our override layer, no vendor edit.
2. **Trailing comma in model JSON output.** Same category as the
   already-documented Arabic-comma quirk — Atlas-Chat-9B occasionally
   leaves a trailing `,` before a closing `]`/`}` on large highlight lists,
   which vendor's strict `json.loads` doesn't tolerate. Added
   `_strip_trailing_commas` alongside the existing
   `_fix_arabic_json_punctuation`, same "known ceiling, not a general JSON
   repair tool" rationale.

Tests: `tests/test_llm_ollama.py` extended (4 new tests: timeout →
empty-string, connection error → empty-string, trailing-comma stripping,
plus the existing suite), 49 passing total, `black`/`ruff` clean.

**Found, NOT fixed (user's call, 2026-07-21): a real vendor bug in
`highlights.py`'s chunking breaks highlight detection on any video ≥30
min.** `chunk_transcript()` keeps each segment's *absolute* video timestamp
when building the LLM prompt (e.g. chunk 2 of the 42-min video shows
`[1146.2s]` through `[2377.9s]`), so the model naturally answers with
absolute-range timestamps. But `call_highlight_api` passes
`chunk["duration"]` (the chunk's *relative* span, e.g. `1200`) as the
`max_end` clamp in `_sanitize_highlights` — any highlight past `1200` gets
clamped to `1200` on both ends and dropped as zero-length, and
`get_highlights` then adds `+offset` again on top of whatever survives.
Net effect: chunks after the first lose nearly all their highlights, then
`call_highlight_api` raises `RuntimeError` after 3 failed attempts and
aborts the whole multi-chunk loop — confirmed directly by capturing
Atlas-Chat-9B's raw output for chunk 2 (`start_time: 7634.95` against a
`max_end` of `1200`, sanitized count 0). Not backend-specific — this is
vendor's own local-mode chunking math, would affect MuAPI/gpt-5-mini the
same way. Root-cause fix would need a `darija_overrides` module that
rebases each chunk's transcript timestamps to 0 before building the prompt
(same monkeypatch pattern as `clipper_stable.py`), but user chose to skip
it for now rather than take on that scope mid-session. **Practical
implication: don't run the full pipeline against videos ≥30 min until this
is fixed — use the 21-min (`CadW5Vyh-hg`) or 7-min (`SkxfKZgy9kw`) cached
test videos instead, both under the chunking threshold.**

Full-pipeline re-test was redirected to the 21-min `CadW5Vyh-hg` video (transcript
already cached, under the 30-min chunking threshold) — see result below if
completed, or "Next up" if still pending as of this write-up.

## highlights.py chunking bug fixed (2026-07-21)

Picked up the chunking bug documented above ("Found, NOT fixed") — root
cause confirmed and fixed via a new override,
`darija_overrides/highlights_chunking.py`.

**Root cause:** vendor's `chunk_transcript` builds each chunk's `segments`
from the transcript's *absolute* (whole-video) timestamps and never rebases
them to the chunk's own start. `build_transcript_text` then shows the model
absolute time labels (chunk 2 of a long video shows `[1146.2s]` onward), so
the model naturally answers with highlights in that same absolute range.
But `call_highlight_api` clamps against `chunk["duration"]` — the chunk's
*relative* span (e.g. `1200`) — so every highlight past the first chunk gets
clamped to zero-length and dropped, and `call_highlight_api` raises after 3
failed attempts, aborting the whole multi-chunk loop.

**Fix:** `chunk_transcript_rebased` delegates to the vendored
`chunk_transcript` for all the actual chunking math (size, overlap, offset)
and only rebases each chunk's segment timestamps to start at 0, matching the
relative `duration`/clamp the rest of the pipeline already assumes.
`install()` monkeypatches the single `chunk_transcript` attribute on
`shorts_generator.highlights`, same one-function-patch pattern as
`clipper_stable.py` — `call_highlight_api`, `dedupe_highlights`, and
`get_highlights` itself stay unmodified vendor logic.

Tests in `tests/test_highlights_chunking.py`: one reproduces the bug against
the *unpatched* vendor function directly (asserts it raises `RuntimeError`),
one confirms `install()` fixes it end-to-end through the real
`get_highlights` (highlights from a later chunk survive and land at the
correct absolute time), plus an `install()`-patches-vendor test and a
rebasing-math test. 53 tests passing total, `black`/`ruff` clean.

Committed on `fix/highlights-chunk-timestamps`, branched off `main`.
Not yet merged — this override still needs to be wired into whatever calls
`get_highlights` once `processor.py` exists (see "Next up"); no such call
site exists in this repo yet to add it to today.

## get_highlights giant-highlight collapse bug root-caused and fixed (2026-07-21)

Picked up the "separate, unfixed issue" logged above (TRK 17): on the 7-min
`SkxfKZgy9kw` video, `get_highlights` sometimes returned a single highlight
spanning almost the entire video instead of several short clips. Not the
chunking bug — this video is well under the 1800s chunking threshold, so
`chunk_transcript` never runs; confirmed as a distinct root cause.

**Reproduced directly against the real transcript and a live Ollama call**
(re-parsed `output/source_SkxfKZgy9kw.srt` into a transcript dict, called
`call_highlight_api` against it 3 times). Result varied run to run
(temperature 0.7): one run returned 4 well-formed short highlights (23-71s
each); another returned 4 highlights where most spanned **226-419 seconds
each** — i.e. Atlas-Chat-9B does not reliably follow
`HIGHLIGHT_SYSTEM_PROMPT`'s own stated "20-180s" duration bounds. Traced the
actual collapse mechanism: `_sanitize_highlights` never checks duration at
all (only `start >= 0`, `end > start`, clamp to video length), so an
oversized highlight sails through untouched; `dedupe_highlights` then keeps
whichever *overlapping* highlight has the highest score — a video-spanning
highlight overlaps essentially every other candidate, so if it scores
highest (or ties), every well-formed short highlight gets evicted as
"overlapping," collapsing the result down to that one giant clip.

**Not a deterministic code bug like the chunking one** — a local 9B model
won't reliably obey a prose duration instruction — so the fix stops
trusting the model on duration and enforces the prompt's own stated bounds
in code instead: new `darija_overrides/highlights_duration_filter.py`
monkeypatches only `_sanitize_highlights` to additionally drop any
highlight outside `[20, 180]` seconds, before `dedupe_highlights` ever sees
it. If every candidate from one attempt is out of bounds, `call_highlight_api`
already treats an empty list the same as "no valid highlights" and retries
— no extra plumbing needed. `dedupe_highlights` and `get_highlights` itself
stay unmodified vendor logic.

Tests in `tests/test_highlights_duration_filter.py`: reproduces the exact
captured real-world scenario (one giant highlight + 3 well-formed short
ones, unpatched vendor collapses to just the giant one; patched keeps the
3 short ones), an `install()`-patches-vendor test, and a parametrized
duration-bounds unit test. 61 tests passing total, `black`/`ruff` clean.

Committed on `fix/highlights-giant-clip-collapse`, branched off `main`.
Not yet merged.

## processor.py built — first orchestration stage, wires in all overrides (2026-07-21)

Built the next architecture-doc component (§3.4 pipeline detail): `processor.py`
wraps vendor's `generate_shorts(mode="local")` and adds caption burn-in on
top, per CLAUDE.md's "call `generate_shorts(...)` rather than reimplementing
its flow."

**Real design conflict, resolved with the user before writing code:** the
architecture doc's §3.5/§4 pipeline diagram puts scene detection *before*
cropping — it's meant to snap each highlight's start/end onto the nearest
real scene cut so a clip doesn't start/end mid-shot. But `generate_shorts`
crops internally with no hook to inject that, and CLAUDE.md forbids
reimplementing its flow to add one. Confirmed the fix: since
`pipeline._run_local` re-resolves `crop_highlights_local` via a fresh
`from .local.clipper import crop_highlights_local` on every call (a plain
module-attribute lookup, not a bound closure — verified this directly with
a throwaway monkeypatch-after-import test before relying on it), a new
override can monkeypatch that one attribute and snap boundaries onto scene
cuts before delegating to the real crop — satisfies both docs at once, same
single-function-patch pattern as every other override in this repo.

**New: `darija_overrides/scene_snap_crop.py`.** Uses `PySceneDetect`
(`scenedetect` package, added to `requirements.txt`) to find scene-cut
timestamps in the source video, then snaps each highlight's `start_time`/
`end_time` to the nearest cut within `MAX_SNAP_DISTANCE_SECONDS = 2.0` —
close cuts are trusted as "the same edit, just imprecise"; anything farther
is left alone rather than snapping a highlight onto an unrelated cut. Falls
back to unsnapped boundaries if scene detection itself fails or if snapping
would collapse the window. 7 tests, all with `_original_crop_highlights_local`
faked (never invokes real scene detection/ffmpeg in the suite).

**New: `processor.py`.** `install_overrides()` installs all six
`darija_overrides` patches (`llm_ollama`, `transcriber_darija`,
`clipper_stable`, `highlights_chunking`, `highlights_duration_filter`,
`scene_snap_crop`) before calling `generate_shorts`; `process_video(video_id, ...)`
then hands each rendered short + the full transcript to
`captioner.burn_captions`, skipping captioning for any clip whose crop
already failed. Writes `state.db`: `source_videos.status` goes
`queued → processing → captioned` (or `failed`), and a `clips` row is
written per rendered short (`clip_id = "{video_id}_{index}"`,
`status = 'pending_qc'` once captioned — ready for `qc_gate.py` to pick up,
`'failed'` otherwise). `source_videos.downloaded_at` is set once the whole
run succeeds, not right after the internal download step — `generate_shorts`
doesn't expose per-step timestamps, documented as a known granularity limit
in the code.

**`db.py` schema extended** with the `clips` table from architecture doc
§7 (`clip_id`, `source_video_id`, `score`, `status`, `posted_video_id`) plus
`title`/`clip_path`/`created_at` for what `processor.py` and later
`qc_gate.py`/`publisher.py` actually need to read/write.

Tests (`tests/test_processor.py`, per CLAUDE.md's spec for this stage):
`generate_shorts` and `captioner.burn_captions` mocked, verifying the
hand-off between them (right args, right order), output format (duration =
`end_time - start_time`, captioned-path presence), `clips`/`source_videos`
row content, and the failure path (`generate_shorts` raising marks
`source_videos.status = 'failed'` and re-raises rather than swallowing).
74 tests passing total, `black`/`ruff` clean.

Verified `processor.py` imports cleanly and `install_overrides()` actually
patches all three previously-separate vendor modules
(`shorts_generator.local.clipper`, `shorts_generator.highlights` ×2) by
inspecting live identity after calling it — not yet run end-to-end against
a real video (that's the top "Next up" item).

Committed on `stage/processor`, branched off `main`. Not yet merged.

## processor.py's first real end-to-end run — found and fixed a third JSON quirk (2026-07-21)

Ran `python processor.py SkxfKZgy9kw --num-clips 3` for real (cached
download + transcript, live Ollama call, live PySceneDetect, live
ffmpeg/OpenCV crop, live caption burn-in) — the first non-mocked run of
`processor.py`.

**First attempt failed**, `RuntimeError: Highlight generator produced
invalid output after 3 attempts: Expecting ',' delimiter`. Confirmed
`source_videos.status` correctly landed on `'failed'` and the exception
wasn't swallowed — the processor's own error handling worked as designed.

**Root-caused the actual JSON bug** by capturing raw Ollama responses
directly (bypassing `llm_ollama.py`'s existing fixes) across ~15 real
calls against the same transcript: Atlas-Chat-9B sometimes stops
generating mid-string and Ollama reports `done_reason: "stop"` (not
`"length"`) — this is the model ending its own generation early, not an
HTTP/token-limit cutoff, so there's no missing text to recover, only
whatever came before the cutoff. Distinct from the two already-fixed
quirks (Arabic comma, trailing comma) — about 2/5 real attempts hit this
in one sampling run, explaining why all 3 of `call_highlight_api`'s
retries can plausibly fail back-to-back.

Also surfaced (once, in the same sampling) a *third* shape: the model
repeating `"highlights":[...]` as several sibling keys instead of one
array with several elements — worth guarding against separately, since
naively concatenating text across that boundary would produce JSON that
`json.loads` accepts but silently resolves by keeping only the *last*
duplicate key's value, dropping the earlier (valid) ones with no error at
all.

**Fix, in `llm_ollama.py`** (same file/pattern as the other two quirks):
new `_salvage_truncated_highlights` scans for the last `}` that closes a
complete element directly inside the top-level `"highlights"` array, then
truncates there and re-closes `]}` — recovering whatever complete
highlights came before a mid-string cutoff. Stops immediately if the array
closes cleanly (handles the repeated-key shape safely: only the genuinely
complete first array survives, nothing gets silently overwritten by a
later duplicate key). No-op for already-well-formed responses and for the
small content-type-detection response (no `[` at all). Chained before the
existing two fixes in `call_local_llm`.

**Re-ran `processor.py` against the same video after the fix — succeeded
end-to-end.** 3 clips produced: `output/short_0{1,2,3}.captioned.mp4`,
9:16 (404×720), durations 23.6s/23.9s/56.1s (all within the 20-180s bounds
from the duration-filter fix), `state.db` has `source_videos.status =
'captioned'` and 3 `clips` rows at `status = 'pending_qc'`. Pulled one
frame (`output/short_01.captioned.mp4` @ 5s) and visually confirmed Arabic
captions render correctly — right-to-left, properly shaped/connected
script, not reversed or disconnected — a positive sign on the long-open
RTL rendering question, though this is one frame, not a full verification
pass.

9 new tests in `tests/test_llm_ollama.py` (unit tests on the salvage
function including the repeated-key case, plus an integration test through
`call_local_llm`'s full chain) — all built from real captured Ollama
output, not synthetic guesses. 80 tests passing total, `black`/`ruff`
clean.

Committed on `fix/llm-json-truncation`, branched off `main`. Not yet
merged.

## qc_gate.py built — QC gate, next real gate after processor.py (2026-07-22)

Built the next architecture-doc component (§3.8/§5): `qc_gate.py`, the last
checkpoint before a clip is eligible for `publisher.py`. Reads `clips` rows
at `status IN ('pending_qc', 'held')` and runs each through, in order: dedup,
score threshold, format validation, source-diversity throttle, then a daily
cap — exactly the doc's flowchart plus its sequence diagram's "filter to top
10/day" step, which the doc leaves ambiguous as to which component owns it.

**Resolved three real ambiguities with the user before writing code** (per
CLAUDE.md's "ask rather than guess on QC thresholds"):
- **Dedup = content-fingerprint match** against already-`posted` clips, not
  same-source-video identity — catches a re-uploaded/re-shared video with the
  same underlying content, not just literal reprocessing of the same
  `video_id`.
- **Score threshold = 60** (vendor's highlight scoring is 0–100, confirmed in
  `highlights.py:154`).
- **Source-diversity cap = 2 clips per source video** per batch (3rd+ is
  `held`, not rejected — matches the flowchart's distinct HOLD outcome).
- **`qc_gate.py` itself owns the daily cap (10/day)**, sorting eligible
  survivors by score; `publisher.py` will stay a dumb consumer of
  `status='queued'`.

**`db.py` schema migration, additive only:** `clips` gains `fingerprint`
(content hash) and `qc_reason` (human-readable reject/hold reason, for the
future `reporter.py`'s "QC rejections and why"). New `_ensure_clip_columns`
runs `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` for whichever columns are
missing, called from `get_connection()` — migrates the real, already-populated
`state.db` in place rather than wiping it.

**`processor.py` now computes each clip's fingerprint at record time,** in
`_caption_short`: hashes the clip's own transcript text (via
`captioner._segments_for_window`, reused rather than re-windowing), not the
LLM-generated title/hook_sentence — those vary run to run at Ollama's
temperature 0.7, so hashing them would fail to recognize the same content
across a reprocessing run.

**`qc_gate.py` design:** `run_qc_gate(conn=None) -> dict` (same
connection-injectable shape as `processor.process_video`) plus an `argparse`
`__main__`. Per-clip checks (dedup → score → format) are independent and
short-circuit on first failure; format validation checks the `.captioned.mp4`
filename suffix (cheap, no ffprobe needed) plus one `ffprobe` call
(`_probe_clip`) for duration/aspect-ratio, same subprocess-call pattern as
`captioner._probe_dimensions`. Clips passing all three go through a
source-diversity pass (score-desc order, `SOURCE_DIVERSITY_CAP` per
`source_video_id`), then a daily-cap pass (score-desc, top `DAILY_CLIP_CAP`
queued). All five thresholds are hardcoded module-level constants, matching
this repo's existing convention for other pipeline thresholds
(`highlights_duration_filter.py`'s [20,180]s bounds, `scene_snap_crop.py`'s
snap distance) rather than reading `config/channels.yaml` — no other
threshold in the codebase reads from that file today, flagged as following
established repo convention over the doc's folder-structure comment.

No bypass flag/parameter anywhere in the file (hard constraint, verified by
inspection — CLAUDE.md is explicit that this can't be added "for testing").

Tests in `tests/test_qc_gate.py`: one passing + one failing case per
architecture-doc branch, per CLAUDE.md's explicit requirement for this stage
— dedup (fingerprint match/mismatch), score threshold (above/below 60), each
format failure mode (missing `.captioned.mp4` suffix, duration ≥ 60s, wrong
aspect ratio) tested separately, source-diversity throttle (3 clips same
source → top 2 queued, 3rd held), daily cap (12 eligible clips across 12
sources → top 10 queued, 2 held), a `held` clip from a simulated previous run
being reconsidered, and a `rejected_*` clip staying terminal (not
reconsidered). `tests/test_processor.py` extended with one assertion that
`_record_clip` writes a non-null `fingerprint`. 93 tests passing total,
`black`/`ruff` clean.

**Dry-run verified against the real `state.db`** from the earlier
`SkxfKZgy9kw` end-to-end run (3 real clips, durations 23.6s/23.9s/56.1s, all
`.captioned.mp4`, 9:16, scores 90/80/75): the two higher-scoring clips
correctly queued, the third correctly held on the source-diversity cap (same
`source_video_id`, cap is 2) with reason `"source diversity cap (2) reached
for SkxfKZgy9kw"` — first real exercise of this stage against genuine
pipeline output, not just mocked tests.

Committed on `stage/qc-gate`, merged to `main` (`11416d0`). Notion: TRK-20
(already existed as a Backlog Tracker item — updated rather than duplicated)
→ Done; added a Docs entry (Category: Decision) recording the four threshold
resolutions above, and a Changelog entry for the merge.

## Clip output-path collision bug fixed (2026-07-23)

Found via a full-pipeline test run: `pipeline.py`'s `_run_local` calls
vendor's `crop_highlights_local(source_path, highlights, aspect_ratio,
out_dir=None)` without ever passing `out_dir`, so every processed video's
clips landed at the same shared `output/short_01/02/03(.captioned).mp4`
filenames — processing a second video silently overwrote the first video's
clip files on disk, corrupting any `clips` DB rows (including already-
`queued` ones) still pointing at those paths.

Since `darija_overrides/scene_snap_crop.py` is already the sole owner of
the `crop_highlights_local` monkeypatch (vendor's plain attribute
assignment means only one override can own a given function — a second
independent patch would just clobber this one depending on install order),
the fix lives there rather than as a separate override: `_video_out_dir`
derives a per-video output dir from `source_path`'s own filename, which
`local/downloader.py` always names `source_{video_id}.ext` — so different
videos never collide. `crop_highlights_local_snapped` now defaults to it
whenever the caller doesn't pass `out_dir` explicitly.

4 new tests in `tests/test_scene_snap_crop.py` (filename parsing, fallback
for non-matching filenames, default-vs-explicit `out_dir` on the patched
crop function). 97 tests passing total, `black`/`ruff` clean on our own
code (vendor/ has pre-existing lint noise, untouched third-party code, not
ours to fix).

Committed on `fix/clip-output-collision`, branched off `main`. Merged to
`main`.

**Real two-video regression check (2026-07-23, post-merge):** ran
`processor.py` back-to-back against two different cached videos
(`SkxfKZgy9kw`, 7-min, then `CadW5Vyh-hg`, 21-min — both already
downloaded/transcribed from earlier sessions, both under the 30-min
chunking threshold), `--num-clips 1` each, to reproduce the actual
collision scenario end-to-end (live Ollama scoring, live PySceneDetect,
live ffmpeg/OpenCV crop, live caption burn-in).

Result: **no collision.** `output/SkxfKZgy9kw/short_01(.captioned).mp4` and
`output/CadW5Vyh-hg/short_01(.captioned).mp4` landed in their own
per-video dirs; sha256 of video 1's clip files was identical before and
after video 2 ran, confirming video 2 never touched video 1's output.
`state.db` shows both `source_videos` rows at `status='captioned'` and
correct distinct `clip_path`s per video. The pre-fix collision artifacts
(loose `output/short_01/02/03(.captioned).mp4` from an earlier run, and
`clips` row `SkxfKZgy9kw_03` still pointing at `output/short_03.captioned.mp4`,
already `rejected_format` from a prior QC pass) were left in place as-is,
untouched by either run — not retroactively cleaned up, just no longer
reachable by the fixed code path.

Along the way, `CadW5Vyh-hg`'s first attempt hit the already-documented
Ollama/Atlas-Chat-9B flakiness (`RuntimeError: ... invalid output after 3
attempts`, one attempt timed out) — unrelated to this fix, failed clean
(`source_videos.status` correctly set to `'failed'`, no `CadW5Vyh-hg`
output dir created since crop was never reached, video 1's files
untouched), and succeeded on a plain retry against the same cached
transcript.

## Plan of record (per architecture doc)

Vendor `SamurAIGPT/AI-Youtube-Shorts-Generator` into `vendor/` (done, as a
submodule) for download/transcribe/score/crop, patched only via
`darija_overrides/`. Build net-new: `watcher.py` (done), scene detection
(done, via `darija_overrides/scene_snap_crop.py`), caption burn-in (done,
`captioner.py`), `qc_gate.py` (done). Still open: `publisher.py`,
`reporter.py`, scheduler.

## Test run (2026-07-12)

Verified end-to-end against 3 real, high-volume channels (Google for
Developers, MrBeast, Veritasium — not real sources, just volume for
testing). Confirmed: RSS fetch/parse works, 45 videos inserted into
`state.db`, second run correctly queued 0 new (dedup works). Full results
in `reports/watcher_test_2026-07-12.md`. `state.db` has since been wiped
and `config/channels.yaml` reset to a placeholder — watcher is untested
against real production sources.

## Long-video (≥30min) full pipeline re-test + chunk-resilience fix (2026-08-02)

Picked up "Re-run the full pipeline against a video ≥30 min now that the
chunking bug is fixed" (was blocked on this since 2026-07-23). Ran
`processor.py Zhj07EXj4HY` (the 42-min video that originally crashed the
machine, per the 2026-07-18 section) — first time this video has gone through
the *full* pipeline (transcribe → highlight scoring → scene-snapped crop →
caption burn-in), not just the transcribe-only memory test from before.

**First two attempts failed on chunk 2/3** with two different JSON errors
(`no valid highlights in response`, then `Extra data: line 8 column 6`) —
looked at first like the usual Atlas-Chat-9B JSON flakiness already covered
by `llm_ollama.py`'s fixes. Reproduced directly by capturing 5 raw Ollama
responses for chunk 2's exact prompt: **all 5 came back as plain conversational
Darija** ("سمح ليا، ما فهمتش شنو باغي تقول" — "sorry, I didn't understand what
you're asking"), not malformed JSON. Ruled out a prompt-structure bug by
re-testing with Ollama's `system` field properly separated from the
transcript (`call_local_llm` currently flattens everything into one
`/api/generate` `prompt` with no `system` field) — still 5/5 conversational.
**Conclusion: genuine Atlas-Chat-9B capability limit on this chunk's content,
not a parseable-JSON bug** — nothing to salvage, unlike the three previously
fixed quirks (Arabic comma, trailing comma, mid-string truncation).

Per CLAUDE.md's "ask rather than guess on Darija model choice," raised this
with the user rather than picking a mitigation silently. Chose: make the
pipeline resilient to a single bad chunk rather than trying to prevent the
model from ever failing (also considered: smaller chunk size, lower
temperature, just log-and-move-on — deferred, can revisit if this recurs
often in production).

**Fix: new `darija_overrides/highlights_chunk_resilience.py`.** Vendored
`get_highlights`'s long-video branch calls `call_highlight_api` per chunk
with no try/except — one chunk exhausting its 3 retries raises `RuntimeError`
straight out of the loop and aborts highlights for the *entire* video, even
chunks that succeeded. This patches `get_highlights`, catching a per-chunk
`RuntimeError` and continuing with the remaining chunks; only raises if
*every* chunk fails. **Patched at `shorts_generator.pipeline.get_highlights`,
not `shorts_generator.highlights.get_highlights`** — pipeline.py does
`from .highlights import get_highlights` at module import time (an
early-bound reference), so patching the source module's attribute (the trick
that works for `chunk_transcript`, referenced by bare name *within* that same
module) wouldn't be seen by pipeline.py's already-bound name. This is a new
wrinkle in the override pattern worth remembering for any future
`get_highlights`-adjacent patch. Wired into `processor.py`'s
`install_overrides()`.

4 new tests in `tests/test_highlights_chunk_resilience.py`: one bad chunk
skipped while others still contribute, all-chunks-failing still raises,
`install()` patches the correct module (and confirms it does *not* touch
`shorts_generator.highlights.get_highlights`), short-video path (<30min)
delegates untouched to vendor's original `get_highlights`. 122 tests passing
total (one pre-existing, unrelated failure in `test_publisher.py` — hardcodes
`posted_at: "2026-07-26"` to test same-day quota logic, now fails simply
because today's date has moved past that; not touched by this fix),
`black`/`ruff` clean.

**Re-ran end-to-end with the fix installed — succeeded.** Chunk 2/3 failed
again (same content, same underlying model limit) and was cleanly skipped;
chunks 1 and 3 produced 2 real clips
(`output/Zhj07EXj4HY/short_0{1,2}.captioned.mp4`), `state.db` shows
`source_videos.status = 'captioned'` and both `clips` rows at
`status = 'pending_qc'`. Closes the "re-run against ≥30min video" item.

**Also visually re-confirmed Arabic/RTL caption rendering** on a real Darija
frame from this run (`short_01.captioned.mp4`) — right-to-left, correctly
shaped/joined script, yellow per-line keyword highlight rendering as
designed. First RTL check against real Darija content through the *current*
caption style (the modern-highlight/word-sync rework happened after the last
RTL check, which was also on English test audio only). Closes the long-open
"visually verify RTL rendering" item.

Committed on `fix/highlights-chunk-resilience`, branched off `main`. Not yet
merged.

## Channel branding: "3la Rassi" identity + overlay burn-in + tags (2026-08-02)

User reprioritized: branding/overlay/channel identity before `reporter.py` +
scheduler, since unbranded clips aren't really shippable content yet. Went
through a design pass together (not guessed silently, per CLAUDE.md — this
touches product identity, not just code):

- **Name/identity: "3la Rassi"** (على راسي — colloquial Darija for
  "trust me" / "for sure"), chosen over two other concepts ("Wach Clips",
  "Darija Dose"). Niche: general Darija commentary/reactions, not
  sports-only (broader than the test channel used so far).
- **Palette:** sunset orange `#FF6B35` + deep purple `#2D1B4E` + cream
  `#F7F3E9`. **Logo v1** (starburst badge + literal head silhouette) was
  rejected as dated on user review; **v2** (flat diagonal-gradient squircle +
  bold "3" monogram, Avenir Next Heavy) was approved. Assets + their
  generator live in `assets/brand/` (`generate.py` — one-off Pillow script,
  not part of the runtime pipeline; Pillow added to `requirements.txt` for
  that reason only).
- **Channel target confirmed:** this is the *same* channel already
  authorized in `config/youtube_token.json` (the one that posted the
  earlier test clip `ffEbx3lEPJk`) — user rebranded it in YouTube Studio
  rather than creating a new one. No publisher.py changes needed for this:
  `videos.insert` always uploads to whichever channel the OAuth token
  grants, implicitly, there's no separate channel-ID parameter anywhere in
  this pipeline. `config/channel_profile.yaml` (new) is the single source of
  truth for name/tagline/hashtags/keywords — `publisher.py` reads it in
  `upload_clip` (new `channel_profile` param, defaults to loading the file)
  to build description/tags; a missing/empty profile falls back to the
  original bare `"#Shorts"` behavior, so this was never a hard dependency.

**`captioner.py`:** `burn_captions` now burns the brand border (ffmpeg
`drawbox`) and, if `assets/brand/overlay_ribbon.png` exists, the logo pill
(ffmpeg `overlay`) in the *same* ffmpeg call as the caption `ass` filter —
one re-encode, not three. `logo_path` param defaults to the brand asset,
pass `None` to skip (border still burns in either way).

Real end-to-end verification: burned a real frame and visually confirmed
border + logo pill + Arabic captions all render together correctly. Then
ran the *full* `processor.py Zhj07EXj4HY` pipeline twice more to get a real
branded, correctly-captioned output (first of these two attempts failed
outright — see below) — final result: 3 real clips, `.captioned.mp4` files
all show the border/logo/captions together correctly (frame-checked).

**Also surfaced: Atlas-Chat-9B flakiness on this video is worse than
previously measured.** Across 5 real `processor.py Zhj07EXj4HY` runs total
today (2 before the branding work, 3 during it), only 2 fully or partially
succeeded — one run failed **all 3 of 3 chunks** (previously only ever saw
1 chunk fail at a time). The chunk-resilience fix (previous section) is
doing its job — no hard crash, clean `RuntimeError` with a clear message,
`source_videos.status` correctly lands on `'failed'` — but a ~1-in-5 total
video success rate is a real production concern once the scheduler is
running unattended. Not investigated further today (out of scope for the
branding work) — worth picking up before or alongside the scheduler ticket
so failed videos don't silently pile up. Logged as a new Tracker item.

4 new tests in `tests/test_captioner.py` (border-only path, logo+overlay
path via `-filter_complex`, missing-logo-file fallback), 4 new tests in
`tests/test_publisher.py` (tagline/hashtags/tags wiring, empty-profile
fallback, missing-file config load, real-config load). 132 tests passing
total (1 pre-existing, unrelated failure — `test_publisher.py`'s same-day
quota test hardcodes `posted_at: "2026-07-26"`, now fails simply because
today's date has moved past that; flagged twice before, not fixed today,
out of scope), `black`/`ruff` clean.

## Atlas-Chat-9B flakiness investigation: chunk size / temperature ruled out, retries bumped instead (2026-08-02)

Picked up the new high-priority Tracker item from the branding session (1-in-5
full-video success rate observed against `Zhj07EXj4HY`). Ran a controlled
sweep: single-attempt success rate across `CHUNK_SIZE_SECONDS` ∈ {1200s
(current), 600s} × temperature ∈ {0.7 (current), 0.2}, against the same real
cached transcript, through the actual production JSON-repair chain
(`llm_ollama.call_local_llm`'s three fixes — first pass of this sweep
accidentally called Ollama's raw API directly, bypassing those fixes, and
got a suspicious 0/16; re-ran through the real fix chain before trusting any
result).

**Neither lever helped:** 1200s/0.7 → 1/3, 600s/0.7 → 1/5, 1200s/0.2 → 1/3,
600s/0.2 → 0/4 (timed out on the 5th, itself a real already-handled failure
mode). Smaller chunks were no better (arguably worse — more, smaller chances
to fail); lower temperature made no measurable difference. Small sample
(13-15 calls), so treat exact percentages as directional — but this
contradicts the original theory that one specific 20-minute span's *content*
was uniquely triggering the conversational drift. Re-slicing the same video
into different chunk boundaries still fails at a similar rate, so this reads
as a general ~1-in-3 per-attempt reliability ceiling for Atlas-Chat-9B on
this content, not something tied to one bad chunk or fixable by prompt
restructuring alone.

**Fix applied instead:** bumped `MAX_HIGHLIGHT_API_ATTEMPTS` 3 → 5 via
`darija_overrides/highlights_chunk_resilience.py` (same file as the earlier
chunk-skip fix — thematically both are "survive highlight-scoring
flakiness"). Same single-attribute monkeypatch pattern as
`chunk_transcript`/`CHUNK_SIZE_SECONDS`: `call_highlight_api` reads
`MAX_HIGHLIGHT_API_ATTEMPTS` as a bare name from its own module's globals at
call time, so patching `shorts_generator.highlights.MAX_HIGHLIGHT_API_ATTEMPTS`
is enough, no vendor edit. At the observed ~33% per-attempt rate, this raises
per-chunk success from ~70% (3 attempts) to ~87% (5 attempts) — the cheapest
lever available given chunk size/temperature didn't move anything, at the
cost of more Ollama time on chunks that are already failing.

New test (`test_install_bumps_max_highlight_api_attempts`) confirms `install()`
patches the constant. A more elaborate integration-style test (simulating a
chunk that only succeeds on the 4th attempt) was started but had a data bug
(mismatched fake `duration` vs. segment timestamps causing a legitimate
highlight to get clamped out) — removed rather than fixed, per direct user
instruction to stop debugging it; the simpler constant-patch test already
covers the actual change. 126 tests passing total (same one pre-existing,
unrelated `test_publisher.py` date-flake), `black`/`ruff` clean.

Not re-verified end-to-end against a real video after this change (that's
naturally slow to confirm given the flakiness itself) — the next real
`processor.py` run against a long video will be the real test.

## Next up

To ship V0 per the architecture doc, in priority order:

- [ ] Get more real Darija source channel IDs from user (only one channel
      in `config/channels.yaml` so far) and re-run `watcher.py` — a scheduler
      is pointless with one source
- [x] Investigate the Atlas-Chat-9B highlight-scoring flakiness rate —
      chunk size / temperature sweep ruled both out as levers; mitigated by
      bumping retries 3→5 instead (see 2026-08-02 section above). Not a full
      fix — real success rate under sustained scheduler use still unverified,
      worth revisiting if failed videos start piling up in practice.
- [ ] `reporter.py` — daily report generation from `state.db`
- [ ] Scheduler — wire the full pipeline into `cron`/`launchd`

Done, not yet in V0 scope but worth doing eventually:

- [ ] Extend `highlights.py`'s system prompt with Darija/code-switch
      few-shot examples (still using the vendored file's default English
      framing today)
- [ ] Face-tracking crop smoothing beyond the debounce fix — reconfirmed
      with real footage 2026-08-02, still visible jiggle
- [ ] Video editing polish (saturation, YouTube description/tags beyond the
      defaults in `config/channel_profile.yaml`) — logo/overlay/branding
      itself shipped 2026-08-02, this is further polish on top

## Open questions

- (Resolved 2026-08-02) Darija/RTL caption rendering correctness — visually
  confirmed correct on real Darija content, see the 2026-08-02 section above.
