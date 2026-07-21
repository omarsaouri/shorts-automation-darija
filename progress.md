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

- [ ] Merge `fix/highlights-giant-clip-collapse` into `main`
- [ ] Wire `darija_overrides.{highlights_chunking,highlights_duration_filter}.install()`
      into whatever calls `get_highlights` once `processor.py` exists — no
      call site exists in this repo yet
- [ ] Re-run the *full* pipeline against a video ≥30 min now that the
      chunking bug is fixed (previously blocked on this)
- [ ] Review the re-run output in `clips/KazZdpoVvio/` (3 captioned clips)
      — confirm captions are now readable phrase-sized cues, not walls of
      text, and that the face-tracking fix visibly holds up over the full
      40 minutes, not just the one window it was verified against
- [ ] **Visually verify `captioner.py`'s Arabic/RTL rendering** on this
      real Darija text — now have real transcript segments to check
      against, no longer just the English-only smoke test (open item,
      deliberately deferred until now)
- [ ] Get more real Darija source channel IDs from user (only one channel
      in `config/channels.yaml` so far) and re-run `watcher.py`
- [ ] Extend `highlights.py`'s system prompt with Darija/code-switch
      few-shot examples (still using the vendored file's default English
      framing today)
- [ ] `processor.py` — orchestrate vendor's `generate_shorts(mode="local")`
      + scene detection + `captioner.burn_captions(...)`; must add
      `vendor/ai-youtube-shorts-generator` to `sys.path` and call
      `darija_overrides.{llm_ollama,transcriber_darija,clipper_stable}.install()`
      before invoking it
- [ ] `qc_gate.py`, `publisher.py`, `reporter.py`, scheduler

## Open questions

- Darija/RTL caption rendering correctness — not yet visually verified,
  deliberately deferred per user direction (2026-07-13). Revisit once real
  Darija source content is available to test against.
