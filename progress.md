# Progress

## Status: watcher merged, base repo vendored, on branch `stage/vendor-base`

- `main`: watcher stage merged (`243ad1d`). Has `db.py`, `config/channels.yaml`,
  `watcher.py`, tests, `requirements.txt` + local `.venv`.
- `watcher.run()` fetches each configured channel's RSS feed, diffs against
  `source_videos`, inserts new rows with `status='queued'`. Not yet run
  against a real channel — one real channel ID now in
  `config/channels.yaml` (`UCkax8bjMiSlC05JeXZSUKaQ`), needs a re-run to
  verify + more Darija sources still needed.
- `stage/vendor-base` (current, not yet merged): vendored
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

- [ ] Merge/review `stage/vendor-base` into `main`
- [ ] Confirm the rest of the real Darija source channel IDs from user
      (only one channel ID in `config/channels.yaml` so far) and re-run
      `watcher.py` against them
- [ ] `darija_overrides/transcriber_darija.py` — swap in Darija fine-tune
- [ ] `darija_overrides/llm_ollama.py` — swap OpenAI call for Ollama; needs
      to shadow-import over `shorts_generator.local.llm` (see note above),
      not just pass a different callback
- [ ] `processor.py` — orchestrate vendor's `generate_shorts(mode="local")`
      + scene detection + caption burn-in
- [ ] `qc_gate.py`, `publisher.py`, `reporter.py`, scheduler

## Open questions

- None currently blocking.
