# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

This repo implements the Darija YouTube Shorts Automation pipeline described in
`darija-shorts-automation-architecture.md` — read that file first, it is the
source of truth for system design, component responsibilities, data flow,
and the DB schema. If anything in this file conflicts with that architecture
doc, ask before proceeding rather than picking one silently.

## Base repository — do not reinvent this

We do **not** build the download/transcribe/score/crop core from scratch.
That logic is forked/vendored from
**[SamurAIGPT/AI-Youtube-Shorts-Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator)**
(MIT license) at `vendor/ai-youtube-shorts-generator/`. See architecture doc
§2 for the full reuse table. In short:

- **Reuse as-is:** `local/downloader.py` (yt-dlp), `local/clipper.py`
  (ffmpeg + OpenCV crop), `pipeline.py` (orchestration — call
  `generate_shorts(...)` rather than reimplementing its flow).
- **Reuse with modification:** `local/transcriber.py` (swap model to a Darija
  fine-tune), `local/llm.py` (swap OpenAI client for Ollama), `highlights.py`
  (extend the system prompt for Darija).
- **Net new, not in the base repo at all:** channel watcher, scene detection,
  caption burn-in, QC gate, publisher, reporter, scheduler.

**Do not edit the vendored repo in place.** Apply modifications through
`darija_overrides/` (model/backend swaps that get imported in place of the
base repo's originals) so the vendored code stays a clean, updatable fork —
if `git pull`-ing upstream changes later, in-place edits would conflict.
If a required change genuinely can't be done as an override (rare), say so
explicitly before editing vendored files directly.

## Environment

- macOS, Apple Silicon (M1 Pro). The vendored base repo's local mode uses
  `faster-whisper`, which supports CPU/CUDA but not Metal directly — verify
  actual inference speed on M1 Pro before assuming it needs replacing; only
  swap the transcription backend if it's genuinely too slow, not preemptively.
- Python is the primary language for pipeline scripts unless a component's
  upstream tool is CLI-only (e.g. `yt-dlp`, `ffmpeg`).
- No Docker requirement — this runs as native processes scheduled via
  `launchd`/`cron`, per the architecture doc.

## Hard constraints — do not violate

- **Fully free / local only.** No paid APIs, no paid SaaS, no cloud LLM calls
  for any pipeline step (transcription, highlight scoring, cropping,
  captioning). This includes the base repo's default local-mode LLM call to
  OpenAI — that must stay swapped to Ollama, never reverted. The only
  external network calls in normal operation are: YouTube's public RSS feed,
  `yt-dlp` downloads, and the YouTube Data/Analytics APIs for publishing and
  reporting.
- **Darija-first transcription.** Default transcription model is a Darija
  fine-tune (see architecture doc §3.3/§4). Do not silently swap in generic
  Whisper as the primary model — it's a documented fallback only, and any
  fallback event should be logged with the video/clip ID.
- **QC gate is non-negotiable.** No clip reaches the publisher without passing
  every check in architecture doc §5 (dedup, score threshold, format
  validation, source-diversity throttle). Do not add a bypass flag "for
  testing" that could be left on accidentally.
- **Respect YouTube quota.** Every call to `videos.insert` costs 1,600 units
  against a 10,000-unit daily budget. Do not write retry logic that could
  loop and burn quota; log and halt on repeated publish failures instead of
  silently retrying indefinitely.
- **State lives in SQLite (`state.db`).** Don't introduce another datastore
  (Redis, Postgres, etc.) without discussing it first — the architecture doc's
  scaling section flags when that tradeoff becomes worth it, and we're not
  there yet.

## Tech stack (per architecture doc §8)

| Step              | Tool                                               | Source                                  |
| ----------------- | -------------------------------------------------- | --------------------------------------- |
| Channel watch     | YouTube RSS feed                                   | New                                     |
| Download          | `yt-dlp`                                           | Base repo, reused as-is                 |
| Transcription     | `faster-whisper` + Darija fine-tune (Hugging Face) | Base repo, model swapped                |
| Highlight scoring | Ollama + Qwen2.5                                   | Base repo, backend swapped (was OpenAI) |
| Scene detection   | PySceneDetect                                      | New                                     |
| Vertical crop     | OpenCV + ffmpeg face tracking                      | Base repo, reused as-is                 |
| Captioning        | ffmpeg + ASS                                       | New                                     |
| Scheduling        | `cron` / `launchd`                                 | New                                     |
| Publishing        | YouTube Data API v3                                | New                                     |
| Analytics         | YouTube Analytics API                              | New                                     |

Don't substitute a different tool in this table without flagging why (e.g. "X
doesn't build on Apple Silicon, switching to Y") rather than swapping silently.

## Folder structure

Follow the layout in architecture doc §10 (`vendor/`, `darija_overrides/`,
`config/`, `raw/`, `clips/`, `reports/`, `state.db`, and one script per new
pipeline stage). Keep each new stage as a separate, independently runnable
script rather than one monolithic file — this matches how the scheduler
invokes them. Base-repo code stays under `vendor/`, untouched except through
`darija_overrides/`.

## Workflow expectations

- Before implementing a new component, check whether the architecture doc
  already specifies its interface (inputs/outputs, DB tables touched) and
  follow that rather than inventing a new shape.
- When a step in the architecture doc is ambiguous or underspecified for
  implementation, ask rather than guessing silently — especially for
  anything touching the QC thresholds, quota handling, or Darija model choice.
- Prefer small, testable scripts per pipeline stage over one large script.
- Log meaningfully at each stage transition (video queued → downloaded →
  transcribed → scored → cropped → captioned → QC result → published →
  reported) so the daily report in §3.10 can be built from real state, not
  re-derived after the fact.
- When editing `ffmpeg`/`faster-whisper` invocations, verify RTL Arabic
  caption rendering doesn't silently break — this is called out as a known
  failure point in the architecture doc.

## Testing requirements

- Each **new** pipeline stage script (`watcher.py`, `processor.py`,
  `qc_gate.py`, `publisher.py`, `reporter.py`) needs unit tests covering its
  core logic independent of external services:
  - `watcher`: RSS diff logic against a fixture feed + a seeded `state.db`.
  - `processor`: verifies `generate_shorts(...)` output is correctly handed
    to scene detection and caption burn-in; assert output format (9:16,
    duration, caption file presence) rather than eyeballing output.
  - `qc_gate`: every branch in architecture doc §5 (dedup, score threshold,
    format validation, source-diversity throttle) needs a passing and a
    failing test case.
  - `publisher`: quota accounting logic and retry/halt behavior, with the
    actual YouTube API call mocked — tests must never hit the real API.
  - `reporter`: report generation from a seeded `state.db` snapshot, checked
    against an expected markdown fixture.
- For `darija_overrides/` (the model/backend swaps on top of the base repo):
  test that `transcriber_darija.py` and `llm_ollama.py` honor the same
  input/output contract as the base repo's originals — that's what keeps them
  drop-in replacements rather than a fork of the whole pipeline.
- The vendored base repo (`vendor/ai-youtube-shorts-generator/`) is treated as
  a third-party dependency — don't write new unit tests for its internals;
  our tests cover our overrides and our new components only.
- Use `pytest`. New stage logic should not be merged without a matching test;
  if something is genuinely hard to test (e.g. actual transcription output
  quality), say so explicitly rather than skipping silently.
- Integration testing (full pipeline on a real short video) is a manual/local
  step, not part of the automated suite — keep it that way so CI-style runs
  stay fast and free of real API calls.

## Code style

- Format with `black`, lint with `ruff`. Run both before considering a change
  done; don't leave formatting/lint fixes for later.
- Type hints on all function signatures in pipeline scripts — this is a
  multi-stage system with a lot of hand-offs between scripts, and typed
  interfaces make those hand-offs easier to trust.
- Keep functions scoped to one responsibility per the component boundaries in
  the architecture doc (e.g. don't let `processor.py` quietly start doing QC
  gate work — that belongs in `qc_gate.py`).
- Docstrings on every public function: what it does, inputs, outputs, and
  which `state.db` tables it reads/writes.

## Git conventions

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
  `chore:` prefixes on every commit message.
- One pipeline stage (or one clearly-scoped fix) per commit — avoid bundling
  unrelated changes across multiple stage scripts in one commit.
- Branch naming: `stage/<name>` for new stage implementation (e.g.
  `stage/qc-gate`), `fix/<short-desc>` for bug fixes.
- No direct commits to `main` for anything beyond trivial doc typos — use a
  branch + review, even in a solo project, so the architecture doc stays the
  reference point for what "correct" looks like.
- Never commit secrets, tokens, or `.env` contents — double check before
  committing anything that touches `config/` or auth setup.
- Never commit with claude as a co-author

## Do not

- Do not reimplement download, transcription-flow, highlight-scoring-flow, or
  crop logic that already exists in the vendored base repo — extend or
  override it, don't duplicate it.
- Do not edit vendored base repo files in place — use `darija_overrides/`.
- Do not add analytics/telemetry calls to third-party services.
- Do not hardcode API keys, tokens, or channel IDs in scripts — use
  `config/channels.yaml` and environment variables / a local `.env` (gitignored).
- Do not commit `raw/`, `clips/`, `state.db`, `vendor/`, or any downloaded
  video content — `vendor/` should be a git submodule or fetched at setup
  time, not committed as a copy.
