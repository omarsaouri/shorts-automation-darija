# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

This repo implements the Darija YouTube Shorts Automation pipeline described in
`docs/darija-shorts-automation-architecture.md` — read that file first, it is the
source of truth for system design, component responsibilities, data flow,
and the DB schema. If anything in this file conflicts with that architecture
doc, ask before proceeding rather than picking one silently.

## Base repository — do not reinvent this

We do **not** build the download/transcribe/score/crop core from scratch.
That logic is forked/vendored from
**[SamurAIGPT/AI-Youtube-Shorts-Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator)**
(MIT license) at `vendor/ai-youtube-shorts-generator/`. See architecture doc
§2 for the full reuse table. In short:

- **Reuse as-is:** `local/downloader.py` (yt-dlp), `pipeline.py`
  (orchestration — call `generate_shorts(mode="local")` rather than
  reimplementing its flow).
- **Reuse, behavior overridden at runtime via `src/darija_overrides/`:**
  `local/transcriber.py` (Darija fine-tune, faster-whisper as fallback
  only), `local/llm.py` (Ollama instead of OpenAI/Gemini), `local/clipper.py`
  (debounced face tracking + scene-cut snapping on top of the original
  ffmpeg+OpenCV crop), `highlights.py` (chunk-timestamp rebasing, duration
  bounds, per-chunk failure resilience on top of the original virality
  prompt/scoring). See each override module's docstring in
  `src/darija_overrides/` for the exact vendor function it patches and why.
  The MuAPI-backed paid-API half of the base repo (`clipper.py`,
  `downloader.py`, `transcriber.py` at the package root, `muapi.py`, and
  `main.py --mode api`) is still present in vendor but never reachable —
  `processor.py` always calls `generate_shorts(mode="local")` — and should
  be deleted outright as dead weight, not reused.
- **Net new, not in the base repo at all:** channel watcher, scene detection,
  caption burn-in, QC gate, publisher, reporter, scheduler.

`vendor/ai-youtube-shorts-generator/` is committed directly (no longer a git
submodule), but it stays a pristine, unedited copy of upstream — all
Darija-specific behavior (Ollama LLM swap, Darija transcriber, face-tracking
stability, scene-cut snapping, highlight chunking/duration/retry fixes)
lives in `src/darija_overrides/` as runtime monkeypatches, one module per
fix, each with an `install()` that's called from `processor.py` before
`generate_shorts(mode="local")` runs. This keeps the diff against upstream
at zero, so re-pulling a newer vendor version (if ever needed) can't
silently clobber our fixes. Editing the vendored files in place is
permitted if a fix genuinely can't be done as a monkeypatch, but prefer the
override layer.

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

Follow the layout in architecture doc §10 (`vendor/`, `config/`, `raw/`,
`clips/`, `reports/`, `state.db`, pipeline stage scripts under `src/`, and
docs under `docs/`). Keep each new stage as a separate, independently
runnable script rather than one monolithic file — this matches how the
scheduler invokes them.

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
- The vendored base repo (`vendor/ai-youtube-shorts-generator/`) is a
  pristine, untouched copy of upstream — don't write unit tests for it.
  `src/darija_overrides/` is ours to test: each module (`llm_ollama.py`,
  `transcriber_darija.py`, `clipper_stable.py`, `scene_snap_crop.py`,
  `highlights_chunking.py`, `highlights_duration_filter.py`,
  `highlights_chunk_resilience.py`) patches one specific vendor function, so
  its tests should verify both the patched behavior and that its
  `install()` correctly shadows/monkeypatches the vendor function it
  targets — keeping it drop-in-compatible with the original vendor contract
  rather than a silent fork of the whole pipeline's expectations.
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
- Do not add analytics/telemetry calls to third-party services.
- Do not hardcode API keys, tokens, or channel IDs in scripts — use
  `config/channels.yaml` and environment variables / a local `.env` (gitignored).
- Do not commit `raw/`, `clips/`, `state.db`, or any downloaded video content.
  `vendor/ai-youtube-shorts-generator/` is the exception — it's committed
  directly as owned code, not a submodule.

# Notion Workspace Guide

This file tells Claude Code how this Notion workspace is built. Read this first before touching any Notion data. It saves time and stops you from creating duplicate pages or guessing at property names.

Workspace name: OS
Owner: Omar Saouri

## The Big Picture

Everything lives under one page called Command Center. That page is the homepage. It shows what is active right now, what to focus on, and what has a deadline coming up. Below that it lists links to every database.

There are 6 databases total. Each one has a clear job. They are connected to each other through relations, so a project automatically shows its own bugs, docs, changelog entries, and session notes.

Command Center page ID: 3a355e5c-51a4-819a-934f-e0f99aa22582
Command Center URL: https://app.notion.com/p/3a355e5c51a4819a934fe0f99aa22582

## The 6 Databases

### 1. Inbox

Job: catch anything fast. Ideas, bug sightings, notes. No friction.
Database ID: 6f3f1793ab854e5dab3597693c01197d
Data source ID: 57eb1fbe-2309-4def-971b-b1c15a48f6c3

Properties:
Name (title)
Type: Idea, Bug sighting, Feature, Note
Triaged: checkbox
Captured: auto timestamp

Rule: new items start as Triaged = false. Once someone turns them into a real Tracker item or Project, mark Triaged = true.

### 2. Projects

Job: track anything bigger than a single task, organized by phase.
Database ID: b395a167f63744c181cae9fd007af96b
Data source ID: 4e702b5a-c5f8-409b-8c36-c3e7ba8d95f5

Properties:
Name (title)
Phase: Brainstorming, Planning, Building, Testing, Shipped, On Hold
Priority: High, Medium, Low
Started: date
Deadline: date
Tags: web, cli, api, mobile, experiment
Tracker Items: relation to Tracker
Docs: relation to Docs
Changelog: relation to Changelog
Sessions: relation to Claude Sessions

Rule: a project moves through Phase in this order. Brainstorming, then Planning, then Building, then Testing, then Shipped. On Hold can happen at any point.

### 3. Tracker

Job: hold every bug, feature, and improvement in one place.
Database ID: e2bf1e1247514711b5655202ffced993
Data source ID: bee0a9ef-acaf-4960-877e-06c6911229eb

Properties:
Name (title)
Type: Bug, Feature, Improvement
Status: New, Backlog, In Progress, Done
Priority: High, Medium, Low
Due: date
Project: relation to Projects
ID: auto number, shows as TRK 1, TRK 2, and so on

Rule: everything in here should link to a Project when possible. Items without a Project are fine too, they just will not show up on that project's page.

### 4. Docs

Job: hold guides, references, snippets, and decisions.
Database ID: 7cd1dbd6240a4d4887616a62a8dcad7c
Data source ID: 17a133fe-7a8b-4403-8298-bf136a599aba

Properties:
Name (title)
Category: Guide, Reference, Snippet, Decision
Project: relation to Projects
Updated: auto timestamp

### 5. Changelog

Job: log what actually shipped.
Database ID: b1ae0bb50dc244ffbe056f146ac4cafa
Data source ID: adf42bcf-3026-4305-849b-7ee7c09ff784

Properties:
Name (title)
Version: text
Date: date
Kind: Release, Fix, Tweak, Announcement
Project: relation to Projects

### 6. Claude Sessions

Job: keep a record of work done together with Claude.
Database ID: 9c1a4a24291a4cc38064dad616cd7e9b
Data source ID: d3d56158-69f9-4271-86ff-5edb2b1666a6

Properties:
Name (title)
Date: date
Focus: Brainstorm, Build, Debug, Design, Research
Project: relation to Projects
Key decisions: text
Outcome: text

Rule: when a work session ends and the person says "log this session", add one row here. Fill in Focus, Key decisions, and Outcome. Link the Project if there is one.

## Naming Convention

Every option, page, and view uses one emoji plus a plain word. For example, "Bug" uses a bug emoji, "Building" uses a hammer emoji, "Shipped" uses a rocket emoji. Keep this pattern when adding new options. Do not add options without an emoji, it will look out of place next to the others.

## How the Databases Connect

Projects is the center of the whole system. Tracker, Docs, Changelog, and Claude Sessions all have a two way relation back to Projects. This means:

If you link a Tracker item to a Project, that item shows up automatically on the Project page.
The same is true for Docs, Changelog entries, and Sessions.

So the normal flow looks like this. Something starts as a quick note in Inbox. Once it is real, it becomes either a Tracker item or a full Project. Work happens. A Claude Session gets logged. When it ships, a Changelog entry gets added.

## Common Tasks

Log a work session
Add a new row to Claude Sessions. Set Date to today. Pick a Focus. Write a short Key decisions note and a short Outcome note. Link the Project if the work was about one.

Add a bug or feature
Add a new row to Tracker. Set Type to Bug or Feature. Set Status to New. Link it to a Project if one exists.

Ship something
Add a new row to Changelog with today's date and a short description. Then go to the related Tracker items and set Status to Done. If a whole Project shipped, set its Phase to Shipped.

Add a diagram
Diagrams go straight into a Project page as a mermaid code block. Do not make a separate database for diagrams, just add the block to the page.

Set a deadline
Tracker items use the Due property. Projects use the Deadline property. Both show up automatically in the Ongoing Deadlines section on the Command Center homepage.

Triage the inbox
Go through Inbox items where Triaged is false. For each one, either turn it into a Tracker item, turn it into a Project, or just delete it if it is no longer useful. Then mark Triaged as true, or just delete the Inbox row once it has been turned into something else.

## Known Bug: Pages Get Moved To Trash

There is a known bug in the Notion connector. It can happen when a page gets edited with a full content rewrite. Read this before editing any page that has embedded databases, sub pages, or linked views inside it. The Command Center page and every database page in this workspace fall into that group.

What causes it
When a page gets updated using a full replace of its content, the connector does not update the page piece by piece. It deletes everything on the page first, then writes the new content back. During that short gap, any database or sub page that was living inside the page loses its parent. Notion treats anything without a parent as orphaned, and it sends orphaned items straight to trash. This can trash the child items, and in some cases the page itself.

How to avoid it
Never use a full content replace on a page that has embedded databases, sub pages, or views inside it. Only add content with an append style command, one that adds new blocks without touching the old ones. If a page truly needs to be reorganized, do it in small steps. Add the new piece first, check that it looks right, then remove the old piece in a separate step. Never do both at once.

If the tool asks for a flag before deleting something, stop and read exactly what it says will be deleted. Only approve it once you are sure it is not about to remove a database or sub page you want to keep.

If something does get trashed
It is not gone forever. Open the page in Notion and use Restore from the trash message. Then check that any views or relations tied to it still work.

## What Not To Do

Do not create a new top level database outside of these 6 without asking first.
Do not remove the emoji prefix from options.
Do not create a second Inbox, Tracker, or Projects database. Use the ones listed above.
Do not skip linking a Project relation when one clearly applies, it breaks the automatic rollups on the Project page.
Do not use a full content replace on the Command Center page or any database page. They all have embedded views inside them. Use an append style edit instead, see the Known Bug section above.
