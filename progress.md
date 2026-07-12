# Progress

## Status: not started — docs only

`CLAUDE.md` and `darija-shorts-automation-architecture.md` are in place. No
code, no `state.db`, no git repo yet.

## Plan of record (per architecture doc)

Vendor `SamurAIGPT/AI-Youtube-Shorts-Generator` into `vendor/` for
download/transcribe/score/crop, patched only via `darija_overrides/`. Build
net-new: `watcher.py`, scene detection, caption burn-in, `qc_gate.py`,
`publisher.py`, `reporter.py`, scheduler.

## Next task: watcher.py

Chosen first because it's the only new stage with no dependency on the
vendored repo, Ollama, or ffmpeg — just RSS + `state.db`. Also forces the
`SOURCE_VIDEOS` schema to exist for real.

- [ ] `git init`, branch `stage/watcher` (repo isn't git-initialized yet —
      confirm with user before doing this)
- [ ] `state.db` schema: `SOURCE_VIDEOS` table (arch doc §7)
- [ ] `config/channels.yaml` — source channel IDs
- [ ] `watcher.py` — poll `/feeds/videos.xml?channel_id=...`, diff against
      `state.db`, enqueue new video IDs
- [ ] `tests/test_watcher.py` — RSS diff logic against a fixture feed + a
      seeded `state.db`

## After that

1. Vendor the base repo under `vendor/`, confirm `local/downloader.py` and
   `pipeline.py` work as documented on this machine.
2. `darija_overrides/transcriber_darija.py` — swap in Darija fine-tune.
3. `darija_overrides/llm_ollama.py` — swap OpenAI call for Ollama.
4. `processor.py` — orchestrate vendor's `generate_shorts()` + scene
   detection + caption burn-in.
5. `qc_gate.py`, `publisher.py`, `reporter.py`, scheduler.

## Open questions

- Repo isn't git-initialized — CLAUDE.md requires branch + review per stage,
  so this needs to happen before the first commit.
