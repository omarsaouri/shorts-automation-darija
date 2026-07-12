# Progress

## Status: watcher stage done, on branch `stage/watcher`

- `main`: initial docs commit (`CLAUDE.md`, architecture doc, `.gitignore`).
- `stage/watcher` (current, not yet merged): `db.py` (state.db schema,
  `source_videos` table), `config/channels.yaml`, `watcher.py`, tests
  (2 passing), `requirements.txt` + local `.venv`.
- `watcher.run()` fetches each configured channel's RSS feed, diffs against
  `source_videos`, inserts new rows with `status='queued'`. Not yet run
  against a real channel — `config/channels.yaml` still has a placeholder
  channel ID.

## Plan of record (per architecture doc)

Vendor `SamurAIGPT/AI-Youtube-Shorts-Generator` into `vendor/` for
download/transcribe/score/crop, patched only via `darija_overrides/`. Build
net-new: `watcher.py` (done), scene detection, caption burn-in, `qc_gate.py`,
`publisher.py`, `reporter.py`, scheduler.

## Test run (2026-07-12)

Verified end-to-end against 3 real, high-volume channels (Google for
Developers, MrBeast, Veritasium — not real sources, just volume for
testing). Confirmed: RSS fetch/parse works, 45 videos inserted into
`state.db`, second run correctly queued 0 new (dedup works). Full results
in `reports/watcher_test_2026-07-12.md`. `state.db` has since been wiped
and `config/channels.yaml` reset to a placeholder — watcher is untested
against real production sources.

## Next up

- [ ] Merge/review `stage/watcher` into `main` (per CLAUDE.md, not a direct
      commit to main)
- [ ] Get real Darija source channel IDs from user (channels they own or
      have explicit permission to re-clip — architecture doc's opening note)
      and put them in `config/channels.yaml`
- [ ] Vendor the base repo under `vendor/`, confirm `local/downloader.py` and
      `pipeline.py` work as documented on this machine
- [ ] `darija_overrides/transcriber_darija.py` — swap in Darija fine-tune
- [ ] `darija_overrides/llm_ollama.py` — swap OpenAI call for Ollama
- [ ] `processor.py` — orchestrate vendor's `generate_shorts()` + scene
      detection + caption burn-in
- [ ] `qc_gate.py`, `publisher.py`, `reporter.py`, scheduler

## Open questions

- None currently blocking.
