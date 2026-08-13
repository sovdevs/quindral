# Offline Quality Evaluator

See EVALUATION.md for the original problem (two models gave contradictory
answers to the same factual question — how do we know which one is
garbage?) and SPEC.md's "Offline Quality Judging" for the design decision:
judging happens as a **batch job against logged data**, never on the live
request path — same cost/environmental reasoning that ruled out a full-LLM
classifier in the first place. This is that batch job.

## What it does

`evaluate.py`:
1. Pulls new outcome records from the deployed API's `GET /outcomes`
   (admin-authenticated — full content, not the public per-user-redacted
   view), resuming from a saved cursor so re-runs only fetch what's new.
2. For each `"accepted"`/`"escalated"` record (the actual execution events —
   deliberately **not** `response_thumbs_*` votes, which carry the same
   prompt/response content and would just judge the same pair twice), asks
   a judge model (default `gpt-4o-mini`) a simple pass/fail question: is
   this response accurate, relevant, and reasonably complete for the
   prompt?
3. Posts each verdict to `POST /judge` (admin-only — see api.py).

From there, `outcome_log.judge_penalty_rate()` reads these back the same
way `negative_response_rate()` reads human votes, and `router.py`'s
`_rank()` adds it as a **separate** soft penalty — a model bad by both the
judge and real human votes gets pushed down more than bad by either alone,
which only works because the two signals are summed, not blended into one
number.

## Running it

```bash
QUINDRAL_API_URL=https://<your-app>.up.railway.app \
QUINDRAL_ADMIN_TOKEN=<the same one set on the Railway service> \
QUINDRAL_JUDGE_OPENAI_KEY=sk-... \
.venv/bin/python3 evaluate.py
```

- `QUINDRAL_JUDGE_OPENAI_KEY` is **not** any end user's BYOK key — those
  only ever live in a browser and are never sent to the server. This is a
  separate key, spent by whoever runs this script, for the judging calls
  themselves.
- No key/network needed to sanity-check the parsing logic:
  `python3 evaluate.py --demo`.
- State (the resume cursor) lives in `local/evaluator_cursor.json`
  (gitignored) — delete it to force a full re-pull from the beginning.

## Design choices worth knowing

- **Manual, not scheduled** — deliberately started this way so you can
  watch it work by hand before wiring up cron/GitHub Actions. Scheduling is
  a small addition once the manual runs look right, not a prerequisite.
- **Pass/fail + one-sentence reasoning**, not a 1-5 scale — easiest to
  validate by eye early on, and maps directly onto the same "fraction
  negative" shape `negative_response_rate` already uses. Can grow into a
  finer scale later if pass/fail turns out too coarse.
- **Admin-only write access to `/judge`** — unlike `/outcome` (any
  anonymous client can log their own real usage), a "quality verdict"
  directly moves ranking. Letting an unauthenticated caller post one would
  let anyone fake the signal, not just report their own activity.
- **Resumable via timestamp cursor**, not a "mark records processed" flag
  on the log itself — keeps `outcome_log.py`'s append-only design intact,
  no read-modify-write needed there.

## Not built yet
- Scheduling (cron / GitHub Actions) — manual only for now, see above.
- Anything beyond simple pass/fail — no confidence score, no
  categorized failure types (hallucination vs. incomplete vs. off-topic),
  no cross-model consistency checking (the original EVALUATION.md scenario
  compared two models' answers directly — this judges each response against
  the prompt alone, not against each other).
- Real exponential backoff on judge API failures — a failed judge call is
  logged to stderr and retried on the **next run** (not immediately): the
  cursor parks just before the earliest failure instead of advancing past
  it, so a later run picks it up again. This does mean a run with any
  failures also re-fetches (and harmlessly re-judges — deduped by
  `(prompt_id, model_used)`, last-write-wins) some already-succeeded records
  with later timestamps; a run isn't fully "caught up" until one comes back
  with zero skips. Verified end-to-end: a record that failed on run 1 was
  confirmed judged on run 2 without manual intervention.
