# Deploying to Railway

This replaces the old RAILWAY.md, which was just Railway's Hobby-tier pricing
copy pasted in — not a deployment plan. This is the actual plan.

## What this app needs, that a generic "deploy a Python app" guide won't tell you

- **A dynamic `$PORT`, bound to `0.0.0.0`.** Railway assigns a port at
  runtime and expects the app to bind it — `api.py`'s `serve()` now does
  this automatically (`os.environ.get("PORT")`, defaulting to `0.0.0.0`
  when `PORT` is set, `127.0.0.1:8000` otherwise for local dev). No flags
  needed on Railway; `--lan` remains for local network sharing only.
- **A persistent volume for `outcomes.jsonl`.** Railway containers are
  ephemeral by default — anything written to local disk is wiped on every
  redeploy. `outcome_log.py` already reads `QUINDRAL_LOG_PATH` from the
  environment (falls back to an in-repo path for local dev), so this is a
  Railway-side volume-mount + env-var step, not a code change.
- **One real Python dependency**: `ecologits` (see `requirements.txt`).
  Railway's Nixpacks builder auto-detects `requirements.txt` and
  `pip install`s it — no extra config needed for the build itself.
- **A start command.** Added `Procfile` (`web: python3 api.py --serve`) —
  Railway reads this directly, no Nixpacks guessing required.

## Steps

### 1. Push the code to GitHub
There's currently no git remote configured on this repo (`git remote -v` is
empty) — Railway's GitHub-based deploy needs a real GitHub repo to connect
to.
```bash
git add -A
git commit -m "..."
gh repo create <name> --private --source=. --push   # or create on github.com and add the remote manually
```
Double-check `.gitignore` before the first push — `outcomes.jsonl` and
`openrouter_cache.json` are already excluded (local/generated data, not
source).

### 2. Create the Railway project
- railway.app → New Project → **Deploy from GitHub repo** → select the repo.
- Railway auto-detects Python via Nixpacks and picks up the `Procfile`.

### 3. Attach a persistent volume
- Project → the service → **Settings → Volumes → New Volume**.
- Mount path: anything outside the app's own code directory, e.g. `/data`.
- This is what makes `outcomes.jsonl` (and therefore the HITL feedback log,
  the quality-penalty ranking signal, and "Recent" chat history — all of
  which read/write through this one file) survive redeploys.

### 4. Set environment variables
Project → the service → **Variables**:

| Variable | Value | Why |
|---|---|---|
| `QUINDRAL_LOG_PATH` | `/data/outcomes.jsonl` | Points logging at the mounted volume from step 3, not the ephemeral container filesystem. |
| `QUINDRAL_ADMIN_TOKEN` | a real secret (generate one, don't reuse anything) | Unlocks `/outcomes`'s full/paginated access mode for the future offline evaluator. Without this set, `/outcomes` runs in its default (unauthenticated, per-user-redacted) mode — fine for the app itself, just means the evaluator's bulk-pull path isn't usable yet. |

`PORT` is injected automatically by Railway — don't set it manually.

### 5. Deploy and verify
- Railway builds and deploys on push (or manually trigger the first deploy).
- Once live, smoke-test the actual behaviors that depend on the steps
  above, not just "does the homepage load":
  - Route a query, refresh the page, confirm **Recent** still shows it
    (proves the volume + `QUINDRAL_LOG_PATH` are wired correctly — if this
    is broken, routing still "works," it just silently loses everything on
    the next redeploy).
  - Check `GET /outcomes` without any auth header — should return
    redacted (no `prompt_text`/`response_text` for records that aren't
    yours) records, not an error and not everything unredacted.
  - If `QUINDRAL_ADMIN_TOKEN` is set, confirm `GET /outcomes` with
    `X-Quindral-Admin-Token: <the token>` returns full records.

## What this does NOT cover (still open, see SPEC.md "Open Risks")
- **Rate limiting.** Railway doesn't add this for you. `/route`, `/feedback`,
  `/outcome`, and `/pricing` are all unauthenticated and uncapped — fine
  for a controlled demo link, not for wide public distribution.
- **HTTPS** is automatic on Railway's provided domain (railway.app subdomain
  or a custom domain with their managed cert) — this part *is* covered for
  free, no action needed.
- **Multi-provider execution, enterprise tier, offline judge pipeline** —
  none of these are deployment concerns, they're unbuilt features (see
  SPEC.md).
