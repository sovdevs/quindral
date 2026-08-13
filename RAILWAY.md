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
There's no "Settings → Volumes" tab — Railway does this two other ways:
- **Command Palette**: `⌘K` → search for "volume" → create it → attach it
  to this service, **or**
- **Right-click the project canvas** → the volume-creation option from that
  context menu → attach it to this service.

Either way, once attached you set its **mount path** in the service config
(e.g. `/data`) — this is what makes `outcomes.jsonl` (and therefore the
HITL feedback log, the quality-penalty ranking signal, and "Recent" chat
history — all of which read/write through this one file) survive redeploys.

One timing gotcha from Railway's own docs: **volumes mount when the
container starts, not during build** — anything written to that path at
build time won't persist. Not a concern here (`outcome_log.py` only ever
writes at request time, never at build time), just don't be surprised if a
build-time smoke test against the mount path looks empty.

Railway auto-injects `RAILWAY_VOLUME_MOUNT_PATH` (the mount path you just
set, e.g. `/data`) as an environment variable on the service — no need to
hardcode it a second time in step 4.

### 4. Set environment variables
Project → the service → **Variables**:

| Variable | Value | Why |
|---|---|---|
| `QUINDRAL_LOG_PATH` | `${{RAILWAY_VOLUME_MOUNT_PATH}}/outcomes.jsonl` | References the volume's actual mount path (auto-injected by Railway, see step 3) instead of hardcoding it a second time — Railway's Variables UI resolves `${{...}}` references to other variables on the same service. |
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
- **Rate limiting is now built** (per-IP, in-memory, see `api.py`'s
  `RateLimiter`) — Railway doesn't add this for you, so it had to be app-side.
  Optional overrides: `QUINDRAL_RATE_LIMIT_MAX` (default 60) and
  `QUINDRAL_RATE_LIMIT_WINDOW` (default 60 seconds). It's per-process/in-memory
  though — fine at today's single instance, wouldn't coordinate correctly if
  this ever scales to multiple Railway replicas.
- **HTTPS** is automatic on Railway's provided domain (railway.app subdomain
  or a custom domain with their managed cert) — this part *is* covered for
  free, no action needed.
- **Multi-provider execution, enterprise tier, offline judge pipeline** —
  none of these are deployment concerns, they're unbuilt features (see
  SPEC.md).
