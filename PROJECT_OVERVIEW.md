# PROJECT OVERVIEW: Quindral

> Combined from `SPEC.md` (what it does, architecture, decisions) and
> `RAILWAY.md` (deployment). Generated as a single reference for downstream
> processing — see those two files for the canonical, independently-updated
> originals if this drifts.

---

# Part 1: What This Is

## Status
**Deployed and live on Railway** (root app, `api.py` + `index.html`) — see
"Part 3: Deployment" below. Classify → route → explain → execute → evaluate
→ log pipeline is built and self-tested (`classifier.py`, `router.py`,
`explain.py`, `providers.js`, `outcome_log.py`, `api.py`). `netlify/v0.7/`
is a **deprecated, frozen** static snapshot — no longer used now that
Railway is live. The offline LLM-as-judge pipeline (`evaluate.py`) is built
and has already run successfully against real production traffic, including
catching a real hallucination in the wild (`gpt-4o-mini` fabricated a name,
`gpt-5.4` got it right, the judge correctly failed the former and passed the
latter).

## Context / Origin
Independent app, inspired by (not affiliated with, not a wrapper of)
[aim2balance.ai](https://aim2balance.ai) — an EU-based orchestrator that
routes prompts across open-source LLMs (Gemma, Mistral, Llama, Qwen, GLM,
Kimi K2), optimizing primarily for environmental efficiency and EU data
residency, using what they describe publicly as "five capability routes"
plus a language-aware routing layer. They do not expose routing reasoning to
end users — that gap is the core differentiator for this project.

## Core Idea
An LLM orchestrator/router that:
1. Selects the best-matched LLM for each incoming query
2. **Shows the user why** that model was chosen (transparent reasoning
   trace), not just the answer
3. Lets users adjust the optimization criteria (cost, environmental impact,
   EU-only/privacy, capability), with **cost as the default primary
   optimization target**
4. Ships as both web and phone app

## Target Platforms
- Web app
- Mobile app (iOS/Android — not yet decided native vs cross-platform)

---

## Architecture Overview

```
User query → Classifier → Candidate scoring → Hard filters → Soft rank
  (cost/env/latency weights + quality-penalty demerit) → Selection + reasoning
  trace → [optional] real execution (BYOK, direct browser→provider) →
  degenerate-response auto-escalation → user feedback (routing vote +
  response-quality vote) → outcome logging (prompt/response text retained)
  → [planned] offline LLM-as-judge → quality-penalty signal → back into rank
```

### 1. Classifier
Decides task type/complexity to determine which models are even eligible.
**Decision: NOT a full LLM call on every request** — running an LLM
classifier on 100% of traffic adds latency/cost/energy to every query just
to save money on the subset that need a big model, undercutting the
cost/environmental pitch.

**Built**: Phase 0 (embedding similarity + heuristics), including a
`needs_current_info` retrieval-necessity signal. `eval_classifier.py`
measures Phase 0 accuracy against the reference set.

Phased approach:
- Phase 0: embedding similarity + heuristics, zero training data required — **built**
- Phase 1: logging real usage outcomes as labeled data — **partially built** (explicit thumbs voting is built; implicit signals defined in schema but not wired to the UI)
- Phase 2: lightweight trained classifier (logistic regression / GBT over embeddings) — **not built**
- Phase 3: confidence-based escalation as a runtime safety net — **built**

Comparable prior art considered: RouteLLM, Semantic Router, RoRF/Not
Diamond, Martian, OpenRouter/Inworld. Dominant pattern: avoid a full
LLM-based classifier — favor embeddings + small trained models.

### 2. Model Registry
`model_registry.py` — **17 models** across 8 providers/families (Mistral,
Meta/Llama, Qwen, Microsoft/Phi, OpenAI, Anthropic, Google, Moonshot/Kimi):
- Cost per input/output token — live-synced via `openrouter_sync.py`
  (falls back to hand-entered placeholders, flagged `cost_is_stale`)
- Energy/water estimate per token (`energy.py`, Ecologits methodology)
- Hosting region (EU / non-EU)
- Context window
- Capability tags (code, vision, long-context, reasoning, creative,
  factual, web_search)
- Measured latency
- Data policy flags (no-train guarantee, encryption, residency)

### 3. Hard Filters vs. Soft Scoring
- **Hard filters** (exclude outright): EU-only toggle, capability floor,
  minimum context length, "only models I have a BYOK key for" (opt-in,
  root app only)
- **Soft scoring** among survivors: weighted sum of normalized
  cost/env/latency (user-adjustable, budget-allocator sliders summing to
  100%, cost weighted highest by default) plus a non-adjustable
  quality-penalty demerit

### 4. Reasoning Trace (core product differentiator)
Every routing decision emits a structured object, decoupled from the
human-readable explanation shown to the user:

```json
{
  "classified_as": "simple",
  "eligible_models": ["mistral-small", "llama-3.1-8b", "gpt-4o-mini"],
  "excluded": [
    {"model": "codestral", "missing_capabilities": ["factual"], "not_eu_hosted": false, "context_deficit": null, "no_key_available": false}
  ],
  "chosen": "mistral-small",
  "binding_criterion": "cost",
  "relaxed_filters": []
}
```

`excluded` entries are structured per-reason booleans/objects so
`explain.py` can compose grammatically-correct sentences and concrete
savings callouts. Structured trace and NL explanation are kept separate for
localization, adjustable verbosity, and click-to-expand detail.

### 5. Escalation (built, two paths, one shared policy)
Both paths use the same provider escalation ladder, newest-model-first
(`PROVIDER_LADDER` in `index.html`/`providers.js`), restricted to models
with a usable execution credential.
- **Automatic**: `isDegenerateResponse()` checks for empty/short output,
  refusal boilerplate, high repetition → one automatic retry on the
  next-best model, logged as `"escalated"`.
- **User-triggered**: 👎 "Poor" reveals a "Try [newest available model]
  instead" button — same ladder, same logging, user decides.

Only OpenAI is wired for execution today, so the ladder has real teeth only
within the `gpt-*` family.

---

## Execution
- **Architecture decision**: direct browser → provider call using the
  user's own BYOK key (`providers.js`), never relayed through Quindral's
  server — the key never touches the backend.
- **Provider adapters** (`providers.js`): one function per provider, added
  incrementally. **Only OpenAI is wired** (`/v1/chat/completions`, Bearer
  auth). Anthropic and others not built yet.
- **Gateways** (OpenRouter, Together, Groq) not wired for execution either.
- **UI**: "Run on this model" appears only when a real execution credential
  resolves for the chosen model.
- Real token usage and actual cost (`/pricing` endpoint) shown after a run,
  distinguished from the fictional demo routing fee.

---

## Monetization
BYOK removes the original implicit model (proxy the call, mark up token
cost). Decisions:
- **Managed/hosted keys: ruled out** — avoids capital exposure, reseller
  ToS risk, support burden.
- **Flat per-route fee, decoupled from token cost**: chosen as the primary
  near-term mechanism. Prototype: fictional per-browser `localStorage`
  balance ($5 demo, $0.02/route), stands in for real Stripe billing
  (deferred until deployed).
- **Enterprise/team governance tier**: identified as the real revenue
  target, **not started** — needs a multi-user/org concept the app doesn't
  have yet.
- Hybrid BYOK+managed and flat subscription: **not pursued**.

---

## HITL Feedback & Quality Evaluation

### Two distinct vote categories
`outcome_log.py` separates **routing-decision votes**
(`thumbs_up`/`thumbs_down`) from **response-quality votes**
(`response_thumbs_up`/`response_thumbs_down`), same `prompt_id`, disjoint
outcome sets. A negative response vote reveals an optional comment box,
logged as its own record without dropping the original vote.

### Quality-penalty ranking (built)
`outcome_log.negative_response_rate(model, route)` computes each model's
real thumbs-down rate among response votes, scoped per route, deduped by
`(user_id, prompt_id)`. Returns `None` below `MIN_VOTES_FOR_QUALITY_SIGNAL =
3`. `router.py`'s `_rank()` adds this as a soft demerit
(`QUALITY_PENALTY_WEIGHT = 1.0`), never a hard exclusion.

### Feedback page (built)
Sidebar view listing every logged HITL record (routing votes, response
votes with comments, auto-escalations), newest first, via `GET /outcomes`.
Expandable `<details>`/`<summary>` preview of full prompt/response text, all
HTML-escaped.

**Multi-user redaction**: `GET /outcomes` only returns prompt/response text
for records whose `user_id` matches the requester's own anonymous client id
(`X-Quindral-Client-Id`). Metadata stays visible for all records.

### Persistence & the offline evaluator's access path
- `outcomes.jsonl` on an ephemeral container gets wiped on redeploy — log
  path configurable via `QUINDRAL_LOG_PATH`, pointed at a mounted Railway
  persistent volume in production.
- `GET /outcomes` supports a shared-secret header
  (`X-Quindral-Admin-Token`, checked against `QUINDRAL_ADMIN_TOKEN`)
  unlocking full/unredacted records with `?since=`/`?limit=` cursor
  pagination — the evaluator's bulk-pull access path.

### Offline Quality Judging (built — see EVALUATOR.md)
**Decision**: no LLM-as-judge on the live request path (same
cost/environmental reasoning as the no-full-LLM-classifier decision).
Judging happens offline as a batch job.

`evaluate.py` pulls new records via admin-authenticated `GET /outcomes`
(resumable cursor), judges each `"accepted"`/`"escalated"` record with a
cheap OpenAI model (pass/fail + one-sentence reasoning), posts verdicts to
admin-only `POST /judge`. Verdicts land as a `"judged"` outcome
(`judge_pass`/`judge_reasoning`/`judge_model`).
`outcome_log.judge_penalty_rate()` mirrors `negative_response_rate`'s shape
(route-scoped, same floor, keyed by `(prompt_id, model_used)`). `router.py`
sums `JUDGE_PENALTY_WEIGHT` with `QUALITY_PENALTY_WEIGHT` — not blended.

**Not built**: scheduling (manual runs only), retry on judge-call failure,
anything beyond simple pass/fail, cross-model consistency checking.

---

## Trust / Transparency
A live network-activity log in Settings — every request the tab makes is
logged with a badge: 🔓 **direct to provider** (key-bearing calls) vs.
**quindral server** (routing/pricing/feedback, never carries the key) —
inviting cross-checking against the browser's own DevTools Network tab. Also
discloses prompt/response content retention distinctly from key privacy.

---

## UI Decisions
- **Budget-allocator sliders**: cost/env/latency linked 0–100% values
  summing to 100 (dragging one rebalances the others). Display-only change;
  underlying weighted-sum ranking is scale-invariant.
- **Preset-gate → free-text unlock**: users try 5 presets before free-text
  unlocks (persisted in `localStorage`). Presets reordered so
  vision/reasoning examples (where slider weight visibly changes outcome)
  lead the list.
- **Response area is a stacked list**: every run action appends a new
  entry rather than replacing the current one; each has independent
  response text, escalation note, cost, and rate/comment/try-better
  controls.
- **"Recent" chat history is fully server-backed**, not `localStorage` —
  reconstructed from `/outcomes` grouped by `prompt_id`
  (`groupRecordsIntoChats()`). A new `"routed"` outcome type logs the full
  trace; `/route` generates `prompt_id` server-side (`uuid4`).
- **Export chats**: exports the server-reconstructed chat groups.

---

## Netlify Demo Scope (`netlify/v0.7/`) — DEPRECATED, frozen
A separate, backend-less static build for a drag-and-drop demo (`router.js`
is a hand-ported JS mirror of the Python classifier/router/explain/registry
logic). No longer actively used or kept in sync now that Railway is live.
Never had real model execution, response voting, the Feedback page,
quality-penalty ranking, or the offline judge — none of that will be
retrofitted here.

---

## User-Facing Criteria & Controls
- **EU-only (privacy)** — hard filter, not a weight
- **Only models I have a key for** — hard filter, opt-in, root app only
- **Cost** — default primary optimization target
- **Environmental impact** — weighted criterion, user-adjustable
- Users can set a default global optimization target and override
  per-session

---

## Classifier Roadmap (Cold Start → Learned)

### Phase 0: Cold start, no training data — **built**
- Embedding similarity to reference examples per capability route
- Heuristic features: token count, code-fence/regex detection, math
  symbols, question vs. imperative phrasing, image/file attached, detected
  language, retrieval-necessity heuristic (`needs_current_info`)
- Decision table mapping signals → route (code, vision, long-context,
  simple/cheap, general reasoning — five routes, finalized)
- `eval_classifier.py` measures accuracy against the reference set

### Phase 1: Log the HITL feedback signal — **explicit built, implicit not**
Explicit thumbs up/down, split into routing votes and response-quality
votes. Implicit signals (regenerate, correction-follow-up, manual
escalation) schema-defined but not UI-triggered.

Log schema per routed request (`outcome_log.py`):
```json
{
  "timestamp": 1786550702.09,
  "route_chosen": "vision",
  "model_used": "gpt-4o-mini",
  "outcome": "response_thumbs_down",
  "criteria_weights_active": {"cost": 0.6, "env": 0.3, "latency": 0.1},
  "user_id": "...", "prompt_id": "...",
  "usage": {"input_tokens": 120, "output_tokens": 340, "actual_cost": 0.0037},
  "comment": "missed the object in the corner",
  "prompt_text": "describe what's happening in this photo",
  "response_text": "..."
}
```

### Phase 2: Train lightweight classifier — **not built**
Once sufficient labeled examples exist per route, train logistic
regression or GBT (XGBoost/LightGBM) on embedding + heuristic features.
Retrain on a schedule, not online-learning from day one. The quality-penalty
mechanism is a simpler, non-learned precursor to this.

### Phase 3: Confidence-based escalation — **built**

---

## Cold-Start Example Data — Sourcing Plan
1. Public benchmark datasets with task-type/difficulty annotations — free,
   fast, generic
2. Synthetic generation via LLM, spot-checked — cheap, fast, one-time cost
3. Real logged usage post-launch — gold standard, exists now that Railway
   is deployed
4. Manual labeling — hand-tag a sample to calibrate/sanity-check

Recommended: seed Phase 0 with public datasets + synthetic generation (this
sourcing is really for Phase 2). Real logged data becomes dominant within
weeks of launch.

---

## Open Risks & Design Questions
- **Class imbalance**: most real traffic skews "simple/cheap"; code/vision/
  long-context rarer. Classifier must handle this skew.
- **Label noise from implicit signals**: not yet an issue — implicit
  signals aren't wired up yet.
- **Feedback loop / exploration-exploitation risk**: nothing currently
  forces occasional traffic to a penalized model to see if it's improved.
- **Failure mode preference**: under-routing vs. over-routing default
  stance not yet decided.
- **Multilingual support**: not yet handled; aim2balance runs a separate
  language-aware layer.
- **Classifier runtime location**: server/backend vs. edge/on-device — not
  decided.
- **Server hardening**: `api.py` is the stdlib prototype server
  (single-process). Closed since deploy: `/outcomes` token auth +
  per-user redaction, per-IP rate limiting on every POST + `GET /outcomes`
  (see RATELIMITER.md). Still open: rate limiter is in-memory/per-process
  (won't coordinate across multiple replicas); static file serving isn't
  rate-limited; a real ASGI server is the eventual answer if traffic grows.
- **Multi-provider execution**: only OpenAI wired. Anthropic, Google,
  Mistral, Moonshot, Microsoft, and gateway execution unbuilt.
- **Enterprise/team tier**: needs an org/multi-user data model that doesn't
  exist yet.

---

## Decisions Already Made
- Independent app, not a wrapper/competitor built on aim2balance's platform
- Routing decisions happen server-side
- Classifier will not be a full LLM call on every request
- Cost is the default primary optimization criterion
- EU-only is a hard filter, not a weighted criterion
- Reasoning trace is structured/logged separately from the NL explanation
- HITL outcome signals inform future routing, but system starts on
  heuristics (bootstrap-then-learn)
- BYOK is free and permanent, not a stepping stone to managed keys
- Model execution happens directly browser → provider, never relayed
  through Quindral's server
- Automated quality judging runs offline against logged data, never on the
  live request path
- Prompt/response text is now retained server-side for offline judging — a
  deliberate, disclosed tradeoff, distinct from key privacy
- Monetization: BYOK stays free; flat per-route fee is the near-term
  mechanism; enterprise governance is the long-term target, not yet built
- Log persistence and evaluator access are env-var-driven
  (`QUINDRAL_LOG_PATH`, `QUINDRAL_ADMIN_TOKEN`)
- Every POST endpoint + `GET /outcomes` is per-IP rate-limited (see
  RATELIMITER.md)
- Offline LLM-judge is built: `evaluate.py`, run manually, pass/fail
  verdicts via admin-only `POST /judge` (see EVALUATOR.md)

## Not Yet Decided
- Classifier runtime: server vs. edge/on-device
- Native vs. cross-platform for mobile
- Whether/how to force occasional exploration traffic to
  quality-penalized models
- Enterprise tier data model (orgs, multi-user, roles)
- Real Stripe wiring details for the routing fee
- Whether/how to schedule the evaluator (cron, GitHub Actions) vs. manual
- Finer-grained judge output (confidence score, failure categorization)
- Cross-model consistency checking (comparing multiple models' answers to
  each other on the same prompt)

---

# Part 2: Where the Files Are

Root of the repo (`/Users/vmac/prog/PY/Quindral`):

| File/Dir | Purpose |
|---|---|
| `api.py` | Stdlib HTTP server — all endpoints (`/route`, `/pricing`, `/outcomes`, `/judge`, `/feedback`, `/outcome`), rate limiting, admin auth |
| `index.html` | Frontend — UI, sliders, chat/response rendering, `PROVIDER_LADDER` |
| `classifier.py` | Phase 0 query classifier (embedding similarity + heuristics) |
| `router.py` | Candidate scoring, hard filters, soft ranking, quality/judge penalties |
| `explain.py` | Converts structured reasoning trace into NL explanation |
| `providers.js` | Browser-side provider adapters (BYOK execution), degenerate-response detection |
| `model_registry.py` | The 17-model table (cost, energy, region, capabilities, latency, data policy) |
| `energy.py` | Ecologits-based energy/water estimate per token |
| `openrouter_sync.py` | Live-syncs cost data from OpenRouter |
| `openrouter_cache.json` | Cached sync output |
| `outcome_log.py` | HITL outcome logging, quality-penalty and judge-penalty rate calculations |
| `outcomes.jsonl` | Local dev log file (production uses a Railway-mounted volume instead, via `QUINDRAL_LOG_PATH`) |
| `evaluate.py` | Offline LLM-as-judge batch script |
| `eval_classifier.py` | Measures classifier Phase 0 accuracy |
| `scripts_export_snapshot.py` | Export/snapshot utility |
| `byok.js` | BYOK key handling (browser-only, never sent to server) |
| `Procfile` | Railway start command (`web: python3 api.py --serve`) |
| `requirements.txt` | Python deps (`ecologits`, etc.) |
| `netlify/v0.7/` | **Deprecated, frozen** static demo snapshot — no longer synced or used |
| `SPEC.md` | Canonical spec/architecture/decisions doc (source for Part 1 above) |
| `RAILWAY.md` | Canonical deployment doc (source for Part 3 below) |
| `EVALUATOR.md` | Offline judge pipeline details |
| `RATELIMITER.md` | Rate limiter details |
| `EVALUATION.md` | Original hallucination-catching evaluation scenario |
| `MONETIZE.md` | Monetization notes |
| `local/`, `logs/` | Local dev artifacts |

---

# Part 3: How It's Deployed (Railway)

## What this app needs from a host
- **A dynamic `$PORT`, bound to `0.0.0.0`** — `api.py`'s `serve()` reads
  `PORT` from the environment, defaulting to `0.0.0.0` when set,
  `127.0.0.1:8000` otherwise for local dev (`--lan` remains for local
  network sharing only).
- **A persistent volume for `outcomes.jsonl`** — containers are ephemeral
  by default; `outcome_log.py` reads `QUINDRAL_LOG_PATH` from the
  environment (falls back to an in-repo path for local dev).
- **One real Python dependency**: `ecologits` (see `requirements.txt`).
- **A start command**: `Procfile` (`web: python3 api.py --serve`).

## Deployment steps (as set up on Railway)
1. **Push the code to GitHub** — Railway's GitHub-based deploy needs a real
   GitHub repo to connect to. `.gitignore` excludes `outcomes.jsonl` and
   `openrouter_cache.json` (local/generated data).
2. **Create the Railway project** — railway.app → New Project → Deploy from
   GitHub repo. Railway auto-detects Python via Nixpacks and picks up the
   `Procfile`.
3. **Attach a persistent volume** — via Command Palette (`⌘K` → "volume")
   or right-click the project canvas. Set its mount path (e.g. `/data`).
   Note: volumes mount when the container starts, not during build —
   `outcome_log.py` only writes at request time, so this isn't an issue.
   Railway auto-injects `RAILWAY_VOLUME_MOUNT_PATH` as an env var.
4. **Set environment variables** (Project → service → Variables):
   | Variable | Value | Why |
   |---|---|---|
   | `QUINDRAL_LOG_PATH` | `${{RAILWAY_VOLUME_MOUNT_PATH}}/outcomes.jsonl` | References the volume's mount path via Railway's `${{...}}` variable resolution |
   | `QUINDRAL_ADMIN_TOKEN` | a real secret | Unlocks `/outcomes`'s full/paginated access mode for the offline evaluator; without it, `/outcomes` runs unauthenticated/per-user-redacted |

   `PORT` is injected automatically — don't set it manually.
5. **Deploy and verify** — Railway builds/deploys on push. Smoke-tests that
   matter: routing a query and confirming "Recent" survives a refresh
   (proves volume + `QUINDRAL_LOG_PATH` wiring); `GET /outcomes` without
   auth returns redacted records; `GET /outcomes` with the admin token
   header returns full records.

## Why Railway (not an inherent requirement — see below)
Chosen for fit at this project's stage, not for anything uniquely Railway
offers:
- GitHub-connected auto-deploy, no CI pipeline to build
- Nixpacks auto-detects Python — zero container config for a single
  stdlib-server app
- Persistent volumes for the stateful `outcomes.jsonl` log
- Free HTTPS out of the box
- Good fit for a single-instance prototype/demo

**Status update**: the Railway trial has expired. Any replacement host just
needs to satisfy the same four requirements above (dynamic `$PORT`,
persistent volume/disk, one pip dependency, a start command) — Fly.io,
Render, and a plain VPS all qualify. No app-side lock-in to Railway
specifically; migration is an open TODO, not yet scheduled.

## What deployment does NOT cover
- **Rate limiting** is app-side, not Railway-provided (`api.py`'s
  `RateLimiter`, per-IP in-memory, `QUINDRAL_RATE_LIMIT_MAX`/
  `QUINDRAL_RATE_LIMIT_WINDOW` env overrides). In-memory/per-process — fine
  at a single instance, wouldn't coordinate across multiple replicas.
- **HTTPS** was free on Railway's domain — confirm equivalent on whichever
  host replaces it.
- **Multi-provider execution, enterprise tier, offline judge scheduling** —
  unbuilt app features, not deployment concerns.
