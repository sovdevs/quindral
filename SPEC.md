# SPEC: Multi-Criteria LLM Orchestrator with Transparent Routing

## Status
Working prototype, not yet deployed. Classify → route → explain → execute →
evaluate → log pipeline is built and self-tested (`classifier.py`, `router.py`,
`explain.py`, `providers.js`, `outcome_log.py`, `api.py`). Root app (`index.html`
+ `api.py`) is the real, backend-connected build; `netlify/v0.7/` is a
static, routing-only demo snapshot with no backend (see "Netlify Demo Scope"
below). Next planned step: deploy the root app to Railway, then build the
offline LLM-as-judge pipeline against real traffic (see "Offline Quality
Judging" below).

## Context / Origin
Independent app, inspired by (not affiliated with, not a wrapper of) [aim2balance.ai](https://aim2balance.ai) — an EU-based orchestrator that routes prompts across open-source LLMs (Gemma, Mistral, Llama, Qwen, GLM, Kimi K2), optimizing primarily for environmental efficiency and EU data residency, using what they describe publicly as "five capability routes" plus a language-aware routing layer. They do not expose routing reasoning to end users — that gap is the core differentiator for this project.

## Core Idea
An LLM orchestrator/router that:
1. Selects the best-matched LLM for each incoming query
2. **Shows the user why** that model was chosen (transparent reasoning trace), not just the answer
3. Lets users adjust the optimization criteria (cost, environmental impact, EU-only/privacy, capability), with **cost as the default primary optimization target**
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
Decides task type/complexity to determine which models are even eligible. **Decision: NOT a full LLM call on every request** — this was a deliberate choice, because running an LLM classifier on 100% of traffic adds latency/cost/energy to every query just to save money on the subset that need a big model, which undercuts the whole cost/environmental pitch.

**Built**: Phase 0 (embedding similarity + heuristics), including a
`needs_current_info` retrieval-necessity signal (does this query need info
newer than any model's training cutoff — orthogonal to task-type route, adds
a `web_search` capability requirement when true). `eval_classifier.py` exists
to measure Phase 0 accuracy against the reference set.

Phased approach (see "Classifier Roadmap" below):
- Phase 0: embedding similarity + heuristics, zero training data required — **built**
- Phase 1: logging real usage outcomes as labeled data — **partially built** (explicit thumbs voting is built; implicit signals like "regenerated" are defined in `outcome_log.py`'s schema but nothing in the UI triggers them yet)
- Phase 2: lightweight trained classifier (logistic regression / gradient-boosted trees over embeddings, e.g. XGBoost/LightGBM) — cheap at inference, no GPU needed — **not built**
- Phase 3: confidence-based escalation as a runtime safety net regardless of classifier maturity — **built**, see "Escalation" below

Comparable prior art considered: RouteLLM (trained weak/strong classifier), Semantic Router (pure embedding similarity, open source), RoRF/Not Diamond (random forest over embeddings), Martian (interpretability-based model mapping), OpenRouter/Inworld (rule-based + confidence escalation). The dominant pattern across serious routers is **avoid a full LLM-based classifier** — favor embeddings + small trained models.

### 2. Model Registry
Table maintained per candidate model (`model_registry.py`), currently 15 models across 8 providers/families (Mistral, Meta/Llama, Qwen, Microsoft/Phi, OpenAI, Anthropic, Google, Moonshot/Kimi):
- Cost per input/output token — **live-synced** via `openrouter_sync.py` (falls back to hand-entered placeholders, flagged `cost_is_stale` when no sync has run)
- Energy/water estimate per token (Ecologits methodology, `energy.py` — public, reusable, same one aim2balance references)
- Hosting region (EU / non-EU)
- Context window
- Capability tags (code, vision, long-context, reasoning, creative, factual, **web_search** — added this session for the retrieval-necessity signal)
- Measured latency
- Data policy flags (no-train guarantee, encryption, residency)

### 3. Hard Filters vs. Soft Scoring
- **Hard filters** (exclude candidates outright, not scored): EU-only toggle, capability floor (e.g. don't route a coding task to a non-code-capable model), minimum context length requirement, **"only models I have a BYOK key for" (opt-in checkbox, root app only)** — this is a genuine hard whitelist intersection, applied after relaxation, never itself relaxed (the user opted in explicitly; silently ignoring it would recommend something unusable)
- **Soft scoring** among survivors: weighted sum of normalized cost, environmental impact, latency — user-adjustable weights (UI: **budget-allocator sliders, always sum to 100%**, decided this session over independent 0–1 weights — see "UI Decisions" below), **cost weighted highest by default**, **plus a quality-penalty demerit** (see "Quality Signal" below) added on top, not user-adjustable — an intrinsic "don't recommend a model with a track record of bad responses for this route" factor, soft not hard (SPEC's exploration/exploitation concern: a model never fully vanishes over a bad run or two)

### 4. Reasoning Trace (core product differentiator)
Every routing decision emits a structured object, decoupled from the human-readable explanation shown to the user. Current real shape (`router.py`/`explain.py`, not the illustrative sketch from the original draft):

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

`excluded` entries are structured per-reason booleans/objects (not a pre-rendered string), so `explain.py` can compose grouped, grammatically-correct sentences ("X, Y and Z aren't hosted in the EU") and compare the chosen model's cost/energy against the priciest *eligible* alternative for a concrete savings callout ("that's roughly 83% cheaper than mistral-large"). Keeping structured trace and NL explanation separate allows localization, adjustable verbosity, and click-to-expand detail without recomputation.

### 5. Escalation (built, two paths — both share one policy)
Cheap models can fail silently on queries above their capability. Both paths use the same **provider escalation ladder**, newest-model-first (`PROVIDER_LADDER` in `index.html`/`providers.js`; decision this session: "next best usually means *newest*," not just "bigger sibling" — e.g. `gpt-4o-mini` escalates straight to `gpt-5.4`, skipping the older-but-larger `gpt-4o`), restricted to models with a usable execution credential.

- **Automatic**: after a real response comes back, `isDegenerateResponse()` (`providers.js`) checks for empty/too-short output, refusal boilerplate, or high word-repetition. If degenerate, one automatic retry on the next-best model in the ladder, logged as outcome `"escalated"`.
- **User-triggered**: voting 👎 "Poor" on a response reveals a "Try [newest available model] instead" button — same ladder, same logging, but the user decides whether to spend the extra call rather than it happening silently.

Only OpenAI is wired for execution today (see "Execution" below), so the ladder currently only has real teeth within the `gpt-*` family.

---

## Execution (built this session — new since original draft)
The original draft only covered *deciding* which model to use. The app now
optionally **actually calls** the chosen model:

- **Architecture decision**: direct browser → provider call using the user's
  own BYOK key (`providers.js`), never relayed through Quindral's own
  server — the key never touches the backend, in any request. Chosen over a
  server-relay proxy specifically to keep that privacy property provable
  (see "Trust / Transparency" below), at the cost of needing a per-provider
  adapter and accepting that some providers may eventually require a
  server-side relay for CORS reasons (not hit yet).
- **Provider adapters** (`providers.js`): one function per provider, added
  incrementally. **Only OpenAI is wired** (`/v1/chat/completions`, Bearer
  auth). Anthropic (`x-api-key`, `/v1/messages`, different response shape)
  and others are explicitly not built yet — each provider is its own small
  increment, not a generic abstraction built ahead of need.
- **Gateways** (OpenRouter, Together, Groq — OpenAI-compatible shape) are
  *not* wired for execution either, despite sharing OpenAI's request format —
  they need provider-prefixed model ids (`"openai/gpt-4o-mini"`), left as a
  follow-up.
- **UI**: a "Run on this model" button appears only when a real execution
  credential resolves for the chosen model (direct key, wired provider) —
  never shown as a dead affordance.
- Real **token usage** and **actual cost** (via a new `/pricing` endpoint
  exposing registry rates) are shown after a run, separate from and
  explicitly distinguished from the fictional demo routing fee (see
  "Monetization" below) — this is the "precise cost per prompt" the fee
  discussion below required.

---

## Monetization (new section — resolved this session)
**BYOK removes the original implicit model** (proxy the call, mark up the
token cost) — once a user supplies their own key, there's no token spend to
mark up. Decisions made, in order of how central they are:

- **Managed/hosted keys: ruled out.** Explicitly decided not to hold
  provider accounts and resell access — avoids capital exposure, reseller
  ToS risk, and the "why does the free tier not cover the model I want"
  support burden of stocking every provider.
- **#4, flat per-route fee, decoupled from token cost**: chosen as the
  primary near-term mechanism. A small flat fee charged per *routed
  decision*, not scaled to tokens — reframes the charge as "paying for the
  routing decision," not re-billing tokens the user already pays their
  provider for directly. This resolves the "BYOK + a fee feels like paying
  twice" objection: two different services (their tokens → their provider;
  the decision → Quindral), not the same spend billed twice.
  **Prototype built**: a fictional per-browser `localStorage` balance
  ($5 demo, $0.02/route, auto-resets at $0, applies to free-text queries
  only — presets stay free) stands in for real Stripe billing, which is
  deferred until the app is actually deployed.
- **#3, enterprise/team governance tier**: identified as the real revenue
  target (admin console, budget caps, policy enforcement, audit logs, team
  analytics — sold to whoever owns the AI spend line, not the individual
  dev) but **not started** — needs a multi-user/org concept the app doesn't
  have yet (today is single-browser, no accounts).
- **#1 (hybrid BYOK + managed keys) and #2 (flat subscription): not
  pursued** — #1 was ruled out per above; #2 couldn't be justified as a
  standalone mechanism (see chat history) though its spirit is echoed in #4
  (both monetize the service rather than the tokens).

---

## HITL Feedback & Quality Evaluation (substantially built this session)

### Two distinct vote categories
`outcome_log.py` deliberately separates **routing-decision votes**
(`thumbs_up`/`thumbs_down` — "was this the right model to pick") from
**response-quality votes** (`response_thumbs_up`/`response_thumbs_down` —
"was the actual answer good"). Same `prompt_id`, disjoint outcome sets, so
they never collide even though both can exist for one prompt. Response
votes are visually distinct in the UI (filled pill buttons vs. outlined
routing thumbs) specifically so users don't conflate the two questions.
A negative response vote reveals an optional comment box ("what went
wrong") — logged as its own record if used, without dropping the vote that
was already logged from the thumbs click itself (a real double-vs-dropped
bug was hit and fixed here).

### Quality-penalty ranking (built)
`outcome_log.negative_response_rate(model, route)` computes each model's
real thumbs-down rate among response votes, **scoped per route** (a model
can be fine at "simple" lookups and bad at "reasoning"). Deduped by
`(user_id, prompt_id)` so a single person's vote-then-comment doesn't
double-count. Returns `None` (no penalty) below `MIN_VOTES_FOR_QUALITY_SIGNAL
= 3` — one bad review can't tank a model. `router.py`'s `_rank()` adds this
as a soft demerit (`QUALITY_PENALTY_WEIGHT = 1.0`) on top of the normalized
cost/env/latency score — never a hard exclusion. Self-checks use an isolated
temp log path so assertions stay deterministic regardless of real production
feedback accumulating in `outcomes.jsonl` (a real test-fragility bug, caught
and fixed).

### Feedback page (built, since expanded)
New sidebar view listing every logged HITL record (routing votes, response
votes with comments, auto-escalations) — newest first, via `GET /outcomes`.
Built specifically as a transparency device: proof feedback is actually
being recorded, not just collected into a void. Each entry with content now
has a native `<details>`/`<summary>` expandable preview of the full
prompt/response text — all rendered content is HTML-escaped (a real XSS gap
was caught and fixed here: model output/user prompts are untrusted content
being inserted via `innerHTML`).

**Multi-user redaction (built)**: since this is a shared feed with no
accounts system, `GET /outcomes` only returns prompt/response *text* for
records whose `user_id` matches the requester's own anonymous client id
(sent as `X-Quindral-Client-Id`) — you can see your own full history, not
everyone else's real content. Metadata (outcome/model/route) stays visible
for all records regardless, which is what the page's "unfiltered
transparency" framing actually needs.

### Persistence & the offline evaluator's access path (built this session)
Two real gaps surfaced once "how would this work once deployed" was asked
directly:
- **`outcomes.jsonl` on an ephemeral container gets wiped on redeploy.**
  Log path is now configurable via `QUINDRAL_LOG_PATH` env var, defaulting
  to the old in-repo location for local dev — point it at a mounted Railway
  persistent volume in production. `log_outcome()` now creates the parent
  directory if missing (first write to a fresh volume shouldn't crash).
- **`GET /outcomes` needed real auth for the evaluator's bulk pull**, since
  it'll be a public URL post-deploy. A shared-secret header
  (`X-Quindral-Admin-Token`, checked against the `QUINDRAL_ADMIN_TOKEN` env
  var; unset in local dev = no auth required, unchanged prior behavior) now
  unlocks a second mode: full records (unredacted), `?since=<unix ts>` +
  `?limit=<n>` for a resumable cursor pull — this is the evaluator's access
  path, distinct from the browser-facing public feed above.

### Offline Quality Judging (data model built, judge itself deferred)
**Decision this session**: don't run an LLM-as-judge (or any automated
fact-checker) on the live request path — it's another API call on every
response, which cuts against the app's cost/environmental positioning the
same way a full-LLM classifier would (SPEC's original classifier
rationale, now applied here too). Instead, judging should happen **offline**,
as a batch job against logged (prompt, response) pairs, once real traffic
exists (post-Railway-deploy).

**What's actually built now**: the data the future judge would need.
`outcome_log.log_outcome()` now accepts and stores `prompt_text` /
`response_text` — real content, not just metadata. Present on any record
tied to a real execution (accepted/escalated/response votes); `None` on
routing-only votes, which never see a response. **This is a genuine privacy
posture change**, called out explicitly rather than folded in quietly:
response text previously never left the browser at all (the entire BYOK
pitch); now it's sent to and retained on Quindral's server for this
purpose. The key still never touches the server, in any request — those are
two independent claims, and the Settings page now states both separately
instead of one implying the other.

**Not built**: the judge itself (LLM-as-judge or other fact-checker), the
batch script that would run it against `outcomes.jsonl`, and the mechanism
that would feed its verdicts back into `_rank()`. When built, its
integration point is designed to be the same shape as
`negative_response_rate` — a per-(model, route) penalty function, just
sourced from automated judging instead of (or alongside) human votes, so no
new routing logic is needed, only a new data source.

---

## Trust / Transparency (new section, built this session)
Direct response to real user feedback ("how do I know you can't see my
key?"): a **live network-activity log** in Settings, not just a paragraph of
prose. Every request the tab makes is logged in order with a badge —
🔓 **direct to provider** (the actual key-bearing calls, e.g.
`api.openai.com`) vs. **quindral server** (routing/pricing/feedback, never
carries the key) — and the copy explicitly invites cross-checking against
the browser's own DevTools Network tab rather than asking for trust.
Updated this session to also disclose the new prompt/response content
retention (see "Offline Quality Judging" above) — the log now distinguishes
"no key involved" from "includes your prompt + response text," since those
are different claims and conflating them would be dishonest by omission.

---

## UI Decisions (this session)
- **Budget-allocator sliders**: cost/env/latency sliders changed from
  independent 0–1 values to linked 0–100% values that always sum to 100
  (dragging one proportionally rebalances the other two). Purely a display
  change — the underlying weighted-sum ranking is scale-invariant, so
  behavior is identical; the old sliders just looked more "maxed out" than
  they meant.
- **Preset-gate → free-text unlock**: users must try 5 presets before a
  free-text query box unlocks (persisted in `localStorage`), replacing an
  earlier "both visible at once" layout that read as confusing. Presets
  reordered so vision/reasoning examples (where slider weight changes
  visibly move the outcome) lead the list — code/simple queries were found
  to have one Pareto-dominant model regardless of weights (real registry
  property, not a bug), so they're de-prioritized as first impressions.
- **Response area is a stacked list, not a single overwritten box**: running
  the main "Run on this model" button, any eligible-model pill's inline
  "▶ Run" (new — every eligible model, not just the chosen one, is now
  runnable), or "Try a better model" all *append* a new entry rather than
  replacing the current one. Each entry has its own independent response
  text, escalation note, cost, and rate/comment/try-better controls.
- **"Recent" chat history rearchitected from localStorage to fully
  server-backed** (a real chain of bugs led here, in order):
  1. Clicking a past "Recent" chat updated internal state but left the
     visible query box blank (fixed: repopulate the input field).
  2. `addChat()` only ever saved the routing *decision*, never the actual
     executed response — reopening a chat showed the explanation again but
     the real answer was gone, only ever held in memory (fixed:
     `saveChatRun()`/`applyStoredRun()`, generalized to an array once the
     response area became a stacked list).
  3. The sidebar's click handlers captured a stale `chats` array snapshot
     at last-render time; `saveChatRun()` updated `localStorage` without
     re-rendering the list, so a chat you'd already interacted with could
     replay pre-run state while a freshly-rendered one looked fine —
     "doesn't always show old responses" (fixed: re-read from storage at
     click time, not a closure).
  4. User then asked the deeper question directly: **why rely on
     `localStorage` at all when a durable server-side log already exists?**
     Two real gaps surfaced: (a) `localStorage` is scoped per browser
     *origin, including port* — switching dev ports looked exactly like
     history vanishing on server restart, though it wasn't actually tied to
     the server; (b) the routing **trace itself** (eligible models,
     exclusions, explanation) was never durably logged anywhere — only
     votes/runs were (`/feedback`, `/outcome`). Resolved by:
     - A new `"routed"` outcome type (`outcome_log.py`) logging the full
       trace (`trace_detail`: eligible_models/excluded/binding_criterion/
       explanation/relaxed_filters) plus `prompt_text`, gated by the same
       per-user redaction as everything else in `/outcomes`.
     - `/route` now generates `prompt_id` **server-side** (`uuid4`, in the
       response) — single source of truth, replacing a client-generated
       `Date.now()` with no durable tie to anything. Best-effort logging:
       a log failure never blocks the actual routing response.
     - "Recent" now reconstructs chats by fetching `/outcomes` and grouping
       records by `prompt_id` (`groupRecordsIntoChats()`) — no client-side
       store at all. `localStorage`-based `loadChats`/`saveChats`/`addChat`/
       `saveChatVote`/`saveChatRun` all removed.
  A genuine integration test (real `api.py` process, only the OpenAI call
  stubbed) driving this end-to-end caught two more real bugs along the way:
  `router.py`'s weighted scoring crashed with an unhandled 500 on `weights`
  containing explicit `None` (not just missing keys — `dict.get`'s default
  only covers the latter; a malformed client payload, e.g. a `NaN` slider
  serializing to JSON `null`, triggered this) — fixed by normalizing weights
  once at the top of `route()`. And "New chat" hid the result card without
  clearing its text (harmless while hidden, but sloppy — same class of bug
  as the `runResponse` one caught earlier) — fixed for consistency.
- **Export chats**: sidebar button, now exports the server-reconstructed
  chat groups (via the same `/outcomes` + grouping path) instead of raw
  `localStorage`.

---

## Netlify Demo Scope (`netlify/v0.7/`)
A separate, intentionally backend-less static build for a drag-and-drop
demo (no server at all — `router.js` is a hand-ported 1:1 JS mirror of
`classifier.py` + `router.py` + `explain.py` + `model_registry.py`'s
`filter_by()`). Kept in sync with the *routing algorithm* only. Explicitly
**does not** include (all require a real backend or were scoped out of the
static build): real model execution (`providers.js` isn't even copied
there), response-quality voting, the Feedback page, quality-penalty
ranking, `/pricing`/`/outcome` endpoints, or the "only models I have a key
for" checkbox (built root-app-only, never ported). It does include the
fictional demo balance, preset-gating, budget-allocator sliders, and the
access-nudge note — the UI/UX decisions that don't need a server.

---

## User-Facing Criteria & Controls

- **EU-only (privacy)** — hard filter, not a weight
- **Only models I have a key for** — hard filter, opt-in, root app only (this session)
- **Cost** — default primary optimization target
- **Environmental impact** — weighted criterion, user-adjustable
- Users can set a default global optimization target and override per-session

Resolved this session: UI mechanism is **budget-allocator sliders** (not presets/toggles) — see "UI Decisions."

---

## Classifier Roadmap (Cold Start → Learned)

### Phase 0: Cold start, no training data — **built**
- Embedding similarity to a small set of reference examples per capability route (few-shot centroid matching), cheap, ~10-50ms, CPU-only
- Heuristic features layered on top: token count, code-fence/regex detection, math symbols, question vs. imperative phrasing, image/file attached, detected language, plus a **retrieval-necessity heuristic** (`needs_current_info`) added this session
- Simple decision table mapping signals → route (code route, vision route, long-context route, simple/cheap route, general reasoning route — mirrors aim2balance's "five capability routes" concept; taxonomy now finalized as these five)
- `eval_classifier.py` exists to measure accuracy against the reference set

### Phase 1: Define and log the HITL feedback signal — **explicit signal built, implicit not**
**Resolved this session** (was an open question): v1 launched with **explicit thumbs up/down**, not implicit-only — and went further, splitting it into routing votes and response-quality votes (see "HITL Feedback" above). Implicit signals (regenerate, correction-follow-up, manual escalation) remain schema-defined in `outcome_log.py` but nothing in the UI triggers them yet.

Log schema per routed request (actual current shape, `outcome_log.py`):
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
- Once sufficient labeled examples exist per route (order of thousands), train logistic regression or GBT (XGBoost/LightGBM) on embedding + heuristic features → predicts probability of successful outcome per candidate model
- Replaces hand-built decision table
- Retrain on a schedule (weekly/monthly), not online-learning from day one — easier to validate and roll back
- The quality-penalty mechanism built this session (route-scoped `negative_response_rate`) is a simpler, non-learned precursor to this — worth noting it already captures some of what Phase 2 would formalize

### Phase 3: Confidence-based escalation — **built**, see "Escalation" above

---

## Cold-Start Example Data — Sourcing Plan
Does NOT need to be entirely hand-generated. In order of effort:
1. **Public benchmark datasets** with existing task-type/difficulty annotations (e.g. datasets used to train RouteLLM, MT-Bench-style prompts, open instruction datasets tagged by category) — free, fast, generic (won't match this app's actual user query mix)
2. **Synthetic generation** — use an LLM to generate diverse example queries per route, label with a stronger model or spot-check manually — cheap, fast, one-time cost, risk of synthetic/real distribution mismatch
3. **Real logged usage post-launch** — the actual gold-standard signal, only exists after traffic starts (blocked on Railway deploy)
4. **Manual labeling** — hand-tag a sample (not the whole set) to calibrate/sanity-check the automated methods

Recommended plan: seed Phase 0 with public datasets + synthetic generation (Phase 0 itself needs zero training data since it's heuristic/embedding-based — this sourcing is really for Phase 2's learned classifier). Real logged data becomes dominant within weeks of launch.

---

## Open Risks & Design Questions (unresolved, need decisions before/during build)

- **Class imbalance**: most real traffic will skew toward "simple/cheap" queries; code/vision/long-context will be rarer. Classifier must handle this skew or will under-route to specialist models.
- **Label noise from implicit signals**: a regenerate doesn't always mean "model was wrong" (could be style preference). Implicit signals need weighting/filtering, not treated as clean ground truth. (Moot until implicit signals are actually wired up — see Phase 1 above.)
- **Feedback loop / exploration-exploitation risk**: if the router only trusts its own past decisions, errors compound (a route never tested on harder queries never learns it could handle them, or vice versa). Consider periodically routing a small % of traffic to alternate models to keep collecting comparison data. The quality-penalty mechanism is soft specifically to mitigate this, but doesn't solve the exploration side — nothing currently forces occasional traffic to a penalized model to see if it's improved.
- **Failure mode preference**: under-routing (cheap model struggles, bad UX) vs. over-routing (expensive model overused, undercuts cost/environmental pitch) — should be a tunable default, not hardcoded, but a default stance is needed.
- **Multilingual support**: aim2balance runs a separate language-aware routing layer on top of their capability routes. If this app supports multiple languages, embeddings/classifier need to generalize across languages or require per-language calibration.
- **Classifier runtime location**: server/backend vs. edge/on-device (mobile) — not yet decided by user.
- **UI mechanism for user-adjustable criteria weights**: ~~sliders vs. presets vs. toggles~~ **resolved — budget-allocator sliders, see "UI Decisions."**
- **Judge model / fact-checking mechanism**: deferred (see "Offline Quality Judging") — LLM-as-judge vs. external fact-checking API vs. cross-model consistency check not yet chosen, blocked on having real traffic to test against.
- **Server hardening**: `api.py` is explicitly the stdlib prototype server (no rate limiting, no HTTPS, single-process) — fine for a controlled demo, not for public/production traffic at scale. `/outcomes` now has token auth + per-user redaction (this session), but `/route`/`/feedback`/`/outcome`/`/pricing` remain fully open with no rate limiting — a real ASGI server is still needed before wide distribution, separate from the Railway deploy itself.
- **Multi-provider execution**: only OpenAI is wired for real calls. Anthropic, Google, Mistral, Moonshot, Microsoft, and gateway (OpenRouter-style) execution are all unbuilt — each is its own increment, not yet scheduled.
- **Enterprise/team tier**: identified as the real monetization target, needs an org/multi-user data model the app doesn't have (today: single browser, no accounts) — not scoped in detail yet.

---

## Decisions Already Made
- Independent app, not a wrapper/competitor built on aim2balance's platform — inspired by it only
- Routing decisions happen **server-side** (not client-side proxy)
- Classifier will **not** be a full LLM call on every request — cheap embedding/heuristic/trained-classifier approach instead
- Cost is the **default** primary optimization criterion
- EU-only is a **hard filter**, not a weighted criterion
- Reasoning trace is structured/logged separately from the natural-language explanation shown to users
- HITL outcome signals will inform future routing decisions, but the system starts on heuristics — this is explicitly a bootstrap-then-learn design, not learned-from-day-one
- **BYOK is free and permanent, not a stepping stone to managed keys** — managed/hosted key reselling was explicitly ruled out as a monetization path
- **Model execution happens directly from the browser to the provider, never relayed through Quindral's server** — chosen specifically so the "we can't see your key" claim is provably true, not just asserted
- **Automated quality judging (LLM-as-judge or similar) will run offline against logged data, never on the live request path** — same cost/environmental reasoning as the original no-full-LLM-classifier decision
- **Prompt and response text are now retained server-side** (for future offline judging) — a deliberate, disclosed privacy tradeoff, distinct from and not implying anything about key privacy (which remains absolute)
- Monetization: **BYOK stays free**; **#4 (flat per-route fee, decoupled from token cost)** is the near-term mechanism (prototyped with a fictional balance); **#3 (enterprise governance)** is the identified long-term revenue target, not yet built; managed keys and flat subscription were considered and not pursued
- **Log persistence and evaluator access are env-var-driven, not hardcoded** — `QUINDRAL_LOG_PATH` (point at a Railway volume) and `QUINDRAL_ADMIN_TOKEN` (unlocks the evaluator's full/paginated `/outcomes` access) are both optional and default to prior local-dev behavior when unset

## Not Yet Decided
- Classifier runtime: server vs. edge/on-device
- Native vs. cross-platform for mobile
- Judge/fact-checking mechanism (once offline judging is actually built)
- Whether/how to force occasional exploration traffic to quality-penalized models
- Enterprise tier data model (orgs, multi-user, roles)
- Real Stripe wiring details for the routing fee (today: fictional `localStorage` balance only)
