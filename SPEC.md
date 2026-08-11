# SPEC: Multi-Criteria LLM Orchestrator with Transparent Routing

## Status
Early design phase. This document captures architecture decisions and open questions from initial planning. No code written yet. Intended to be picked up and extended by another Claude session or a developer.

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
User query → Classifier → Candidate scoring → Hard filters → Rank → Selection + reasoning trace → LLM call → Response + explanation → Outcome logging (HITL signal)
```

### 1. Classifier
Decides task type/complexity to determine which models are even eligible. **Decision: NOT a full LLM call on every request** — this was a deliberate choice, because running an LLM classifier on 100% of traffic adds latency/cost/energy to every query just to save money on the subset that need a big model, which undercuts the whole cost/environmental pitch.

Phased approach (see "Classifier Roadmap" below):
- Phase 0: embedding similarity + heuristics, zero training data required
- Phase 1: logging real usage outcomes as labeled data
- Phase 2: lightweight trained classifier (logistic regression / gradient-boosted trees over embeddings, e.g. XGBoost/LightGBM) — cheap at inference, no GPU needed
- Phase 3: confidence-based escalation as a runtime safety net regardless of classifier maturity

Comparable prior art considered: RouteLLM (trained weak/strong classifier), Semantic Router (pure embedding similarity, open source), RoRF/Not Diamond (random forest over embeddings), Martian (interpretability-based model mapping), OpenRouter/Inworld (rule-based + confidence escalation). The dominant pattern across serious routers is **avoid a full LLM-based classifier** — favor embeddings + small trained models.

### 2. Model Registry
Table maintained per candidate model:
- Cost per input/output token
- Energy/water estimate per token (Ecologits methodology — public, reusable, same one aim2balance references)
- Hosting region (EU / non-EU)
- Context window
- Capability tags (code, vision, long-context, reasoning, creative, factual)
- Measured latency
- Data policy flags (no-train guarantee, encryption, residency)

### 3. Hard Filters vs. Soft Scoring
- **Hard filters** (exclude candidates outright, not scored): EU-only toggle, capability floor (e.g. don't route a coding task to a non-code-capable model), minimum context length requirement
- **Soft scoring** among survivors: weighted sum of normalized cost, environmental impact, latency — user-adjustable weights, **cost weighted highest by default**

### 4. Reasoning Trace (core product differentiator)
Every routing decision emits a structured object, decoupled from the human-readable explanation shown to the user:

```json
{
  "classified_as": "simple_factual_lookup",
  "eligible_models": ["mistral-small", "llama-3-8b", "gpt-4o-mini"],
  "excluded": [
    {"model": "gpt-4-turbo", "reason": "capability floor exceeded (overkill), cost 40x higher"}
  ],
  "chosen": "mistral-small",
  "binding_criterion": "cost",
  "estimated_savings": {"vs_default_large_model": "~85% energy, ~90% cost"}
}
```

A separate template layer converts this into a user-facing one-liner (e.g. "Routed to Mistral-Small because this looked like a simple factual lookup — a smaller model was sufficient, saving ~85% energy vs. a large model, and it met your EU-only filter"). Keeping structured trace and NL explanation separate allows localization, adjustable verbosity, and click-to-expand detail without recomputation.

### 5. Fallback / Escalation Loop
Cheap models can fail silently on queries above their capability. Include a lightweight confidence/quality check on the response; auto-escalate once to a stronger model if the output looks degenerate (very short, repetitive, refusal boilerplate, or classifier confidence was low). This protects trust during the period the learned router is still immature, and doubles as a labeled training example ("this route needed escalation").

---

## User-Facing Criteria & Controls

- **EU-only (privacy)** — hard filter, not a weight
- **Cost** — default primary optimization target
- **Environmental impact** — weighted criterion, user-adjustable
- Users can set a default global optimization target and override per-session

Open question: exact UI mechanism (sliders vs. presets vs. toggles) — not yet decided.

---

## Classifier Roadmap (Cold Start → Learned)

### Phase 0: Cold start, no training data
- Embedding similarity to a small set of reference examples per capability route (few-shot centroid matching), cheap, ~10-50ms, CPU-only
- Heuristic features layered on top: token count, code-fence/regex detection, math symbols, question vs. imperative phrasing, image/file attached, detected language
- Simple decision table mapping signals → route (code route, vision route, long-context route, simple/cheap route, general reasoning route — mirrors aim2balance's "five capability routes" concept, exact taxonomy still TBD)
- Expected accuracy: ~70-80%, good enough to launch and start collecting real signal

### Phase 1: Define and log the HITL feedback signal
Implicit signals (preferred for v1 — no extra UI needed):
- User regenerates/retries the response → negative signal
- User's immediate follow-up reads as a correction/rejection (cheap sentiment/keyword check on next message) → negative signal
- User manually escalates to a bigger model for the same query after the routed one answered → strong, specifically actionable negative signal ("this route under-provisioned")
- Conversation continues normally → weak positive signal

Explicit signal (optional addition, higher quality but lower volume): thumbs up/down control in UI.

**Decision needed:** whether v1 launches with implicit-only signals or adds explicit thumbs up/down from day one. (Open — not yet decided by user as of this doc.)

Log schema per routed request:
```json
{
  "query_features": {...},
  "route_chosen": "cheap_general",
  "model_used": "mistral-small",
  "outcome": "regenerated" | "escalated" | "accepted" | "thumbs_down" | "continued",
  "criteria_weights_active": {"cost": 0.7, "env": 0.3}
}
```

### Phase 2: Train lightweight classifier
- Once sufficient labeled examples exist per route (order of thousands), train logistic regression or GBT (XGBoost/LightGBM) on embedding + heuristic features → predicts probability of successful outcome per candidate model
- Replaces hand-built decision table
- Retrain on a schedule (weekly/monthly), not online-learning from day one — easier to validate and roll back

### Phase 3: Confidence-based escalation
Runtime safety net regardless of classifier maturity (see Architecture §5 above).

---

## Cold-Start Example Data — Sourcing Plan

Does NOT need to be entirely hand-generated. In order of effort:
1. **Public benchmark datasets** with existing task-type/difficulty annotations (e.g. datasets used to train RouteLLM, MT-Bench-style prompts, open instruction datasets tagged by category) — free, fast, generic (won't match this app's actual user query mix)
2. **Synthetic generation** — use an LLM to generate diverse example queries per route, label with a stronger model or spot-check manually — cheap, fast, one-time cost, risk of synthetic/real distribution mismatch
3. **Real logged usage post-launch** — the actual gold-standard signal, only exists after traffic starts
4. **Manual labeling** — hand-tag a sample (not the whole set) to calibrate/sanity-check the automated methods

Recommended plan: seed Phase 0 with public datasets + synthetic generation (Phase 0 itself needs zero training data since it's heuristic/embedding-based — this sourcing is really for Phase 2's learned classifier). Real logged data becomes dominant within weeks of launch.

---

## Open Risks & Design Questions (unresolved, need decisions before/during build)

- **Class imbalance**: most real traffic will skew toward "simple/cheap" queries; code/vision/long-context will be rarer. Classifier must handle this skew or will under-route to specialist models.
- **Label noise from implicit signals**: a regenerate doesn't always mean "model was wrong" (could be style preference). Implicit signals need weighting/filtering, not treated as clean ground truth.
- **Feedback loop / exploration-exploitation risk**: if the router only trusts its own past decisions, errors compound (a route never tested on harder queries never learns it could handle them, or vice versa). Consider periodically routing a small % of traffic to alternate models to keep collecting comparison data.
- **Privacy/EU data policy**: given the EU-only/privacy positioning, decide whether query content used for training is anonymized, aggregated, or excluded entirely — affects both data policy and what can legally/ethically be logged.
- **Failure mode preference**: under-routing (cheap model struggles, bad UX) vs. over-routing (expensive model overused, undercuts cost/environmental pitch) — should be a tunable default, not hardcoded, but a default stance is needed.
- **Multilingual support**: aim2balance runs a separate language-aware routing layer on top of their capability routes. If this app supports multiple languages, embeddings/classifier need to generalize across languages or require per-language calibration.
- **Classifier runtime location**: server/backend vs. edge/on-device (mobile) — not yet decided by user.
- **Exact capability route taxonomy**: "five capability routes" mirrors aim2balance's public description but this app's actual route categories are not yet finalized.
- **UI mechanism for user-adjustable criteria weights**: sliders vs. presets vs. toggles — not yet decided.

---

## Decisions Already Made
- Independent app, not a wrapper/competitor built on aim2balance's platform — inspired by it only
- Routing decisions happen **server-side** (not client-side proxy)
- Classifier will **not** be a full LLM call on every request — cheap embedding/heuristic/trained-classifier approach instead
- Cost is the **default** primary optimization criterion
- EU-only is a **hard filter**, not a weighted criterion
- Reasoning trace is structured/logged separately from the natural-language explanation shown to users
- HITL outcome signals will inform future routing decisions, but the system starts on heuristics — this is explicitly a bootstrap-then-learn design, not learned-from-day-one

## Not Yet Decided
- Implicit-only vs. implicit+explicit (thumbs up/down) feedback for v1
- Classifier runtime: server vs. edge/on-device
- Exact capability taxonomy (beyond the rough five-route sketch)
- UI mechanism for criteria weight adjustment
- Native vs. cross-platform for mobile
- Data retention/anonymization policy for logged queries used in training
