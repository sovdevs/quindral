"""API entrypoint:
  POST /route    {"query": "...", "eu_only": bool, "weights": {...}, "user_id"?} ->
                 routing trace + explanation + a server-issued "prompt_id" (single
                 source of truth for grouping everything about this prompt). If
                 user_id is present, the full trace is durably logged as a "routed"
                 outcome (see outcome_log.py) — this is what makes a server-backed
                 "Recent" history possible; without user_id, routing still works,
                 it's just never persisted (best-effort, never blocks the response).
  POST /feedback {"prompt_id", "user_id", "route_chosen", "model_used",
                  "outcome": "thumbs_up"|"thumbs_down"|"response_thumbs_up"|"response_thumbs_down",
                  "comment"?, "prompt_text"?, "response_text"?} — routing-decision votes and
                 response-quality votes, kept as separate categories (see outcome_log.py) even
                 though they share a prompt_id.
  POST /pricing  {"models": [...]} -> {model_name: {cost_per_1k_input, cost_per_1k_output}}
  POST /outcome  {"route_chosen", "model_used", "outcome", "criteria_weights_active", "usage"?,
                  "prompt_text"?, "response_text"?}
                 non-vote implicit signals (e.g. "escalated") — see SPEC.md Phase 1
  GET  /outcomes -> last 200 logged records, newest first — powers the Feedback page.
                 Unauthenticated: prompt_text/response_text are stripped EXCEPT for
                 records whose user_id matches the X-Quindral-Client-Id header (i.e.
                 you can see your own full history, not everyone else's real content
                 — this is a public multi-user feed once deployed, not a private log).
                 Authenticated (X-Quindral-Admin-Token matching QUINDRAL_ADMIN_TOKEN
                 env var): full records incl. all users' text, plus ?since=<unix ts>
                 and ?limit=<n> for a resumable pull — this is the offline evaluator's
                 access path (see outcome_log.py docstring), not meant for browsers.
  POST /judge    {"prompt_id", "model_used", "route_chosen"?, "judge_pass": bool,
                  "judge_reasoning"?, "judge_model"?} — ADMIN-ONLY (X-Quindral-Admin-Token
                 required, 403 otherwise). The offline evaluator's verdict on one
                 (prompt, response) pair, logged as a "judged" outcome — see
                 outcome_log.judge_penalty_rate and router.py's _rank for how this
                 feeds into ranking as its own signal, separate from human votes.
  GET  /model-selector/categories -> task categories model-selector ranks within (see
                 model-selector/src/model_selector/taxonomy.py) — powers the Task Finder view's
                 category dropdown.
  POST /model-selector/rank {"description", "sample_input"?, "category"?, "difficulty"?} ->
                 cheapest-that-works ranking for a task, from model-selector's local catalog
                 (model-selector/README.md), plus "samples" (the actual probe prompts backing
                 any pass-rate evidence shown, for transparency) and each candidate's "released"
                 date (when OpenAI's /v1/models first listed it — NOT a training-data knowledge
                 cutoff, which no discovery source here exposes; see
                 model_selector/discovery/openai_docs.py's models_from_api docstring) and
                 "description" (short catalog label, e.g. "GPT-4.1 nano") —
                 appends the request to MS_HISTORY_PATH (task_history.jsonl) as a side effect.
                 Empty/no recommendation until `model-selector discover` (+ probes) has run.
  GET  /model-selector/history?limit=<n> -> past Task Finder requests, newest first — powers the
                 "previously run" list in the UI.
  POST /model-selector/trial {"description", "sample_input", "rubric"? or "expect"?, "category"?,
                 "difficulty"?, "max_models"?} -> like /model-selector/rank, but ACTUALLY CALLS
                 real OpenAI models (cheapest-first, stops at first PASS; Google/Gemini has no
                 execution adapter yet) with the given sample and grades the real response —
                 spends real money on this server's own OPENAI_API_KEY (never a user's BYOK key).
                 Requires OPENAI_API_KEY set on the server; 400 if missing. Both rank and trial
                 responses carry a server-minted "result_id" for /model-selector/feedback below.
  POST /model-selector/feedback {"result_id", "vote": "up"|"down", "client_id"?} -> thumbs
                 up/down on one specific Task Finder result (rank or trial), keyed by
                 (result_id, client_id, timestamp) — appended to task_feedback.jsonl. client_id
                 falls back to the X-Quindral-Client-Id header (same anonymous per-browser id
                 used by /feedback's routing votes) if not in the body.
  GET  /         serves the web UI (index.html)

  PRIVACY: /feedback and /outcome accept prompt_text/response_text for a future
  offline quality-judging pipeline (see outcome_log.py docstring) — this is real
  content retention, including model responses that otherwise never reach this
  server at all (BYOK calls go straight from the browser to the provider). The
  frontend's network-activity log (index.html) discloses this to the user live.

  RATE LIMITING: per-IP sliding window (see RateLimiter) on every POST endpoint
  and on GET /outcomes (except admin-authenticated requests, which are exempt —
  see _rate_limited). Default 60 requests / 60s, overridable via
  QUINDRAL_RATE_LIMIT_MAX / QUINDRAL_RATE_LIMIT_WINDOW. Static file serving
  (/, /logo.png, /byok.js, /providers.js) is NOT limited.

Stdlib http.server only, no framework dep installed yet — swap for
FastAPI/Flask when request volume or middleware needs (auth, validation,
docs) outgrow this. Run: python3 api.py --serve
"""
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from router import route, DEFAULT_WEIGHTS
from explain import explain
from outcome_log import log_outcome, get_user_vote, read_outcomes, VOTE_OUTCOMES, RESPONSE_VOTE_OUTCOMES, ALL_VOTE_OUTCOMES, VALID_OUTCOMES
from model_registry import REGISTRY


def _load_dotenv(path: Path):
    """Same minimal .env loader as evaluate.py (kept as a separate copy —
    it's ten lines, not worth a shared module for). Needed so `OPENAI_API_KEY`
    (added to .env for model-selector) actually reaches this process; nothing
    else here previously read .env at all."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).parent / ".env")

# model-selector lives as a standalone sibling project (its own pyproject.toml,
# tests, catalog) rather than merged into this package — see
# model-selector/HANDOFF.md. Add its src/ to the path instead of installing it,
# so Quindral's own deps/build stay untouched.
sys.path.insert(0, str(Path(__file__).parent / "model-selector" / "src"))
from model_selector.config import Settings as MSSettings
from model_selector.store import Store as MSStore
from model_selector.ranker import rank as ms_rank, trial as ms_trial, TaskSpec as MSTaskSpec, CATEGORY_SUITE
from model_selector.probes.runner import load_suite, ProbeRunner
from model_selector.taxonomy import CATEGORIES as MS_CATEGORIES

# Same catalog Quindral's own data dir convention would use; empty/no
# recommendation until `model-selector discover` (+ probes) has actually
# been run against it — see model-selector/README.md.
_ms_settings = MSSettings(home=Path(os.environ.get("QUINDRAL_DATA_DIR", "./data")) / "model_selector").ensure()
_ms_store = MSStore(_ms_settings.db_path)

# Every Task Finder rank request gets appended here — this is the "previously
# selected tasks" history the UI lists, and it's a plain JSONL append (same
# pattern as outcome_log.py's outcomes.jsonl) rather than a new sqlite table,
# since it's just a browsing convenience, not something ranked/queried.
MS_HISTORY_PATH = _ms_settings.home / "task_history.jsonl"


def _log_task_finder_history(task: "MSTaskSpec", result: dict) -> None:
    """`result` is the dict shape returned to the client (category/
    difficulty/classified_by/recommendation) — same for both a plain rank
    and a combined multi-provider rank, so this doesn't care which produced
    it. Mints and stamps `result["result_id"]` in place — the same id the
    client gets back, so a later thumbs vote on this specific result
    (see _handle_model_selector_feedback) can reference it."""
    result_id = uuid.uuid4().hex
    result["result_id"] = result_id
    rec = result.get("recommendation")
    recommendation = {"model_id": rec["model_id"], "est_cost_per_call": rec.get("est_cost_per_call")} if rec else None
    rec = {
        "timestamp": time.time(),
        "result_id": result_id,
        "description": task.description,
        "sample_input": task.sample_input,
        "category": result["category"],
        "difficulty": result["difficulty"],
        "classified_by": result["classified_by"],
        "recommendation": recommendation,
    }
    try:
        with open(MS_HISTORY_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass  # best-effort, never blocks the actual rank response


MS_FEEDBACK_PATH = _ms_settings.home / "task_feedback.jsonl"


def _log_task_finder_feedback(result_id: str, vote: str, client_id: str) -> dict:
    """Thumbs up/down on one specific Task Finder result. Uniquely identified
    by (result_id, client_id, timestamp) — result_id ties it back to the
    exact recommendation shown (see _log_task_finder_history), client_id is
    the same anonymous per-browser id the rest of the app uses (CLIENT_ID in
    index.html, sent as X-Quindral-Client-Id — no accounts system, so this is
    the same "who" concept as everywhere else in Quindral, not a new one)."""
    rec = {"timestamp": time.time(), "result_id": result_id, "vote": vote, "client_id": client_id}
    with open(MS_FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _read_task_finder_history(limit: int = 20) -> list[dict]:
    if not MS_HISTORY_PATH.exists():
        return []
    with open(MS_HISTORY_PATH) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines[-limit:]][::-1]  # newest first


def _task_finder_samples(category: str, difficulty: str) -> list[dict]:
    """The full probe suite for this category — this is the "what sample was
    used" transparency behind any pass-rate shown. Deliberately NOT filtered
    to >= difficulty: ranker.rank() itself falls back to evidence from ANY
    difficulty when nothing exists at/above the task's level (see
    ranker.py's `relevant`/`or` fallback), so a same-filtered list here would
    silently hide the very prompts a shown pass_rate is actually backed by."""
    suite_name = CATEGORY_SUITE.get(category)
    if not suite_name:
        return []
    try:
        suite = load_suite(suite_name)
    except Exception:
        return []
    return [
        {"id": p["id"], "difficulty": p["difficulty"], "input": p.get("input") or f"[generated: {p.get('generator')}]"}
        for p in suite.get("prompts", [])
    ]


# OpenAI id patterns that carry real web-search access even though they sit
# in the "reasoning" category, not "deep_research" (e.g. gpt-4o-search-preview)
# — used only to annotate search_grounding honestly, not to change routing.
_OPENAI_WEB_SEARCH_ID_RE = re.compile(r"search-preview|deep-research")


def _attach_model_card(candidates: list[dict]) -> None:
    """Tags each candidate with what model-selector's catalog actually knows
    about it beyond price/pass-rate, honestly reflecting that OpenAI's and
    Google's docs expose different things:
    - `released`: OpenAI's /v1/models "first listed" date, or Google's own
      "Latest update" field (the closest per-provider analogue — neither is
      a training cutoff) — whichever this candidate's provider has.
    - `knowledge_cutoff`: Google and Anthropic's docs both publish this per-
      model (Anthropic's overview table even splits it into "reliable" vs.
      "training data" cutoff — the reliable one is used here); OpenAI's
      don't (see gemini.py/claude.py module docstrings) — None for OpenAI
      candidates rather than a fabricated guess.
    - `search_grounding`: real for Google (its own "Search grounding"
      capability field); inferred for OpenAI from category/id (deep_research
      category models and *-search-preview ids genuinely get a live
      web_search tool wired in providers.js/adapters.py); unknown for
      Anthropic (its overview table doesn't list a web-search capability
      column at all — left False rather than guessed, same "absent data,
      not a false claim" policy as Gemini's Deep Research agent pages).
    - `description`: short catalog label, from whichever provider's docs.
    This catalog is much thinner than Quindral's own 17-model registry (no
    region here) — this is the ceiling of what's available across OpenAI
    (~150 models), Google (~50 models), and Anthropic (4 current models)."""
    for c in candidates:
        provider = c.get("provider", "openai")
        row = _ms_store.db.execute(
            "SELECT description, extra FROM models WHERE provider=? AND model_id=?", (provider, c["model_id"])
        ).fetchone()
        extra = json.loads(row["extra"]) if row and row["extra"] else {}
        c["description"] = (row["description"] if row else None) or None
        if provider == "google":
            c["released"] = extra.get("latest_update")
            c["knowledge_cutoff"] = extra.get("knowledge_cutoff")
            c["search_grounding"] = bool(extra.get("search_grounding"))
        elif provider == "anthropic":
            c["released"] = None  # no "first listed"/"latest update" field on Anthropic's overview page
            c["knowledge_cutoff"] = extra.get("reliable_knowledge_cutoff")
            c["search_grounding"] = False  # not published on this page — absence of data, not "unsupported"
        else:
            c["released"] = extra.get("api_created")
            c["knowledge_cutoff"] = None
            c["search_grounding"] = bool(
                c.get("category") == "deep_research" or _OPENAI_WEB_SEARCH_ID_RE.search(c["model_id"]))


MS_PROVIDERS = ("openai", "google", "anthropic")


def _combined_rank(task: "MSTaskSpec") -> dict:
    """Ranks across every provider model-selector knows about, not just
    OpenAI — this is the "clever enough to figure it out" piece: a task that
    needs live/current information (routed to the deep_research category by
    taxonomy.py's recency keywords) is only useful on a model that can
    actually reach the live web. OpenAI candidates qualify by construction
    (being in the deep_research category on the OpenAI side already implies
    the adapter wires in a real web_search tool, see providers.js) — every
    other provider is filtered by its own `search_grounding` flag, which is
    real+verified for Google and unpublished (defaults False, never assumed
    True) for Anthropic; a candidate without it is excluded outright for a
    deep_research task rather than shown as a false option that would just
    hallucinate.
    This is a heuristic, not a full capability-requirements engine — if
    real usage shows it guessing wrong often enough, the next step is an
    explicit per-category capability-requirements table (or a user-facing
    "what does this task need" checklist) rather than more keyword tuning."""
    per_provider = {p: ms_rank(_ms_store, task, provider=p) for p in MS_PROVIDERS}
    base = per_provider["openai"]  # classify() doesn't depend on provider; any provider's category/difficulty is representative
    combined: list[dict] = []
    for provider, r in per_provider.items():
        for c in r.candidates:
            d = asdict(c)
            d["provider"] = provider
            combined.append(d)
    _attach_model_card(combined)
    if base.category == "deep_research":
        combined = [d for d in combined if d["provider"] == "openai" or d["search_grounding"]]
    combined.sort(key=lambda d: (d["est_cost_per_call"] is None, d["est_cost_per_call"] or 0))
    rec = next((d for d in combined if d.get("meets_bar")), None)
    unprobed = sorted({m for r in per_provider.values() for m in r.unprobed})
    return {
        "category": base.category, "difficulty": base.difficulty, "classified_by": base.classified_by,
        "candidates": combined, "recommendation": rec, "unprobed": unprobed,
    }

# Shared-secret for the evaluator's bulk-pull access to /outcomes — there's no
# real accounts/auth system yet, so this is the simplest thing that isn't
# "wide open." Unset in local dev (no token required, matches prior behavior).
ADMIN_TOKEN = os.environ.get("QUINDRAL_ADMIN_TOKEN")


class RateLimiter:
    """Per-IP sliding-window limiter, in-memory. No locking: `serve()` uses
    the stdlib's single-threaded HTTPServer (not Threading), so requests are
    already handled one at a time — this would need a lock if that ever
    changes. Not shared across replicas either (fine at today's single-
    instance Railway scale; would need a real store, e.g. Redis, to work
    correctly with >1 instance)."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        q = self._hits[key]
        while q and now - q[0] > self.window_seconds:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True


# Defaults: generous enough for normal interactive use (routing a bunch of
# queries, voting, running models in one session), restrictive enough to
# stop a script hammering the server. Env-configurable since "right" depends
# on real traffic patterns we don't have yet.
RATE_LIMIT_MAX = int(os.environ.get("QUINDRAL_RATE_LIMIT_MAX", "60"))
RATE_LIMIT_WINDOW = float(os.environ.get("QUINDRAL_RATE_LIMIT_WINDOW", "60"))
_rate_limiter = RateLimiter(RATE_LIMIT_MAX, RATE_LIMIT_WINDOW)

INDEX_HTML = Path(__file__).parent / "index.html"
LOGO_PNG = Path(__file__).parent / "logo.png"
BYOK_JS = Path(__file__).parent / "byok.js"
PROVIDERS_JS = Path(__file__).parent / "providers.js"


class Handler(BaseHTTPRequestHandler):
    def _rate_limited(self) -> bool:
        """Sends a 429 and returns True if this client is over the limit —
        callers should `return` immediately when this is True."""
        if not _rate_limiter.allow(self.client_address[0]):
            self._send(429, {"error": f"rate limit exceeded, max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW:g}s"})
            return True
        return False

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self._send_file(INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/logo.png":
            self._send_file(LOGO_PNG, "image/png")
        elif path == "/byok.js":
            self._send_file(BYOK_JS, "text/javascript; charset=utf-8")
        elif path == "/providers.js":
            self._send_file(PROVIDERS_JS, "text/javascript; charset=utf-8")
        elif path == "/outcomes":
            self._handle_list_outcomes()
        elif path == "/model-selector/categories":
            self._send(200, {"categories": [{"key": k, "label": c.label} for k, c in MS_CATEGORIES.items()]})
        elif path == "/model-selector/model":
            mid = parse_qs(urlsplit(self.path).query).get("id", [""])[0]
            res = _ms_store.lookup(mid) if mid else {"error": "'id' query param is required"}
            self._send(200 if res.get("found") else 404 if mid else 400, res)
        elif path == "/model-selector/history":
            limit = int(parse_qs(urlsplit(self.path).query).get("limit", ["20"])[0])
            self._send(200, {"tasks": _read_task_finder_history(limit)})
        else:
            self._send(404, {"error": "not found"})

    def _handle_list_outcomes(self):
        query = parse_qs(urlsplit(self.path).query)
        is_admin = bool(ADMIN_TOKEN) and self.headers.get("X-Quindral-Admin-Token") == ADMIN_TOKEN
        # Admin (the evaluator, holding the real secret) is exempt — it's a
        # trusted, known caller doing legitimate paginated bulk pulls, not
        # the anonymous public traffic this limiter exists to blunt.
        if not is_admin and self._rate_limited():
            return

        if is_admin:
            # Evaluator path: full content, resumable via ?since=<unix ts>,
            # oldest-first so a cursor (max timestamp seen) works naturally.
            since = query.get("since", [None])[0]
            limit = int(query.get("limit", ["1000"])[0])
            records = list(read_outcomes())
            if since is not None:
                records = [r for r in records if r.get("timestamp", 0) > float(since)]
            self._send(200, {"records": records[:limit]})
            return

        # Public path: last 200, newest first, and only YOUR OWN records keep
        # prompt_text/response_text — this is a shared multi-user feed once
        # deployed, not a private log, so other users' real content is redacted.
        client_id = self.headers.get("X-Quindral-Client-Id")
        records = list(read_outcomes())[-200:]
        records.reverse()
        redacted = []
        for r in records:
            if r.get("user_id") != client_id:
                r = {**r, "prompt_text": None, "response_text": None}
            redacted.append(r)
        self._send(200, {"records": redacted})

    def _send_file(self, path: Path, content_type: str):
        if not path.exists():
            self._send(404, {"error": "not found"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/judge":
            # Admin-only, same exemption pattern as GET /outcomes: a
            # correctly-authenticated evaluator is trusted, known traffic,
            # not the anonymous public traffic this limiter exists to blunt.
            # An unauthenticated caller still gets rate-limited before the
            # 403, so this can't be used to bypass the limiter by hammering
            # a wrong token.
            self._handle_judge()
            return
        if self._rate_limited():
            return
        if self.path == "/route":
            self._handle_route()
        elif self.path == "/feedback":
            self._handle_feedback()
        elif self.path == "/pricing":
            self._handle_pricing()
        elif self.path == "/outcome":
            self._handle_outcome()
        elif self.path == "/model-selector/rank":
            self._handle_model_selector_rank()
        elif self.path == "/model-selector/trial":
            self._handle_model_selector_trial()
        elif self.path == "/model-selector/feedback":
            self._handle_model_selector_feedback()
        else:
            self._send(404, {"error": "not found"})

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _handle_route(self):
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return

        query = body.get("query")
        if not query:
            self._send(400, {"error": "'query' is required"})
            return

        allowed_models = body.get("allowed_models")
        trace = route(
            query,
            weights=body.get("weights"),
            eu_only=body.get("eu_only", False),
            min_context=body.get("min_context", 0),
            allowed_models=set(allowed_models) if allowed_models is not None else None,
        )
        trace["explanation"] = explain(trace, eu_only=body.get("eu_only", False))

        # Server issues the prompt_id — single source of truth for grouping
        # everything about this prompt (routed decision, later runs, votes)
        # server-side, instead of the frontend minting its own (previously
        # Date.now(), never durably tied to anything).
        trace["prompt_id"] = uuid.uuid4().hex
        user_id = body.get("user_id")
        if user_id:
            # Best-effort: a logging failure should never break the actual
            # routing response the user is waiting on.
            try:
                log_outcome(
                    trace, "routed", body.get("weights") or DEFAULT_WEIGHTS,
                    user_id=user_id, prompt_id=trace["prompt_id"], prompt_text=query,
                    trace_detail={
                        "eligible_models": trace.get("eligible_models"),
                        "excluded": trace.get("excluded"),
                        "binding_criterion": trace.get("binding_criterion"),
                        "explanation": trace.get("explanation"),
                        "relaxed_filters": trace.get("relaxed_filters"),
                    },
                )
            except Exception:
                pass

        self._send(200, trace)

    def _handle_feedback(self):
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return

        outcome = body.get("outcome")
        user_id = body.get("user_id")
        prompt_id = body.get("prompt_id")
        if outcome not in ALL_VOTE_OUTCOMES:
            self._send(400, {"error": f"outcome must be one of {sorted(ALL_VOTE_OUTCOMES)}"})
            return
        if not user_id or not prompt_id:
            self._send(400, {"error": "'user_id' and 'prompt_id' are required"})
            return

        comment = body.get("comment")
        vote_group = VOTE_OUTCOMES if outcome in VOTE_OUTCOMES else RESPONSE_VOTE_OUTCOMES
        if comment is None and get_user_vote(user_id, prompt_id, outcomes=vote_group) == outcome:
            # already this exact vote with no new comment — no-op rather than
            # appending a redundant duplicate row (repeated votes with a
            # DIFFERENT outcome, i.e. changing your mind, still append
            # normally, further down). A comment is always new information —
            # e.g. thumbs_down logged immediately, then a comment added a
            # moment later — so it always gets its own record, never dropped.
            self._send(200, {"outcome": outcome, "unchanged": True})
            return

        fake_trace = {"classified_as": body.get("route_chosen"), "chosen": body.get("model_used")}
        try:
            record = log_outcome(
                fake_trace, outcome, body.get("criteria_weights_active") or {},
                user_id=user_id, prompt_id=prompt_id, comment=comment,
                prompt_text=body.get("prompt_text"), response_text=body.get("response_text"),
            )
        except ValueError as e:
            self._send(400, {"error": str(e)})
            return
        self._send(200, record)

    def _handle_pricing(self):
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return

        names = body.get("models") or []
        pricing = {}
        for name in names:
            m = REGISTRY.get(name)
            if m is not None:
                pricing[name] = {
                    "cost_per_1k_input": m.cost_per_1k_input,
                    "cost_per_1k_output": m.cost_per_1k_output,
                }
        self._send(200, pricing)

    def _handle_outcome(self):
        """Non-vote implicit signals (e.g. auto-escalation) — unlike /feedback,
        no user_id/prompt_id required, no idempotency check (each escalation
        event is a distinct occurrence, not a resettable vote)."""
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return

        outcome = body.get("outcome")
        if outcome not in VALID_OUTCOMES or outcome in ALL_VOTE_OUTCOMES:
            self._send(400, {"error": f"outcome must be one of {sorted(VALID_OUTCOMES - ALL_VOTE_OUTCOMES)}"})
            return

        fake_trace = {"classified_as": body.get("route_chosen"), "chosen": body.get("model_used")}
        try:
            record = log_outcome(
                fake_trace, outcome, body.get("criteria_weights_active") or {},
                user_id=body.get("user_id"), prompt_id=body.get("prompt_id"),
                usage=body.get("usage"),
                prompt_text=body.get("prompt_text"), response_text=body.get("response_text"),
            )
        except ValueError as e:
            self._send(400, {"error": str(e)})
            return
        self._send(200, record)

    def _handle_model_selector_rank(self):
        """Cheapest-that-works ranking for a task, via model-selector's local
        catalog — a separate tool from this app's own router (see
        model-selector/README.md), not a replacement for it."""
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return

        description = body.get("description")
        if not description:
            self._send(400, {"error": "'description' is required"})
            return

        task = MSTaskSpec(
            description=description,
            sample_input=body.get("sample_input"),
            category=body.get("category") or None,
            difficulty=body.get("difficulty") or None,
        )
        try:
            result = _combined_rank(task)
        except Exception as e:
            self._send(500, {"error": f"model-selector rank failed: {e}"})
            return
        _log_task_finder_history(task, result)
        result["samples"] = _task_finder_samples(result["category"], result["difficulty"])
        self._send(200, result)

    def _handle_model_selector_trial(self):
        """Actually calls real models with the user's own sample, cheapest-
        first, stopping at the first PASS — unlike /model-selector/rank
        (free, estimate-only), this spends real money on the API key in this
        server's own environment (OPENAI_API_KEY, NOT a user's BYOK key —
        BYOK never reaches this server at all, same boundary as everywhere
        else in this app). Bounded by max_models (default 3, capped at 5) so
        a request can't cascade through the whole catalog."""
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return

        description = body.get("description")
        sample_input = body.get("sample_input")
        if not description:
            self._send(400, {"error": "'description' is required"})
            return
        if not sample_input:
            self._send(400, {"error": "'sample_input' is required for a trial — it's what actually gets sent to models"})
            return
        rubric = body.get("rubric")
        expect = body.get("expect")
        if not rubric and not expect:
            self._send(400, {"error": "'rubric' or 'expect' is required — a trial needs a way to grade the response"})
            return
        if not os.environ.get("OPENAI_API_KEY"):
            self._send(400, {"error": "server has no OPENAI_API_KEY configured — a trial can't call real models without one"})
            return

        task = MSTaskSpec(
            description=description, sample_input=sample_input,
            category=body.get("category") or None, difficulty=body.get("difficulty") or None,
        )
        max_models = max(1, min(int(body.get("max_models", 3)), 5))
        try:
            # Trial only calls real OpenAI models (no Gemini execution
            # adapter exists yet — see providers.js/adapters.py, OpenAI-only)
            # so the cascade itself stays OpenAI-scoped, cheapest-first
            # within that provider only.
            ranking = ms_rank(_ms_store, task, provider="openai")
            runner = ProbeRunner(_ms_store, _ms_settings, log=lambda *_: None)
            outcomes = ms_trial(_ms_store, runner, ranking, rubric=rubric, expect=expect, max_models=max_models)
            # Re-rank across ALL providers after the trial: the trial just
            # wrote fresh OpenAI probe evidence for this exact task (suite
            # "custom:<hash>"), which ranker.rank() prioritizes over generic
            # suite evidence — but the displayed comparison still includes
            # Gemini candidates (untested by this trial, shown as such).
            result = _combined_rank(task)
        except Exception as e:
            self._send(500, {"error": f"model-selector trial failed: {e}"})
            return
        _log_task_finder_history(task, result)
        result["samples"] = _task_finder_samples(result["category"], result["difficulty"])
        result["trial_outcomes"] = [asdict(o) for o in outcomes]
        self._send(200, result)

    def _handle_model_selector_feedback(self):
        """Thumbs up/down on one Task Finder result (was this recommendation
        actually right?). No accounts system, so identity is the same
        anonymous per-browser client id used everywhere else in this app
        (X-Quindral-Client-Id), same pattern as /feedback's routing votes."""
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return
        result_id = body.get("result_id")
        vote = body.get("vote")
        client_id = body.get("client_id") or self.headers.get("X-Quindral-Client-Id")
        if not result_id or vote not in ("up", "down"):
            self._send(400, {"error": "'result_id' and 'vote' ('up' or 'down') are required"})
            return
        if not client_id:
            self._send(400, {"error": "'client_id' is required (or X-Quindral-Client-Id header)"})
            return
        record = _log_task_finder_feedback(result_id, vote, client_id)
        self._send(200, record)

    def _handle_judge(self):
        """Offline evaluator writes verdicts here — admin-token required.
        Unlike /outcome (any anonymous client can log "accepted"/"escalated"
        for their own real usage), a "quality verdict" directly moves
        ranking (see router.py's judge_penalty_rate) — letting an
        unauthenticated caller write these would let anyone fake the
        signal, not just log their own activity."""
        is_admin = bool(ADMIN_TOKEN) and self.headers.get("X-Quindral-Admin-Token") == ADMIN_TOKEN
        if not is_admin:
            if self._rate_limited():
                return
            self._send(403, {"error": "admin token required"})
            return

        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return

        prompt_id = body.get("prompt_id")
        model_used = body.get("model_used")
        judge_pass = body.get("judge_pass")
        if not prompt_id or not model_used or judge_pass is None:
            self._send(400, {"error": "'prompt_id', 'model_used', and 'judge_pass' are required"})
            return

        fake_trace = {"classified_as": body.get("route_chosen"), "chosen": model_used}
        try:
            record = log_outcome(
                fake_trace, "judged", {}, prompt_id=prompt_id,
                judge_pass=bool(judge_pass),
                judge_reasoning=body.get("judge_reasoning"),
                judge_model=body.get("judge_model"),
            )
        except ValueError as e:
            self._send(400, {"error": str(e)})
            return
        self._send(200, record)

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # ponytail: silence default stderr access log; add real logging if this ships


def serve(port: int = None, host: str = None):
    """port/host default from the environment when unset: Railway (and most
    PaaS hosts) inject a $PORT to bind and expect 0.0.0.0, not localhost —
    without this, a "successful" deploy is still unreachable. Explicit
    args (e.g. --lan below) still override."""
    port = port if port is not None else int(os.environ.get("PORT", 8000))
    host = host if host is not None else ("0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
    HTTPServer((host, port), Handler).serve_forever()


def _lan_ip() -> str:
    """Best-effort local network IP for printing a shareable URL."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # doesn't actually send anything
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _demo():
    import threading
    import urllib.request

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/route",
            data=json.dumps({"query": "what is the capital of france?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        assert data["chosen"] is not None
        assert "explanation" in data

        bad_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/route",
            data=json.dumps({}).encode(),
            method="POST",
        )
        try:
            urllib.request.urlopen(bad_req)
            assert False, "should have raised HTTPError for missing query"
        except urllib.error.HTTPError as e:
            assert e.code == 400

        feedback_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/feedback",
            data=json.dumps({
                "prompt_id": "test-prompt-1", "user_id": "test-user",
                "route_chosen": data["classified_as"], "model_used": data["chosen"],
                "outcome": "thumbs_up",
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(feedback_req) as resp:
            fb = json.loads(resp.read())
        assert fb["outcome"] == "thumbs_up"

        # voting the SAME outcome again is a no-op, not a duplicate log entry
        with urllib.request.urlopen(feedback_req) as resp:
            fb_repeat = json.loads(resp.read())
        assert fb_repeat.get("unchanged") is True

        bad_outcome_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/feedback",
            data=json.dumps({"prompt_id": "p", "user_id": "u", "outcome": "bogus"}).encode(),
            method="POST",
        )
        try:
            urllib.request.urlopen(bad_outcome_req)
            assert False, "should have raised HTTPError for invalid outcome"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        server.shutdown()

    print("api self-check: all cases passed")


if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        if "--lan" in sys.argv:
            lan_port = int(os.environ.get("PORT", 8000))
            print(f"Listening on http://0.0.0.0:{lan_port}  (share: http://{_lan_ip()}:{lan_port})")
            serve(host="0.0.0.0", port=lan_port)
        elif "PORT" in os.environ:
            # deployed (Railway etc.) — bind 0.0.0.0:$PORT automatically
            print(f"Listening on http://0.0.0.0:{os.environ['PORT']} (deployed)")
            serve()
        else:
            print("Listening on http://127.0.0.1:8000")
            serve()
    else:
        _demo()
