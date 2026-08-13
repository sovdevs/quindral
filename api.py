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
import time
import uuid
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from router import route, DEFAULT_WEIGHTS
from explain import explain
from outcome_log import log_outcome, get_user_vote, read_outcomes, VOTE_OUTCOMES, RESPONSE_VOTE_OUTCOMES, ALL_VOTE_OUTCOMES, VALID_OUTCOMES
from model_registry import REGISTRY

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
