# Rate Limiting

Added to protect the live Railway deployment — `api.py` is the stdlib
`http.server` prototype (see its module docstring), which has no built-in
protection against a script hammering it, and it's now a real public URL.

## What's limited

- **Every POST endpoint**: `/route`, `/feedback`, `/pricing`, `/outcome`
- **`GET /outcomes`** — except admin-authenticated requests (see below)

**Not limited**: static file serving (`/`, `/logo.png`, `/byok.js`,
`/providers.js`) — these are cheap reads, not compute or log writes, and
rate-limiting page loads would just break normal browsing.

## How it works

`RateLimiter` in `api.py` — a per-IP sliding window, in memory:

```python
class RateLimiter:
    def __init__(self, max_requests, window_seconds): ...
    def allow(self, key: str) -> bool: ...
```

Each client IP (`self.client_address[0]`) gets its own deque of recent
request timestamps. On each check, timestamps older than `window_seconds`
are dropped from the front; if what's left is still `>= max_requests`, the
request is denied (`429`), otherwise it's recorded and allowed.

No locking: `serve()` uses the stdlib's single-threaded `HTTPServer` (not
`ThreadingHTTPServer`), so requests are already handled one at a time. If
that ever changes, this needs a lock around `allow()`.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `QUINDRAL_RATE_LIMIT_MAX` | `60` | Max requests per window, per IP |
| `QUINDRAL_RATE_LIMIT_WINDOW` | `60` (seconds) | Window length |

Defaults are picked to be generous for normal interactive use (routing
several queries, voting, running models in one session) while still
stopping a script from hammering the server. Not yet tuned against real
traffic — revisit if either false-positives on real users or doesn't
actually stop abuse.

## Admin exemption

Requests to `GET /outcomes` with a valid `X-Quindral-Admin-Token` header
(matching the `QUINDRAL_ADMIN_TOKEN` env var) skip the limiter entirely —
this is the future offline evaluator's bulk-pull path (see
`outcome_log.py`'s docstring and "Offline Quality Judging" in SPEC.md), a
trusted known caller doing legitimate paginated pulls, not the anonymous
public traffic this limiter exists to blunt.

## Known limitation

In-memory and per-process. Correct at today's scale (one Railway instance).
If this ever scales to multiple replicas, each instance would track its own
independent counts — a client could effectively get `max_requests × replica
count` before being limited anywhere. Would need a shared store (e.g. Redis)
to coordinate correctly across replicas. Not a concern until that's
actually the deployment shape.

## Verification

Confirmed with real threaded-server tests (not just unit-level): requests
allowed up to the limit, `429` beyond it, allowed again once the window
rolls off, and admin-token requests bypass the limit entirely regardless of
volume. See the session history / git blame around `api.py`'s
`RateLimiter` and `_rate_limited` for the test transcripts if useful as a
reference for future changes to this logic.
