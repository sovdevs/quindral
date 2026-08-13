"""Offline LLM-as-judge evaluator. See EVALUATOR.md for the full writeup.

Pulls new (prompt, response) pairs from the deployed API's GET /outcomes
(admin-authenticated), judges each with a cheap OpenAI model, and posts
pass/fail verdicts back to POST /judge — which is what makes
outcome_log.judge_penalty_rate (and therefore router.py's ranking) see them.

Only judges "accepted"/"escalated" records (the actual execution events, one
per model run) — not response_thumbs_* votes, which carry the same
prompt/response content and would just judge the same pair a second time
for no reason.

Resumable: tracks the last-processed timestamp in local/evaluator_cursor.json
(gitignored) so re-runs only fetch what's new via ?since=<cursor>.

Run manually — either export the env vars inline, or (easier, avoids
shell-quoting headaches) put them in a `.env` file next to this script:
    QUINDRAL_API_URL=https://<your-app>.up.railway.app
    QUINDRAL_ADMIN_TOKEN=...
    QUINDRAL_JUDGE_OPENAI_KEY=sk-...
then just: .venv/bin/python3 evaluate.py
`.env`/`.env.*` are already gitignored — never commit real secrets there.

QUINDRAL_JUDGE_OPENAI_KEY is deliberately its own variable, distinct from
any end user's BYOK key (which only ever lives in a browser and is never
sent to this server) — this key is spent by whoever runs this script.

Self-check (no network calls, no keys needed): python3 evaluate.py --demo
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _load_dotenv(path: Path):
    """Minimal .env loader — KEY=value per line, blank/#-comment lines
    skipped, doesn't override already-set real env vars. No new dependency
    for what's a handful of lines; not a general-purpose parser (no quoting,
    no multiline values) — fine for this script's own few keys."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).parent / ".env")

API_URL = os.environ.get("QUINDRAL_API_URL", "http://127.0.0.1:8000").rstrip("/")
ADMIN_TOKEN = os.environ.get("QUINDRAL_ADMIN_TOKEN")
JUDGE_OPENAI_KEY = os.environ.get("QUINDRAL_JUDGE_OPENAI_KEY")
JUDGE_MODEL = os.environ.get("QUINDRAL_JUDGE_MODEL", "gpt-4o-mini")

CURSOR_PATH = Path(__file__).parent / "local" / "evaluator_cursor.json"

JUDGEABLE_OUTCOMES = {"accepted", "escalated"}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair evaluator of AI assistant responses. Given a "
    "user prompt and a model's response, judge whether the response is "
    "accurate, relevant, and reasonably complete. Respond with ONLY a JSON "
    'object: {"pass": true or false, "reasoning": "one short sentence"}. '
    "Fail responses that are factually wrong, refuse without good reason, "
    "are empty or near-empty, or don't actually address the prompt."
)


def _load_cursor():
    if CURSOR_PATH.exists():
        return json.loads(CURSOR_PATH.read_text()).get("since")
    return None


def _save_cursor(since):
    CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURSOR_PATH.write_text(json.dumps({"since": since}))


def _api_get_outcomes(since):
    url = f"{API_URL}/outcomes"
    if since is not None:
        url += f"?since={since}&limit=1000"
    req = urllib.request.Request(url, headers={"X-Quindral-Admin-Token": ADMIN_TOKEN})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["records"]


def _api_post_judge(prompt_id, model_used, route_chosen, verdict):
    req = urllib.request.Request(
        f"{API_URL}/judge",
        data=json.dumps({
            "prompt_id": prompt_id, "model_used": model_used, "route_chosen": route_chosen,
            "judge_pass": verdict["pass"], "judge_reasoning": verdict.get("reasoning"),
            "judge_model": JUDGE_MODEL,
        }).encode(),
        headers={"Content-Type": "application/json", "X-Quindral-Admin-Token": ADMIN_TOKEN},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _parse_judge_response(raw_content: str) -> dict:
    """Pure parsing logic, split out from judge_response() so it's testable
    without a network call — this is the part a judge-model API/prompt
    change would actually break."""
    parsed = json.loads(raw_content)
    return {"pass": bool(parsed["pass"]), "reasoning": str(parsed.get("reasoning", ""))[:500]}


def judge_response(prompt_text: str, response_text: str) -> dict:
    """Calls the judge model, returns {"pass": bool, "reasoning": str}. Raises on API error."""
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"PROMPT:\n{prompt_text}\n\nRESPONSE:\n{response_text}"},
            ],
            "response_format": {"type": "json_object"},
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {JUDGE_OPENAI_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    return _parse_judge_response(data["choices"][0]["message"]["content"])


def run():
    if not ADMIN_TOKEN:
        sys.exit("QUINDRAL_ADMIN_TOKEN is required")
    if not JUDGE_OPENAI_KEY:
        sys.exit("QUINDRAL_JUDGE_OPENAI_KEY is required")

    since = _load_cursor()
    records = _api_get_outcomes(since)
    print(f"pulled {len(records)} record(s) since {since if since is not None else 'the beginning'}")

    judged_count, skipped_count = 0, 0
    max_timestamp = since or 0
    for r in records:
        max_timestamp = max(max_timestamp, r.get("timestamp", 0))
        if r.get("outcome") not in JUDGEABLE_OUTCOMES:
            continue
        if not r.get("prompt_text") or not r.get("response_text"):
            continue  # nothing to judge (shouldn't happen for admin-authed pulls, but be defensive)
        try:
            verdict = judge_response(r["prompt_text"], r["response_text"])
            _api_post_judge(r["prompt_id"], r["model_used"], r.get("route_chosen"), verdict)
            judged_count += 1
            print(f"  {'PASS' if verdict['pass'] else 'FAIL'} {r['model_used']} ({r.get('route_chosen')}): {verdict['reasoning']}")
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            skipped_count += 1
            print(f"  skipped one record due to error: {e}", file=sys.stderr)

    _save_cursor(max_timestamp)
    print(f"done: judged {judged_count}, skipped {skipped_count}, cursor advanced to {max_timestamp}")


def _demo():
    result = _parse_judge_response('{"pass": false, "reasoning": "hallucinated a fact"}')
    assert result == {"pass": False, "reasoning": "hallucinated a fact"}

    result2 = _parse_judge_response('{"pass": true, "reasoning": "correct and complete"}')
    assert result2["pass"] is True

    # missing reasoning -> defaults to empty string, doesn't crash
    result3 = _parse_judge_response('{"pass": true}')
    assert result3["reasoning"] == ""

    print("evaluate.py self-check: all cases passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        run()
