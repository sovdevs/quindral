"""Phase 1 HITL outcome logging: append routing decisions + outcomes as JSONL.

See SPEC.md "Phase 1: Define and log the HITL feedback signal". Now includes
explicit thumbs up/down (one vote per prompt per user) alongside the original
implicit signals (regenerated/escalated/accepted/continued).

"One vote per prompt per user" is enforced as last-write-wins, not by
rejecting repeat votes: the log stays a plain append-only file (cheap, no
read-before-write), and get_user_vote() only ever looks at the most recent
entry for a given (user_id, prompt_id) pair. A user changing their mind and
re-voting just appends another line; old votes become inert, not deleted.

ponytail: flat JSONL file, not a DB — swap for a real store once volume or
querying needs outgrow `grep`/`wc -l`.

PRIVACY: prompt_text/response_text are retained here (see log_outcome) so an
offline LLM-as-judge pipeline can score real (prompt, response) pairs without
touching the live request path — no extra API call cost per request, judging
happens as a batch job against this log. This is a deliberate retention
decision: response text previously never left the browser (BYOK's whole
point), and now it does, for this purpose. The Settings page's network-log
disclosure must stay in sync with this — see index.html's NETWORK_LOG notes.
The judge itself isn't built yet — this file only carries the data it would
need. Its output would plug into router.py the same way negative_response_rate
does: a per-(model, route) penalty in _rank(), just sourced from automated
judging instead of (or alongside) human votes.
"""
import json
import os
import time
from pathlib import Path

# Two distinct vote categories, kept separate on purpose: routing-decision
# votes ("was this the right model to pick") and response-quality votes
# ("was the actual answer good"). Both use the same prompt_id, so they must
# stay in disjoint outcome sets or get_user_vote() couldn't tell them apart.
VOTE_OUTCOMES = {"thumbs_up", "thumbs_down"}
RESPONSE_VOTE_OUTCOMES = {"response_thumbs_up", "response_thumbs_down"}
ALL_VOTE_OUTCOMES = VOTE_OUTCOMES | RESPONSE_VOTE_OUTCOMES
# "routed" logs the routing DECISION itself (trace + explanation), separate
# from "accepted"/"escalated" which log an actual executed response. Without
# this, the trace only ever lived in the /route HTTP response and the
# browser's localStorage — nothing durable — so a server-backed "Recent"
# history had no routing explanation to reconstruct from.
ROUTED_OUTCOME = "routed"
# "judged" is the offline LLM-as-judge's verdict on one (prompt, response)
# pair — written by the evaluator script (see evaluate.py), not by end
# users. Kept as its own outcome type rather than mixed into
# response_thumbs_*: it's a different kind of signal (automated, not human)
# and router.py scores them as two SEPARATE penalties, not one blended rate.
JUDGED_OUTCOME = "judged"
VALID_OUTCOMES = {"regenerated", "escalated", "accepted", "continued", ROUTED_OUTCOME, JUDGED_OUTCOME} | ALL_VOTE_OUTCOMES
# Outcomes that need a stable prompt_id/user_id to be meaningful — vote
# outcomes (for "one vote per prompt per user") and "routed" (so a
# server-backed history can group everything for one prompt together).
# "judged" needs prompt_id too (to know which run it's about) but not
# user_id (it's a system verdict, not a user's) — validated separately,
# see log_outcome.
REQUIRES_IDS_OUTCOMES = ALL_VOTE_OUTCOMES | {ROUTED_OUTCOME}
# QUINDRAL_LOG_PATH: point this at a mounted persistent volume in
# production (e.g. Railway) — the default lives next to the code, which is
# fine locally but gets wiped on every redeploy of an ephemeral container.
LOG_PATH = Path(os.environ.get("QUINDRAL_LOG_PATH") or (Path(__file__).parent / "outcomes.jsonl"))


def log_outcome(
    trace: dict,
    outcome: str,
    criteria_weights_active: dict,
    user_id: str = None,
    prompt_id: str = None,
    log_path: Path = LOG_PATH,
    usage: dict = None,
    comment: str = None,
    prompt_text: str = None,
    response_text: str = None,
    trace_detail: dict = None,
    judge_pass: bool = None,
    judge_reasoning: str = None,
    judge_model: str = None,
) -> dict:
    """Append one outcome record derived from a router trace. Returns the record written.

    usage: optional {"input_tokens", "output_tokens", "actual_cost"} from a
    real provider call (see providers.js) — measured, not the registry
    estimate. Only present once real execution (not just routing) has run.
    comment: optional free-text, meant for response_thumbs_down ("what went
    wrong") — the UI only collects it on a negative response vote.
    prompt_text / response_text: the actual query and model response, for a
    future offline quality-judging pipeline (see module docstring). Only
    present on records tied to real execution (accepted/escalated/response
    votes) — routing-only votes never see a response, so have neither.
    trace_detail: only for outcome="routed" — {"eligible_models", "excluded",
    "binding_criterion", "explanation", "relaxed_filters"}, the rest of the
    routing trace beyond route_chosen/model_used, so a server-backed history
    can reconstruct the full explanation, not just which model was picked.
    judge_pass / judge_reasoning / judge_model: only for outcome="judged" —
    the offline evaluator's pass/fail verdict on one (prompt, response) pair,
    a short reason, and which model did the judging. See router.py's
    judge_penalty_rate for how this feeds into ranking.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {VALID_OUTCOMES}, got {outcome!r}")
    if outcome in REQUIRES_IDS_OUTCOMES and not (user_id and prompt_id):
        raise ValueError(f"{outcome!r} requires both user_id and prompt_id")
    if outcome == JUDGED_OUTCOME and not prompt_id:
        raise ValueError(f"{outcome!r} requires prompt_id (no user_id needed — it's a system verdict)")

    record = {
        "timestamp": time.time(),
        "route_chosen": trace["classified_as"],
        "model_used": trace["chosen"],
        "outcome": outcome,
        "criteria_weights_active": criteria_weights_active,
        "user_id": user_id,
        "prompt_id": prompt_id,
        "usage": usage,
        "comment": comment,
        "prompt_text": prompt_text,
        "response_text": response_text,
        "trace_detail": trace_detail,
        "judge_pass": judge_pass,
        "judge_reasoning": judge_reasoning,
        "judge_model": judge_model,
    }
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_outcomes(log_path: Path = LOG_PATH):
    """Yield logged records in order. Missing file = no records."""
    if not log_path.exists():
        return
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_user_vote(user_id: str, prompt_id: str, log_path: Path = LOG_PATH, outcomes: set = VOTE_OUTCOMES):
    """Return this user's current vote for a prompt within the given outcome
    category (routing votes by default; pass RESPONSE_VOTE_OUTCOMES for
    response-quality votes), or None. Last-write-wins within that category."""
    current = None
    for record in read_outcomes(log_path):
        if record.get("outcome") in outcomes and record.get("user_id") == user_id and record.get("prompt_id") == prompt_id:
            current = record["outcome"]
    return current


MIN_VOTES_FOR_QUALITY_SIGNAL = 3  # ponytail: fixed floor, not confidence-interval math — revisit if votes stay sparse


def negative_response_rate(model_name: str, route: str = None, log_path: Path = LOG_PATH) -> float | None:
    """Fraction of response-quality votes for this model (optionally scoped to
    one route, e.g. "code" vs "simple" — a model can be fine at one and bad
    at another) that were response_thumbs_down. Returns None if there aren't
    enough votes yet to trust the signal (MIN_VOTES_FOR_QUALITY_SIGNAL)
    rather than letting one early bad review tank a model's ranking.

    This is the only "is this model returning garbage" signal that exists
    right now — real human feedback, not automated fact-checking (no
    external verification pipeline is wired up — see EVALUATION.md).
    """
    # Dedupe by (user_id, prompt_id), last-write-wins — same as get_user_vote.
    # Necessary because a single vote can be logged twice (Poor click logs
    # immediately with no comment, then adding a comment logs again) — count
    # each person's opinion on a response once, not once per click.
    latest = {}  # (user_id, prompt_id) -> outcome
    for record in read_outcomes(log_path):
        if record.get("model_used") != model_name:
            continue
        if route is not None and record.get("route_chosen") != route:
            continue
        outcome = record.get("outcome")
        if outcome in RESPONSE_VOTE_OUTCOMES:
            latest[(record.get("user_id"), record.get("prompt_id"))] = outcome

    total = len(latest)
    if total < MIN_VOTES_FOR_QUALITY_SIGNAL:
        return None
    negative = sum(1 for o in latest.values() if o == "response_thumbs_down")
    return negative / total


def judge_penalty_rate(model_name: str, route: str = None, log_path: Path = LOG_PATH) -> float | None:
    """Fraction of offline judge verdicts for this model (optionally scoped
    to one route) that were a fail. Same shape as negative_response_rate —
    same MIN_VOTES_FOR_QUALITY_SIGNAL floor, same None-if-not-enough-data
    behavior — but a SEPARATE signal, not blended with human votes: a model
    that's bad by both measures should be penalized more than bad by either
    alone, which only works if they're kept apart (see router.py's _rank).

    Dedupe key is (prompt_id, model_used) rather than (user_id, prompt_id) —
    a judge verdict isn't "per user," it's per (prompt, response) pair; if
    the evaluator ever re-judges the same pair, last-write-wins like votes.
    """
    latest = {}  # (prompt_id, model_used) -> judge_pass
    for record in read_outcomes(log_path):
        if record.get("outcome") != JUDGED_OUTCOME:
            continue
        if record.get("model_used") != model_name:
            continue
        if route is not None and record.get("route_chosen") != route:
            continue
        latest[(record.get("prompt_id"), record.get("model_used"))] = record.get("judge_pass")

    total = len(latest)
    if total < MIN_VOTES_FOR_QUALITY_SIGNAL:
        return None
    failed = sum(1 for v in latest.values() if v is False)
    return failed / total


def _demo():
    import tempfile
    from router import route, DEFAULT_WEIGHTS

    with tempfile.TemporaryDirectory() as tmp:
        test_log = Path(tmp) / "test_outcomes.jsonl"

        trace = route("what is the capital of france?")
        record = log_outcome(trace, "accepted", DEFAULT_WEIGHTS, log_path=test_log)
        assert record["outcome"] == "accepted"
        assert record["model_used"] == trace["chosen"]

        log_outcome(trace, "escalated", DEFAULT_WEIGHTS, log_path=test_log)

        records = list(read_outcomes(test_log))
        assert len(records) == 2
        assert records[0]["outcome"] == "accepted"
        assert records[1]["outcome"] == "escalated"

        try:
            log_outcome(trace, "bogus", DEFAULT_WEIGHTS, log_path=test_log)
            assert False, "should have rejected invalid outcome"
        except ValueError:
            pass

        assert list(read_outcomes(Path(tmp) / "nonexistent.jsonl")) == []

        # thumbs voting requires user_id + prompt_id
        try:
            log_outcome(trace, "thumbs_up", DEFAULT_WEIGHTS, log_path=test_log)
            assert False, "should have rejected a vote with no user_id/prompt_id"
        except ValueError:
            pass

        # no vote yet
        assert get_user_vote("alice", "prompt-1", test_log) is None

        # alice votes thumbs_up on prompt-1
        log_outcome(trace, "thumbs_up", DEFAULT_WEIGHTS, user_id="alice", prompt_id="prompt-1", log_path=test_log)
        assert get_user_vote("alice", "prompt-1", test_log) == "thumbs_up"

        # a different user's vote on the same prompt is independent
        log_outcome(trace, "thumbs_down", DEFAULT_WEIGHTS, user_id="bob", prompt_id="prompt-1", log_path=test_log)
        assert get_user_vote("alice", "prompt-1", test_log) == "thumbs_up"
        assert get_user_vote("bob", "prompt-1", test_log) == "thumbs_down"

        # alice changes her mind -> last-write-wins, one active vote per prompt per user
        log_outcome(trace, "thumbs_down", DEFAULT_WEIGHTS, user_id="alice", prompt_id="prompt-1", log_path=test_log)
        assert get_user_vote("alice", "prompt-1", test_log) == "thumbs_down"

        # same user, different prompt -> independent
        assert get_user_vote("alice", "prompt-2", test_log) is None

        # response-quality votes are a separate category from routing votes —
        # same prompt_id, different outcome set, don't collide
        log_outcome(trace, "response_thumbs_down", DEFAULT_WEIGHTS, user_id="alice", prompt_id="prompt-1",
                    log_path=test_log, comment="answer was factually wrong")
        assert get_user_vote("alice", "prompt-1", test_log) == "thumbs_down"  # routing vote unchanged
        assert get_user_vote("alice", "prompt-1", test_log, outcomes=RESPONSE_VOTE_OUTCOMES) == "response_thumbs_down"
        last_record = list(read_outcomes(test_log))[-1]
        assert last_record["comment"] == "answer was factually wrong"
        assert "timestamp" in last_record

        # negative_response_rate: not enough votes yet -> None, not a rate
        assert negative_response_rate("gpt-4o-mini", log_path=test_log) is None

        # 3 distinct users vote on gpt-4o-mini for the "code" route: 2 down, 1 up
        code_trace = {"classified_as": "code", "chosen": "gpt-4o-mini"}
        for i, (uid, vote) in enumerate([("u1", "response_thumbs_down"), ("u2", "response_thumbs_down"), ("u3", "response_thumbs_up")]):
            log_outcome(code_trace, vote, DEFAULT_WEIGHTS, user_id=uid, prompt_id=f"code-prompt-{i}", log_path=test_log)
        assert negative_response_rate("gpt-4o-mini", route="code", log_path=test_log) == 2 / 3
        # unrelated route has no votes for this model -> still None
        assert negative_response_rate("gpt-4o-mini", route="vision", log_path=test_log) is None

        # a user re-voting (Poor click then a comment) must count ONCE, not twice
        log_outcome(code_trace, "response_thumbs_down", DEFAULT_WEIGHTS, user_id="u1", prompt_id="code-prompt-0",
                    log_path=test_log, comment="still wrong")
        assert negative_response_rate("gpt-4o-mini", route="code", log_path=test_log) == 2 / 3

        # prompt_text/response_text: present when supplied (real execution),
        # absent (None) otherwise (routing-only votes never saw a response)
        record_with_text = log_outcome(
            trace, "accepted", DEFAULT_WEIGHTS, log_path=test_log,
            prompt_text="what is the capital of france?", response_text="Paris.",
        )
        assert record_with_text["prompt_text"] == "what is the capital of france?"
        assert record_with_text["response_text"] == "Paris."
        assert record["prompt_text"] is None and record["response_text"] is None

        # "routed" logs the routing decision itself, requires ids like votes,
        # and carries trace_detail so a server-backed history can reconstruct
        # the full explanation (not just which model got picked)
        try:
            log_outcome(trace, "routed", DEFAULT_WEIGHTS, log_path=test_log)
            assert False, "should have rejected 'routed' with no user_id/prompt_id"
        except ValueError:
            pass

        routed_record = log_outcome(
            trace, "routed", DEFAULT_WEIGHTS, user_id="alice", prompt_id="prompt-routed-1",
            log_path=test_log, prompt_text="what is the capital of france?",
            trace_detail={"eligible_models": ["mistral-small"], "excluded": [], "binding_criterion": "cost",
                          "explanation": "Routed to mistral-small...", "relaxed_filters": []},
        )
        assert routed_record["trace_detail"]["binding_criterion"] == "cost"
        assert routed_record["prompt_text"] == "what is the capital of france?"

        # "judged" needs prompt_id but NOT user_id (system verdict, not a user's)
        try:
            log_outcome(trace, "judged", DEFAULT_WEIGHTS, log_path=test_log)
            assert False, "should have rejected 'judged' with no prompt_id"
        except ValueError:
            pass
        judged_record = log_outcome(
            trace, "judged", DEFAULT_WEIGHTS, prompt_id="judge-prompt-1", log_path=test_log,
            judge_pass=False, judge_reasoning="hallucinated a fact", judge_model="gpt-4o-mini",
        )
        assert judged_record["user_id"] is None
        assert judged_record["judge_pass"] is False
        assert judged_record["judge_reasoning"] == "hallucinated a fact"

        # judge_penalty_rate: separate signal from negative_response_rate,
        # same shape (route-scoped, min-sample floor, last-write-wins per
        # (prompt_id, model_used))
        assert judge_penalty_rate("gpt-4o-mini", log_path=test_log) is None  # only 1 verdict so far
        judge_trace = {"classified_as": "reasoning", "chosen": "gpt-4o-mini"}
        for i, verdict in enumerate([False, False, True]):
            log_outcome(judge_trace, "judged", DEFAULT_WEIGHTS, prompt_id=f"judge-reasoning-{i}",
                        log_path=test_log, judge_pass=verdict, judge_model="gpt-4o-mini")
        assert judge_penalty_rate("gpt-4o-mini", route="reasoning", log_path=test_log) == 2 / 3
        # re-judging the SAME prompt_id+model overwrites, doesn't double-count
        log_outcome(judge_trace, "judged", DEFAULT_WEIGHTS, prompt_id="judge-reasoning-0",
                    log_path=test_log, judge_pass=True, judge_model="gpt-4o-mini")
        assert judge_penalty_rate("gpt-4o-mini", route="reasoning", log_path=test_log) == 1 / 3
        # this signal is independent of negative_response_rate's data
        assert negative_response_rate("gpt-4o-mini", route="reasoning", log_path=test_log) is None

    print("outcome_log self-check: all cases passed")


if __name__ == "__main__":
    _demo()
