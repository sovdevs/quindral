"""Phase 1 HITL outcome logging: append routing decisions + outcomes as JSONL.

See SPEC.md "Phase 1: Define and log the HITL feedback signal". Implicit
signals only for v1 (regenerated/escalated/accepted/continued/thumbs_down) —
explicit thumbs up/down left as an open decision per SPEC, but the enum
already accounts for it.

ponytail: flat JSONL file, not a DB — swap for a real store once volume or
querying needs outgrow `grep`/`wc -l`.
"""
import json
from pathlib import Path

VALID_OUTCOMES = {"regenerated", "escalated", "accepted", "thumbs_down", "continued"}
LOG_PATH = Path(__file__).parent / "outcomes.jsonl"


def log_outcome(trace: dict, outcome: str, criteria_weights_active: dict, log_path: Path = LOG_PATH) -> dict:
    """Append one outcome record derived from a router trace. Returns the record written."""
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {VALID_OUTCOMES}, got {outcome!r}")

    record = {
        "route_chosen": trace["classified_as"],
        "model_used": trace["chosen"],
        "outcome": outcome,
        "criteria_weights_active": criteria_weights_active,
    }
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

    print("outcome_log self-check: all cases passed")


if __name__ == "__main__":
    _demo()
