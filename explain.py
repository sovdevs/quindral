"""NL explanation template: turns a router trace dict into a user-facing one-liner.

Kept separate from router.py per SPEC.md ("structured trace and NL explanation
separate allows localization, adjustable verbosity... without recomputation").
"""
from model_registry import get

ROUTE_PHRASING = {
    "simple": "a simple factual lookup",
    "code": "a coding task",
    "vision": "a task involving an image",
    "long_context": "a task needing a long context window",
    "reasoning": "a task needing multi-step reasoning",
}

CRITERION_PHRASING = {
    "cost": "cost",
    "env": "environmental impact",
    "latency": "response speed",
}


def explain(trace: dict, eu_only: bool = False) -> str:
    """Return a one-line human-readable explanation for a router trace."""
    if trace["chosen"] is None:
        return f"No model met your requirements for this {ROUTE_PHRASING.get(trace['classified_as'], 'task')}."

    chosen = get(trace["chosen"])
    route_desc = ROUTE_PHRASING.get(trace["classified_as"], "your request")
    criterion_desc = CRITERION_PHRASING.get(trace["binding_criterion"], trace["binding_criterion"])

    parts = [f"Routed to {chosen.name} because this looked like {route_desc}"]
    parts.append(f"optimizing primarily for {criterion_desc}")

    if trace["excluded"]:
        overkill = [e for e in trace["excluded"] if "overkill" in e["reason"] or "capability" not in e["reason"]]
        if overkill:
            parts.append(f"a smaller model was sufficient vs. {len(trace['excluded'])} larger option(s) considered")

    if eu_only:
        parts.append("it met your EU-only filter")

    return ", ".join(parts) + "."


def _demo():
    from router import route

    result = route("what is the capital of france?")
    text = explain(result)
    assert result["chosen"] in text
    assert text.endswith(".")

    eu_result = route("what is the capital of france?", eu_only=True)
    eu_text = explain(eu_result, eu_only=True)
    assert "EU-only filter" in eu_text

    no_match = route("summarize this huge document", min_context=99999999)
    no_match_text = explain(no_match)
    assert "No model met" in no_match_text

    print("explain self-check: all cases passed")


if __name__ == "__main__":
    _demo()
