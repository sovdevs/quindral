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

RELAXED_FILTER_PHRASING = {
    "min_context": "the context-length requirement",
    "capability": "the capability requirement",
    "eu_only": "the EU-only filter",
}


def explain(trace: dict, eu_only: bool = False) -> str:
    """Return a one-line human-readable explanation for a router trace."""
    route_desc = ROUTE_PHRASING.get(trace["classified_as"], "your request")

    if trace["chosen"] is None:
        return f"No model met your requirements for {route_desc}, even after relaxing every filter — try loosening your constraints."

    chosen = get(trace["chosen"])
    criterion_desc = CRITERION_PHRASING.get(trace["binding_criterion"], trace["binding_criterion"])

    if trace.get("relaxed_filters"):
        dropped = ", ".join(RELAXED_FILTER_PHRASING.get(f, f) for f in trace["relaxed_filters"])
        return (
            f"No model matched all your filters for {route_desc}, so we relaxed {dropped} "
            f"and suggest {chosen.name} as the closest match, optimizing for {criterion_desc}."
        )

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

    fallback = route("describe what's happening in this photo", eu_only=True)
    fallback_text = explain(fallback, eu_only=True)
    assert "relaxed" in fallback_text
    assert fallback["chosen"] in fallback_text

    print("explain self-check: all cases passed")


if __name__ == "__main__":
    _demo()
