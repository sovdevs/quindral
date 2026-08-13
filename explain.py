"""NL explanation template: turns a router trace dict into user-facing prose.

Kept separate from router.py per SPEC.md ("structured trace and NL explanation
separate allows localization, adjustable verbosity... without recomputation").
router.py's "excluded" entries are structured (missing_capabilities,
not_eu_hosted, context_deficit) rather than pre-rendered English — this file
is where that becomes a sentence.
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
    "web_search": "the live-search preference",
    "capability": "the capability requirement",
    "eu_only": "the EU-only filter",
}

CAPABILITY_PHRASING = {
    "code": "generate or debug code",
    "vision": "understand images",
    "long_context": "handle very long documents",
    "reasoning": "work through complex multi-step reasoning",
    "factual": "reliably answer general knowledge questions",
    "web_search": "search the web for current information",
}


def _join_and(items: list) -> str:
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} or {items[-1]}"


def _exclusion_reason_phrases(entry: dict) -> list:
    reasons = []
    if entry.get("missing_capabilities"):
        phrases = [CAPABILITY_PHRASING.get(c, c) for c in entry["missing_capabilities"]]
        reasons.append(f"can't {_join_and(phrases)}")
    if entry.get("not_eu_hosted"):
        reasons.append("isn't hosted in the EU")
    if entry.get("context_deficit"):
        d = entry["context_deficit"]
        reasons.append(
            f"only supports {d['actual']:,} tokens of context, short of the "
            f"{d['required']:,} this query needs"
        )
    if entry.get("no_key_available"):
        reasons.append("you don't have a key for it")
    return reasons


def explain_exclusion(entry: dict) -> str:
    """Render one router.py 'excluded' entry as a friendly sentence."""
    reasons = _exclusion_reason_phrases(entry)
    if not reasons:
        return f"{entry['model']} was excluded."
    return f"{entry['model']} {' and '.join(reasons)}."


def explain_exclusions(excluded: list) -> str:
    """Render all excluded models as grouped sentences — models sharing the
    exact same reason are named together (e.g. "X, Y and Z aren't hosted in
    the EU") instead of one repetitive sentence per model, which reads badly
    once more than a couple of models share a filter."""
    groups = {}  # reason text -> [model names], insertion-ordered
    for entry in excluded:
        reason = " and ".join(_exclusion_reason_phrases(entry)) or "was excluded"
        groups.setdefault(reason, []).append(entry["model"])

    sentences = []
    for reason, models in groups.items():
        reason_for_group = reason
        if len(models) > 1:
            # reason phrases are written singular ("isn't hosted", "only
            # supports") — fix subject-verb agreement when naming a group
            reason_for_group = reason_for_group.replace("isn't hosted", "aren't hosted")
            reason_for_group = reason_for_group.replace("only supports", "only support")
        sentences.append(f"{_join_and(models)} {reason_for_group}.")
    return " ".join(sentences)


def explain(trace: dict, eu_only: bool = False) -> str:
    """Return a short paragraph explaining a router trace to an end user."""
    route_desc = ROUTE_PHRASING.get(trace["classified_as"], "your request")

    if trace["chosen"] is None:
        return (
            f"No model met your requirements for {route_desc}, even after relaxing "
            f"every filter — try loosening your constraints."
        )

    chosen = get(trace["chosen"])
    criterion_desc = CRITERION_PHRASING.get(trace["binding_criterion"], trace["binding_criterion"])

    if trace.get("relaxed_filters"):
        dropped = ", ".join(RELAXED_FILTER_PHRASING.get(f, f) for f in trace["relaxed_filters"])
        return (
            f"No model matched every filter for {route_desc}, so we relaxed {dropped} "
            f"and suggest {chosen.name} as the closest match, still optimizing for {criterion_desc}."
        )

    sentences = [f"Routed to {chosen.name} for {route_desc}, optimizing primarily for {criterion_desc}."]

    # Compare the chosen model against the priciest/least-efficient ELIGIBLE
    # alternative (not the excluded ones — those failed a hard filter, they
    # were never in the running) to give a concrete sense of what was saved.
    eligible = [get(name) for name in trace["eligible_models"]]
    if len(eligible) > 1:
        if trace["binding_criterion"] == "env":
            costliest = max(eligible, key=lambda m: m.energy_wh_per_1k_tokens)
            if costliest.name != chosen.name and costliest.energy_wh_per_1k_tokens > 0:
                savings = 1 - (chosen.energy_wh_per_1k_tokens / costliest.energy_wh_per_1k_tokens)
                if savings > 0.05:
                    sentences.append(
                        f"That's roughly {savings:.0%} less energy per token than "
                        f"{costliest.name}, the most energy-intensive model that also qualified."
                    )
        else:
            costliest = max(eligible, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
            chosen_cost = chosen.cost_per_1k_input + chosen.cost_per_1k_output
            costliest_cost = costliest.cost_per_1k_input + costliest.cost_per_1k_output
            if costliest.name != chosen.name and costliest_cost > 0:
                savings = 1 - (chosen_cost / costliest_cost)
                if savings > 0.05:
                    sentences.append(
                        f"That's roughly {savings:.0%} cheaper per token than "
                        f"{costliest.name}, the priciest model that also qualified."
                    )

    if trace["excluded"]:
        sentences.append("Ruled out: " + explain_exclusions(trace["excluded"]))

    if eu_only:
        sentences.append(f"{chosen.name} is EU-hosted, meeting your privacy filter.")

    return " ".join(sentences)


def _demo():
    from router import route

    result = route("what is the capital of france?")
    text = explain(result)
    assert result["chosen"] in text
    assert text.endswith(".")

    eu_result = route("what is the capital of france?", eu_only=True)
    eu_text = explain(eu_result, eu_only=True)
    assert "EU-hosted" in eu_text

    coding_result = route("write a python function to reverse a linked list")
    coding_text = explain(coding_result)
    assert "Ruled out" in coding_text
    assert "can't" in coding_text  # some excluded model lacks the 'code' capability

    fallback = route("describe what's happening in this photo", eu_only=True)
    fallback_text = explain(fallback, eu_only=True)
    assert "relaxed" in fallback_text
    assert fallback["chosen"] in fallback_text

    current_info_fallback = route("what's the weather in Tokyo right now", eu_only=True)
    current_info_text = explain(current_info_fallback, eu_only=True)
    assert "live-search preference" in current_info_text

    print("explain self-check: all cases passed")


if __name__ == "__main__":
    _demo()
