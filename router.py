"""Scoring pipeline: classify -> hard filter -> weighted rank -> pick + trace.

See SPEC.md "Hard Filters vs. Soft Scoring" and "Reasoning Trace".
"""
from classifier import classify
from model_registry import ModelInfo, REGISTRY, filter_by

DEFAULT_WEIGHTS = {"cost": 0.6, "env": 0.3, "latency": 0.1}

# route -> minimum capability a model must have to be eligible
ROUTE_CAPABILITY = {
    "code": "code",
    "vision": "vision",
    "long_context": "long_context",
    "reasoning": "reasoning",
    "simple": "factual",
}


def _cost(m: ModelInfo) -> float:
    return m.cost_per_1k_input + m.cost_per_1k_output


def _normalize(models, key):
    values = [key(m) for m in models]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {m.name: 0.0 for m in models}
    return {m.name: (key(m) - lo) / (hi - lo) for m in models}


def route(query: str, weights: dict = None, eu_only: bool = False, min_context: int = 0) -> dict:
    """Run the full pipeline and return a structured reasoning trace (SPEC.md format)."""
    weights = weights or DEFAULT_WEIGHTS
    classification = classify(query)
    required_capability = ROUTE_CAPABILITY[classification["route"]]

    all_models = _all_models()
    survivors = filter_by(capability=required_capability, eu_only=eu_only, min_context=min_context)
    excluded = []
    for m in all_models:
        if m not in survivors:
            reasons = []
            if required_capability not in m.capabilities:
                reasons.append(f"missing required capability '{required_capability}'")
            if eu_only and m.region != "EU":
                reasons.append("not EU-hosted")
            if m.context_window < min_context:
                reasons.append(f"context window {m.context_window} < required {min_context}")
            excluded.append({"model": m.name, "reason": "; ".join(reasons)})

    relaxed_filters = []
    if not survivors:
        survivors, relaxed_filters = _relax_and_retry(required_capability, eu_only, min_context)

    if not survivors:
        return {
            "classified_as": classification["route"],
            "eligible_models": [],
            "excluded": excluded,
            "chosen": None,
            "binding_criterion": None,
            "error": "no candidates survive hard filters, even after relaxing every filter",
        }

    chosen = _rank(survivors, weights)[0]
    binding_criterion = max(weights, key=weights.get)

    result = {
        "classified_as": classification["route"],
        "eligible_models": [m.name for m in survivors],
        "excluded": excluded,
        "chosen": chosen.name,
        "binding_criterion": binding_criterion,
    }
    if relaxed_filters:
        result["relaxed_filters"] = relaxed_filters
    return result


def _rank(models, weights):
    """Return models sorted best-first by weighted normalized cost/env/latency."""
    cost_norm = _normalize(models, _cost)
    env_norm = _normalize(models, lambda m: m.energy_wh_per_1k_tokens)
    latency_norm = _normalize(models, lambda m: m.latency_ms_p50)

    def score(m):
        return (
            weights.get("cost", 0) * cost_norm[m.name]
            + weights.get("env", 0) * env_norm[m.name]
            + weights.get("latency", 0) * latency_norm[m.name]
        )

    return sorted(models, key=score)  # lower normalized score = cheaper/greener/faster = better


# Order to drop hard filters in when nothing survives, least-consequential first.
# EU-only is dropped last since it's a privacy commitment, not just a preference.
_RELAXATION_ORDER = ["min_context", "capability", "eu_only"]


def _relax_and_retry(required_capability, eu_only, min_context):
    """Drop hard filters one at a time (then all together) until something survives.

    Returns (survivors, relaxed_filter_names). This is the "second best" fallback:
    rather than a flat no-match, tell the user what constraint had to give.
    """
    active = {"capability": required_capability, "eu_only": eu_only, "min_context": min_context}
    dropped = []
    for filter_name in _RELAXATION_ORDER:
        if filter_name == "capability" and required_capability is None:
            continue
        if filter_name == "eu_only" and not eu_only:
            continue
        if filter_name == "min_context" and min_context == 0:
            continue
        dropped.append(filter_name)
        trial = dict(active)
        for name in dropped:
            trial[name] = None if name == "capability" else (False if name == "eu_only" else 0)
        survivors = filter_by(capability=trial["capability"], eu_only=trial["eu_only"], min_context=trial["min_context"])
        if survivors:
            return survivors, dropped
    return [], dropped


def _all_models():
    return list(REGISTRY.values())


def _demo():
    result = route("what is the capital of france?")
    assert result["chosen"] is not None
    assert result["classified_as"] == "simple"

    result_eu = route("what is the capital of france?", eu_only=True)
    from model_registry import get
    assert get(result_eu["chosen"]).region == "EU"

    # every hard filter gets relaxed in turn, so as long as the registry is
    # non-empty there's always a "closest match" fallback rather than a dead end
    result_extreme = route("summarize this huge document", min_context=99999999)
    assert result_extreme["chosen"] is not None
    assert "min_context" in result_extreme["relaxed_filters"]

    # no EU-hosted model has vision capability -> must fall back
    result_fallback = route("describe what's happening in this photo", eu_only=True)
    assert result_fallback["chosen"] is not None
    assert "capability" in result_fallback["relaxed_filters"]

    # EU-hosted coding now exists (codestral), so this should NOT need a fallback
    result_eu_code = route("write a python function to reverse a list", eu_only=True)
    assert result_eu_code["chosen"] == "codestral"
    assert "relaxed_filters" not in result_eu_code

    print("router self-check: all cases passed")


if __name__ == "__main__":
    _demo()
