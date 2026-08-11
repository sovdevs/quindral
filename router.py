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

    if not survivors:
        return {
            "classified_as": classification["route"],
            "eligible_models": [],
            "excluded": excluded,
            "chosen": None,
            "binding_criterion": None,
            "error": "no candidates survive hard filters",
        }

    cost_norm = _normalize(survivors, _cost)
    env_norm = _normalize(survivors, lambda m: m.energy_wh_per_1k_tokens)
    latency_norm = _normalize(survivors, lambda m: m.latency_ms_p50)

    scored = []
    for m in survivors:
        score = (
            weights.get("cost", 0) * cost_norm[m.name]
            + weights.get("env", 0) * env_norm[m.name]
            + weights.get("latency", 0) * latency_norm[m.name]
        )
        scored.append((m, score))
    scored.sort(key=lambda pair: pair[1])  # lower normalized score = cheaper/greener/faster = better
    chosen, _ = scored[0]

    binding_criterion = max(weights, key=weights.get)

    return {
        "classified_as": classification["route"],
        "eligible_models": [m.name for m in survivors],
        "excluded": excluded,
        "chosen": chosen.name,
        "binding_criterion": binding_criterion,
    }


def _all_models():
    return list(REGISTRY.values())


def _demo():
    result = route("what is the capital of france?")
    assert result["chosen"] is not None
    assert result["classified_as"] == "simple"

    result_eu = route("what is the capital of france?", eu_only=True)
    from model_registry import get
    assert get(result_eu["chosen"]).region == "EU"

    result_impossible = route("summarize this huge document", min_context=99999999)
    assert result_impossible["chosen"] is None
    assert result_impossible["error"]

    print("router self-check: all cases passed")


if __name__ == "__main__":
    _demo()
