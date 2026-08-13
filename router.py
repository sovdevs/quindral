"""Scoring pipeline: classify -> hard filter -> weighted rank -> pick + trace.

See SPEC.md "Hard Filters vs. Soft Scoring" and "Reasoning Trace".
"""
from classifier import classify
from model_registry import ModelInfo, REGISTRY, filter_by
from outcome_log import negative_response_rate, LOG_PATH as DEFAULT_OUTCOME_LOG_PATH

DEFAULT_WEIGHTS = {"cost": 0.6, "env": 0.3, "latency": 0.1}

# How hard a model's real negative-feedback rate (see outcome_log.py) pushes
# it down in ranking. Not a user-adjustable weight — this is an intrinsic
# quality demerit, not a cost/env/latency tradeoff a user would choose to
# de-prioritize. 1.0 means a model with a 100% negative rate on this route
# gets pushed down by as much as being worst-in-class on cost+env+latency
# combined — soft, not a hard ban (SPEC.md's exploration/exploitation note:
# a model never fully vanishes just because it had a bad run early on).
QUALITY_PENALTY_WEIGHT = 1.0

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


def route(query: str, weights: dict = None, eu_only: bool = False, min_context: int = 0,
          allowed_models: set = None, outcome_log_path=None) -> dict:
    """Run the full pipeline and return a structured reasoning trace (SPEC.md format).

    allowed_models: optional whitelist of model names (e.g. "only models I
    have a BYOK key for" from the UI). Applied as a final hard intersection,
    after capability/eu/context filtering and after relaxation — deliberately
    NOT part of _RELAXATION_ORDER, since the user opted into this constraint
    explicitly and relaxing it silently would recommend something unusable.

    outcome_log_path: which outcomes.jsonl to read the quality-penalty signal
    from (see _rank). Defaults to the real log; tests/self-checks pass an
    isolated path so assertions stay deterministic instead of depending on
    whatever real feedback has accumulated in production.
    """
    weights = weights or DEFAULT_WEIGHTS
    # Guard against explicit None values (not just missing keys) — a
    # malformed client payload (e.g. a NaN slider serializing to JSON null)
    # used to crash the whole request with an unhandled 500 downstream.
    weights = {k: (v if v is not None else 0) for k, v in weights.items()}
    classification = classify(query)
    required_capabilities = {ROUTE_CAPABILITY[classification["route"]]}
    if classification["needs_current_info"]:
        # retrieval-necessity signal (see classifier.needs_current_info) — the
        # query likely needs info newer than any model's training cutoff
        required_capabilities.add("web_search")

    all_models = _all_models()
    survivors = filter_by(capability=required_capabilities, eu_only=eu_only, min_context=min_context)

    relaxed_filters = []
    if not survivors:
        survivors, relaxed_filters = _relax_and_retry(required_capabilities, eu_only, min_context)

    if allowed_models is not None:
        survivors = [m for m in survivors if m.name in allowed_models]

    # Structured, not pre-rendered English — explain.py owns turning this
    # into a sentence (SPEC.md: trace and NL explanation stay separate so
    # phrasing/localization can change independently). Built against the
    # final survivor set so it reflects every filter, including allowed_models.
    excluded = [
        {
            "model": m.name,
            "missing_capabilities": sorted(required_capabilities - m.capabilities),
            "not_eu_hosted": bool(eu_only and m.region != "EU"),
            "context_deficit": (
                {"actual": m.context_window, "required": min_context}
                if m.context_window < min_context else None
            ),
            "no_key_available": bool(allowed_models is not None and m.name not in allowed_models),
        }
        for m in all_models if m not in survivors
    ]

    if not survivors:
        return {
            "classified_as": classification["route"],
            "eligible_models": [],
            "excluded": excluded,
            "chosen": None,
            "binding_criterion": None,
            "error": "no candidates survive hard filters, even after relaxing every filter",
        }

    log_path = outcome_log_path or DEFAULT_OUTCOME_LOG_PATH
    chosen = _rank(survivors, weights, route=classification["route"], log_path=log_path)[0]
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


def _rank(models, weights, route=None, log_path=None):
    """Return models sorted best-first by weighted normalized cost/env/latency,
    plus a soft demerit for models with a real negative-feedback track record
    on this route (see outcome_log.negative_response_rate)."""
    cost_norm = _normalize(models, _cost)
    env_norm = _normalize(models, lambda m: m.energy_wh_per_1k_tokens)
    latency_norm = _normalize(models, lambda m: m.latency_ms_p50)

    def score(m):
        quality_penalty = negative_response_rate(m.name, route=route, log_path=log_path or DEFAULT_OUTCOME_LOG_PATH) or 0.0
        # `or 0` guards against an explicit None (not just a missing key) —
        # dict.get's default only covers the latter. A malformed client
        # payload (e.g. a NaN slider value serializing to JSON null) used to
        # crash this with an unhandled 500 instead of just defaulting.
        return (
            (weights.get("cost", 0) or 0) * cost_norm[m.name]
            + (weights.get("env", 0) or 0) * env_norm[m.name]
            + (weights.get("latency", 0) or 0) * latency_norm[m.name]
            + QUALITY_PENALTY_WEIGHT * quality_penalty
        )

    return sorted(models, key=score)  # lower normalized score = cheaper/greener/faster = better


# Order to drop hard filters in when nothing survives, least-consequential first.
# "web_search" (a nice-to-have freshness boost) gives way before the core
# route capability; EU-only is dropped last since it's a privacy commitment.
_RELAXATION_ORDER = ["min_context", "web_search", "capability", "eu_only"]


def _relax_and_retry(required_capabilities, eu_only, min_context):
    """Drop hard filters one at a time (then all together) until something survives.

    Returns (survivors, relaxed_filter_names). This is the "second best" fallback:
    rather than a flat no-match, tell the user what constraint had to give.
    """
    capabilities = set(required_capabilities)
    dropped = []
    for filter_name in _RELAXATION_ORDER:
        if filter_name == "web_search" and "web_search" not in capabilities:
            continue
        if filter_name == "capability" and not (capabilities - {"web_search"}):
            continue
        if filter_name == "eu_only" and not eu_only:
            continue
        if filter_name == "min_context" and min_context == 0:
            continue
        dropped.append(filter_name)

        trial_capabilities = set(required_capabilities)
        if "web_search" in dropped:
            trial_capabilities.discard("web_search")
        if "capability" in dropped:
            trial_capabilities.clear()  # drop the core route capability too
        trial_eu_only = False if "eu_only" in dropped else eu_only
        trial_min_context = 0 if "min_context" in dropped else min_context

        survivors = filter_by(capability=trial_capabilities, eu_only=trial_eu_only, min_context=trial_min_context)
        if survivors:
            return survivors, dropped
    return [], dropped


def _all_models():
    return list(REGISTRY.values())


def _demo():
    import tempfile
    import functools
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        # isolated outcome log — self-check assertions must be deterministic,
        # not dependent on whatever real feedback has accumulated in
        # production outcomes.jsonl (see _rank's quality penalty)
        log_path = Path(tmp) / "test_outcomes.jsonl"
        route = functools.partial(globals()["route"], outcome_log_path=log_path)

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

        # current-info queries require web_search capability
        result_current = route("what's the weather in Tokyo right now")
        assert result_current["chosen"] is not None
        assert "web_search" in get(result_current["chosen"]).capabilities
        assert "relaxed_filters" not in result_current

        # no EU-hosted model has web_search -> must fall back
        result_current_eu = route("what's the weather in Tokyo right now", eu_only=True)
        assert result_current_eu["chosen"] is not None
        assert "web_search" in result_current_eu["relaxed_filters"]
        assert get(result_current_eu["chosen"]).region == "EU"

        # quality penalty: log enough negative response votes against the
        # cheapest "simple" model that it should lose its cost advantage and
        # get passed over for an alternative with a clean track record
        from outcome_log import log_outcome

        default_pick = route("what is the capital of germany?")["chosen"]
        bad_trace = {"classified_as": "simple", "chosen": default_pick}
        for i in range(4):
            log_outcome(bad_trace, "response_thumbs_down", DEFAULT_WEIGHTS,
                        user_id=f"quality-test-user-{i}", prompt_id=f"quality-test-prompt-{i}",
                        log_path=log_path)
        result_after_bad_feedback = route("what is the capital of germany?")
        assert result_after_bad_feedback["chosen"] != default_pick

        # malformed weights (explicit None, e.g. a NaN slider serialized to
        # JSON null) must not crash the whole request with a 500
        result_none_weights = route("what is the capital of france?", weights={"cost": None, "env": None, "latency": None})
        assert result_none_weights["chosen"] is not None

    print("router self-check: all cases passed")


if __name__ == "__main__":
    _demo()
