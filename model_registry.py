"""Model registry: static table of candidate LLMs and their routing-relevant attributes.

See SPEC.md "Model Registry". Plain dataclasses + a dict — no DB, this is
config data that changes rarely and fits in memory. Energy/water figures are
computed from Ecologits at import time (see energy.py) rather than hardcoded;
cost/region/capability/context/latency are still hand-entered placeholders —
the OpenRouter sync (next step) is what should replace those.
"""
from dataclasses import dataclass

from energy import estimate_energy_and_water
from openrouter_sync import load_cached_pricing

FALLBACK_ENERGY_WH = 0.2   # ponytail: used only when Ecologits has no data for a model
FALLBACK_WATER_ML = 0.4    # (e.g. qwen isn't in its dataset) — flagged via energy_is_estimated

_openrouter_pricing = load_cached_pricing()  # {} until openrouter_sync.py has run at least once


@dataclass(frozen=True)
class ModelInfo:
    name: str
    cost_per_1k_input: float   # USD
    cost_per_1k_output: float  # USD
    energy_wh_per_1k_tokens: float  # from Ecologits, or FALLBACK_ENERGY_WH
    water_ml_per_1k_tokens: float
    energy_is_estimated: bool  # True if Ecologits had no data and we used the fallback
    cost_is_stale: bool  # True if no OpenRouter sync has run yet (using hand-entered placeholder)
    region: str  # "EU" or "non-EU"
    context_window: int
    capabilities: frozenset  # e.g. {"code", "vision", "long_context", "reasoning", "creative", "factual"}
    latency_ms_p50: float
    no_train_guarantee: bool


def _make(name, ecologits_provider, ecologits_name, cost_per_1k_input, cost_per_1k_output, **kwargs):
    result = estimate_energy_and_water(ecologits_provider, ecologits_name)
    if result is None:
        energy_wh, water_ml, estimated = FALLBACK_ENERGY_WH, FALLBACK_WATER_ML, True
    else:
        energy_wh, water_ml, estimated = result[0], result[1], False

    priced = _openrouter_pricing.get(name)
    if priced is not None:
        cost_per_1k_input = priced["cost_per_1k_input"]
        cost_per_1k_output = priced["cost_per_1k_output"]

    return ModelInfo(
        name=name, cost_per_1k_input=cost_per_1k_input, cost_per_1k_output=cost_per_1k_output,
        energy_wh_per_1k_tokens=energy_wh, water_ml_per_1k_tokens=water_ml,
        energy_is_estimated=estimated, cost_is_stale=priced is None, **kwargs,
    )


# Curated current-gen roster across providers/tiers (replaces the earlier
# test placeholders). Capability tags and region are still hand-judged —
# OpenRouter's API doesn't expose hosting region, and its capability data
# is too coarse to trust for hard filters, so this stays manually curated
# rather than auto-ingesting all ~400 OpenRouter models.
REGISTRY = {
    m.name: m for m in [
        _make(
            "mistral-small", "mistralai", "mistral-small-latest",
            cost_per_1k_input=0.00009, cost_per_1k_output=0.00025,
            region="EU", context_window=256000,
            capabilities=frozenset({"factual", "creative", "long_context"}),
            latency_ms_p50=400, no_train_guarantee=True,
        ),
        _make(
            "codestral", "mistralai", "codestral-2508",
            cost_per_1k_input=0.0003, cost_per_1k_output=0.0009,
            region="EU", context_window=256000,
            capabilities=frozenset({"code", "long_context"}),
            latency_ms_p50=450, no_train_guarantee=True,
        ),
        _make(
            "mistral-large", "mistralai", "mistral-large-2512",
            cost_per_1k_input=0.0005, cost_per_1k_output=0.0015,
            region="EU", context_window=262144,
            capabilities=frozenset({"factual", "creative", "reasoning", "long_context", "code"}),
            latency_ms_p50=700, no_train_guarantee=True,
        ),
        _make(
            "llama-3.1-8b", "huggingface_hub", "meta-llama/Meta-Llama-3.1-8B-Instruct",
            cost_per_1k_input=0.00005, cost_per_1k_output=0.00008,
            region="non-EU", context_window=131072,
            capabilities=frozenset({"factual"}),
            latency_ms_p50=350, no_train_guarantee=False,
        ),
        _make(
            # Ecologits has no 3.3 entry yet; 3.1-70B is the closest same-size
            # proxy for the energy estimate (pricing/context still from 3.3).
            "llama-3.3-70b", "huggingface_hub", "meta-llama/Meta-Llama-3.1-70B-Instruct",
            cost_per_1k_input=0.0001, cost_per_1k_output=0.00032,
            region="non-EU", context_window=131072,
            capabilities=frozenset({"factual", "reasoning"}),
            latency_ms_p50=550, no_train_guarantee=False,
        ),
        _make(
            "qwen-2.5-coder", "huggingface_hub", "Qwen/Qwen2.5-Coder-32B-Instruct",
            cost_per_1k_input=0.00066, cost_per_1k_output=0.001,
            region="non-EU", context_window=32768,
            capabilities=frozenset({"code", "reasoning"}),
            latency_ms_p50=500, no_train_guarantee=False,
        ),
        _make(
            "gpt-4o-mini", "openai", "gpt-4o-mini",
            cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
            region="non-EU", context_window=128000,
            capabilities=frozenset({"factual", "code", "vision", "long_context"}),
            latency_ms_p50=600, no_train_guarantee=False,
        ),
        _make(
            "gpt-4o", "openai", "gpt-4o",
            cost_per_1k_input=0.0025, cost_per_1k_output=0.01,
            region="non-EU", context_window=128000,
            capabilities=frozenset({"factual", "code", "vision", "long_context", "reasoning", "creative"}),
            latency_ms_p50=900, no_train_guarantee=False,
        ),
        _make(
            "gpt-5.4-mini", "openai", "gpt-5.4-mini",
            cost_per_1k_input=0.00075, cost_per_1k_output=0.0045,
            region="non-EU", context_window=400000,
            capabilities=frozenset({"factual", "code", "vision", "long_context"}),
            latency_ms_p50=600, no_train_guarantee=False,
        ),
        _make(
            "gpt-5.4", "openai", "gpt-5.4",
            cost_per_1k_input=0.0025, cost_per_1k_output=0.015,
            region="non-EU", context_window=1050000,
            capabilities=frozenset({"factual", "code", "vision", "long_context", "reasoning", "creative"}),
            latency_ms_p50=900, no_train_guarantee=False,
        ),
        _make(
            "claude-haiku-4.5", "anthropic", "claude-haiku-4-5-20251001",
            cost_per_1k_input=0.001, cost_per_1k_output=0.005,
            region="non-EU", context_window=200000,
            capabilities=frozenset({"factual", "code", "vision", "long_context"}),
            latency_ms_p50=500, no_train_guarantee=True,
        ),
        _make(
            "claude-sonnet-4.5", "anthropic", "claude-sonnet-4-5-20250929",
            cost_per_1k_input=0.003, cost_per_1k_output=0.015,
            region="non-EU", context_window=1000000,
            capabilities=frozenset({"factual", "code", "vision", "long_context", "reasoning", "creative"}),
            latency_ms_p50=900, no_train_guarantee=True,
        ),
        _make(
            "gemini-2.5-flash", "google_genai", "gemini-2.5-flash",
            cost_per_1k_input=0.0003, cost_per_1k_output=0.0025,
            region="non-EU", context_window=1048576,
            capabilities=frozenset({"factual", "code", "vision", "long_context"}),
            latency_ms_p50=500, no_train_guarantee=False,
        ),
        _make(
            # moonshotai isn't in Ecologits' provider list yet -> energy_is_estimated
            "kimi-k2", "moonshotai", "kimi-k2-0905",
            cost_per_1k_input=0.0006, cost_per_1k_output=0.0025,
            region="non-EU", context_window=262144,
            capabilities=frozenset({"factual", "code", "long_context", "reasoning"}),
            latency_ms_p50=700, no_train_guarantee=False,
        ),
    ]
}


def get(name: str) -> ModelInfo:
    return REGISTRY[name]


def filter_by(capability: str = None, eu_only: bool = False, min_context: int = 0):
    """Hard-filter candidates. See SPEC.md 'Hard Filters vs Soft Scoring'."""
    result = []
    for m in REGISTRY.values():
        if capability and capability not in m.capabilities:
            continue
        if eu_only and m.region != "EU":
            continue
        if m.context_window < min_context:
            continue
        result.append(m)
    return result


def _demo():
    assert get("mistral-small").region == "EU"
    eu_models = filter_by(eu_only=True)
    assert all(m.region == "EU" for m in eu_models)
    assert eu_models  # at least one EU model exists

    eu_coders = filter_by(capability="code", eu_only=True)
    assert eu_coders and all("code" in m.capabilities for m in eu_coders)
    assert "codestral" in {m.name for m in eu_coders}  # closes the EU+code gap

    coders = filter_by(capability="code")
    assert all("code" in m.capabilities for m in coders)
    assert {"codestral", "mistral-large", "qwen-2.5-coder", "gpt-4o-mini", "gpt-4o", "gpt-5.4-mini", "gpt-5.4",
            "claude-haiku-4.5", "claude-sonnet-4.5", "gemini-2.5-flash", "kimi-k2"} == {m.name for m in coders}

    assert get("gpt-4o-mini").energy_wh_per_1k_tokens > 0
    assert get("gpt-4o-mini").energy_is_estimated is False
    assert get("kimi-k2").energy_is_estimated is True  # moonshotai not in Ecologits' dataset

    print("model_registry self-check: all cases passed")
    for m in REGISTRY.values():
        energy_flag = " (estimated)" if m.energy_is_estimated else ""
        cost_flag = " (stale, no OpenRouter sync yet)" if m.cost_is_stale else ""
        print(
            f"  {m.name}: ${m.cost_per_1k_input:.6f}/${m.cost_per_1k_output:.6f} per 1k in/out{cost_flag}, "
            f"{m.energy_wh_per_1k_tokens:.4f} Wh, {m.water_ml_per_1k_tokens:.4f} mL / 1k tokens{energy_flag}"
        )


if __name__ == "__main__":
    _demo()
