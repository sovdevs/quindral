"""Energy/water estimates via Ecologits (see SPEC.md "Model Registry":
"Ecologits methodology - public, reusable, same one aim2balance references").

Ecologits doesn't ship static per-model wh/token numbers - it computes impacts
from a model's parameter count + token count + a datacenter/electricity-mix
profile. We use the world-average electricity mix (zone "WOR") and Ecologits'
own default datacenter PUE/WUE, then normalize to per-1k-output-token units
to match model_registry.ModelInfo's fields.
"""
from ecologits.electricity_mix_repository import ElectricityMixRepository
from ecologits.impacts.llm import compute_llm_impacts
from ecologits.model_repository import models as ecologits_models
from ecologits.utils.range_value import RangeValue

DATACENTER_PUE = 1.2  # Ecologits default (impacts/llm.py IF_ELECTRICITY_MIX defaults)
DATACENTER_WUE = 1.8  # L/kWh, Ecologits default

_electricity_mixes = ElectricityMixRepository.from_json()
_WOR = _electricity_mixes.find_electricity_mix("WOR")


def _midpoint(value):
    if isinstance(value, RangeValue):
        return (value.min + value.max) / 2
    return value


def estimate_energy_and_water(provider: str, model_name: str, output_tokens: int = 1000):
    """Return (energy_wh, water_ml) for generating `output_tokens` tokens.

    Returns None if Ecologits has no data for this provider/model_name pair.
    """
    model = ecologits_models.find_model(provider, model_name)
    if model is None:
        return None

    params = model.architecture.parameters
    if hasattr(params, "total"):  # MoE: has .total/.active
        total_params = _midpoint(params.total)
        active_params = _midpoint(params.active)
    else:
        total_params = active_params = _midpoint(params)

    impacts = compute_llm_impacts(
        model_active_parameter_count=active_params,
        model_total_parameter_count=total_params,
        output_token_count=output_tokens,
        if_electricity_mix_adpe=_WOR.adpe,
        if_electricity_mix_pe=_WOR.pe,
        if_electricity_mix_gwp=_WOR.gwp,
        if_electricity_mix_wue=_WOR.wue,
        datacenter_pue=DATACENTER_PUE,
        datacenter_wue=DATACENTER_WUE,
    )

    energy_kwh = _midpoint(impacts.energy.value)
    water_l = _midpoint(impacts.usage.wcf.value) if impacts.usage else 0.0
    return energy_kwh * 1000, water_l * 1000  # -> Wh, mL


def _demo():
    result = estimate_energy_and_water("openai", "gpt-4o-mini")
    assert result is not None
    energy_wh, water_ml = result
    assert energy_wh > 0 and water_ml > 0

    bigger = estimate_energy_and_water("anthropic", "claude-haiku-4-5")
    assert bigger is not None

    assert estimate_energy_and_water("nonexistent", "not-a-model") is None

    print(f"gpt-4o-mini: {energy_wh:.4f} Wh, {water_ml:.4f} mL per 1k output tokens")
    print("energy self-check: all cases passed")


if __name__ == "__main__":
    _demo()
