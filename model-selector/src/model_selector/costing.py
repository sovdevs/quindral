"""Cost computation from usage + price rows (prices are USD per 1M tokens unless unit says otherwise)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

M = 1_000_000


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0          # includes reasoning tokens (OpenAI bills them as output)
    reasoning_tokens: int = 0
    units: float | None = None      # e.g. audio minutes for per-minute models
    # modality splits for multimodal token billing (e.g. image/audio tokens)
    input_by_modality: dict[str, int] | None = None
    output_by_modality: dict[str, int] | None = None


def token_cost(price: dict[str, Any], u: Usage, long_ctx: bool = False) -> float | None:
    pin = price.get("long_input" if long_ctx else "input")
    pcache = price.get("long_cached_input" if long_ctx else "cached_input")
    pout = price.get("long_output" if long_ctx else "output")
    if pin is None and pout is None:
        return None
    uncached = max(u.input_tokens - u.cached_tokens, 0)
    c = uncached * (pin or 0) + u.cached_tokens * (pcache if pcache is not None else (pin or 0))
    c += u.output_tokens * (pout or 0)
    return c / M


def cost(prices: list[dict[str, Any]] | dict[str, Any] | None, u: Usage,
         long_threshold: int | None = None) -> tuple[float | None, str]:
    """Return (usd, basis). `prices` may be one row or all modality rows for a model."""
    if not prices:
        return None, "no_price"
    rows = [prices] if isinstance(prices, dict) else list(prices)
    by_mod = {r.get("modality", "text"): r for r in rows}
    per_unit = next((r for r in rows if r.get("per_unit") is not None), None)
    if per_unit and u.units is not None:
        return per_unit["per_unit"] * u.units, f"per_{per_unit.get('unit', 'unit')}"
    long_ctx = bool(long_threshold and u.input_tokens > long_threshold)
    if u.input_by_modality or u.output_by_modality:
        total, basis = 0.0, []
        for mod, n in (u.input_by_modality or {}).items():
            r = by_mod.get(mod) or by_mod.get("text")
            if r and r.get("input") is not None:
                total += n * r["input"] / M; basis.append(f"in:{mod}")
        for mod, n in (u.output_by_modality or {}).items():
            r = by_mod.get(mod) or by_mod.get("text")
            if r and r.get("output") is not None:
                total += n * r["output"] / M; basis.append(f"out:{mod}")
        return total, "tokens[" + ",".join(basis) + "]"
    r = by_mod.get("text") or rows[0]
    c = token_cost(r, u, long_ctx)
    if c is None and per_unit:
        return None, "needs_units"
    return c, "tokens_long" if long_ctx else "tokens"


def estimate(prices: list[dict[str, Any]] | dict[str, Any] | None, in_tokens: int, out_tokens: int,
             units: float | None = None) -> float | None:
    return cost(prices, Usage(input_tokens=in_tokens, output_tokens=out_tokens, units=units))[0]
