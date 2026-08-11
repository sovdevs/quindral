"""Weekly OpenRouter pricing sync. See SPEC.md "Model Registry" (cost per token)
and the decision to refresh this weekly via launchd.

Fetches https://openrouter.ai/api/v1/models (no API key needed for pricing),
pulls prompt/completion cost for each registry model, and writes a small
JSON cache that model_registry.py reads at import time. Run standalone:
    .venv/bin/python3 openrouter_sync.py
"""
import json
import urllib.request
from pathlib import Path

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_PATH = Path(__file__).parent / "openrouter_cache.json"

# our registry name -> OpenRouter model id
OPENROUTER_IDS = {
    "mistral-small": "mistralai/mistral-small-3.2-24b-instruct",
    "llama-3-8b": "meta-llama/llama-3.1-8b-instruct",
    "qwen-2.5-coder": "qwen/qwen-2.5-coder-32b-instruct",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",
}


def fetch_pricing() -> dict:
    """Return {registry_name: {"cost_per_1k_input": ..., "cost_per_1k_output": ...}}."""
    with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=15) as resp:
        data = json.load(resp)["data"]

    by_id = {m["id"]: m for m in data}
    pricing = {}
    for registry_name, or_id in OPENROUTER_IDS.items():
        model = by_id.get(or_id)
        if model is None:
            continue
        pricing[registry_name] = {
            "cost_per_1k_input": float(model["pricing"]["prompt"]) * 1000,
            "cost_per_1k_output": float(model["pricing"]["completion"]) * 1000,
        }
    return pricing


def sync(cache_path: Path = CACHE_PATH) -> dict:
    pricing = fetch_pricing()
    cache_path.write_text(json.dumps(pricing, indent=2))
    return pricing


def load_cached_pricing(cache_path: Path = CACHE_PATH) -> dict:
    """Read the last synced pricing. Returns {} if never synced."""
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text())


def _demo():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_cache = Path(tmp) / "test_cache.json"
        assert load_cached_pricing(test_cache) == {}

        pricing = sync(test_cache)
        assert "gpt-4o-mini" in pricing
        assert pricing["gpt-4o-mini"]["cost_per_1k_input"] > 0

        reloaded = load_cached_pricing(test_cache)
        assert reloaded == pricing

    print("openrouter_sync self-check: all cases passed")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _demo()
    else:
        pricing = sync()
        print(f"synced pricing for {len(pricing)} models -> {CACHE_PATH}")
