from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _home() -> Path:
    return Path(os.environ.get("MODEL_SELECTOR_HOME", "./data/model_selector")).expanduser()


@dataclass
class Settings:
    home: Path = field(default_factory=_home)
    # Discovery sources (markdown variants are served by appending .md to the doc URL).
    openai_pricing_urls: tuple[str, ...] = (
        "https://developers.openai.com/api/docs/pricing.md",
        "https://developers.openai.com/api/docs/pricing",
    )
    openai_models_urls: tuple[str, ...] = (
        "https://developers.openai.com/api/docs/models.md",
        "https://developers.openai.com/api/docs/models",
    )
    http_timeout_s: float = 30.0
    # Probing. Reasoning effort (this axis) is separate from task difficulty
    # (easy/medium/hard, in TaskSpec/prompts/*.yaml) — effort is a per-call
    # knob a model exposes, difficulty is a property of the task/prompt.
    # Recent OpenAI models (gpt-5.x/gpt-6-*) added "xhigh"/"max" above the
    # original low/medium/high; older models will just fail fast on an
    # unsupported effort and get skipped (see runner.py's `unsupported` guard)
    # so listing them here is safe even for models that don't support them.
    default_efforts: tuple[str | None, ...] = (None, "low", "medium", "high", "xhigh", "max")
    judge_model: str | None = None          # None -> auto: mid-priced reasoning model
    tts_model_for_fixtures: str | None = None  # None -> auto: cheapest tts
    max_probe_cost_usd: float = 0.50        # per single call guard (estimated pre-call)
    allow_code_exec: bool = False
    # Pricing: token threshold above which "long context" rates apply (None = unknown -> short rates)
    long_context_threshold: int | None = None

    @property
    def db_path(self) -> Path:
        return self.home / "catalog.sqlite3"

    @property
    def audio_dir(self) -> Path:
        return self.home / "audio"

    def ensure(self) -> "Settings":
        self.home.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        return self
