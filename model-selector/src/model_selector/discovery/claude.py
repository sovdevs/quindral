"""Anthropic Claude discovery: the models-overview page's comparison table.

Unlike Gemini (needed a per-model detail-page scrape — see gemini.py) and
unlike OpenAI (exposes no cutoff at all — see openai_docs.py), Anthropic's
overview page puts everything in ONE clean markdown table per current
model: pricing, context window, max output, and — better than either other
provider — BOTH a "Reliable knowledge cutoff" and a broader "Training data
cutoff". One page fetch, no browser rendering needed (a `.md` variant is
served directly, same convention as OpenAI's docs).

Scope limit, stated rather than worked around: the table only carries full
detail for the CURRENT lineup (4 models at the time this was written). Older
"Legacy models" are named with links at the bottom of the page but with no
pricing/cutoff/context-window data on this page — guessing their API ID
string from the display name risks a wrong id that silently fails at call
time, so they're deliberately left out rather than guessed.
"""
from __future__ import annotations

import re
from typing import Callable

import httpx

from ..config import Settings
from .base import DiscoveryResult, ModelRecord, PriceRecord

PROVIDER = "anthropic"
OVERVIEW_URL = "https://platform.claude.com/docs/en/models/overview.md"

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BACKTICK = re.compile(r"`([a-z0-9.\-]+)`")
_DOLLAR = re.compile(r"\$\s*([0-9][0-9,]*\.?[0-9]*)")
_TOKENS = re.compile(r"([\d.]+)\s*([MK])\s*tokens", re.I)


def http_fetch(settings: Settings) -> Callable[[str], str]:
    client = httpx.Client(timeout=settings.http_timeout_s, follow_redirects=True,
                          headers={"User-Agent": "model-selector/0.1"})

    def _fetch(url: str) -> str:
        r = client.get(url)
        r.raise_for_status()
        return r.text
    return _fetch


def _clean_label(label: str) -> str:
    """"[Pricing](url)" -> "Pricing" — several row labels are links."""
    return _LINK.sub(r"\1", label).strip()


def parse_overview_table(md_text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Returns (model_names, {clean_label: [value_per_model...]})."""
    lines = md_text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith("| Feature")), None)
    if start is None:
        raise ValueError("comparison table not found (page format may have changed)")
    header = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    model_names = [_clean_label(c) for c in header[1:]]
    table: dict[str, list[str]] = {}
    i = start + 2  # skip the "---" separator row
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if len(cells) == len(header):
            table[_clean_label(cells[0])] = cells[1:]
        i += 1
    return model_names, table


def _tokens(cell: str) -> int | None:
    m = _TOKENS.search(cell or "")
    if not m:
        return None
    n = float(m.group(1))
    return int(n * (1_000_000 if m.group(2).upper() == "M" else 1_000))


def parse_models(md_text: str) -> tuple[list[ModelRecord], list[PriceRecord]]:
    model_names, table = parse_overview_table(md_text)
    models: list[ModelRecord] = []
    prices: list[PriceRecord] = []
    for j, name in enumerate(model_names):
        api_id_cell = (table.get("Claude API ID") or [None] * len(model_names))[j]
        m = _BACKTICK.search(api_id_cell or "")
        if not m:
            continue  # no usable id for this column, skip rather than guess
        model_id = m.group(1)
        description = (table.get("Description") or [None] * len(model_names))[j]
        extra = {
            "reliable_knowledge_cutoff": (table.get("Reliable knowledge cutoff") or [None] * len(model_names))[j],
            "training_data_cutoff": (table.get("Training data cutoff") or [None] * len(model_names))[j],
            "retirement": (table.get("Retirement") or [None] * len(model_names))[j],
            "comparative_latency": (table.get("Comparative latency") or [None] * len(model_names))[j],
            "default_effort": ((table.get("Default effort") or [None] * len(model_names))[j] or "").strip("`") or None,
            "context_window_tokens": _tokens((table.get("Context window") or [""] * len(model_names))[j]),
            "max_output_tokens": _tokens((table.get("Max output") or [""] * len(model_names))[j]),
        }
        # All current Claude models are general-purpose (text/code/reasoning
        # in one line, unlike OpenAI/Gemini's separate tts/image/embedding
        # model families) — one category covers the whole current lineup.
        models.append(ModelRecord(model_id, PROVIDER, "reasoning", description=description,
                                  source="docs", extra=extra))

        pricing_cell = (table.get("Pricing") or [""] * len(model_names))[j]
        dollars = _DOLLAR.findall(pricing_cell)
        if len(dollars) >= 2:
            prices.append(PriceRecord(model_id, PROVIDER, tier="Standard",
                                      input=float(dollars[0]), output=float(dollars[1]),
                                      section="Current lineup"))
    return models, prices


class ClaudeDiscovery:
    provider = PROVIDER

    def __init__(self, settings: Settings, fetch: Callable[[str], str] | None = None):
        self.settings = settings
        self.fetch = fetch or http_fetch(settings)

    def run(self) -> DiscoveryResult:
        try:
            md_text = self.fetch(OVERVIEW_URL)
        except Exception as e:  # noqa: BLE001
            return DiscoveryResult(PROVIDER, [], [], {}, [f"overview fetch failed: {e}"])
        try:
            models, prices = parse_models(md_text)
        except Exception as e:  # noqa: BLE001
            return DiscoveryResult(PROVIDER, [], [], {OVERVIEW_URL: md_text}, [f"parse failed: {e}"])
        warnings = [] if models else ["no models parsed — page format may have changed"]
        return DiscoveryResult(PROVIDER, models, prices, {OVERVIEW_URL: md_text}, warnings)
