"""Google Gemini discovery: the model index page + each model's own detail
page (both server-rendered plain HTML — no browser/JS rendering needed,
confirmed by fetching with plain httpx) plus the pricing page.

Unlike OpenAI (see openai_docs.py's models_from_api docstring — no discovery
source there exposes a knowledge-data cutoff at all), Gemini's per-model
detail pages genuinely publish both "Latest update" and "Knowledge cutoff"
dates, plus a real capability list including "Search grounding" (whether the
model can ground its answer in live Google Search results) — this is where
that data actually comes from, not an approximation. Not every model
publishes a Knowledge cutoff (e.g. TTS/embedding models often don't) — left
as None rather than guessed.

No API key needed for any of this (unlike OpenAI's optional /v1/models
call) — model discovery, capabilities, and pricing are all public docs.
"""
from __future__ import annotations

import re
from typing import Callable

import httpx
from markdownify import markdownify as _to_markdown

from ..config import Settings
from ..taxonomy import category_for_gemini_id
from .base import DiscoveryResult, ModelRecord, PriceRecord

PROVIDER = "google"
Fetcher = Callable[[str], str]

MODELS_INDEX_URL = "https://ai.google.dev/gemini-api/docs/models"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
MODEL_DETAIL_URL = "https://ai.google.dev/gemini-api/docs/models/{model_id}"

_MODEL_LINK = re.compile(r"/gemini-api/docs/models/([a-z0-9.\-]+)")
# Match by label PREFIX only, not the full label — some labels (e.g. "Token
# limits[[*]](/gemini-api/docs/tokens)") carry a markdown-escaped footnote
# suffix ("\[\[\*\]\]...") that's brittle to reproduce exactly; the row's
# left-hand cell is unambiguous from the prefix alone.
_ROW = lambda label_prefix: re.compile(rf"\|\s*\S*{re.escape(label_prefix)}[^|]*\|\s*(.+?)\s*\|\s*\n")
_BACKTICK_ID = re.compile(r"`([a-z0-9.\-]+)`")
_INT = lambda s: int(s.replace(",", "")) if s and s.replace(",", "").isdigit() else None


def http_fetch(settings: Settings) -> Fetcher:
    client = httpx.Client(timeout=settings.http_timeout_s, follow_redirects=True,
                          headers={"User-Agent": "model-selector/0.1"})

    def _fetch(url: str) -> str:
        r = client.get(url)
        r.raise_for_status()
        return r.text
    return _fetch


def discover_model_ids(index_html: str) -> list[str]:
    """Model detail-page links found on the index page — this alone doesn't
    need JS rendering (confirmed: plain httpx sees the full link list even
    though some OTHER content on this same page is client-rendered)."""
    return sorted(set(_MODEL_LINK.findall(index_html)))


def parse_model_detail(model_id: str, html: str) -> tuple[ModelRecord, dict]:
    """Returns (ModelRecord, token_limits) — token_limits kept separate since
    ModelRecord has no dedicated field for it (stored in `extra` by the
    caller alongside everything else Gemini-specific)."""
    text = _to_markdown(html)

    def field(label: str) -> str | None:
        m = _ROW(label).search(text)
        return m.group(1).strip() if m else None

    code_cell = field("Model code") or ""
    code_match = _BACKTICK_ID.search(code_cell)
    real_id = code_match.group(1) if code_match else model_id

    # Description: the last real prose paragraph before the property table's
    # header row ("| Property | Description |"). The setext-style heading
    # right above the table (model id, then a "----" underline on the next
    # line) renders as its own "paragraph" too — excluded explicitly, or it
    # wins as "last" and the actual description is lost.
    before_table = text.split("| Property | Description |")[0]
    # Exclude nav links ("[...]") and bullet-list lines ("* " with a space —
    # NOT bold text like "**Warning:**", which starts with "*" too but isn't
    # a list item; an earlier version of this filter conflated the two and
    # discarded real deprecation-warning descriptions along with actual nav
    # cruft) and heading markers ("#").
    paras = [p.strip() for p in before_table.split("\n\n")
             if p.strip() and not p.strip().startswith(("[", "* ", "#"))]
    # A setext-style heading ("gemini-2.5-pro\n--------------", or "===" for
    # an H1) renders as its own paragraph directly above the table — its last
    # line is a run of dashes/equals signs; exclude any paragraph shaped like
    # that rather than real prose.
    paras = [p for p in paras if not re.fullmatch(r"[-=]{3,}", p.splitlines()[-1])]
    description = paras[-1] if paras else None

    caps_cell = field("Capabilities") or ""
    search_grounding = bool(re.search(r"\[Search grounding\]\([^)]*\)\*\*\s*Supported", caps_cell))

    token_cell = field("Token limits") or ""
    in_tok = _INT((re.search(r"Input token limit\*\*\s*([\d,]+)", token_cell) or [None, None])[1])
    out_tok = _INT((re.search(r"Output token limit\*\*\s*([\d,]+)", token_cell) or [None, None])[1])

    extra = {
        "latest_update": field("Latest update"),
        "knowledge_cutoff": field("Knowledge cutoff"),
        "search_grounding": search_grounding,
    }
    cat = category_for_gemini_id(real_id)
    rec = ModelRecord(real_id, PROVIDER, cat, description=description, source="docs", extra=extra)
    return rec, {"input_token_limit": in_tok, "output_token_limit": out_tok}


_PRICE_ROW = re.compile(r"\|\s*(Input price|Output price[^|]*)\s*\|\s*([^|]*)\|\s*([^|]*)\|")
_DOLLAR = re.compile(r"\$\s*([0-9][0-9,]*\.?[0-9]*)")


def parse_pricing(pricing_html: str, model_ids: set[str]) -> list[PriceRecord]:
    """Best-effort: current Standard-tier, paid-tier price per model (first
    dollar figure in the cell — Gemini's pricing table often lists a
    scheduled future increase in the same cell, e.g. "$0.75 through Dec 31,
    2026. $1.50 starting Jan 1, 2027." — this takes the CURRENT rate, not
    the future one). Deliberately narrower than OpenAI's parser: doesn't
    capture Batch/Flex/Priority tiers or the separate Search-grounding
    per-request fee, both real gaps left for a follow-up rather than
    guessed at."""
    text = _to_markdown(pricing_html)
    sections = re.split(r"\n([A-Z][A-Za-z0-9 .\-]+)\n-{3,}\n", text)
    out: list[PriceRecord] = []
    # re.split with a capturing group interleaves [pre, heading, body, heading, body, ...]
    for i in range(1, len(sections) - 1, 2):
        heading, body = sections[i], sections[i + 1]
        mid = _slugify_heading(heading)
        if mid not in model_ids:
            continue
        standard = body.split("### Batch")[0].split("### Flex")[0]
        rows = {m.group(1).strip(): m.group(3).strip() for m in _PRICE_ROW.finditer(standard)}
        in_price = _first_dollar(rows.get("Input price"))
        out_price = _first_dollar(rows.get("Output price (including thinking tokens)")) or _first_dollar(rows.get("Output price"))
        if in_price is None and out_price is None:
            continue
        out.append(PriceRecord(mid, PROVIDER, tier="Standard", input=in_price, output=out_price,
                               section="Standard (current, paid tier)"))
    return out


def _first_dollar(cell: str | None) -> float | None:
    if not cell:
        return None
    m = _DOLLAR.search(cell)
    return float(m.group(1).replace(",", "")) if m else None


def _slugify_heading(heading: str) -> str:
    # "Gemini 3.8 Flash" -> "gemini-3.8-flash". Version numbers are dotted in
    # real model ids (gemini-3.8-flash, not gemini-3-8-flash) — an earlier
    # version of this collapsed "." to "-" like every other separator, which
    # silently broke the match for almost every versioned model (only the
    # rare heading with no dotted version number matched). Dots are kept;
    # everything else non-alphanumeric becomes "-".
    return re.sub(r"[^a-z0-9.]+", "-", heading.strip().lower()).strip("-")


class GeminiDiscovery:
    provider = PROVIDER

    def __init__(self, settings: Settings, fetch: Fetcher | None = None):
        self.settings = settings
        self.fetch = fetch or http_fetch(settings)

    def run(self) -> DiscoveryResult:
        warnings: list[str] = []
        docs: dict[str, str] = {}
        models: list[ModelRecord] = []

        try:
            index_html = self.fetch(MODELS_INDEX_URL)
        except Exception as e:  # noqa: BLE001
            return DiscoveryResult(PROVIDER, [], [], {}, [f"models index fetch failed: {e}"])
        docs[MODELS_INDEX_URL] = index_html
        ids = discover_model_ids(index_html)

        for mid in ids:
            url = MODEL_DETAIL_URL.format(model_id=mid)
            try:
                html = self.fetch(url)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"{mid}: detail fetch failed: {e}")
                continue
            docs[url] = html
            try:
                rec, tok = parse_model_detail(mid, html)
                rec.extra.update(tok)
                models.append(rec)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"{mid}: parse failed: {e}")

        prices: list[PriceRecord] = []
        try:
            pricing_html = self.fetch(PRICING_URL)
            docs[PRICING_URL] = pricing_html
            prices = parse_pricing(pricing_html, {m.model_id for m in models})
        except Exception as e:  # noqa: BLE001
            warnings.append(f"pricing fetch/parse failed: {e}")

        return DiscoveryResult(PROVIDER, models, prices, docs, warnings)
