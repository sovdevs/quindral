"""OpenAI discovery: pricing page (markdown), model catalog page, and /v1/models (optional).

Sources are fetched as markdown (`<url>.md`) with an HTML->markdown fallback, parsed into
PriceRecord/ModelRecord, and returned alongside raw documents for local snapshotting.
"""
from __future__ import annotations

import os
import re
from typing import Callable

import httpx

from ..config import Settings
from ..taxonomy import SUBCAT_MAP, category_for_id, category_for_section
from .base import DiscoveryResult, ModelRecord, PriceRecord
from .mdtables import MdTable, parse_price, parse_tables

PROVIDER = "openai"
Fetcher = Callable[[str], str]

# column name -> PriceRecord field (after header merge)
_COLS = {
    "input": "input", "cached input": "cached_input", "cache writes": "cache_write", "output": "output",
    "output / cost": "output",
    "short context input": "input", "short context cached input": "cached_input",
    "short context cache writes": "cache_write", "short context output": "output",
    "long context input": "long_input", "long context cached input": "long_cached_input",
    "long context cache writes": "long_cache_write", "long context output": "long_output",
}
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9.\-_:]*$")
_MODEL_LINK = re.compile(r"\[([^\]]*)\]\((?:https://developers\.openai\.com)?/api/docs/models/([a-z0-9.\-_]+)\)")


def http_fetch(settings: Settings) -> Fetcher:
    def fetch(url: str) -> str:
        r = httpx.get(url, timeout=settings.http_timeout_s, follow_redirects=True,
                      headers={"User-Agent": "model-selector/0.1 (+discovery)"})
        r.raise_for_status()
        text = r.text
        if "html" in r.headers.get("content-type", "") or text.lstrip().startswith("<"):
            from markdownify import markdownify
            text = markdownify(text, heading_style="ATX")
        return text
    return fetch


def _first_ok(fetch: Fetcher, urls: tuple[str, ...], warnings: list[str]) -> tuple[str, str] | None:
    for u in urls:
        try:
            return u, fetch(u)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"fetch failed {u}: {e}")
    return None


def prices_from_tables(tables: list[MdTable]) -> tuple[list[PriceRecord], list[ModelRecord]]:
    prices: list[PriceRecord] = []
    models: dict[str, ModelRecord] = {}
    for t in tables:
        cat = category_for_section(t.section, t.headings)
        if cat is None:
            continue
        model_col = next((c for c in t.columns if c.lower() == "model"), None)
        if not model_col:
            continue
        last_model = None
        for row in t.rows:
            mid = row.get(model_col, "").strip().strip("`")
            if not mid:
                mid = last_model  # continuation row (e.g. another modality)
            if not mid or not _MODEL_ID.match(mid.split()[0]):
                continue
            mid = mid.split()[0]  # "o4-mini-2025-04-16   with data sharing" -> id
            last_model = mid
            rec = PriceRecord(model_id=mid, provider=PROVIDER, tier=t.tier or "Standard",
                              section=t.section, raw=dict(row))
            modality = row.get("Modality", "").strip().lower()
            if modality:
                rec.modality = modality
            sub = None
            for key in ("Category", "Use case"):
                if row.get(key):
                    sub = row[key].strip().lower()
            for col, val in row.items():
                field = _COLS.get(col.strip().lower())
                if field:
                    num, unit = parse_price(val)
                    if num is not None and unit is None:
                        setattr(rec, field, num)
            for col in ("Estimated cost", "Price per minute", "Training"):
                if row.get(col):
                    num, unit = parse_price(row[col])
                    if num is not None and col != "Training":
                        rec.per_unit = num
                        rec.unit = unit or "minute"
            if rec.per_unit is None and all(
                getattr(rec, f) is None for f in ("input", "cached_input", "output")
            ):
                continue
            prices.append(rec)
            mcat = SUBCAT_MAP.get(sub or "", cat) if sub else cat
            if mcat == "other":
                mcat = category_for_id(mid)
            if mid not in models:
                models[mid] = ModelRecord(mid, PROVIDER, mcat, subcategory=sub, source="pricing",
                                          extra={"section": t.section})
    return prices, list(models.values())


def models_from_catalog(md: str) -> list[ModelRecord]:
    """Extract model ids linked from the model catalog page + the nearest group label."""
    out: dict[str, ModelRecord] = {}
    group = ""
    for line in md.splitlines():
        s = line.strip()
        h = re.match(r"^#{1,6}\s+(.*)", s)
        if h:
            group = h.group(1).strip()
        elif s and len(s) < 40 and not s.startswith(("[", "|", "-", "*", "!")) and "](" not in s:
            group = s  # short label lines like "Realtime", "Transcription"
        for text, mid in _MODEL_LINK.findall(s):
            # Links to a model's own page come in a plain and a `.md` (markdown-
            # source) variant with the same href stem — without stripping this,
            # every such model is ingested twice as distinct ids ("gpt-4o-mini"
            # and "gpt-4o-mini.md"), a real bug that inflated the catalog to 240
            # entries when only 140 were genuinely unique models.
            if mid.endswith(".md"):
                mid = mid[:-3]
            if mid in out:
                continue
            desc = text[len(mid):].strip() if text.startswith(mid) else text
            cat = category_for_id(mid)
            gcat = category_for_section(group) if group else None
            if cat in ("other", "reasoning") and gcat not in (None, "other"):
                cat = gcat
            out[mid] = ModelRecord(mid, PROVIDER, cat, description=desc or None,
                                   source="catalog", extra={"group": group})
    return list(out.values())


def models_from_api(api_key: str | None = None) -> list[tuple[str, int | None]]:
    """(model_id, created_unix_ts) pairs from /v1/models. `created` is the
    real, honest signal available here — NOT the same thing as a model's
    training-data knowledge cutoff (OpenAI doesn't expose that via any
    discovery source this scrapes; no page fetched by this module lists it).
    Fabricating cutoff dates would be worse than not having them, so this
    stores what's actually verifiable — when the model was added to the
    API — as `extra["api_created"]`, labeled as such everywhere it's shown."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    return sorted(((m.id, getattr(m, "created", None)) for m in client.models.list()), key=lambda t: t[0])


class OpenAIDiscovery:
    provider = PROVIDER

    def __init__(self, settings: Settings, fetch: Fetcher | None = None,
                 list_api_models: Callable[[], list[str]] | None = None, use_api: bool = True):
        self.settings = settings
        self.fetch = fetch or http_fetch(settings)
        self.list_api_models = list_api_models or (
            models_from_api if use_api and os.environ.get("OPENAI_API_KEY") else None
        )

    def run(self) -> DiscoveryResult:
        warnings: list[str] = []
        docs: dict[str, str] = {}
        prices: list[PriceRecord] = []
        models: dict[str, ModelRecord] = {}

        got = _first_ok(self.fetch, self.settings.openai_pricing_urls, warnings)
        if got:
            url, md = got
            docs[url] = md
            p, m = prices_from_tables(parse_tables(md))
            prices += p
            models.update({r.model_id: r for r in m})
            if not p:
                warnings.append(f"no price tables parsed from {url} — page format may have changed")

        got = _first_ok(self.fetch, self.settings.openai_models_urls, warnings)
        if got:
            url, md = got
            docs[url] = md
            for r in models_from_catalog(md):
                if r.model_id in models:
                    models[r.model_id].description = models[r.model_id].description or r.description
                    if models[r.model_id].category in ("other",):
                        models[r.model_id].category = r.category
                else:
                    models[r.model_id] = r

        if self.list_api_models:
            try:
                for entry in self.list_api_models():
                    # tuple from the real models_from_api(); existing tests/
                    # callers may still pass plain id strings — accept both.
                    mid, created_ts = entry if isinstance(entry, tuple) else (entry, None)
                    if mid not in models:
                        models[mid] = ModelRecord(mid, PROVIDER, category_for_id(mid), source="api")
                    else:
                        models[mid].extra["api_visible"] = True
                    if created_ts:
                        from datetime import datetime, timezone
                        models[mid].extra["api_created"] = datetime.fromtimestamp(
                            created_ts, tz=timezone.utc).date().isoformat()
            except Exception as e:  # noqa: BLE001
                warnings.append(f"/v1/models failed: {e}")

        return DiscoveryResult(PROVIDER, list(models.values()), prices, docs, warnings)
