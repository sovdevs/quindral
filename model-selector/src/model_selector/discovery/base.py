from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class ModelRecord:
    model_id: str
    provider: str
    category: str
    subcategory: str | None = None
    display_name: str | None = None
    description: str | None = None
    source: str = ""          # "pricing", "catalog", "api"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PriceRecord:
    model_id: str
    provider: str
    tier: str = "Standard"
    modality: str = "text"
    unit: str = "1M tokens"   # or "minute", "call", "hour"
    input: float | None = None
    cached_input: float | None = None
    cache_write: float | None = None
    output: float | None = None
    long_input: float | None = None
    long_cached_input: float | None = None
    long_cache_write: float | None = None
    long_output: float | None = None
    per_unit: float | None = None   # flat price per `unit` (minute-billed models)
    section: str = ""
    raw: dict[str, str] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw")
        return d


@dataclass
class DiscoveryResult:
    provider: str
    models: list[ModelRecord]
    prices: list[PriceRecord]
    documents: dict[str, str]           # url -> raw content (for snapshots)
    warnings: list[str] = field(default_factory=list)


class Discovery(Protocol):
    provider: str

    def run(self) -> DiscoveryResult: ...
