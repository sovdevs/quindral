"""Markdown table extraction with heading context and pricing-tier tab mapping.

Docs pages render tiered prices as tab labels ("Standard", "Batch", "Flex", "Fast mode")
followed by one table per tab, in the same order. We map tables to tabs positionally.
Two-row headers ("| | Short context | | | Long context |" then "| Model | Input | ...")
are merged into "Short context Input" etc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TIER_LABELS = {"standard", "batch", "flex", "fast mode", "priority", "scale"}
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


@dataclass
class MdTable:
    headings: list[str]
    columns: list[str]
    rows: list[dict[str, str]]
    tier: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def section(self) -> str:
        return self.headings[-1] if self.headings else ""


def _cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _merge_headers(group_row: list[str], name_row: list[str]) -> list[str]:
    out, current = [], ""
    for i, name in enumerate(name_row):
        g = group_row[i] if i < len(group_row) else ""
        if g:
            current = g
        out.append(f"{current} {name}".strip() if current and name.lower() != "model" else name)
    return out


def parse_tables(md: str) -> list[MdTable]:
    lines = md.splitlines()
    heads: list[tuple[int, str]] = []
    tabs: list[str] = []
    tab_idx = 0
    group_keys: list[tuple[set[str], str | None]] = []  # (row keys, tier) of tables under current heading
    tables: list[MdTable] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            heads = [h for h in heads if h[0] < level] + [(level, m.group(2).strip())]
            tabs, tab_idx, group_keys = [], 0, []
            i += 1
            continue
        plain = line.strip()
        if plain.lower() in TIER_LABELS:
            if plain not in tabs:
                tabs.append(plain)
            i += 1
            continue
        if plain.startswith("|") and i + 1 < len(lines) and _SEP.match(lines[i + 1].strip()):
            header = _cells(plain)
            j = i + 2
            body: list[list[str]] = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                body.append(_cells(lines[j]))
                j += 1
            # two-row header: first body row holds the real column names
            if body and any(c.lower() == "model" for c in body[0]) and not any(
                c.lower() == "model" for c in header
            ):
                columns = _merge_headers(header, body[0])
                body = body[1:]
            else:
                columns = header
            rows = []
            for r in body:
                r = r + [""] * (len(columns) - len(r))
                rows.append(dict(zip(columns, r)))
            keys = {next(iter(r.values()), "") for r in rows} - {""}
            if group_keys and keys and all(not (keys & k) for k, _ in group_keys):
                # disjoint model set => an additional model list, not another pricing tier
                tier = group_keys[0][1]
            else:
                tier = tabs[tab_idx] if tab_idx < len(tabs) else None
                tab_idx += 1
            group_keys.append((keys, tier))
            tables.append(MdTable([h for _, h in heads], columns, rows, tier))
            i = j
            continue
        i += 1
    return tables


_PRICE = re.compile(r"\$\s*([0-9][0-9,]*\.?[0-9]*)(?:\s*/\s*([A-Za-z0-9 ]+))?")


def parse_price(cell: str) -> tuple[float | None, str | None]:
    """'$10.00' -> (10.0, None); '$0.034 / minute' -> (0.034, 'minute'); '-' -> (None, None)."""
    if not cell:
        return None, None
    m = _PRICE.search(cell)
    if not m:
        return None, None
    val = float(m.group(1).replace(",", ""))
    unit = m.group(2).strip().lower() if m.group(2) else None
    return val, unit
