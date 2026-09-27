"""Local SQLite catalog: snapshots, models, price history, probe results."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .discovery.base import DiscoveryResult, PriceRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots(
  id INTEGER PRIMARY KEY, provider TEXT, url TEXT, fetched_at TEXT, sha256 TEXT, content TEXT);
CREATE TABLE IF NOT EXISTS models(
  model_id TEXT, provider TEXT, category TEXT, subcategory TEXT, display_name TEXT, description TEXT,
  source TEXT, extra TEXT, first_seen TEXT, last_seen TEXT, status TEXT DEFAULT 'active',
  category_locked INTEGER DEFAULT 0, PRIMARY KEY(provider, model_id));
CREATE TABLE IF NOT EXISTS prices(
  id INTEGER PRIMARY KEY, run_at TEXT, model_id TEXT, provider TEXT, tier TEXT, modality TEXT, unit TEXT,
  input REAL, cached_input REAL, cache_write REAL, output REAL,
  long_input REAL, long_cached_input REAL, long_cache_write REAL, long_output REAL,
  per_unit REAL, section TEXT);
CREATE INDEX IF NOT EXISTS ix_prices ON prices(provider, model_id, tier, modality, run_at);
CREATE TABLE IF NOT EXISTS probes(
  id INTEGER PRIMARY KEY, ts TEXT, provider TEXT, model_id TEXT, effort TEXT, suite TEXT, prompt_id TEXT,
  difficulty TEXT, ok INTEGER, score REAL, latency_s REAL, input_tokens INTEGER, cached_tokens INTEGER,
  output_tokens INTEGER, reasoning_tokens INTEGER, units REAL, cost_usd REAL, cost_basis TEXT,
  output_text TEXT, error TEXT, meta TEXT);
CREATE INDEX IF NOT EXISTS ix_probes ON probes(provider, model_id, suite, difficulty);
"""

PRICE_FIELDS = ("input", "cached_input", "cache_write", "output", "long_input", "long_cached_input",
                "long_cache_write", "long_output", "per_unit")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---------- discovery ----------
    def ingest(self, res: DiscoveryResult) -> dict[str, Any]:
        ts = now()
        before_models = {r["model_id"] for r in self.models(res.provider, include_retired=True)}
        before_prices = {(p["model_id"], p["tier"], p["modality"]): p for p in self.current_prices(res.provider)}

        for url, content in res.documents.items():
            h = hashlib.sha256(content.encode()).hexdigest()
            last = self.db.execute(
                "SELECT sha256 FROM snapshots WHERE provider=? AND url=? ORDER BY id DESC LIMIT 1",
                (res.provider, url)).fetchone()
            if not last or last["sha256"] != h:
                self.db.execute("INSERT INTO snapshots(provider,url,fetched_at,sha256,content) VALUES(?,?,?,?,?)",
                                (res.provider, url, ts, h, content))

        for m in res.models:
            row = self.db.execute("SELECT category_locked FROM models WHERE provider=? AND model_id=?",
                                  (m.provider, m.model_id)).fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO models(model_id,provider,category,subcategory,display_name,description,source,"
                    "extra,first_seen,last_seen) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (m.model_id, m.provider, m.category, m.subcategory, m.display_name, m.description,
                     m.source, json.dumps(m.extra), ts, ts))
            else:
                cat_sql = "" if row["category_locked"] else "category=?,"
                params = ([] if row["category_locked"] else [m.category]) + [
                    m.subcategory, m.description, m.source, json.dumps(m.extra), ts, m.provider, m.model_id]
                self.db.execute(
                    f"UPDATE models SET {cat_sql} subcategory=COALESCE(?,subcategory), "
                    "description=COALESCE(?,description), source=?, extra=?, last_seen=?, status='active' "
                    "WHERE provider=? AND model_id=?", params)

        changes = []
        for p in res.prices:
            key = (p.model_id, p.tier, p.modality)
            old = before_prices.get(key)
            if old and any((old[f] or 0) != (getattr(p, f) or 0) for f in PRICE_FIELDS):
                changes.append({"model": p.model_id, "tier": p.tier, "modality": p.modality,
                                "old": {f: old[f] for f in PRICE_FIELDS if old[f] is not None},
                                "new": {f: getattr(p, f) for f in PRICE_FIELDS if getattr(p, f) is not None}})
            row = p.as_row()
            cols = ["run_at", *row.keys()]
            self.db.execute(f"INSERT INTO prices({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                            [ts, *row.values()])

        seen = {m.model_id for m in res.models}
        retired = sorted(before_models - seen)
        if res.models:  # don't retire everything on a failed fetch
            for mid in retired:
                self.db.execute("UPDATE models SET status='not_listed' WHERE provider=? AND model_id=?",
                                (res.provider, mid))
        self.db.commit()
        return {"new_models": sorted(seen - before_models), "not_listed": retired,
                "price_changes": changes, "warnings": res.warnings,
                "n_models": len(seen), "n_prices": len(res.prices)}

    # ---------- queries ----------
    def models(self, provider: str | None = None, category: str | None = None,
               include_retired: bool = False) -> list[sqlite3.Row]:
        q, a = "SELECT * FROM models WHERE 1=1", []
        if provider:
            q += " AND provider=?"; a.append(provider)
        if category:
            q += " AND category=?"; a.append(category)
        if not include_retired:
            q += " AND status='active'"
        return self.db.execute(q + " ORDER BY category, model_id", a).fetchall()

    def set_category(self, provider: str, model_id: str, category: str) -> None:
        self.db.execute("UPDATE models SET category=?, category_locked=1 WHERE provider=? AND model_id=?",
                        (category, provider, model_id))
        self.db.commit()

    def current_prices(self, provider: str | None = None, model_id: str | None = None,
                       tier: str | None = None) -> list[sqlite3.Row]:
        q = """SELECT p.* FROM prices p JOIN (
                 SELECT provider, model_id, tier, modality, MAX(run_at) AS r FROM prices GROUP BY 1,2,3,4
               ) l ON p.provider=l.provider AND p.model_id=l.model_id AND p.tier=l.tier
                  AND p.modality=l.modality AND p.run_at=l.r WHERE 1=1"""
        a: list[Any] = []
        if provider:
            q += " AND p.provider=?"; a.append(provider)
        if model_id:
            q += " AND p.model_id=?"; a.append(model_id)
        if tier:
            q += " AND p.tier=?"; a.append(tier)
        return self.db.execute(q + " GROUP BY p.provider,p.model_id,p.tier,p.modality", a).fetchall()

    def price_for(self, provider: str, model_id: str, tier: str = "Standard",
                  modality: str | None = None) -> tuple[dict[str, Any] | None, str]:
        """Return (price_row, basis). Falls back to family/sibling pricing for unseen model ids."""
        rows = self.current_prices(provider, model_id, tier) or self.current_prices(provider, model_id)
        if rows:
            pick = next((r for r in rows if modality and r["modality"] == modality), None) or \
                   next((r for r in rows if r["modality"] == "text"), rows[0])
            return dict(pick), "listed"
        # sibling inference: strip date / version suffixes
        base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model_id)
        cands = [base] + [base.rsplit("-", k)[0] for k in (1, 2) if "-" in base]
        for c in cands:
            if c != model_id:
                r = self.current_prices(provider, c, tier)
                if r:
                    return dict(r[0]), f"inferred:{c}"
        return None, "unknown"

    # ---------- probes ----------
    def add_probe(self, **kw: Any) -> None:
        kw.setdefault("ts", now())
        if isinstance(kw.get("meta"), (dict, list)):
            kw["meta"] = json.dumps(kw["meta"])
        cols = list(kw)
        self.db.execute(f"INSERT INTO probes({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                        [kw[c] for c in cols])
        self.db.commit()

    def probe_stats(self, provider: str | None = None, suite: str | None = None) -> list[sqlite3.Row]:
        q = """SELECT provider, model_id, COALESCE(effort,'') AS effort, suite, difficulty,
                      COUNT(*) n, AVG(ok) pass_rate, AVG(score) score, AVG(cost_usd) avg_cost,
                      AVG(input_tokens) avg_in, AVG(output_tokens) avg_out, AVG(reasoning_tokens) avg_reason,
                      AVG(latency_s) avg_latency, SUM(error IS NOT NULL) errors
               FROM probes WHERE 1=1"""
        a: list[Any] = []
        if provider:
            q += " AND provider=?"; a.append(provider)
        if suite:
            q += " AND suite=?"; a.append(suite)
        return self.db.execute(q + " GROUP BY 1,2,3,4,5", a).fetchall()

    def probed_models(self) -> set[tuple[str, str]]:
        return {(r[0], r[1]) for r in self.db.execute("SELECT DISTINCT provider, model_id FROM probes")}


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]
