"""One-off exporter: dumps the current model_registry.REGISTRY to a static JSON
snapshot for the Netlify demo build (netlify/v0.5/). Not part of the live app —
run manually whenever you want to refresh a demo snapshot:
    .venv/bin/python3 scripts_export_snapshot.py netlify/v0.5/registry.json
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

from model_registry import REGISTRY


def export_registry() -> list:
    models = []
    for m in REGISTRY.values():
        d = asdict(m)
        d["capabilities"] = sorted(m.capabilities)
        models.append(d)
    return models


if __name__ == "__main__":
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("registry_snapshot.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(export_registry(), indent=2))
    print(f"exported {len(REGISTRY)} models -> {out_path}")
