"""CLI: model-selector {discover,models,prices,probe,stats,rank,set-category}"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .store import Store


def _fmt(v, nd=6):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _table(rows: list[dict], cols: list[str]) -> str:
    w = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) if rows else len(c) for c in cols}
    lines = ["  ".join(c.ljust(w[c]) for c in cols), "  ".join("-" * w[c] for c in cols)]
    lines += ["  ".join(_fmt(r.get(c)).ljust(w[c]) for c in cols) for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="model-selector")
    ap.add_argument("--home", help="data dir (default $MODEL_SELECTOR_HOME or ./data/model_selector)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="fetch latest models + prices and store a snapshot")
    d.add_argument("--no-api", action="store_true", help="skip /v1/models listing")

    m = sub.add_parser("models"); m.add_argument("--category")
    p = sub.add_parser("prices"); p.add_argument("--category"); p.add_argument("--tier", default="Standard")

    pr = sub.add_parser("probe", help="run a prompt suite against models")
    pr.add_argument("suite")
    pr.add_argument("--models", nargs="*", help="default: all models in the suite's category")
    pr.add_argument("--unprobed-only", action="store_true", help="only models never probed (new/unseen)")
    pr.add_argument("--efforts", nargs="*", default=None, help="e.g. none low medium high ('default' = omit)")
    pr.add_argument("--max-difficulty", default="hard", choices=["easy", "medium", "hard"])
    pr.add_argument("--allow-expensive", action="store_true")
    pr.add_argument("--allow-code-exec", action="store_true")
    pr.add_argument("--max-call-usd", type=float)

    st = sub.add_parser("stats"); st.add_argument("--suite")

    r = sub.add_parser("rank", help="rank models for a novel task, cheapest first")
    r.add_argument("description")
    r.add_argument("--sample", help="sample input text (or @path)")
    r.add_argument("--category"); r.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    r.add_argument("--in-tokens", type=int); r.add_argument("--out-tokens", type=int)
    r.add_argument("--audio-minutes", type=float); r.add_argument("--calls", type=int, default=1)
    r.add_argument("--min-pass", type=float, default=0.8)
    r.add_argument("--llm-classify", help="model id to use for task classification")
    r.add_argument("--trial", action="store_true", help="cascade-run the sample cheapest-first until PASS")
    r.add_argument("--rubric"); r.add_argument("--expect")
    r.add_argument("--json", action="store_true")

    sc = sub.add_parser("set-category"); sc.add_argument("model_id"); sc.add_argument("category")

    a = ap.parse_args(argv)
    s = Settings()
    if a.home:
        from pathlib import Path
        s.home = Path(a.home)
    s.ensure()
    store = Store(s.db_path)

    if a.cmd == "discover":
        from .discovery import OpenAIDiscovery, GeminiDiscovery, ClaudeDiscovery
        rep = {
            "openai": store.ingest(OpenAIDiscovery(s, use_api=not a.no_api).run()),
            # No API key needed for Gemini — model list, capabilities
            # (including Search grounding) and Knowledge cutoff/Latest update
            # come straight from Google's public per-model docs pages, which
            # OpenAI's equivalent docs simply don't expose (see gemini.py's
            # module docstring).
            "google": store.ingest(GeminiDiscovery(s).run()),
            # Also no API key needed — Anthropic's overview page publishes a
            # clean comparison table with real pricing + BOTH a "reliable"
            # and "training data" knowledge cutoff per current model (see
            # claude.py's module docstring). Only the current lineup (4
            # models at time of writing) is captured — legacy model API ids
            # aren't guessed from their display names.
            "anthropic": store.ingest(ClaudeDiscovery(s).run()),
        }
        print(json.dumps(rep, indent=2))
        return 0

    if a.cmd == "models":
        rows = [dict(x) for x in store.models(category=a.category)]
        print(_table(rows, ["category", "model_id", "source", "status", "first_seen"]))
        return 0

    if a.cmd == "prices":
        cats = {x["model_id"]: x["category"] for x in store.models()}
        rows = [dict(x) | {"category": cats.get(x["model_id"])} for x in store.current_prices(tier=a.tier)]
        if a.category:
            rows = [x for x in rows if x["category"] == a.category]
        rows.sort(key=lambda x: (x["category"] or "", x["per_unit"] if x["per_unit"] is not None else (x["input"] or 0)))
        print(_table(rows, ["category", "model_id", "modality", "input", "cached_input", "output",
                            "per_unit", "unit"]))
        return 0

    if a.cmd == "probe":
        from .probes import ProbeRunner, load_suite
        suite = load_suite(a.suite)
        if a.allow_code_exec:
            s.allow_code_exec = True
        if a.max_call_usd:
            s.max_probe_cost_usd = a.max_call_usd
        models = a.models or [x["model_id"] for x in store.models("openai", suite["category"])]
        if a.unprobed_only:
            seen = {mid for _, mid in store.probed_models()}
            models = [x for x in models if x not in seen]
        efforts = [None if e in ("default", "") else e for e in (a.efforts or ["default"])]
        ProbeRunner(store, s).run(suite, models, efforts, a.max_difficulty, a.allow_expensive)
        return 0

    if a.cmd == "stats":
        rows = [dict(x) for x in store.probe_stats(suite=a.suite)]
        rows.sort(key=lambda x: (x["suite"], x["difficulty"], x["avg_cost"] or 0))
        print(_table(rows, ["suite", "difficulty", "model_id", "effort", "n", "pass_rate", "avg_cost",
                            "avg_out", "avg_reason", "avg_latency", "errors"]))
        return 0

    if a.cmd == "rank":
        from .ranker import TaskSpec, rank, ranking_to_dict, trial
        sample = a.sample
        if sample and sample.startswith("@"):
            sample = open(sample[1:]).read()
        task = TaskSpec(a.description, sample, a.category, a.difficulty, a.in_tokens, a.out_tokens,
                        a.audio_minutes, a.calls, a.min_pass)
        llm = None
        if a.llm_classify:
            from .probes.adapters import OpenAIAdapters
            ad = OpenAIAdapters()
            llm = lambda q: ad.text(a.llm_classify, q, max_output_tokens=200).text  # noqa: E731
        res = rank(store, task, llm=llm)
        if a.trial:
            from .probes import ProbeRunner
            trial(store, ProbeRunner(store, s), res, a.rubric, a.expect)
            res = rank(store, task, llm=llm)
        if a.json:
            print(json.dumps(ranking_to_dict(res), indent=2))
            return 0
        print(f"category={res.category} difficulty={res.difficulty} (by {res.classified_by})")
        rows = [{"model": c.model_id, "effort": c.effort, "usd/call": c.est_cost_per_call,
                 "usd/total": c.est_total, "pass": c.pass_rate, "n": c.n_probes,
                 "ok": {True: "yes", False: "no"}.get(c.meets_bar, "?"), "notes": "; ".join(c.notes)}
                for c in res.candidates]
        print(_table(rows, ["model", "effort", "usd/call", "usd/total", "pass", "n", "ok", "notes"]))
        if res.recommendation:
            rc = res.recommendation
            print(f"\n→ cheapest meeting bar: {rc.model_id} (effort={rc.effort or 'default'}) "
                  f"~${rc.est_cost_per_call:.6f}/call")
        elif res.unprobed:
            print(f"\nno evidence yet — probe: model-selector probe <suite> --models {' '.join(res.unprobed[:6])}")
        return 0

    if a.cmd == "set-category":
        store.set_category("openai", a.model_id, a.category)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
