"""Rank models for a (possibly novel) task: cheapest -> most expensive, annotated with evidence.

Cost per call is estimated from *observed* output tokens per model/effort (probe history) where
available — this captures reasoning-token inflation, which is why a cheap-per-token model at high
effort can cost more than a pricier model at low effort.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .costing import estimate
from .probes.runner import DIFF_ORDER, load_suite
from .store import Store
from .taxonomy import CATEGORIES, classify_task_heuristic

DEFAULT_OUT_TOKENS = {"easy": 150, "medium": 600, "hard": 2500}
CATEGORY_SUITE = {"reasoning": "reasoning", "coding": "coding", "deep_research": "deep_research",
                  "transcription": "transcription", "tts": "tts", "image_generation": "image_generation",
                  "realtime_voice": "realtime", "realtime_transcription": None, "realtime_translation": None}


@dataclass
class TaskSpec:
    description: str
    sample_input: str | None = None
    category: str | None = None
    difficulty: str | None = None       # easy | medium | hard
    in_tokens: int | None = None
    out_tokens: int | None = None
    audio_minutes: float | None = None  # per call, for audio/realtime categories
    calls: int = 1                      # volume multiplier for totals
    min_pass_rate: float = 0.8


@dataclass
class Candidate:
    model_id: str
    effort: str | None
    category: str
    est_cost_per_call: float | None
    est_total: float | None
    price_basis: str
    pass_rate: float | None = None      # on probes at >= task difficulty
    n_probes: int = 0
    avg_latency_s: float | None = None
    meets_bar: bool | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Ranking:
    task: TaskSpec
    category: str
    difficulty: str
    classified_by: str
    candidates: list[Candidate]
    recommendation: Candidate | None
    unprobed: list[str]


def classify(task: TaskSpec, llm: Callable[[str], str] | None = None) -> tuple[str, str, str]:
    if task.category:
        return task.category, task.difficulty or "medium", "user"
    text = task.description + " " + (task.sample_input or "")[:500]
    if llm:
        cats = ", ".join(f"{k} ({c.label})" for k, c in CATEGORIES.items())
        raw = llm(f"Classify this task. Categories: {cats}. Difficulty: easy (lookup/format/extract), "
                  f"medium (short multi-step), hard (long reasoning, tricky math/code, long context).\n"
                  f"Task: {text}\nReply ONLY JSON: {{\"category\": \"...\", \"difficulty\": \"...\"}}")
        try:
            obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            if obj.get("category") in CATEGORIES:
                return obj["category"], task.difficulty or obj.get("difficulty", "medium"), "llm"
        except Exception:  # noqa: BLE001
            pass
    scores = classify_task_heuristic(text)
    cat = scores[0][0] if scores else "reasoning"
    diff = task.difficulty or ("hard" if re.search(r"prove|complex|multi-step|long document|olympiad|"
                                                  r"architecture|research", text, re.I) else "medium")
    return cat, diff, "heuristic"


def _realtime_units(minutes: float) -> dict[str, float]:
    a = load_suite("realtime")["assumptions"]
    return {"in_audio": minutes * 60 * a["user_audio_tokens_per_second"] * 0.5,
            "out_audio": minutes * 60 * a["model_audio_tokens_per_second"] * 0.4}


def rank(store: Store, task: TaskSpec, provider: str = "openai",
         llm: Callable[[str], str] | None = None, include_efforts: bool = True) -> Ranking:
    cat, diff, how = classify(task, llm)
    suite = CATEGORY_SUITE.get(cat)
    stats = [dict(r) for r in store.probe_stats(provider, suite)] if suite else []
    # evidence from trials of *this* task outranks generic suite evidence
    import hashlib
    custom = [dict(r) for r in store.probe_stats(
        provider, "custom:" + hashlib.md5(task.description.encode()).hexdigest()[:8])]
    if custom:
        tried = {(s["model_id"], s["effort"]) for s in custom}
        stats = custom + [s for s in stats if (s["model_id"], s["effort"]) not in tried]
    in_tok = task.in_tokens or (len(task.sample_input) // 4 if task.sample_input else 500)

    cands: list[Candidate] = []
    unprobed: list[str] = []
    for m in store.models(provider, cat):
        mid = m["model_id"]
        rows = [dict(r) for r in store.current_prices(provider, mid, "Standard")] or \
               [dict(r) for r in store.current_prices(provider, mid)]
        basis = "listed"
        if not rows:
            row, basis = store.price_for(provider, mid)
            rows = [row] if row else []
        mstats = [s for s in stats if s["model_id"] == mid]
        efforts = sorted({s["effort"] for s in mstats}) if (include_efforts and mstats) else [""]
        if not mstats:
            unprobed.append(mid)
        for eff in efforts:
            relevant = [s for s in mstats if s["effort"] == eff
                        and DIFF_ORDER.get(s["difficulty"], 1) >= DIFF_ORDER[diff]] or \
                       [s for s in mstats if s["effort"] == eff]
            n = sum(s["n"] for s in relevant)
            graded = [s for s in relevant if s["pass_rate"] is not None]
            pr = (sum(s["pass_rate"] * s["n"] for s in graded) / sum(s["n"] for s in graded)) if graded else None
            out_tok = task.out_tokens
            if out_tok is None:
                obs = [s["avg_out"] for s in relevant if s["avg_out"]]
                out_tok = int(sum(obs) / len(obs)) if obs else DEFAULT_OUT_TOKENS[diff]
            notes = []
            if cat.startswith("realtime"):
                mins = task.audio_minutes or 5
                per_min = next((r for r in rows if r.get("per_unit") is not None), None)
                if per_min:
                    c = per_min["per_unit"] * mins
                else:
                    ru = _realtime_units(mins)
                    audio = next((r for r in rows if r.get("modality") == "audio"), None)
                    c = ((ru["in_audio"] * (audio["input"] or 0) + ru["out_audio"] * (audio["output"] or 0)) / 1e6
                         if audio else None)
                    notes.append("realtime cost from assumed audio-token rates")
            else:
                c = estimate(rows, in_tok, out_tok, units=task.audio_minutes)
            lat = [s["avg_latency"] for s in relevant if s["avg_latency"]]
            cand = Candidate(mid, eff or None, cat, c, c * task.calls if c is not None else None,
                             basis, pr, n, sum(lat) / len(lat) if lat else None, notes=notes)
            if pr is not None:
                cand.meets_bar = pr >= task.min_pass_rate
            if basis.startswith("inferred"):
                cand.notes.append(f"price {basis}")
            if not rows:
                cand.notes.append("no price found")
            cands.append(cand)

    cands.sort(key=lambda c: (c.est_cost_per_call is None, c.est_cost_per_call or 0))
    rec = next((c for c in cands if c.meets_bar), None)
    return Ranking(task, cat, diff, how, cands, rec, unprobed)


def ranking_to_dict(r: Ranking) -> dict[str, Any]:
    return {"category": r.category, "difficulty": r.difficulty, "classified_by": r.classified_by,
            "recommendation": asdict(r.recommendation) if r.recommendation else None,
            "unprobed": r.unprobed, "candidates": [asdict(c) for c in r.candidates]}


def trial(store: Store, runner: Any, ranking: Ranking, rubric: str | None = None, expect: str | None = None,
          max_models: int = 5, efforts: tuple[str | None, ...] = (None,)) -> list[Any]:
    """Cascade: run the novel task's sample on candidates cheapest-first; stop at the first PASS.

    Results are stored under suite 'custom:<hash>' so later rankings for the same task reuse them.
    """
    import hashlib
    if not ranking.task.sample_input:
        raise ValueError("trial needs task.sample_input")
    grader = ({"type": "contains", "expect": expect} if expect else
              {"type": "llm_judge", "rubric": rubric or f"PASS if the response correctly and completely "
                                                         f"accomplishes: {ranking.task.description}"})
    sid = "custom:" + hashlib.md5(ranking.task.description.encode()).hexdigest()[:8]
    suite = {"suite": sid, "category": ranking.category,
             "prompts": [{"id": "sample", "difficulty": ranking.difficulty,
                          "input": ranking.task.sample_input, "grader": grader,
                          "max_output_tokens": ranking.task.out_tokens or 4000}]}
    tried, results = set(), []
    for c in ranking.candidates:
        if c.model_id in tried or len(tried) >= max_models or c.est_cost_per_call is None:
            continue
        tried.add(c.model_id)
        outs = runner.run(suite, [c.model_id], efforts=efforts, suite_name=sid)
        results += outs
        if any(o.ok for o in outs):
            break
    return results
