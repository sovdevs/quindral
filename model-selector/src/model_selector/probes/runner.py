"""Probe runner: execute prompt suites against models, grade, cost, and persist results."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from ..config import Settings
from ..costing import Usage, cost, estimate
from ..store import Store
from ..taxonomy import CATEGORIES
from .adapters import CallResult, OpenAIAdapters
from .graders import grade

DIFF_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def load_suite(path_or_name: str | Path) -> dict[str, Any]:
    p = Path(path_or_name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return yaml.safe_load(p.read_text())
    text = resources.files("model_selector.prompts").joinpath(f"{path_or_name}.yaml").read_text()
    return yaml.safe_load(text)


def load_suites() -> dict[str, dict[str, Any]]:
    out = {}
    for f in resources.files("model_selector.prompts").iterdir():
        if f.name.endswith(".yaml"):
            s = yaml.safe_load(f.read_text())
            out[s["suite"]] = s
    return out


def expand_prompt(p: dict[str, Any]) -> str:
    if p.get("generator") == "needle":
        prm = p["params"]
        rng = random.Random(hashlib.md5(p["id"].encode()).hexdigest())
        n_sent = max(1, int(prm.get("haystack_tokens", 8000)) // 14)
        subjects = ["The committee", "A courier", "The archive", "Our auditor", "The harbour office"]
        verbs = ["reviewed", "logged", "misplaced", "approved", "rescheduled"]
        objs = ["shipment", "ledger", "permit", "survey", "manifest"]
        sents = [f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objs)} #{rng.randint(1000, 9999)} "
                 f"on day {rng.randint(1, 365)}." for _ in range(n_sent)]
        sents.insert(int(n_sent * prm.get("depth", 0.6)), prm["needle"])
        return " ".join(sents) + "\n\n" + prm["question"]
    return p["input"]


@dataclass
class ProbeOutcome:
    model_id: str
    effort: str | None
    prompt_id: str
    difficulty: str
    ok: bool | None
    score: float | None
    cost_usd: float | None
    detail: str
    error: str | None = None


class ProbeRunner:
    def __init__(self, store: Store, settings: Settings, adapters: OpenAIAdapters | None = None,
                 provider: str = "openai", log: Callable[[str], None] = print):
        self.store, self.s, self.provider, self.log = store, settings, provider, log
        self._adapters = adapters

    @property
    def a(self) -> OpenAIAdapters:
        if self._adapters is None:
            self._adapters = OpenAIAdapters()
        return self._adapters

    # ---- helpers ----
    def _prices(self, model: str) -> tuple[list[dict[str, Any]], str]:
        rows = [dict(r) for r in self.store.current_prices(self.provider, model, "Standard")] or \
               [dict(r) for r in self.store.current_prices(self.provider, model)]
        if rows:
            return rows, "listed"
        row, basis = self.store.price_for(self.provider, model)
        return ([row] if row else []), basis

    def _cheapest(self, category: str) -> str | None:
        best = None
        for m in self.store.models(self.provider, category):
            rows, _ = self._prices(m["model_id"])
            c = estimate(rows, 1000, 1000, units=1.0)
            if c is not None and (best is None or c < best[0]):
                best = (c, m["model_id"])
        return best[1] if best else None

    def _judge_model(self) -> str | None:
        if self.s.judge_model:
            return self.s.judge_model
        ranked = sorted(
            ((estimate(self._prices(m["model_id"])[0], 1000, 1000), m["model_id"])
             for m in self.store.models(self.provider, "reasoning")),
            key=lambda x: (x[0] is None, x[0] or 0))
        ranked = [r for r in ranked if r[0] is not None]
        return ranked[len(ranked) // 2][1] if ranked else None  # median-priced model as judge

    def _judge(self) -> Callable[[str], str] | None:
        jm = self._judge_model()
        return (lambda q: self.a.text(jm, q, max_output_tokens=300).text) if jm else None

    def _vision_judge(self) -> Callable[[str, str], str] | None:
        jm = self._judge_model()
        return (lambda q, img: self.a.vision_ask(jm, q, img)) if jm else None

    def _audio_fixture(self, text: str, pid: str) -> Path:
        path = self.s.audio_dir / f"{pid}-{hashlib.md5(text.encode()).hexdigest()[:8]}.wav"
        if not path.exists():
            tts = self.s.tts_model_for_fixtures or self._cheapest("tts")
            if not tts:
                raise RuntimeError("no TTS model in catalog to synthesize audio fixtures; supply `audio:`")
            path.write_bytes(self.a.tts(tts, text).audio or b"")
        return path

    # ---- main ----
    def run(self, suite: dict[str, Any], models: Iterable[str], efforts: Iterable[str | None] = (None,),
            max_difficulty: str = "hard", allow_expensive: bool = False, suite_name: str | None = None
            ) -> list[ProbeOutcome]:
        name = suite_name or suite["suite"]
        kind = CATEGORIES[suite["category"]].probe_kind
        if suite.get("expensive") and not allow_expensive:
            raise PermissionError(f"suite {name} is marked expensive; pass allow_expensive=True")
        prompts = [p for p in suite.get("prompts", [])
                   if DIFF_ORDER.get(p.get("difficulty", "medium"), 1) <= DIFF_ORDER[max_difficulty]]
        judge, vjudge = None, None
        out: list[ProbeOutcome] = []
        for model in models:
            prices, basis = self._prices(model)
            for effort in (efforts if kind in ("text", "deep_research") else (None,)):
                unsupported = False
                for p in prompts:
                    if unsupported:
                        break
                    diff = p.get("difficulty", "medium")
                    text_in = expand_prompt(p) if "input" in p or "generator" in p else p.get("reference", "")
                    guess = estimate(prices, len(text_in) // 4, p.get("max_output_tokens", 1000), units=1.0)
                    if guess is not None and guess > self.s.max_probe_cost_usd:
                        self.log(f"skip {model}/{p['id']}: est ${guess:.3f} > guard")
                        continue
                    try:
                        res, ref, img = self._call(kind, model, p, text_in, effort)
                        if p["grader"]["type"] == "llm_judge" and judge is None:
                            judge = self._judge()
                        if p["grader"]["type"] == "vision_judge" and vjudge is None:
                            vjudge = self._vision_judge()
                        ok, score, detail = grade(p["grader"], res.text, reference=ref, judge=judge,
                                                  allow_code_exec=self.s.allow_code_exec,
                                                  image_b64=img, vision_judge=vjudge)
                        c, cbasis = cost(prices, res.usage, self.s.long_context_threshold)
                        if c is not None:
                            c += res.extra_cost_usd
                        cbasis = f"{cbasis}|price:{basis}"
                        err = None
                        if res.meta.get("status") == "incomplete":
                            detail += " [incomplete: raise max_output_tokens]"
                    except Exception as e:  # noqa: BLE001
                        msg = str(e)
                        ok, score, detail, c, cbasis, err = False, 0.0, "", None, "error", msg[:500]
                        res = CallResult()
                        if effort and ("reasoning" in msg.lower() or "effort" in msg.lower()):
                            unsupported = True
                    u: Usage = res.usage
                    self.store.add_probe(
                        provider=self.provider, model_id=model, effort=effort, suite=name, prompt_id=p["id"],
                        difficulty=diff, ok=None if ok is None else int(ok), score=score,
                        latency_s=res.latency_s, input_tokens=u.input_tokens, cached_tokens=u.cached_tokens,
                        output_tokens=u.output_tokens, reasoning_tokens=u.reasoning_tokens, units=u.units,
                        cost_usd=c, cost_basis=cbasis, output_text=(res.text or "")[:4000], error=err,
                        meta={**res.meta, "detail": detail})
                    o = ProbeOutcome(model, effort, p["id"], diff, ok, score, c, detail, err)
                    out.append(o)
                    self.log(f"{model:<28} {str(effort or '-'):<7} {p['id']:<24} "
                             f"{'PASS' if ok else ('SKIP' if ok is None else 'FAIL')} "
                             f"${(c or 0):.5f} {detail[:60] if not err else 'ERR ' + err[:60]}")
        return out

    def _call(self, kind: str, model: str, p: dict[str, Any], text_in: str, effort: str | None
              ) -> tuple[CallResult, str | None, str | None]:
        mot = p.get("max_output_tokens")
        if kind == "text":
            return self.a.text(model, text_in, effort, mot), None, None
        if kind == "deep_research":
            return self.a.deep_research(model, text_in, effort, mot), None, None
        if kind == "transcription":
            ref = p["reference"]
            audio = Path(p["audio"]) if p.get("audio") else self._audio_fixture(ref, p["id"])
            return self.a.transcribe(model, audio), ref, None
        if kind == "tts":
            res = self.a.tts(model, text_in)
            path = self.s.audio_dir / f"tts-{model}-{p['id']}.wav"
            path.write_bytes(res.audio or b"")
            stt = self._cheapest("transcription")
            if stt:
                res.text = self.a.transcribe(stt, path).text
            return res, text_in, None
        if kind == "image":
            prm = p.get("params", {})
            res = self.a.image(model, text_in, prm.get("size", "1024x1024"), prm.get("quality"))
            return res, None, res.image_b64
        raise NotImplementedError(f"probing not implemented for kind={kind}")
