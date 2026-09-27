"""Thin adapters over the OpenAI SDK, one per probe kind. All return CallResult.

Kept small and duck-typed so tests can inject a fake client, and so other providers can
implement the same methods later.
"""
from __future__ import annotations

import base64
import io
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..costing import Usage

WEB_SEARCH_USD_PER_CALL = 0.01  # $10 / 1k calls (pricing page, Tools table) — re-check on discovery


@dataclass
class CallResult:
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    extra_cost_usd: float = 0.0          # e.g. tool-call fees
    meta: dict[str, Any] = field(default_factory=dict)
    audio: bytes | None = None
    image_b64: str | None = None


def _g(obj: Any, *path: str, default: Any = 0) -> Any:
    for p in path:
        if obj is None:
            return default
        obj = obj.get(p) if isinstance(obj, dict) else getattr(obj, p, None)
    return default if obj is None else obj


def wav_minutes(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
            frames = w.getnframes()
        # streamed WAVs often carry a bogus (max) length header: derive from payload size instead
        by_size = max(len(data) - 44, 0) / float(rate * width * ch)
        secs = frames / float(rate)
        return (by_size if secs <= 0 or secs > by_size * 1.5 else secs) / 60.0
    except Exception:  # noqa: BLE001
        return None


class OpenAIAdapters:
    def __init__(self, client: Any = None):
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.c = client

    # ---- text / reasoning / coding ----
    def text(self, model: str, prompt: str, effort: str | None = None,
             max_output_tokens: int | None = None, tools: list | None = None,
             content: list | None = None) -> CallResult:
        kw: dict[str, Any] = {"model": model, "input": prompt}
        if content is not None:
            kw["input"] = [{"role": "user", "content": content}]
        if effort:
            kw["reasoning"] = {"effort": effort}
        if max_output_tokens:
            kw["max_output_tokens"] = max_output_tokens
        if tools:
            kw["tools"] = tools
        t0 = time.perf_counter()
        r = self.c.responses.create(**kw)
        dt = time.perf_counter() - t0
        u = _g(r, "usage", default=None)
        usage = Usage(
            input_tokens=int(_g(u, "input_tokens")),
            cached_tokens=int(_g(u, "input_tokens_details", "cached_tokens")),
            output_tokens=int(_g(u, "output_tokens")),
            reasoning_tokens=int(_g(u, "output_tokens_details", "reasoning_tokens")),
        )
        n_search = sum(1 for item in (_g(r, "output", default=[]) or [])
                       if _g(item, "type", default="") == "web_search_call")
        return CallResult(text=_g(r, "output_text", default="") or "", usage=usage, latency_s=dt,
                          extra_cost_usd=n_search * WEB_SEARCH_USD_PER_CALL,
                          meta={"status": _g(r, "status", default=None), "web_search_calls": n_search})

    def deep_research(self, model: str, prompt: str, effort: str | None = None,
                      max_output_tokens: int | None = None) -> CallResult:
        return self.text(model, prompt, effort, max_output_tokens, tools=[{"type": "web_search"}])

    # ---- audio ----
    def tts(self, model: str, text: str, voice: str = "alloy") -> CallResult:
        t0 = time.perf_counter()
        r = self.c.audio.speech.create(model=model, voice=voice, input=text, response_format="wav")
        data = r.read() if hasattr(r, "read") else getattr(r, "content", r)
        mins = wav_minutes(data)
        usage = Usage(input_tokens=max(1, len(text) // 4), units=mins)
        return CallResult(audio=data, usage=usage, latency_s=time.perf_counter() - t0,
                          meta={"audio_minutes": mins, "usage_estimated": True})

    def transcribe(self, model: str, audio_path: Path) -> CallResult:
        data = Path(audio_path).read_bytes()
        mins = wav_minutes(data)
        t0 = time.perf_counter()
        with open(audio_path, "rb") as f:
            r = self.c.audio.transcriptions.create(model=model, file=f)
        dt = time.perf_counter() - t0
        u = _g(r, "usage", default=None)
        usage = Usage(units=mins)
        if u is not None and _g(u, "type", default="") == "tokens":
            usage.input_tokens = int(_g(u, "input_tokens"))
            usage.output_tokens = int(_g(u, "output_tokens"))
        elif u is not None and _g(u, "type", default="") == "duration":
            usage.units = float(_g(u, "seconds")) / 60.0
        return CallResult(text=_g(r, "text", default="") or "", usage=usage, latency_s=dt,
                          meta={"audio_minutes": usage.units})

    # ---- images ----
    def image(self, model: str, prompt: str, size: str = "1024x1024", quality: str | None = None) -> CallResult:
        kw: dict[str, Any] = {"model": model, "prompt": prompt, "size": size}
        if quality:
            kw["quality"] = quality
        t0 = time.perf_counter()
        r = self.c.images.generate(**kw)
        dt = time.perf_counter() - t0
        u = _g(r, "usage", default=None)
        text_in = int(_g(u, "input_tokens_details", "text_tokens")) or int(_g(u, "input_tokens"))
        img_in = int(_g(u, "input_tokens_details", "image_tokens"))
        out = int(_g(u, "output_tokens"))
        usage = Usage(input_tokens=text_in + img_in, output_tokens=out,
                      input_by_modality={"text": text_in, **({"image": img_in} if img_in else {})},
                      output_by_modality={"image": out})
        b64 = _g(r, "data", default=[None])[0]
        return CallResult(image_b64=_g(b64, "b64_json", default=None), usage=usage, latency_s=dt,
                          meta={"size": size, "quality": quality})

    def vision_ask(self, model: str, question: str, image_b64: str) -> str:
        content = [{"type": "input_text", "text": question},
                   {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"}]
        return self.text(model, question, content=content, max_output_tokens=400).text


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
