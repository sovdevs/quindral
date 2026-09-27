"""Task categories and model -> category classification.

Rule order: explicit pricing-section mapping, then model-id patterns. Categories are the unit
of comparison: models are only ranked against others in the same category.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    probe_kind: str          # adapter used to probe: text | deep_research | transcription | tts | image | realtime | none
    keywords: tuple[str, ...]  # for classifying novel task descriptions


CATEGORIES: dict[str, Category] = {c.key: c for c in [
    Category("reasoning", "Text / reasoning", "text",
             ("reason", "analy", "summar", "extract", "classif", "math", "logic", "answer", "write",
              "rewrite", "translate text", "json", "plan", "agent", "tool call")),
    Category("coding", "Code generation", "text",
             ("code", "function", "refactor", "bug", "python", "typescript", "sql", "regex", "unit test")),
    Category("deep_research", "Deep research", "deep_research",
             ("research", "literature", "survey", "investigate", "sources", "cite", "market scan", "report on")),
    Category("realtime_voice", "Realtime voice (speech-to-speech)", "realtime",
             ("voice agent", "phone call", "call center", "speech-to-speech", "voice assistant", "live conversation")),
    Category("realtime_transcription", "Live transcription", "realtime",
             ("live transcription", "live captions", "streaming transcription", "real-time captions")),
    Category("realtime_translation", "Live speech translation", "realtime",
             ("live translation", "interpret", "simultaneous translation")),
    Category("transcription", "File transcription", "transcription",
             ("transcribe", "transcript", "speech to text", "audio file", "podcast", "subtitle", "captions")),
    Category("tts", "Text to speech", "tts",
             ("text to speech", "tts", "narrat", "voiceover", "read aloud", "synthesize speech")),
    Category("image_generation", "Image generation / editing", "image",
             ("image", "illustration", "logo", "picture", "photo edit", "render", "poster", "thumbnail")),
    Category("embeddings", "Embeddings", "none", ("embedding", "semantic search", "vector", "similarity")),
    Category("moderation", "Moderation", "none", ("moderat", "toxicity", "content filter")),
    Category("cyber", "Cybersecurity (restricted)", "none", ("vulnerability", "pentest", "exploit analysis")),
    Category("life_sciences", "Life sciences (restricted)", "none", ("protein", "genomic", "biology research")),
    Category("other", "Uncategorized", "none", ()),
]}

# pricing-page section heading -> category (lowercased substring match)
SECTION_MAP: list[tuple[str, str]] = [
    ("flagship", "reasoning"),
    ("cyber", "cyber"),
    ("gpt-live", "realtime_voice"),
    ("realtime and audio", "realtime_voice"),
    ("image generation", "image_generation"),
    ("transcription", "transcription"),
    ("embedding", "embeddings"),
    ("moderation", "moderation"),
    ("deep research", "deep_research"),
    ("speech generation", "tts"),
]
SKIP_SECTIONS = ("finetuning", "fine-tuning", "tools")

# Specialized table "Category" column / transcription "Use case" column
SUBCAT_MAP = {
    "codex": "coding", "chatgpt": "reasoning", "life sciences": "life_sciences",
    "live translation": "realtime_translation", "live transcription": "realtime_transcription",
    "transcription": "transcription",
}

ID_PATTERNS: list[tuple[str, str]] = [
    (r"deep-research", "deep_research"),
    (r"embedding", "embeddings"),
    (r"moderation", "moderation"),
    (r"(^|-)tts", "tts"),
    (r"realtime-translate", "realtime_translation"),
    (r"(live-transcribe|realtime-whisper)", "realtime_transcription"),
    (r"(transcribe|whisper)", "transcription"),
    (r"(realtime|gpt-live)", "realtime_voice"),
    (r"(image|dall-e)", "image_generation"),
    (r"codex", "coding"),
    (r"(cyber|daybreak)", "cyber"),
    (r"rosalind", "life_sciences"),
    (r"^(gpt-|o\d|chat-latest)", "reasoning"),
]


def category_for_section(section: str, headings: list[str] | None = None) -> str | None:
    hay = " / ".join([*(headings or []), section]).lower()
    if any(s in section.lower() for s in SKIP_SECTIONS):
        return None
    for needle, cat in SECTION_MAP:
        if needle in section.lower():
            return cat
    for needle, cat in SECTION_MAP:
        if needle in hay:
            return cat
    return "other"


def category_for_id(model_id: str) -> str:
    mid = model_id.lower()
    for pat, cat in ID_PATTERNS:
        if re.search(pat, mid):
            return cat
    return "other"


def classify_task_heuristic(text: str) -> list[tuple[str, int]]:
    """Score categories by keyword hits. Returns [(category, score)] sorted desc."""
    t = text.lower()
    scores = []
    for c in CATEGORIES.values():
        s = sum(1 for k in c.keywords if k in t)
        if s:
            scores.append((c.key, s))
    # more specific categories win ties over generic reasoning
    scores.sort(key=lambda x: (-x[1], x[0] == "reasoning"))
    return scores
