"""Phase 0 classifier: heuristics + word-overlap similarity to route a query.

No training data, no ML deps (stdlib only). Routes: code, vision, long_context,
simple, reasoning. See SPEC.md "Classifier Roadmap / Phase 0".
"""
import re
from collections import Counter

ROUTES = ["code", "vision", "long_context", "simple", "reasoning"]

# A few reference phrases per route for centroid word-overlap matching.
REFERENCE_EXAMPLES = {
    "code": [
        "write a python function that sorts a list",
        "fix this bug in my javascript code",
        "how do I implement a binary search tree",
        "debug this stack trace",
    ],
    "vision": [
        "what is in this image",
        "describe this photo",
        "analyze the attached screenshot",
    ],
    "long_context": [
        "summarize this entire document",
        "read through this long report and extract key points",
    ],
    "reasoning": [
        "explain step by step why this proof works",
        "solve this complex multi-step math problem",
        "plan a strategy considering multiple tradeoffs",
    ],
    "simple": [
        "what is the capital of france",
        "what time is it in tokyo",
        "define the word ubiquitous",
    ],
}


_STOPWORDS = {
    "a", "an", "the", "is", "in", "of", "this", "what", "how", "do", "i",
    "to", "and", "that", "it", "for", "on", "my", "with", "why",
}


def _tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _bow(text):
    return Counter(_tokenize(text))


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    mag_a = sum(v * v for v in a.values()) ** 0.5
    mag_b = sum(v * v for v in b.values()) ** 0.5
    if not mag_a or not mag_b:
        return 0.0
    return dot / (mag_a * mag_b)


_REFERENCE_BOWS = {
    route: [_bow(ex) for ex in exs] for route, exs in REFERENCE_EXAMPLES.items()
}


def _embedding_similarity_scores(query: str) -> dict:
    """Word-overlap stand-in for real embedding similarity.

    ponytail: bag-of-words cosine, not a real embedding model — swap in
    sentence-transformers centroid matching once that dependency is justified
    by measured accuracy needs (see SPEC.md Phase 0).
    """
    q = _bow(query)
    scores = {}
    for route, ref_bows in _REFERENCE_BOWS.items():
        scores[route] = max((_cosine(q, ref) for ref in ref_bows), default=0.0)
    return scores


def _heuristic_boosts(query: str) -> dict:
    boosts = {route: 0.0 for route in ROUTES}
    if re.search(r"```|def |function\s*\(|class \w+|import \w+|;\s*$", query, re.M):
        boosts["code"] += 0.3
    if re.search(r"\b(image|photo|picture|screenshot|diagram)\b", query, re.I):
        boosts["vision"] += 0.3
    word_count = len(_tokenize(query))
    if word_count > 200:
        boosts["long_context"] += 0.3
    if re.search(r"[=+\-*/^]|\bsolve\b|\bprove\b", query, re.I):
        boosts["reasoning"] += 0.15
    if word_count < 12 and "?" in query:
        boosts["simple"] += 0.15
    return boosts


def classify(query: str) -> dict:
    """Return {"route": str, "confidence": float, "scores": {route: score}}."""
    sim_scores = _embedding_similarity_scores(query)
    boosts = _heuristic_boosts(query)
    scores = {route: sim_scores[route] + boosts[route] for route in ROUTES}
    # No signal for any route (all scores 0, e.g. topics we have no reference
    # examples for like translation) -> default to "simple" rather than
    # letting max() arbitrarily pick whichever route happens to be first.
    if max(scores.values()) == 0.0:
        route = "simple"
    else:
        route = max(scores, key=scores.get)
    return {"route": route, "confidence": round(scores[route], 3), "scores": scores}


def _demo():
    cases = {
        "write a python function that reverses a string": "code",
        "what is the capital of spain?": "simple",
        "describe what's happening in this photo": "vision",
        "prove that the square root of 2 is irrational": "reasoning",
        "translate this book into German": "simple",  # no matching route -> safe default, not "code"
    }
    for query, expected in cases.items():
        result = classify(query)
        assert result["route"] == expected, (
            f"expected {expected}, got {result['route']} for {query!r}: {result['scores']}"
        )
    print("classifier self-check: all cases passed")


if __name__ == "__main__":
    _demo()
