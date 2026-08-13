"""Phase 0 classifier: heuristics + word-overlap similarity to route a query.

No training data, no ML deps (stdlib only). Routes: code, vision, long_context,
simple, reasoning. See SPEC.md "Classifier Roadmap / Phase 0".
"""
import re
from collections import Counter

ROUTES = ["code", "vision", "long_context", "simple", "reasoning"]

# Reference phrases per route for centroid word-overlap matching. Deliberately
# varied phrasing/vocabulary per route (not just near-duplicates of each
# other) since bag-of-words similarity only catches queries sharing literal
# words with these — see eval_classifier.py for measured accuracy impact.
REFERENCE_EXAMPLES = {
    "code": [
        "write a python function that sorts a list",
        "fix this bug in my javascript code",
        "how do I implement a binary search tree",
        "debug this stack trace",
        "why does my css layout keep breaking on mobile",
        "this sql query is returning duplicate rows, what's wrong",
        "my docker container keeps exiting immediately, help",
        "convert this react class component to use hooks",
        "write a shell script to automate this deployment",
        "what's the correct syntax for a git rebase",
    ],
    "vision": [
        "what is in this image",
        "describe this photo",
        "analyze the attached screenshot",
        "can you read the text in this scanned document",
        "is there anything unusual in this x-ray",
        "what's wrong with the layout in this UI mockup",
        "identify the objects visible in this picture",
        "transcribe the handwriting shown here",
    ],
    "long_context": [
        "summarize this entire document",
        "read through this long report and extract key points",
        "go through this whole contract and flag anything unusual",
        "compare these lengthy research papers for contradictions",
        "extract every action item from this long email thread",
    ],
    "reasoning": [
        "explain step by step why this proof works",
        "solve this complex multi-step math problem",
        "plan a strategy considering multiple tradeoffs",
        "work out a word problem involving relative speed and distance",
        "determine whether this argument is logically valid",
        "calculate the probability of a specific card-drawing outcome",
        "figure out the most efficient path between several locations",
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
    "are", "there", "was", "were", "be", "been", "can", "does", "did",
}


def _tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    # drop single-char tokens too — mostly contraction debris ("what's" -> "s",
    # "don't" -> "t") that carries no signal and causes spurious collisions
    # between any two queries that happen to both use a contraction
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


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
    if re.search(
        r"```|def |function\s*\(|class \w+|import \w+|;\s*$|"
        r"\b(css|sql|docker|kubernetes|regex|npm|git|api|react|hooks|"
        r"container|component|query|syntax|compile|stack trace|repo)\b",
        query, re.I | re.M,
    ):
        boosts["code"] += 0.3
    if re.search(r"\b(image|photo|picture|screenshot|diagram|x-ray|scan(ned)?|mockup|handwriting)\b", query, re.I):
        boosts["vision"] += 0.3
    word_count = len(_tokenize(query))
    if word_count > 200:
        boosts["long_context"] += 0.3
    if re.search(r"[=+\-*/^]|\b(solve|prove|probability|logically valid|logical fallacy|derive)\b", query, re.I):
        boosts["reasoning"] += 0.15
    if word_count < 12 and "?" in query:
        boosts["simple"] += 0.15
    return boosts


# Signals that a query needs information newer than any model's training
# cutoff (current events, prices, scores, "who is currently X") — the
# retrieval-necessity / knowledge-boundary problem. This is orthogonal to
# the task-type route above: a query can be "simple" AND need live search
# ("what's the weather right now") or "simple" and not ("capital of Spain").
_CURRENT_INFO_PATTERN = re.compile(
    r"\b(today|tonight|currently|current|right now|as of|this week|this month|this year|"
    r"latest|breaking|recent(ly)?|up[- ]to[- ]date|"
    r"who (is|are) the (current|latest)|who won|who is winning|"
    r"stock price|exchange rate|weather (in|today|right now)|"
    r"score of|live score|"
    r"20[2-9]\d)\b",
    re.I,
)


def needs_current_info(query: str) -> bool:
    """Heuristic: does this query likely need retrieval beyond training-cutoff knowledge?"""
    return bool(_CURRENT_INFO_PATTERN.search(query))


def classify(query: str) -> dict:
    """Return {"route", "confidence", "scores", "needs_current_info"}."""
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
    return {
        "route": route,
        "confidence": round(scores[route], 3),
        "scores": scores,
        "needs_current_info": needs_current_info(query),
    }


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

    current_info_cases = {
        "what's the weather in Tokyo right now": True,
        "who won the game last night": True,
        "what's the latest news on the election": True,
        "what is the capital of Spain": False,
        "write a python function that reverses a string": False,
        "explain how photosynthesis works": False,
    }
    for query, expected in current_info_cases.items():
        result = classify(query)
        assert result["needs_current_info"] == expected, (
            f"expected needs_current_info={expected} for {query!r}, got {result['needs_current_info']}"
        )

    print("classifier self-check: all cases passed")


if __name__ == "__main__":
    _demo()
