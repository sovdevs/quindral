"""Classifier accuracy eval: labeled test set -> per-route accuracy + confusion
matrix. Answers "are we routing correctly?" before wiring real model calls.

Phrasing here is deliberately different from classifier.py's REFERENCE_EXAMPLES
(the centroids classify() matches against) — testing against near-identical
phrasing would just confirm the lookup table works, not that it generalizes.

Run: .venv/bin/python3 eval_classifier.py
"""
from classifier import classify

# (query, expected_route)
ROUTE_DATASET = [
    # code
    ("how do I center a div in css", "code"),
    ("my react component keeps re-rendering, help me debug it", "code"),
    ("write a bash script that backs up a folder nightly", "code"),
    ("what's wrong with this SQL query, it returns duplicate rows", "code"),
    ("refactor this function to use async/await", "code"),
    ("implement quicksort in rust", "code"),
    ("why is my docker container exiting immediately", "code"),
    ("add type hints to this python module", "code"),
    ("write a regex that matches email addresses", "code"),
    ("convert this class-based react component to hooks", "code"),
    # vision
    ("what breed is this dog in the picture", "vision"),
    ("can you read the text in this scanned document", "vision"),
    ("is there anything unusual in this x-ray", "vision"),
    ("identify the landmark shown in this picture", "vision"),
    ("what's wrong with the layout in this UI mockup", "vision"),
    ("count how many people are in this photo", "vision"),
    ("transcribe the handwriting in this image", "vision"),
    # long_context
    ("summarize the key arguments across these 40 pages of meeting notes", "long_context"),
    ("go through this entire contract and flag anything unusual", "long_context"),
    ("extract every action item from this long email thread", "long_context"),
    ("compare these three lengthy research papers and note contradictions", "long_context"),
    ("read through this whole codebase's changelog and summarize breaking changes", "long_context"),
    # reasoning
    ("if a train leaves at 3pm going 60mph and another at 4pm going 90mph, when do they meet", "reasoning"),
    ("walk through the logic of why this proof by induction holds", "reasoning"),
    ("what's the optimal strategy for this game theory scenario", "reasoning"),
    ("derive the formula for compound interest step by step", "reasoning"),
    ("is this argument logically valid or does it have a flaw", "reasoning"),
    ("work out the probability of drawing two aces in a row", "reasoning"),
    ("plan the most efficient route visiting these 6 cities", "reasoning"),
    # simple
    ("how tall is mount everest", "simple"),
    ("who wrote pride and prejudice", "simple"),
    ("what's the boiling point of water in fahrenheit", "simple"),
    ("define serendipity", "simple"),
    ("how many continents are there", "simple"),
    ("what year did world war 2 end", "simple"),
    ("what's the chemical symbol for gold", "simple"),
]

# (query, expected_needs_current_info)
CURRENT_INFO_DATASET = [
    ("what's the score of tonight's game", True),
    ("who is currently leading the election polls", True),
    ("what's the exchange rate for USD to EUR today", True),
    ("give me the latest news on the merger", True),
    ("what happened in the news this week", True),
    ("who won the game last night", True),
    ("what's bitcoin's price right now", True),
    ("how tall is mount everest", False),
    ("define serendipity", False),
    ("write a python function that sorts a list", False),
    ("explain how photosynthesis works", False),
    ("derive the formula for compound interest", False),
]


def run_eval():
    confusion = {}  # expected -> {predicted: count}
    correct = 0
    misses = []
    for query, expected in ROUTE_DATASET:
        predicted = classify(query)["route"]
        confusion.setdefault(expected, {}).setdefault(predicted, 0)
        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1
        else:
            misses.append((query, expected, predicted))

    total = len(ROUTE_DATASET)
    print(f"Route classification: {correct}/{total} = {correct/total:.1%} overall\n")

    routes = sorted(confusion)
    print("Per-route accuracy:")
    for route in routes:
        row = confusion[route]
        route_total = sum(row.values())
        route_correct = row.get(route, 0)
        print(f"  {route:14s} {route_correct}/{route_total} = {route_correct/route_total:.1%}")

    print("\nConfusion matrix (rows=expected, cols=predicted):")
    header = "".join(f"{r[:8]:>10s}" for r in routes)
    print(f"{'':14s}{header}")
    for expected in routes:
        row = "".join(f"{confusion[expected].get(r, 0):>10d}" for r in routes)
        print(f"{expected:14s}{row}")

    if misses:
        print(f"\nMisclassified ({len(misses)}):")
        for query, expected, predicted in misses:
            print(f"  {query!r} -> expected {expected}, got {predicted}")

    ci_correct = 0
    ci_misses = []
    for query, expected in CURRENT_INFO_DATASET:
        predicted = classify(query)["needs_current_info"]
        if predicted == expected:
            ci_correct += 1
        else:
            ci_misses.append((query, expected, predicted))
    ci_total = len(CURRENT_INFO_DATASET)
    print(f"\nneeds_current_info: {ci_correct}/{ci_total} = {ci_correct/ci_total:.1%}")
    if ci_misses:
        print("Misclassified:")
        for query, expected, predicted in ci_misses:
            print(f"  {query!r} -> expected {expected}, got {predicted}")

    return correct / total, ci_correct / ci_total


if __name__ == "__main__":
    route_acc, ci_acc = run_eval()
    # SPEC.md's own Phase 0 expectation is ~70-80% — this is the check that
    # keeps that claim honest rather than assumed.
    assert route_acc >= 0.70, f"route accuracy {route_acc:.1%} below SPEC's stated Phase 0 floor of 70%"
