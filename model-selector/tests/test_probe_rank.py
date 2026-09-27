from types import SimpleNamespace as NS

from model_selector.probes import ProbeRunner, load_suite, load_suites
from model_selector.probes.graders import grade, wer
from model_selector.ranker import TaskSpec, rank, trial

ANSWERS = {"extract_invoice_json": '{"invoice_number": "INV-20931", "total": 1284.50}',
           "sentiment_one_word": "Negative.", "date_to_iso": "2024-03-03", "unit_price_word_problem": "18",
           "ordering_logic": "Bob", "code_trace": "35", "train_arrival": "17:10",
           "inclusion_exclusion": "401", "crt_smallest": "301", "weekday": "Thursday", "needle_20k": "7-4-1-9-Q"}


class FakeResponses:
    """luna gets hard items wrong unless effort=high (and then burns reasoning tokens); sol always right."""
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        prompt = kw["input"] if isinstance(kw["input"], str) else "judge"
        pid = next((k for k, v in PROMPT_TEXT.items() if v.strip()[:60] in prompt), None)
        eff = (kw.get("reasoning") or {}).get("effort")
        ans = ANSWERS.get(pid, "PASS ok")
        reason = 0
        if kw["model"] == "gpt-6-luna" and pid in ("inclusion_exclusion", "crt_smallest", "weekday"):
            ans, reason = (ans, 3000) if eff == "high" else ("42", 50)
        usage = NS(input_tokens=len(prompt) // 4, input_tokens_details=NS(cached_tokens=0),
                   output_tokens=20 + reason, output_tokens_details=NS(reasoning_tokens=reason))
        return NS(output_text=ans, usage=usage, status="completed", output=[])


SUITE = load_suite("reasoning")
from model_selector.probes.runner import expand_prompt  # noqa: E402
PROMPT_TEXT = {p["id"]: expand_prompt(p) for p in SUITE["prompts"]}


def test_all_suites_load():
    s = load_suites()
    assert {"reasoning", "coding", "transcription", "tts", "image_generation", "deep_research", "realtime"} <= set(s)


def test_graders():
    assert grade({"type": "exact", "expect": "bob", "normalize": True}, "The answer is Bob.")[0]
    assert grade({"type": "numeric", "expect": 18}, "$18")[0]
    assert wer("the cat sat", "the cat sat") == 0 and abs(wer("a b c d", "a x c d") - 0.25) < 1e-9
    code = "```python\ndef slugify(s):\n    import re\n    return re.sub(r'[^a-z0-9]+','-',s.lower()).strip('-')\n```"
    spec = load_suite("coding")["prompts"][0]["grader"]
    assert grade(spec, code, allow_code_exec=True)[0] is True
    assert grade(spec, code)[0] is None  # exec disabled by default


def test_probe_and_rank(store, settings):
    from model_selector.probes.adapters import OpenAIAdapters
    fake = NS(responses=FakeResponses())
    runner = ProbeRunner(store, settings, adapters=OpenAIAdapters(client=fake), log=lambda *_: None)
    outs = runner.run(SUITE, ["gpt-6-luna", "gpt-6-sol"], efforts=[None, "high"])
    assert len(outs) == 2 * 2 * len(SUITE["prompts"])
    assert all(o.cost_usd is not None for o in outs)

    r = rank(store, TaskSpec("count integers satisfying divisibility constraints", difficulty="hard",
                             category="reasoning", in_tokens=200))
    by = {(c.model_id, c.effort): c for c in r.candidates}
    assert by[("gpt-6-luna", None)].meets_bar is False
    assert by[("gpt-6-luna", "high")].meets_bar is True
    # observed reasoning tokens flow into the estimate
    assert by[("gpt-6-luna", "high")].est_cost_per_call > by[("gpt-6-luna", None)].est_cost_per_call
    costs = [c.est_cost_per_call for c in r.candidates if c.est_cost_per_call is not None]
    assert costs == sorted(costs)
    # cheap-per-token model at high effort burns reasoning tokens -> pricier than sol at default
    assert by[("gpt-6-luna", "high")].est_cost_per_call > by[("gpt-6-sol", None)].est_cost_per_call
    assert r.recommendation.model_id == "gpt-6-sol" and r.recommendation.effort is None
    assert "gpt-6-astra" in r.unprobed

    easy = rank(store, TaskSpec("extract invoice fields to json", category="reasoning", difficulty="easy"))
    assert easy.recommendation.model_id == "gpt-6-luna"


def test_trial_cascade(store, settings):
    from model_selector.probes.adapters import OpenAIAdapters
    fake = NS(responses=FakeResponses())
    runner = ProbeRunner(store, settings, adapters=OpenAIAdapters(client=fake), log=lambda *_: None)
    task = TaskSpec("summarize support tickets", sample_input="Ticket: printer on fire", category="reasoning")
    r = rank(store, task)
    outs = trial(store, runner, r, rubric="PASS if it summarizes")
    assert outs and outs[-1].ok and len({o.model_id for o in outs}) == 1  # stopped at cheapest pass
    r2 = rank(store, task)
    assert r2.recommendation is not None


def test_classify_heuristic():
    from model_selector.ranker import classify
    assert classify(TaskSpec("transcribe 200 hours of podcast audio files"))[0] == "transcription"
    assert classify(TaskSpec("build a phone call voice agent for bookings"))[0] == "realtime_voice"
    assert classify(TaskSpec("generate a logo image for my bakery"))[0] == "image_generation"
