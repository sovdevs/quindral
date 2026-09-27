"""Graders: return (ok, score in [0,1], detail)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Callable

Judge = Callable[[str], str]  # prompt -> text (used by llm_judge / vision_judge)


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^[\"'`*\s]+|[\"'`*.\s!]+$", "", s)
    return s


def _last_number(s: str) -> float | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def _extract_json(s: str) -> Any:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        raise ValueError("no JSON object")
    return json.loads(m.group(0))


def _extract_code(s: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", s, re.S)
    return max(blocks, key=len) if blocks else s


def wer(ref: str, hyp: str) -> float:
    tok = lambda x: re.findall(r"[a-z0-9']+", x.lower())  # noqa: E731
    r, h = tok(ref), tok(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def grade(spec: dict[str, Any], output: str, *, reference: str | None = None,
          judge: Judge | None = None, allow_code_exec: bool = False,
          image_b64: str | None = None, vision_judge: Callable[[str, str], str] | None = None
          ) -> tuple[bool | None, float | None, str]:
    t = spec["type"]
    exp = spec.get("expect")
    if t == "exact":
        a = _norm(output) if spec.get("normalize") else output.strip()
        e = _norm(str(exp)) if spec.get("normalize") else str(exp)
        ok = a == e or (spec.get("normalize") and a.split()[-1:] == [e])
        return bool(ok), float(bool(ok)), f"got={output.strip()[:80]!r}"
    if t == "contains":
        ok = str(exp).lower() in output.lower()
        return ok, float(ok), ""
    if t == "regex":
        ok = re.search(str(exp), output) is not None
        return ok, float(ok), ""
    if t == "numeric":
        v = _last_number(output)
        ok = v is not None and abs(v - float(exp)) <= float(spec.get("tol", 1e-9))
        return ok, float(ok), f"got={v}"
    if t == "json_fields":
        try:
            obj = _extract_json(output)
        except Exception as e:  # noqa: BLE001
            return False, 0.0, f"json: {e}"
        hits = 0
        for k, v in exp.items():
            got = obj.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    hits += abs(float(str(got).replace(",", "").replace("$", "")) - v) < 1e-6
                except (TypeError, ValueError):
                    pass
            else:
                hits += str(got).strip() == str(v)
        score = hits / len(exp)
        return score == 1.0, score, f"fields {hits}/{len(exp)}"
    if t == "all_of":
        res = [grade(c, output, reference=reference, judge=judge) for c in spec["checks"]]
        oks = [r[0] for r in res]
        return all(oks), sum(map(bool, oks)) / len(oks), "; ".join(r[2] for r in res if r[2])
    if t == "wer":
        w = wer(reference or "", output)
        return w <= float(spec.get("max_wer", 0.1)), max(0.0, 1 - w), f"wer={w:.3f}"
    if t == "code_exec":
        if not allow_code_exec:
            return None, None, "code_exec disabled (set allow_code_exec)"
        return _run_code(_extract_code(output), spec["entry"], spec["tests"])
    if t == "llm_judge":
        if judge is None:
            return None, None, "no judge configured"
        verdict = judge(textwrap.dedent(f"""\
            You are a strict grader. Rubric:
            {spec['rubric']}
            ---- RESPONSE ----
            {output[:20000]}
            ---- END ----
            Reply with PASS or FAIL on the first line, then one sentence of reasons."""))
        ok = verdict.strip().upper().startswith("PASS")
        return ok, float(ok), verdict.strip()[:200]
    if t == "vision_judge":
        if vision_judge is None or image_b64 is None:
            return None, None, "no vision judge / image"
        items = spec["checklist"]
        q = ("For each numbered item, answer yes or no based only on the image, one per line as '<n>: yes|no'.\n"
             + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(items)))
        ans = vision_judge(q, image_b64).lower()
        yes = sum(1 for i in range(len(items)) if re.search(rf"^\s*{i + 1}\s*[:.)-]\s*yes", ans, re.M))
        return yes == len(items), yes / len(items), f"{yes}/{len(items)} checklist"
    raise ValueError(f"unknown grader {t}")


def _run_code(code: str, entry: str, tests: list) -> tuple[bool, float, str]:
    harness = code + "\n\nimport json,sys\n_t=json.loads(sys.stdin.read())\n_r=[]\n" \
        f"for a,e in _t:\n    try:\n        _r.append({entry}(*a)==e)\n    except Exception:\n        _r.append(False)\n" \
        "print(json.dumps(_r))\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(harness)
    try:
        p = subprocess.run([sys.executable, "-I", f.name], input=json.dumps(tests), capture_output=True,
                           text=True, timeout=10)
        res = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else []
    except Exception as e:  # noqa: BLE001
        return False, 0.0, f"exec error: {e}"
    passed = sum(res)
    return passed == len(tests), passed / len(tests), f"tests {passed}/{len(tests)}"
