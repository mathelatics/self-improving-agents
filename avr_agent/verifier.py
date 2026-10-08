"""Verifier — deterministic ground-truth checks reused from HW1.

  * MathVerifier: regex answer extraction + canonical numeric equality.
    (In eval mode we compare against gold; in agent mode with no gold we use
     self-consistency across candidates as a soft verifier.)
  * CodeVerifierAdapter: wraps hw1.CodeVerifier (subprocess sandbox) and runs
    hidden unit tests against each candidate completion.
"""

from __future__ import annotations

import os
import sys

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
for _p in (os.path.join(ROOT, "hw1"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from grading import extract_answer, answers_equal          # noqa: E402
from hw1_ex2 import CodeVerifier                          # noqa: E402


class Verifier:
    def verify(self, candidate, task=None) -> dict:
        raise NotImplementedError


class MathVerifier(Verifier):
    """Deterministic math verifier.

    With gold available (evals): passed = extracted answer == gold.
    Without gold (agent runtime): majority-vote agreement fraction is used as
    the verification signal (>= ``agree_min`` of candidates).
    """

    def __init__(self, agree_min: float = 0.5):
        self.agree_min = agree_min

    def verify(self, candidate, task=None) -> dict:
        ans = extract_answer(candidate.text)
        candidate.answer = ans
        if task is not None and task.get("gold") is not None:
            ok = answers_equal(ans, task["gold"])
            return {"passed": bool(ok), "answer": ans,
                    "error": None if ok else f"answer {ans} != gold {task['gold']}"}
        return {"passed": ans is not None, "answer": ans,
                "error": None if ans else "no answer extracted"}

    def vote_verify(self, candidates) -> dict:
        """No-gold mode: an answer is 'verified' if enough candidates agree."""
        from collections import Counter
        from grading import _norm
        vals = [extract_answer(c.text) for c in candidates]
        for c, v in zip(candidates, vals):
            c.answer = v
        valid = [v for v in vals if v is not None]
        if not valid:
            return {"winner": None, "per_candidate": [False] * len(candidates)}
        winner, votes = Counter(valid).most_common(1)[0]
        frac = votes / len(candidates)
        per = [answers_equal(v, winner) and frac >= self.agree_min
               for v in vals]
        return {"winner": winner, "per_candidate": per, "votes": votes}


class CodeVerifierAdapter(Verifier):
    """HumanEval-style verification inside the HW1 subprocess sandbox."""

    def __init__(self, timeout_seconds: int = 6):
        self.sandbox = CodeVerifier(timeout_seconds=timeout_seconds)

    def verify(self, candidate, task=None) -> dict:
        assert task is not None and task.get("prompt"), "code task required"
        body = candidate.meta.get("code")
        if body is None:
            try:
                from .generator import Generator
            except ImportError:
                from generator import Generator
            body = Generator.strip_fences(candidate.text)
            candidate.meta["code"] = body
        script = (task["prompt"].rstrip() + "\n" + body + "\n\n" +
                  task["test"] + f"\ncheck({task['entry_point']})\n")
        res = self.sandbox.verify("", script)
        candidate.answer = body[:80]
        return {"passed": res["passed"], "answer": None, "error": res["error"]}
