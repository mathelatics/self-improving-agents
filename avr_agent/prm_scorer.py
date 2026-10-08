"""HW2 Exercise 1 — Mini Process Reward Model (step-level math verifier).

Instead of only checking the FINAL answer, we split a chain-of-thought roll-out
into steps and score each one. Two interchangeable backends:

  * HeuristicPRM  — free, deterministic, runs anywhere (Colab/local).
      - arithmetic re-computation: parse "a op b = c" equations and check them;
      - number carry-forward: a step may only introduce numbers that were
        either in the problem statement or produced by an earlier step;
      - NaN / negative-count sanity flags.
  * LLMJudgePRM   — a small LLM call per step ("is this step correct? +0/-1").
      Swap-in ready for Phase 2's Skywork-PRM-8B (same interface).

Interface:  prm.score(problem, steps) -> list[StepScore]
            prm.first_bad_index(scores) -> int | None   # first negative step
Used by the AVR agent to trigger backtracking/refinement when a step is flagged.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
for _p in (os.path.join(ROOT, "hw1"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass
class StepScore:
    index: int
    text: str
    score: float          # +1 good, 0 neutral, -1 flagged
    reason: str = ""

    def to_dict(self) -> dict:
        return {"index": self.index, "score": self.score,
                "reason": self.reason, "text": self.text[:160]}


# ---------------------------------------------------------------- splitting --
_STEP_SPLIT_RE = re.compile(r"\n\s*\n+|(?=(?:Step\s*\d+|Therefore|Finally))")

def split_steps(text: str) -> list:
    """Split a CoT roll-out into reasoning steps (blank-line / 'Step N' based)."""
    # strip common final-answer markers so they don't count as reasoning steps
    text = re.sub(r"####.*$", "", text, flags=re.M)
    parts = [p.strip() for p in _STEP_SPLIT_RE.split(text) if p and p.strip()]
    # merge tiny fragments (<3 tokens) into previous step
    merged = []
    for p in parts:
        if merged and len(p.split()) < 3:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


# ------------------------------------------------------------- heuristic PRM --
_EQ_RE = re.compile(
    r"(-?\d[\d,]*(?:\.\d+)?)\s*([+\-*/])\s*"
    r"(-?\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _eval_eq(a, op, b):
    a, b = float(a.replace(",", "")), float(b.replace(",", ""))
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b if b else float("nan")
    return None


class HeuristicPRM:
    """Free, deterministic step scorer. Returns StepScore list."""

    def score(self, problem: str, steps: list) -> list:
        given = set(_NUM_RE.findall(problem.replace(",", "")))
        available = set(given)
        out = []
        for i, s in enumerate(steps):
            sc, reason = 0.0, "neutral"
            eqs = _EQ_RE.findall(s)
            bad_eqs = []
            for a, op, b, res in eqs:
                expected = _eval_eq(a, op, b)
                got = float(res.replace(",", ""))
                if expected is None or abs(expected - got) > max(1e-6, abs(expected) * 1e-6):
                    bad_eqs.append(f"{a}{op}{b}={res} (should be {expected:g})")
            if bad_eqs:
                sc, reason = -1.0, "invalid arithmetic: " + "; ".join(bad_eqs)
            else:
                # results of valid equations count as available *within* this step
                produced = {res.replace(",", "") for a, op, b, res in eqs}
                new_nums = [n for n in _NUM_RE.findall(s.replace(",", ""))
                            if n not in available and n not in produced
                            and not _is_small(n)]
                for p in produced:
                    available.add(p)
                if new_nums:
                    sc, reason = -1.0, ("numbers not carried forward: "
                                        + ", ".join(sorted(set(new_nums))[:4]))
                elif eqs:
                    sc, reason = 1.0, "verified arithmetic"
            # update available set with all numbers seen in this step
            available.update(n.replace(",", "") for n in _NUM_RE.findall(s))
            out.append(StepScore(i, s, sc, reason))
        return out

    def first_bad_index(self, scores: list):
        for s in scores:
            if s.score < 0:
                return s.index
        return None


def _is_small(num: str) -> bool:
    """Allow small integers (2, 3, ...) that are implicit constants (doubling etc.)."""
    try:
        return abs(float(num)) <= 10 and float(num).is_integer()
    except ValueError:
        return False


# ------------------------------------------------------------------ LLM judge --
JUDGE_SYS = ("You are a strict math process-reward model. For the given step, "
             "reply exactly 'CORRECT' or 'INCORRECT' followed by a short reason.")
JUDGE_TMPL = ("Problem:\n{problem}\n\nSteps so far:\n{prefix}\n\n"
              "Candidate next step:\n{step}\n\nIs the candidate step correct?")


class LLMJudgePRM:
    def __init__(self, client=None, model: str = None):
        from llm_client import get_client, DEFAULT_MODEL
        self.client = client or get_client()
        self.model = model or DEFAULT_MODEL

    def score(self, problem: str, steps: list) -> list:
        from llm_client import complete
        out = []
        for i, s in enumerate(steps):
            prefix = "\n".join(steps[:i]) or "(none)"
            c = complete(self.client, JUDGE_TMPL.format(problem=problem,
                                                        prefix=prefix, step=s),
                         model=self.model, temperature=0.0, max_tokens=48,
                         system=JUDGE_SYS, logprobs=False)
            verdict = c.text.strip().upper()
            if verdict.startswith("INCORRECT"):
                out.append(StepScore(i, s, -1.0, c.text.strip()))
            elif verdict.startswith("CORRECT"):
                out.append(StepScore(i, s, 1.0, c.text.strip()))
            else:
                out.append(StepScore(i, s, 0.0, "unparseable: " + verdict[:40]))
        return out

    def first_bad_index(self, scores: list):
        return HeuristicPRM.first_bad_index(self, scores)


# convenience factory used by the agent
def get_prm(kind: str = "heuristic", **kw):
    return LLMJudgePRM(**kw) if kind == "llm" else HeuristicPRM()


if __name__ == "__main__":
    import json
    problem = ("Janet has 16 hours a day. She sleeps for 1/4 of the day and "
               "works 8 hours. How many hours does she have free?")
    good = ["She sleeps 16 / 4 = 4 hours.",
            "Time left after sleeping: 16 - 4 = 12 hours.",
            "Free time: 12 - 8 = 4 hours.",
            "The answer is 4"]
    bad = ["She sleeps 16 / 4 = 5 hours.",           # wrong arithmetic
           "Time left: 16 - 5 = 11 hours.",
           "She then buys 37 widgets for 90 dollars.",  # invented numbers
           "The answer is 90"]
    prm = HeuristicPRM()
    for name, steps in [("GOOD", good), ("BAD", bad)]:
        scores = prm.score(problem, steps)
        print(name, "first_bad:", prm.first_bad_index(scores))
        print(json.dumps([s.to_dict() for s in scores], indent=1))
