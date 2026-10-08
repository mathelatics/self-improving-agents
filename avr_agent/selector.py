"""Selector — filter failing candidates, break ties by avg log-probability."""

from __future__ import annotations

from dataclasses import dataclass, field


FALLBACK_MESSAGE = "I cannot solve this."


@dataclass
class SelectionResult:
    answer: object            # extracted math answer or code body
    candidate: object         # winning Candidate (or None)
    verified: bool            # did the winner pass verification?
    n_candidates: int
    n_passed: int
    fallback: bool
    per_candidate: list = field(default_factory=list)


class Selector:
    """PRD rule: drop all candidates failing verification; among survivors pick
    highest average token log-prob; if nobody survives -> fallback message."""

    def select(self, candidates, verdicts) -> SelectionResult:
        passed = [(c, v) for c, v in zip(candidates, verdicts) if v["passed"]]
        per = [v["passed"] for v in verdicts]
        if not passed:
            return SelectionResult(FALLBACK_MESSAGE, None, False,
                                   len(candidates), 0, True, per)
        # rank survivors by confidence proxy (avg logprob), then vote weight
        best_c, best_v = max(passed, key=lambda cv: cv[0].avg_logprob)
        answer = best_v.get("answer") or best_c.answer or best_c.text
        return SelectionResult(answer, best_c, True, len(candidates),
                               len(passed), False, per)
