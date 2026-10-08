"""AVRAgent — Phase 1 orchestration loop: Router -> Generator -> Verifier -> Selector."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
for _p in (os.path.join(ROOT, "hw1"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from llm_client import get_client, DEFAULT_MODEL  # noqa: E402
from .router import Router                        # noqa: E402
from .generator import Generator                  # noqa: E402
from .verifier import MathVerifier, CodeVerifierAdapter  # noqa: E402
from .selector import Selector                    # noqa: E402


@dataclass
class AgentResult:
    task_id: str
    task_type: str
    difficulty: str
    n_generations: int
    temperature: float
    probe_avg_logprob: float
    verified: bool
    fallback: bool
    answer: object
    correct: bool | None          # only when gold available (eval mode)
    tokens_generated: int
    per_candidate_passed: list = field(default_factory=list)


class AVRAgent:
    def __init__(self, client=None, model: str = DEFAULT_MODEL,
                 max_tokens: int = 512):
        self.client = client or get_client()
        self.model = model
        self.router = Router(self.client, model=model)
        self.generator = Generator(self.client, model=model,
                                   max_tokens=max_tokens)
        self.math_verifier = MathVerifier()
        self.code_verifier = CodeVerifierAdapter()
        self.selector = Selector()

    def solve(self, prompt: str, task_type: str = "math",
              task: dict | None = None) -> AgentResult:
        task = dict(task or {})
        task.setdefault("type", task_type)

        decision = self.router.route(prompt, task_type=task_type)
        cands = self.generator.generate(prompt, decision.n,
                                        decision.temperature,
                                        task_type=task_type)
        verifier = (self.code_verifier if task_type == "code"
                    else self.math_verifier)
        # eval mode for math uses gold; agent mode falls back to extraction-only
        verdicts = [verifier.verify(c, task if task_type == "code" or
                                    task.get("gold") is not None else None)
                    for c in cands]
        sel = self.selector.select(cands, verdicts)

        correct = None
        if task.get("gold") is not None and not sel.fallback:
            from grading import answers_equal
            correct = bool(answers_equal(sel.answer, task["gold"]))
        elif task_type == "code":
            correct = sel.verified  # hidden tests ARE the ground truth

        return AgentResult(
            task_id=task.get("id", "adhoc"), task_type=task_type,
            difficulty=decision.difficulty.value,
            n_generations=len(cands), temperature=decision.temperature,
            probe_avg_logprob=decision.probe_avg_logprob,
            verified=sel.verified, fallback=sel.fallback,
            answer=sel.answer, correct=correct,
            tokens_generated=sum(c.n_tokens for c in cands) +
                               (96 if task_type == "math" else 0),
            per_candidate_passed=sel.per_candidate)
