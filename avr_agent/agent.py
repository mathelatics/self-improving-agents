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
from .prm_scorer import HeuristicPRM, split_steps  # noqa: E402
from .state_manager import Trajectory             # noqa: E402


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
    # ---- Phase-1 v2.0 additions -------------------------------------
    prm_first_bad_step: int | None = None     # flagged step index (math)
    prm_scores: list = field(default_factory=list)
    refine_loops: int = 0                     # ReAct loops used (code)
    trajectory: dict | None = None            # serialised state_manager log


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
        self.prm = HeuristicPRM()   # swap for get_prm("llm") / Skywork-PRM in P2

    def solve(self, prompt: str, task_type: str = "math",
              task: dict | None = None) -> AgentResult:
        task = dict(task or {})
        task.setdefault("type", task_type)

        decision = self.router.route(prompt, task_type=task_type)
        cands = self.generator.generate(prompt, decision.n,
                                        decision.temperature,
                                        task_type=task_type)
        traj = Trajectory(task_id=task.get("id", "adhoc"),
                          task_type=task_type, prompt=prompt)
        traj.add("route", difficulty=decision.difficulty.value, n=decision.n,
                 temperature=decision.temperature,
                 probe_avg_logprob=decision.probe_avg_logprob)
        verifier = (self.code_verifier if task_type == "code"
                    else self.math_verifier)
        # eval mode for math uses gold; agent mode falls back to extraction-only
        verdicts = [verifier.verify(c, task if task_type == "code" or
                                    task.get("gold") is not None else None)
                    for c in cands]
        for c, v in zip(cands, verdicts):
            traj.add("verify", passed=v["passed"], tokens=c.n_tokens,
                     error=(v.get("error") or "")[:500])

        sel = self.selector.select(cands, verdicts)

        # ---- v2.0 step-level verification (Mini-PRM) -------------------
        # Dense check applied to the SELECTED trajectory only (cost-aware):
        # a flagged step means we do NOT trust the final answer even if the
        # verifier passed it -> triggers repair instead of acceptance.
        first_bad = None
        prm_scores_out = []
        prm_rejected = False
        if task_type == "math" and sel.candidate is not None:
            steps = split_steps(sel.candidate.text)
            scores = self.prm.score(task.get("question", prompt), steps)
            first_bad = self.prm.first_bad_index(scores)
            prm_scores_out = [s.to_dict() for s in scores]
            traj.add("prm", first_bad=first_bad, scores=prm_scores_out[:8])
            if first_bad is not None and sel.verified:
                # gold-verifier agreed but a reasoning step is invalid ->
                # treat as unverified; Selector fallback will be overridden
                # below by majority-vote among PRM-clean candidates if any.
                clean_idx = [i for i, c in enumerate(cands)
                             if self.prm.first_bad_index(
                                 self.prm.score(task.get("question", prompt),
                                                split_steps(c.text))) is None]
                traj.add("prm_reject", step=first_bad, clean_candidates=clean_idx)
                if clean_idx:
                    best_clean = max((cands[i] for i in clean_idx),
                                     key=lambda c: c.avg_logprob)
                    v = self.math_verifier.verify(
                        best_clean, task if task.get("gold") else None)
                    sel = self.selector.select([best_clean], [v])
                prm_rejected = True

        # ---- v2.0 ReAct self-correction on failure (code) --------------
        refine_loops = 0
        if task_type == "code" and not any(v["passed"] for v in verdicts):
            from .react_executor import ReActCoder
            coder = ReActCoder(client=self.client, model=self.model,
                               max_loops=3, sandbox_timeout=6)
            rr = coder.solve({**task, "task_id": task.get("id", "adhoc")})
            refine_loops = rr.loops_used
            traj.steps.extend(rr.trajectory.steps)   # merge sub-trajectory
            if rr.passed:
                fixed = type(cands[0])(text=rr.final_code, avg_logprob=0.0,
                                       n_tokens=rr.tokens,
                                       meta={"code": rr.final_code})
                cands.append(fixed)
                verdicts.append({"passed": True, "answer": None, "error": None})
                sel = self.selector.select(cands, verdicts)
                traj.add("refine", note="ReAct repair succeeded",
                         loops=refine_loops, tokens=rr.tokens)

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
                               (96 if task_type == "math" else 0) +
                               (refine_loops and
                                sum(int(s_.payload.get("tokens", 0))
                                    for s_ in traj.steps
                                    if s_.kind == "reflect") or 0),
            per_candidate_passed=sel.per_candidate,
            prm_first_bad_step=first_bad, prm_scores=prm_scores_out,
            refine_loops=refine_loops, trajectory=traj.to_dict())
