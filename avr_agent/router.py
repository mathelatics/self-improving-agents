"""Router — cheap difficulty estimation before spending generation compute.

Phase 1 implementation (per PRD): a *single cheap forward pass* whose mean
token log-probability is combined with prompt-length heuristics to produce a
difficulty score in {Low, Medium, High}.

Formalisation: given a short probe roll-out y_1..y_T on the real prompt with
    lp(y) = (1/T) * sum_t log P(y_t | y_<t, x)
and structural features of the prompt x (length, arithmetic-operator count,
multi-hop cue words), we map to difficulty via calibrated thresholds tuned on
the HW1 GSM8K distribution (see results/ notebooks).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from enum import Enum

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
for _p in (os.path.join(ROOT, "hw1"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from llm_client import complete, DEFAULT_MODEL  # noqa: E402


class Difficulty(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Budget per difficulty tier: (N samples, temperature)  — PRD section 3.
ROUTING_POLICY = {
    Difficulty.LOW:    {"n": 1, "temperature": 0.0},
    Difficulty.MEDIUM: {"n": 3, "temperature": 0.5},
    Difficulty.HIGH:   {"n": 8, "temperature": 0.8},
}


@dataclass
class RouteDecision:
    difficulty: Difficulty
    n: int
    temperature: float
    probe_avg_logprob: float
    heuristic_score: float
    reason: str


class Router:
    def __init__(self, client, model: str = DEFAULT_MODEL,
                 lp_easy: float = -0.25, lp_hard: float = -0.9,
                 probe_max_tokens: int = 96):
        """lp_easy / lp_hard calibrate the logprob part of the score;
        defaults chosen so that trivially-formatted answers (~-0.1) are Low and
        rambling/unconfident probes (<-0.9) push toward High."""
        self.client = client
        self.model = model
        self.lp_easy, self.lp_hard = lp_easy, lp_hard
        self.probe_max_tokens = probe_max_tokens

    # ------------------------------------------------------------ heuristics
    @staticmethod
    def _prompt_features(prompt: str) -> float:
        p = prompt.lower()
        score = 0.0
        score += min(len(p.split()) / 120.0, 1.0) * 0.4          # length
        score += min(len(re.findall(r"\d+", p)) / 12.0, 1.0) * 0.3  # entities
        hops = len(re.findall(
            r"(?:after|then|each|more than|less than|total|left|"
            r"combined|difference|twice|half)", p))
        score += min(hops / 6.0, 1.0) * 0.3                       # multi-hop cues
        return round(min(score, 1.0), 3)

    def _logprob_component(self, avg_lp: float) -> float:
        """Map avg token logprob to [0,1] (0 = very confident)."""
        if avg_lp >= self.lp_easy:
            return 0.0
        if avg_lp <= self.lp_hard:
            return 1.0
        return (self.lp_easy - avg_lp) / (self.lp_easy - self.lp_hard)

    # ------------------------------------------------------------------ route
    def route(self, prompt: str, task_type: str = "math") -> RouteDecision:
        h = self._prompt_features(prompt)
        if task_type == "code":
            # Code tasks: never treat as free — verification is the gate.
            base = Difficulty.MEDIUM if h < 0.5 else Difficulty.HIGH
            n, temp = ROUTING_POLICY[base]["n"], ROUTING_POLICY[base]["temperature"]
            return RouteDecision(base, n, temp, float("nan"), h,
                                 f"code task heuristic={h}")

        probe = complete(self.client, prompt, model=self.model,
                         temperature=0.7, max_tokens=self.probe_max_tokens,
                         logprobs=True)
        lp = probe.avg_logprob if probe.token_logprobs else self.lp_hard
        l = self._logprob_component(lp)
        s = 0.6 * l + 0.4 * h                     # blended difficulty score
        if s < 0.33:
            d = Difficulty.LOW
        elif s < 0.66:
            d = Difficulty.MEDIUM
        else:
            d = Difficulty.HIGH
        pol = ROUTING_POLICY[d]
        return RouteDecision(d, pol["n"], pol["temperature"], round(lp, 4), h,
                             f"score={s:.2f} (lp={l:.2f}, heur={h:.2f})")
