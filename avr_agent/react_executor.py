"""HW2 Exercise 2 — ReAct self-correction loop for code generation.

ReActCoder implements: Generate -> Execute (sandboxed verifier) ->
Capture stderr -> Reflect (forced Root-Cause-Analysis) -> Regenerate,
with a hard cap of ``max_loops`` refinements (spec: 3).

Pitfall mitigation (per spec): before every refinement the model MUST emit a
"Root Cause Analysis:" block; if it skips it we re-ask once with a stricter
instruction, and the RCA is stored in the trajectory so we can measure whether
refinements actually change strategy instead of tweaking symptoms.

Multi-turn state lives in ``state_manager.Trajectory`` (all thoughts/actions/
observations), so context is never lost between refinement turns.
"""

from __future__ import annotations

import os
import re
import sys

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
for _p in (os.path.join(ROOT, "hw1"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_client import get_client, DEFAULT_MODEL            # noqa: E402
from .generator import Generator, Candidate                 # noqa: E402
from .verifier import CodeVerifierAdapter                   # noqa: E402
from .state_manager import Trajectory                       # noqa: E402

REFINE_TMPL = (
    "The code failed with this error. Analyze the error and rewrite the code "
    "to fix it.\n\n"
    "First output a one-paragraph 'Root Cause Analysis:' explaining WHY the "
    "previous implementation was wrong at its core (not just the symptom). "
    "Then output ONLY the corrected function code (no fences, no prose).\n\n"
    "Task header:\n```python\n{header}\n```\n\n"
    "Your previous code:\n```python\n{code}\n```\n\n"
    "Error / traceback:\n```\n{error}\n```"
)
RCA_RE = re.compile(r"root cause analysis\s*:", re.I)


class ReActResult:
    def __init__(self, task_id):
        self.task_id = task_id
        self.passed = False
        self.loops_used = 0          # refinement iterations actually run
        self.final_code = ""
        self.rcas = []               # root-cause analyses, one per refinement
        self.tokens = 0
        self.llm_calls = 0
        self.traceback_seen = None
        self.trajectory: Trajectory | None = None

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "passed": self.passed,
                "loops_used": self.loops_used, "rcas": self.rcas,
                "tokens": self.tokens, "llm_calls": self.llm_calls,
                "first_error": (self.traceback_seen or "")[:500]}


class ReActCoder:
    def __init__(self, client=None, model: str = DEFAULT_MODEL,
                 max_loops: int = 3, temperature: float = 0.3,
                 max_tokens: int = 512, sandbox_timeout: int = 6):
        self.client = client or get_client()
        self.model = model
        self.max_loops = max_loops
        self.temperature = temperature
        self.generator = Generator(self.client, model=model,
                                   max_tokens=max_tokens)
        self.verifier = CodeVerifierAdapter(timeout_seconds=sandbox_timeout)

    # ------------------------------------------------------------------ core --
    def solve(self, task: dict) -> ReActResult:
        """task = {id, prompt(header), test, entry_point} (HumanEval schema)."""
        tid = task.get("task_id") or task.get("id") or "adhoc"
        res = ReActResult(tid)
        traj = Trajectory(task_id=res.task_id, task_type="code",
                          prompt=task["prompt"])
        res.trajectory = traj

        # ---- Thought/Action 1: generate initial candidate ------------------
        cands = self.generator.generate(task["prompt"], n=1,
                                        temperature=self.temperature,
                                        task_type="code")
        cand = cands[0]
        code = Generator.strip_fences(cand.text)
        cand.meta["code"] = code
        res.tokens += cand.n_tokens
        res.llm_calls += 1
        traj.add("generate", tokens=cand.n_tokens, best_text=code,
                 request_prompt=task["prompt"])

        # ---- Observation 1: execute hidden tests in the sandbox ------------
        verdict = self.verifier.verify(cand, task)
        traj.add("verify", passed=verdict["passed"],
                 error=(verdict["error"] or "")[:2000])
        res.traceback_seen = verdict["error"]

        loop = 0
        while not verdict["passed"] and loop < self.max_loops:
            loop += 1
            res.loops_used = loop

            # ---- Reflection: forced Root Cause Analysis + rewrite ----------
            refine_prompt = REFINE_TMPL.format(header=task["prompt"],
                                               code=code,
                                               error=(verdict["error"] or
                                                      "unknown failure")[:3000])
            rca, new_code = self._reflect(refine_prompt, traj)
            res.rcas.append(rca)
            res.llm_calls += 1

            # guard against the classic pitfall: identical rewrite => stop early
            if new_code.strip() == code.strip():
                traj.add("refine", note="model returned identical code; aborting loop")
                break
            code = new_code
            res.tokens += getattr(self, "_last_tokens", 0)

            cand = Candidate(text=code, avg_logprob=getattr(cand, "avg_logprob", 0.0),
                             n_tokens=getattr(self, "_last_tokens", 0))
            cand.meta["code"] = code
            verdict = self.verifier.verify(cand, task)
            traj.add("verify", passed=verdict["passed"],
                     error=(verdict["error"] or "")[:2000], loop=loop)
            if not verdict["passed"]:
                res.last_error = verdict["error"]

        res.passed = bool(verdict["passed"])
        res.final_code = code
        traj.add("accept" if res.passed else "fallback",
                 loops=loop, tokens=res.tokens)
        return res

    # ------------------------------------------------------------- reflection --
    def _reflect(self, refine_prompt: str, traj: Trajectory):
        """Return (root_cause_analysis, code). Re-asks once if RCA missing."""
        from llm_client import complete
        c = complete(self.client, refine_prompt, model=self.model,
                     temperature=max(self.temperature, 0.5),
                     max_tokens=self.generator.max_tokens, logprobs=False)
        self._last_tokens = c.n_tokens
        text = c.text
        m = RCA_RE.search(text)
        if not m:                                     # enforce the RCA gate
            traj.add("reflect", note="missing RCA, re-asking", tokens=c.n_tokens)
            strict = ("You did not provide the required 'Root Cause Analysis:' "
                      "section. Rewrite your answer starting with "
                      "'Root Cause Analysis:' followed by the corrected code.\n\n"
                      + refine_prompt)
            c2 = complete(self.client, strict, model=self.model,
                          temperature=0.7,
                          max_tokens=self.generator.max_tokens, logprobs=False)
            self._last_tokens += c2.n_tokens
            text = c2.text
            m = RCA_RE.search(text)
        traj.add("reflect", tokens=self._last_tokens,
                 has_rca=bool(m), raw=text[:800])
        rca = ""
        code_part = text
        if m:
            rca = text[m.end():].strip()
            # code follows the first ``` fence after the RCA, else whole tail
            fence = re.search(r"```(?:python)?\s*(.*?)```", rca, flags=re.S)
            if fence:
                code_part = fence.group(1)
                rca = rca[:fence.start()].strip()
            else:
                idx = rca.find("def ")
                code_part = rca[idx:] if idx >= 0 else rca
        code = Generator.strip_fences(code_part)
        return rca[:600], code


# ------------------------------------------------------------------- demo/self-test --
if __name__ == "__main__":
    import json
    sys.path.insert(0, os.path.join(ROOT, "hw1"))
    from data import load_humaneval

    tasks = load_humaneval()[:3]
    coder = ReActCoder(max_loops=3)
    out = []
    for t in tasks:
        r = coder.solve(t)
        out.append(r.to_dict())
        print(json.dumps(r.to_dict(), indent=1))
        if r.trajectory:
            os.makedirs("results", exist_ok=True)
            r.trajectory.to_json(f"results/react_{r.task_id.replace('/', '_')}.json")
