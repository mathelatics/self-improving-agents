"""State manager — the agent's trajectory store (Phase 1 v2.0).

Every thought (LLM roll-out), action (generation / verification / refinement)
and observation (verifier verdict, PRM step scores, stderr traceback) is
appended to a ``Trajectory`` so multi-turn self-correction never loses context
and so runs are fully replayable/serialisable for the results log.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Step:
    """One entry in the trajectory."""
    kind: str                      # generate | verify | reflect | refine | route | accept | fallback
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ts": round(self.ts, 3), **self.payload}


@dataclass
class Trajectory:
    task_id: str
    task_type: str
    prompt: str
    steps: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- writing -------------------------------------------------------
    def add(self, kind: str, **payload) -> Step:
        s = Step(kind=kind, payload=payload)
        self.steps.append(s)
        return s

    # ---- reading -------------------------------------------------------
    def last(self, kind: str) -> Optional[Step]:
        for s in reversed(self.steps):
            if s.kind == kind:
                return s
        return None

    def kinds(self) -> list:
        return [s.kind for s in self.steps]

    @property
    def n_llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind in ("generate", "reflect"))

    @property
    def total_tokens(self) -> int:
        return sum(int(s.payload.get("tokens", 0)) for s in self.steps)

    def messages_for_refinement(self) -> list:
        """Rebuild the conversation for a ReAct refinement turn.

        Returns [{role, content}, ...] containing the original generation
        request, the candidate that failed, and the verifier observation
        (stderr) of the most recent failed verification.
        """
        msgs: list = []
        gen = self.last("generate")
        ver = None
        for s in reversed(self.steps):          # newest FAILED verification
            if s.kind == "verify" and not s.payload.get("passed", False):
                ver = s
                break
        if gen is not None:
            msgs.append({"role": "user", "content": gen.payload.get("request_prompt", self.prompt)})
            msgs.append({"role": "assistant", "content": gen.payload.get("best_text", "")})
        if ver is not None:
            err = ver.payload.get("error") or "verification failed"
            msgs.append({"role": "user",
                         "content": "The code failed with this error. Analyze the "
                                    "error and rewrite the code to fix it.\n\n"
                                    f"```\n{err[:4000]}\n```"})
        return msgs

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "task_type": self.task_type,
                "prompt": self.prompt, "meta": self.meta,
                "steps": [s.to_dict() for s in self.steps]}

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        t = cls(task_id=d["task_id"], task_type=d["task_type"],
                prompt=d["prompt"], meta=d.get("meta", {}))
        for s in d.get("steps", []):
            s = dict(s)
            t.add(s.pop("kind"), **{k: v for k, v in s.items() if k != "ts"})
        return t
