"""Generator — wraps the base LLM; samples N candidates at budgeted temperature."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
for _p in (os.path.join(ROOT, "hw1"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from llm_client import complete_many, DEFAULT_MODEL  # noqa: E402


@dataclass
class Candidate:
    text: str
    avg_logprob: float
    n_tokens: int
    answer: str | None = None          # extracted (math) or code body (code)
    meta: dict = field(default_factory=dict)


CODE_PROMPT_TEMPLATE = (
    "You are an expert Python programmer. Complete the function below.\n"
    "Output ONLY the code that continues after the given header lines "
    "(correctly indented), no explanations, no markdown fences.\n\n"
    "```python\n{prompt}\n```"
)


class Generator:
    def __init__(self, client, model: str = DEFAULT_MODEL, max_tokens: int = 512):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str, n: int, temperature: float,
                 task_type: str = "math") -> list:
        if task_type == "code":
            user_prompt = CODE_PROMPT_TEMPLATE.format(prompt=prompt)
        else:
            user_prompt = prompt
        comps = complete_many(self.client, [user_prompt] * max(n, 1),
                              model=self.model, temperature=max(temperature, 1e-3),
                              max_tokens=self.max_tokens, logprobs=True,
                              workers=min(n, 6))
        out = []
        for c in comps:
            out.append(Candidate(text=c.text,
                                 avg_logprob=c.avg_logprob,
                                 n_tokens=c.n_tokens,
                                 meta={"error": c.error}))
        return out

    @staticmethod
    def strip_fences(text: str) -> str:
        """Remove ```python fences and prose preamble from a code completion."""
        import re
        m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
        body = m.group(1) if m else text
        # drop leading non-code prose lines
        lines = body.split("\n")
        while lines and lines[0].strip() and not (
                lines[0].startswith(("def ", "class ", "import ", "from ", "@"))
                or lines[0].startswith("    ")):
            lines.pop(0)
        return "\n".join(lines).strip("\n")
