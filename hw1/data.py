"""Dataset loading for HW1 / AVR evals (Colab-friendly, offline-capable).

GSM8K is loaded via HuggingFace ``datasets`` when the network allows it;
otherwise we fall back to a bundled JSONL snapshot (data/gsm8k_mini.jsonl) so
the code runs in locked-down Colab/notebook environments.

HumanEval: prefer ``human-eval``/``evalplus`` if installed, else bundled
snapshot data/humaneval_mini.jsonl.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Think step by step. At the very end, output the final numerical answer "
    "on its own line in the exact format:\n#### <answer>"
)


def load_gsm8k(n: int = 100, split: str = "test", seed: int = 42):
    """Return list of dicts: {id, question, gold, prompt}."""
    rows = None
    try:  # preferred path: real dataset
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=split)
        rows = [{"id": f"gsm8k-{i}", "question": r["question"],
                 "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
                for i, r in enumerate(ds)]
    except Exception:  # noqa: BLE001 - offline / gated repo -> bundled fallback
        with open(os.path.join(DATA, "gsm8k_mini.jsonl")) as f:
            rows = [json.loads(line) for line in f]
    # deterministic subset
    import random
    rng = random.Random(seed)
    subset = rng.sample(rows, min(n, len(rows)))
    for r in subset:
        r["prompt"] = PROMPT_TEMPLATE.format(question=r["question"])
    return subset


def load_humaneval(n: int = 50):
    """Return list of dicts: {task_id, prompt, canonical_solution, tests, entry_point}."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        rows = []
        for r in ds:
            tests = "\n".join(r.get("test", "").split("\n")[1:]) or \
                    f"assert {r['entry_point']} is not None"
            rows.append({"task_id": r["task_id"], "prompt": r["prompt"],
                         "canonical_solution": r["canonical_solution"],
                         "test": r.get("test", ""), "tests": tests,
                         "entry_point": r["entry_point"]})
        return rows[:n]
    except Exception:  # noqa: BLE001
        with open(os.path.join(DATA, "humaneval_mini.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        return rows[:n]


def gsm8k_prompt(row: dict) -> str:
    return row.get("prompt") or PROMPT_TEMPLATE.format(question=row["question"])
