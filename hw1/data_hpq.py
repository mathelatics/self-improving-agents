"""HotpotQA loader for HW2 Exercise 1 (Mini-LATS) — Colab-friendly.

Prefers the bundled snapshot ``data/hotpotqa_mini.jsonl`` (5,901 validation
questions with their distractor contexts, scraped from the official HotpotQA
validation split).  Falls back to streaming a parquet mirror on HuggingFace if
the snapshot is missing.

Each item: {id, question, gold, contexts:[{title, sentences:[...]}]}.
"""

from __future__ import annotations

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SNAPSHOT = os.path.join(DATA, "hotpotqa_mini.jsonl")


def load_hotpotqa(n: int = 20, seed: int = 42, split_start: int = 0):
    rows = None
    if os.path.exists(SNAPSHOT):
        with open(SNAPSHOT) as f:
            rows = [json.loads(line) for line in f]
    else:  # network fallback (Colab): official-style parquet mirror
        import pandas as pd
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("lucadiliello/hotpotqa",
                            "data/validation-00000-of-00001-01498301e5b78982.parquet",
                            repo_type="dataset")
        df = pd.read_parquet(p)
        rows = []
        for _, r in df.iterrows():
            ctxs = [c.strip() for c in str(r["context"]).split("[PAR]") if c.strip()]
            gold = r["answers"][0] if isinstance(r["answers"], list) and r["answers"] else str(r["answers"])
            rows.append({"id": f"hpq-{len(rows)}", "question": r["question"], "gold": gold,
                         "contexts": [{"title": c.split("[SEP]")[0].replace("[TLE]", "").strip(),
                                       "sentences": [s.strip() for s in c.split("[SEP]")[1].split(". ") if s.strip()]}
                                      for c in ctxs[:10]]})
        with open(SNAPSHOT, "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
    rng = random.Random(seed)
    subset = rng.sample(rows, min(n + split_start, len(rows)))
    return subset[split_start:n + split_start]
