"""Phase-1 evaluation harness: run the AVR agent on GSM8K (math) and
HumanEval subset (code), logging prompt, difficulty score, number of
generations, verification outcomes and final accuracy to results/.

Colab usage:
    %env NVIDIA_API_KEY nvapi-XXXX
    !python evals/run_avr_eval.py --math-limit 25 --code-limit 10
Local usage:
    export NVIDIA_API_KEY=nvapi-XXXX
    python evals/run_avr_eval.py --math-limit 100 --code-limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hw1"))

from llm_client import get_client, DEFAULT_MODEL   # noqa: E402
from data import load_gsm8k, load_humaneval, gsm8k_prompt  # noqa: E402
from avr_agent.agent import AVRAgent               # noqa: E402


def row_to_record(r, outdir=None):
    d = dict(r.__dict__)
    if isinstance(d.get("answer"), str) and len(d["answer"]) > 300:
        d["answer"] = d["answer"][:300] + "..."
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-limit", type=int, default=25)
    ap.add_argument("--code-limit", type=int, default=10)
    ap.add_argument("--model", default=os.environ.get("AVR_MODEL", DEFAULT_MODEL))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    client = get_client()
    agent = AVRAgent(client=client, model=args.model, max_tokens=args.max_tokens)

    records = []
    t0 = time.time()

    print(f"=== GSM8K math ({args.math_limit} prompts) ===")
    for i, row in enumerate(load_gsm8k(n=args.math_limit)):
        res = agent.solve(gsm8k_prompt(row), task_type="math",
                          task={"id": row["id"], "gold": row["gold"]})
        rec = row_to_record(res)
        records.append(rec)
        print(f"  {i+1}/{args.math_limit} {row['id']} diff={rec['difficulty']} "
              f"N={rec['n_generations']} correct={rec['correct']}", flush=True)

    print(f"=== HumanEval code ({args.code_limit} tasks) ===")
    for i, task in enumerate(load_humaneval(n=args.code_limit)):
        res = agent.solve(task["prompt"], task_type="code",
                          task={"id": task["task_id"], "prompt": task["prompt"],
                                "test": task["test"],
                                "entry_point": task["entry_point"]})
        rec = row_to_record(res)
        records.append(rec)
        print(f"  {i+1}/{args.code_limit} {task['task_id']} "
              f"diff={rec['difficulty']} N={rec['n_generations']} "
              f"passed={rec['verified']}", flush=True)

    wall = time.time() - t0
    jpath = os.path.join(args.out, "avr_phase1_results.json")
    with open(jpath, "w") as f:
        json.dump({"config": vars(args), "wall_time_s": round(wall, 1),
                   "records": records}, f, indent=2)
    cpath = os.path.join(args.out, "avr_phase1_results.csv")
    keys = ["task_id", "task_type", "difficulty", "n_generations",
            "temperature", "probe_avg_logprob", "verified", "fallback",
            "correct", "tokens_generated"]
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    math_recs = [r for r in records if r["task_type"] == "math"]
    code_recs = [r for r in records if r["task_type"] == "code"]
    summary = {
        "model": args.model, "wall_time_s": round(wall, 1),
        "math": {"n": len(math_recs),
                 "accuracy": round(sum(bool(r["correct"]) for r in math_recs) /
                                   max(len(math_recs), 1), 4),
                 "avg_generations": round(sum(r["n_generations"] for r in math_recs) /
                                          max(len(math_recs), 1), 2)},
        "code": {"n": len(code_recs),
                 "pass_at_verified": round(sum(bool(r["verified"]) for r in code_recs) /
                                           max(len(code_recs), 1), 4),
                 "avg_generations": round(sum(r["n_generations"] for r in code_recs) /
                                          max(len(code_recs), 1), 2)},
        "difficulty_distribution": {
            d: sum(1 for r in records if r["difficulty"] == d)
            for d in ("low", "medium", "high")},
    }
    spath = os.path.join(args.out, "avr_phase1_summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("->", jpath, "\n->", cpath)


if __name__ == "__main__":
    main()
