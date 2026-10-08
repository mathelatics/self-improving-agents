"""HW2 Exercise 2 driver — single-shot vs ReAct self-correction on HumanEval.

For each task we run two arms with the SAME model/sandbox:
  * single_shot : one generation, verified once in the HW1 subprocess sandbox.
  * react       : avr_agent.ReActCoder (Generate -> Execute -> stderr ->
                  forced Root-Cause-Analysis -> Regenerate, max 3 loops).
Reports pass-rate delta and token cost, writes results/hw2_ex2_results.json
(+ chart results/hw2_ex2_chart.png) and per-task trajectories.

Usage: NVIDIA_API_KEY=... python hw1/hw2_ex2_react.py --limit 20 [--model M]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "avr_agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_client import get_client, DEFAULT_MODEL                    # noqa: E402
from data import load_humaneval                                     # noqa: E402
from avr_agent.generator import Generator                           # noqa: E402
from avr_agent.verifier import CodeVerifierAdapter                  # noqa: E402
from avr_agent.react_executor import ReActCoder                     # noqa: E402


def single_shot(client, verifier, task, temperature=0.3):
    gen = Generator(client, max_tokens=512)
    cands = gen.generate(task["prompt"], n=1, temperature=temperature,
                         task_type="code")
    v = verifier.verify(cands[0], task)
    return {"task_id": task["task_id"], "passed": bool(v["passed"]),
            "tokens": cands[0].n_tokens, "llm_calls": 1,
            "error": (v["error"] or "")[:400]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-loops", type=int, default=3)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    client = get_client()
    verifier = CodeVerifierAdapter()
    coder = ReActCoder(client=client, model=args.model or DEFAULT_MODEL,
                       max_loops=args.max_loops)
    tasks = load_humaneval()[: args.limit]

    rows, t0 = [], time.time()
    for i, t in enumerate(tasks):
        s = single_shot(client, verifier, t)
        r = coder.solve(t)
        row = {"task_id": t["task_id"],
               "single_pass": s["passed"], "react_pass": r.passed,
               "loops_used": r.loops_used,
               "single_tokens": s["tokens"], "react_tokens": r.tokens,
               "fixed_by_react": (not s["passed"]) and r.passed,
               "rcas": r.rcas}
        rows.append(row)
        r.trajectory.to_json(os.path.join(ROOT, "results",
                               f"react_{t['task_id'].replace('/', '_')}.json"))
        print(f"[{i+1}/{len(tasks)}] {t['task_id']} single={s['passed']} "
              f"react={r.passed} loops={r.loops_used} "
              f"tok(s/r)={s['tokens']}/{r.tokens}", flush=True)

    n = len(rows)
    sp = sum(r["single_pass"] for r in rows) / n
    rp = sum(r["react_pass"] for r in rows) / n
    summary = {"n": n, "seconds": round(time.time() - t0, 1),
               "single_shot_pass_rate": round(sp, 3),
               "react_pass_rate": round(rp, 3),
               "delta": round(rp - sp, 3),
               "tasks_fixed_by_react": sum(r["fixed_by_react"] for r in rows),
               "mean_loops_on_failures": round(
                   sum(r["loops_used"] for r in rows) /
                   max(sum(1 for r in rows if not r["single_pass"]), 1), 2),
               "total_tokens_single": sum(r["single_tokens"] for r in rows),
               "total_tokens_react": sum(r["react_tokens"] for r in rows)}
    outdir = os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    json.dump({"summary": summary, "per_task": rows},
              open(os.path.join(outdir, "hw2_ex2_results.json"), "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Single-shot", f"ReAct (+{summary['delta']*100:.0f}pp)"],
                      [sp, rp], color=["#888", "#55a868"])
        ax.set_ylabel("HumanEval pass rate"); ax.set_ylim(0, 1)
        ax.set_title(f"HW2 Ex2: ReAct self-correction (n={n}, max 3 loops)")
        for b, v in zip(bars, [sp, rp]):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "hw2_ex2_chart.png"))
        print("chart -> results/hw2_ex2_chart.png")
    except Exception as e:
        print("no chart:", e)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
