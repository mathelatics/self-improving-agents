"""AVR Agent v2.0 evaluation harness — Phase 1 (Router + Verifier + Selector)
   upgraded with dense step-level feedback (Mini-PRM) and ReAct self-correction.

Arms:
  math : baseline_n{N}          — Router forced to fixed N (HW1 ablation grid)
         avr                    — full agent: Router -> Gen -> Verify -> Select
                                  -> Mini-PRM step check (backtrack signal)
  code : single_shot            — one generation, sandbox verification
         avr_react              — agent w/ ReAct repair loop (max 3 refinements,
                                  forced Root Cause Analysis per refinement)

Outputs (all under results/):
  avr_eval_results.json  — machine-readable log for the write-up
  avr_eval_chart.png     — accuracy-vs-N + PRM/refine comparison chart
Usage:
  NVIDIA_API_KEY=... python evals/run_avr_eval.py --math-limit 40 --code-limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "hw1"), ROOT, os.path.join(ROOT, "avr_agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_client import get_client, DEFAULT_MODEL                      # noqa: E402
from data import load_gsm8k, load_humaneval                           # noqa: E402
from grading import extract_answer, answers_equal                     # noqa: E402
from avr_agent.agent import AVRAgent                                  # noqa: E402
from avr_agent.router import ROUTING_POLICY, Difficulty               # noqa: E402
from avr_agent.generator import Generator                             # noqa: E402
from avr_agent.verifier import CodeVerifierAdapter                    # noqa: E402


def run_math(client, rows, arms, verbose=False):
    """Returns {arm: [correct_bool,...]} plus token ledger."""
    res = {a: [] for a in arms}
    tokens = {a: 0 for a in arms}
    agents = {}
    for arm in arms:
        if arm == "avr":
            agents[arm] = AVRAgent(client=client)
    for i, row in enumerate(rows):
        outs = {}
        gold = row["gold"]
        for arm in arms:
            if arm.startswith("baseline_n"):
                n = int(arm.split("n")[1])
                temp = ROUTING_POLICY[[Difficulty.LOW, Difficulty.MEDIUM,
                                      Difficulty.HIGH][min(n // 4, 2)]
                                     ]["temperature"] if n > 1 else 0.0
                gen = Generator(client, max_tokens=512)
                cands = gen.generate(row["question"], n, temp or 0.7,
                                     task_type="math")
                from collections import Counter
                vals = [extract_answer(c.text) for c in cands]
                valid = [v for v in vals if v is not None]
                pick = Counter(valid).most_common(1)[0][0] if valid else None
                ok = answers_equal(pick, gold)
                tokens[arm] += sum(c.n_tokens for c in cands)
            else:  # full AVR agent (router + verifier + selector + PRM)
                r = agents[arm].solve(row["question"], "math",
                                      {"id": row["id"], "gold": gold,
                                       "question": row["question"]})
                ok = bool(r.correct)
                outs[row["id"]] = {"difficulty": r.difficulty,
                                   "prm_first_bad": r.prm_first_bad_step,
                                   "fallback": r.fallback}
                tokens[arm] += r.tokens_generated
            res[arm].append(ok)
        if verbose:
            print(f"  [{i+1}/{len(rows)}] {row['id']} " +
                  " ".join(f"{a}={int(res[a][-1])}" for a in arms), flush=True)
    return res, tokens, outs


def run_code(client, tasks, max_loops=3, verbose=False):
    verifier = CodeVerifierAdapter()
    agent = AVRAgent(client=client)
    rows = []
    for i, t in enumerate(tasks):
        task = {"id": t["task_id"], "prompt": t["prompt"],
                "test": t["tests"], "entry_point": t["entry_point"]}
        gen = Generator(client, max_tokens=512)
        cands = gen.generate(t["prompt"], 1, 0.3, task_type="code")
        v = verifier.verify(cands[0], task)
        single_ok = bool(v["passed"])
        r = agent.solve(t["prompt"], "code", task)   # includes ReAct repair
        rows.append({"task_id": t["task_id"], "single_shot": single_ok,
                     "avr_react": bool(r.correct), "refine_loops": r.refine_loops,
                     "fixed_by_react": (not single_ok) and bool(r.correct)})
        if verbose:
            print(f"  [{i+1}/{len(tasks)}] {t['task_id']} "
                  f"single={single_ok} react={r.correct} "
                  f"loops={r.refine_loops}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-limit", type=int, default=40)
    ap.add_argument("--code-limit", type=int, default=10)
    ap.add_argument("--arms", default="baseline_n1,baseline_n8,avr")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    client = get_client()
    if args.model:
        os.environ["AVR_MODEL"] = args.model
    arms = args.arms.split(",")
    math_rows = load_gsm8k()[: args.math_limit]
    code_tasks = load_humaneval()[: args.code_limit]

    t0 = time.time()
    mres, mtoks, extra = run_math(client, math_rows, arms, verbose=True)
    cres = run_code(client, code_tasks, verbose=True)

    summary = {
        "model": os.environ.get("AVR_MODEL", DEFAULT_MODEL),
        "n_math": len(math_rows), "n_code": len(code_tasks),
        "seconds": round(time.time() - t0, 1),
        "math": {a: {"accuracy": round(sum(v) / len(v), 3),
                     "tokens": mtoks[a]} for a, v in mres.items()},
        "code": {"single_shot_pass": sum(r["single_shot"] for r in cres) / len(cres),
                 "avr_react_pass": sum(r["avr_react"] for r in cres) / len(cres),
                 "tasks_fixed_by_react": sum(r["fixed_by_react"] for r in cres),
                 "total_refine_loops": sum(r["refine_loops"] for r in cres)},
        "per_task_code": cres,
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out_json = os.path.join(ROOT, "results", "avr_eval_results.json")
    json.dump({"summary": summary, "math_extra": extra}, open(out_json, "w"),
              indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        names = list(mres)
        accs = [sum(v) / len(v) for v in mres.values()]
        toks = [mtoks[a] for a in names]
        b = ax[0].bar(names, accs, color="#4c72b0")
        ax[0].set_ylabel("GSM8K accuracy"); ax[0].set_ylim(0, 1)
        ax[0].set_title(f"Math arms (n={len(math_rows)})")
        for p, t in zip(b, toks):
            ax[0].text(p.get_x()+p.get_width()/2, p.get_height()+0.02,
                       f"{t//1000}k tok", ha="center", fontsize=8)
        sp = summary["code"]["single_shot_pass"]
        rp = summary["code"]["avr_react_pass"]
        b2 = ax[1].bar(["single-shot", "AVR+ReAct"], [sp, rp], color="#55a868")
        ax[1].set_ylabel("HumanEval pass rate"); ax[1].set_ylim(0, 1)
        ax[1].set_title(f"Code self-correction (n={len(code_tasks)})")
        for p, v in zip(b2, [sp, rp]):
            ax[1].text(p.get_x()+p.get_width()/2, v+0.02, f"{v:.2f}", ha="center")
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "results", "avr_eval_chart.png"))
        print("chart -> results/avr_eval_chart.png")
    except Exception as e:
        print("no chart:", e)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
