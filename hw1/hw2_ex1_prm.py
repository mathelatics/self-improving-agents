"""HW2 Exercise 1 driver — Mini-PRM step-level verification on GSM8K.

Pipeline per problem:
  1. Sample N chain-of-thought roll-outs (T=0.7).
  2. Split each into steps; score every step with the HeuristicPRM
     (arithmetic re-computation + number carry-forward).
  3. BACKTRACK POLICY: if a step is flagged negative, truncate the reasoning
     prefix *before* the bad step and re-generate the continuation from there
     (up to `max_backtracks` times) instead of trusting a flawed final answer.
  4. Compare accuracy/tokens of:
       plain      : first roll-out, no PRM
       prm_filter : accept roll-outs only if NO step is flagged (else majority
                    vote over unflagged ones)
       prm_bt     : backtracking regeneration until clean trajectory
Usage:  NVIDIA_API_KEY=... python hw1/hw2_ex1_prm.py --limit 25 [--model M]
Writes results to results/hw2_ex1_results.json (+ chart).
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

from llm_client import get_client, complete, DEFAULT_MODEL          # noqa: E402
from grading import extract_answer, answers_equal                   # noqa: E402
from data import load_gsm8k                                         # noqa: E402
from prm_scorer import HeuristicPRM, split_steps                    # noqa: E402

SOLVE_TMPL = ("{question}\n\nSolve this step by step, one calculation per "
              "line. End with 'The answer is X'.")


def run_one(client, prm, row, n_samples=3, max_backtracks=2, model=None):
    q, gold = row["question"], row["gold"]
    stats = {"id": row["id"], "tokens": 0, "calls": 0}

    def gen(prompt, temp=0.7):
        c = complete(client, prompt, model=model or DEFAULT_MODEL,
                     temperature=temp, max_tokens=400)
        stats["tokens"] += c.n_tokens
        stats["calls"] += 1
        return c.text

    # ---- sample N trajectories -----------------------------------------
    trajs = []
    for _ in range(n_samples):
        text = gen(SOLVE_TMPL.format(question=q))
        steps = split_steps(text)
        scores = prm.score(q, steps)
        trajs.append((text, steps, scores))

    # ---- plain baseline: first roll-out ---------------------------------
    plain_ok = answers_equal(extract_answer(trajs[0][0]), gold)

    # ---- prm_filter: keep only fully-clean trajectories -----------------
    clean = [t for t in trajs if prm.first_bad_index(t[2]) is None]
    filter_src = clean if clean else trajs
    filt_answers = [extract_answer(t[0]) for t in filter_src]
    from collections import Counter
    valid = [a for a in filt_answers if a is not None]
    filt_ans = Counter(valid).most_common(1)[0][0] if valid else None
    filter_ok = answers_equal(filt_ans, gold)

    # ---- prm_bt: backtrack at first flagged step ------------------------
    bt_text, bt_steps, bt_scores = trajs[0]
    attempts = 0
    while True:
        bad = prm.first_bad_index(bt_scores)
        if bad is None or attempts >= max_backtracks:
            break
        attempts += 1
        prefix = "\n".join(bt_steps[:bad]) or "(start fresh)"
        redo = gen(f"{SOLVE_TMPL.format(question=q)}\n\n"
                   f"A previous attempt went wrong. Verified so far:\n{prefix}\n\n"
                   f"The next step was flagged as WRONG: {bt_steps[bad]!r}\n"
                   f"Reason: {bt_scores[bad].reason}\n"
                   f"Continue from the verified prefix with a CORRECT next step.")
        bt_text = (prefix + "\n" + redo).strip() if prefix != "(start fresh)" else redo
        bt_steps = split_steps(bt_text)
        bt_scores = prm.score(q, bt_steps)
        stats["backtracks"] = stats.get("backtracks", 0) + 1
    bt_ok = answers_equal(extract_answer(bt_text), gold)

    stats.update(plain=plain_ok, prm_filter=filter_ok, prm_backtrack=bt_ok,
                 flagged_steps=[s.to_dict() for t in trajs for s in t[2]
                                if s.score < 0][:6],
                 gold=gold)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    client = get_client()
    prm = HeuristicPRM()
    rows = load_gsm8k()[: args.limit]
    out, t0 = [], time.time()
    for i, row in enumerate(rows):
        r = run_one(client, prm, row, n_samples=args.samples, model=args.model)
        out.append(r)
        print(f"[{i+1}/{len(rows)}] {r['id']} plain={r['plain']} "
              f"filter={r['prm_filter']} bt={r['prm_backtrack']} "
              f"tok={r['tokens']} bt_n={r.get('backtracks',0)}", flush=True)

    acc = lambda k: sum(1 for r in out if r[k]) / len(out)
    summary = {
        "n": len(out), "seconds": round(time.time() - t0, 1),
        "accuracy_plain": round(acc("plain"), 3),
        "accuracy_prm_filter": round(acc("prm_filter"), 3),
        "accuracy_prm_backtrack": round(acc("prm_backtrack"), 3),
        "mean_tokens": round(sum(r["tokens"] for r in out) / len(out), 1),
        "total_backtracks": sum(r.get("backtracks", 0) for r in out),
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", "hw2_ex1_results.json")
    json.dump({"summary": summary, "per_problem": out}, open(path, "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["Plain\n(1-shot)", "PRM filter\n(majority of clean)", "PRM backtrack\n(regenerate)"]
        vals = [summary["accuracy_plain"], summary["accuracy_prm_filter"],
                summary["accuracy_prm_backtrack"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, vals, color=["#888", "#4c72b0", "#55a868"])
        ax.set_ylabel("GSM8K accuracy"); ax.set_ylim(0, 1)
        ax.set_title(f"HW2 Ex1: Mini-PRM step verification (n={summary['n']})")
        for x, v in zip(names, vals):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center")
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "results", "hw2_ex1_chart.png"))
        print("chart -> results/hw2_ex1_chart.png")
    except Exception as e:  # plotting optional
        print("no chart:", e)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
