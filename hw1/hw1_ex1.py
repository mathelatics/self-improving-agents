"""HW1 Exercise 1 — Static Best-of-N vs Compute-Optimal (dynamic) scaling.

Strategies compared on GSM8K (NVIDIA OpenAI-compatible endpoint, logprobs):

  A. Static Best-of-N (majority vote): N=8 samples @ T=0.7 for EVERY prompt.
  B. Compute-optimal heuristic: sample once @ T=0.7; if mean token logprob
     >= threshold (-1.5 nats/token) accept immediately, else fall back to the
     remaining N-1 samples + majority vote.

Metrics: accuracy and total completion tokens generated ("compute cost").

Colab usage:
    %env NVIDIA_API_KEY nvapi-XXXX
    !git clone <this-repo> && cd repo/hw1
    !python hw1_ex1.py --n-samples 8 --limit 20          # quick smoke test
    !python hw1_ex1.py --n-samples 8 --limit 100         # full HW run

Local usage:
    export NVIDIA_API_KEY=nvapi-XXXX
    python hw1_ex1.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import get_client, complete, sample_n, DEFAULT_MODEL  # noqa
from grading import extract_answer, answers_equal                      # noqa
from data import load_gsm8k, gsm8k_prompt                               # noqa


def majority_vote(cands):
    """cands: list of extracted answer strings (None dropped).
    Returns (winner, votes, total_valid)."""
    vals = [c for c in cands if c is not None]
    if not vals:
        return None, 0, 0
    winner, votes = Counter(vals).most_common(1)[0]
    return winner, votes, len(vals)


# --------------------------------------------------------------- strategies
def run_static(client, rows, model, n_samples, temp, max_tokens, save_dir=None):
    recs = []
    for i, row in enumerate(rows):
        prompt = gsm8k_prompt(row)
        comps = sample_n(client, prompt, n_samples, model=model,
                         temperature=temp, max_tokens=max_tokens)
        answers = [extract_answer(c.text) for c in comps]
        winner, votes, valid = majority_vote(answers)
        correct = answers_equal(winner, row["gold"])
        toks = sum(c.n_tokens for c in comps)
        recs.append({"id": row["id"], "strategy": "static_best_of_n",
                     "gold": row["gold"], "selected": winner,
                     "correct": bool(correct), "tokens": int(toks),
                     "generations": len(comps), "votes": votes,
                     "valid_answers": valid})
        if save_dir:
            for j, c in enumerate(comps):
                with open(os.path.join(save_dir, f"static_{row['id']}_{j}.txt"),
                          "w") as f:
                    f.write(c.text or f"[ERROR {c.error}]")
        print(f"  [static] {i+1}/{len(rows)} {row['id']} "
              f"correct={correct} tokens={toks}", flush=True)
    return recs


def run_dynamic(client, rows, model, n_samples, temp, max_tokens,
                threshold=-1.5, save_dir=None):
    recs = []
    for i, row in enumerate(rows):
        prompt = gsm8k_prompt(row)
        first = complete(client, prompt, model=model, temperature=temp,
                         max_tokens=max_tokens)
        a1 = extract_answer(first.text)
        avg_lp = first.avg_logprob
        confident = avg_lp >= threshold
        if confident:
            winner, votes, valid, comps = a1, 1, int(a1 is not None), [first]
        else:
            rest = sample_n(client, prompt, n_samples - 1, model=model,
                            temperature=temp, max_tokens=max_tokens)
            comps = [first] + rest
            winner, votes, valid = majority_vote(
                [extract_answer(c.text) for c in comps])
        correct = answers_equal(winner, row["gold"])
        toks = sum(c.n_tokens for c in comps)
        recs.append({"id": row["id"], "strategy": "compute_optimal",
                     "gold": row["gold"], "selected": winner,
                     "correct": bool(correct), "tokens": int(toks),
                     "avg_logprob": round(avg_lp, 4), "confident": confident,
                     "generations": len(comps), "votes": votes,
                     "valid_answers": valid})
        if save_dir:
            for j, c in enumerate(comps):
                with open(os.path.join(save_dir, f"dyn_{row['id']}_{j}.txt"),
                          "w") as f:
                    f.write(c.text or f"[ERROR {c.error}]")
        print(f"  [dynamic] {i+1}/{len(rows)} {row['id']} "
              f"conf={confident} lp={avg_lp:.3f} correct={correct} "
              f"tokens={toks}", flush=True)
    return recs


def summarise(recs, strategy):
    n = len(recs)
    acc = sum(r["correct"] for r in recs) / n
    tok = sum(r["tokens"] for r in recs)
    gens = sum(r["generations"] for r in recs)
    out = {"strategy": strategy, "n_prompts": n, "accuracy": round(acc, 4),
           "total_tokens": int(tok), "avg_tokens_per_prompt": round(tok / n, 1),
           "total_generations": int(gens)}
    if strategy == "compute_optimal":
        conf = [r for r in recs if r["confident"]]
        out["pct_routed_easy"] = round(len(conf) / n, 4)
        out["accuracy_on_confident"] = round(
            sum(r["correct"] for r in conf) / max(len(conf), 1), 4)
    return out


def make_chart(static_sum, dyn_sum, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Static\nBest-of-N", "Compute-Optimal\n(logprob routing)"]
    accs = [static_sum["accuracy"] * 100, dyn_sum["accuracy"] * 100]
    toks = [static_sum["total_tokens"], dyn_sum["total_tokens"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, vals, title, unit in [
            (axes[0], accs, "Accuracy (%)", "%"),
            (axes[1], toks, "Total tokens generated (compute cost)", "")]:
        bars = ax.bar(labels, vals, color=["#c0504d", "#4f81bd"], width=0.55)
        ax.set_title(title, fontsize=11)
        ax.bar_label(bars, fmt="%.1f" if unit == "%" else "%.0f")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("HW1 Ex1: Best-of-N vs Compute-Optimal Scaling (GSM8K)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--threshold", type=float, default=-1.5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--model", default=os.environ.get("AVR_MODEL", DEFAULT_MODEL))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--save-responses", action="store_true")
    args = ap.parse_args()

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    respdir = os.path.join(outdir, "responses") if args.save_responses else None
    if respdir:
        os.makedirs(respdir, exist_ok=True)

    client = get_client()
    rows = load_gsm8k(n=args.limit)
    print(f"Model={args.model}  prompts={len(rows)}  N={args.n_samples}  "
          f"T={args.temperature}  threshold={args.threshold}")

    t0 = time.time()
    static_recs = run_static(client, rows, args.model, args.n_samples,
                             args.temperature, args.max_tokens, respdir)
    dyn_recs = run_dynamic(client, rows, args.model, args.n_samples,
                           args.temperature, args.max_tokens,
                           args.threshold, respdir)
    wall = time.time() - t0

    s_sum, d_sum = (summarise(static_recs, "static_best_of_n"),
                    summarise(dyn_recs, "compute_optimal"))
    savings = 1 - d_sum["total_tokens"] / max(s_sum["total_tokens"], 1)
    report = {"config": vars(args), "wall_time_s": round(wall, 1),
              "static": s_sum, "compute_optimal": d_sum,
              "token_savings_pct": round(savings * 100, 1)}
    with open(os.path.join(outdir, "hw1_ex1_results.json"), "w") as f:
        json.dump({"summary": report, "records": static_recs + dyn_recs},
                  f, indent=2)
    chart = os.path.join(outdir, "hw1_ex1_chart.png")
    make_chart(s_sum, d_sum, chart)
    print(json.dumps(report, indent=2))
    print("chart ->", chart)


if __name__ == "__main__":
    main()
