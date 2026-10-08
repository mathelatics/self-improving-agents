"""HW2 Exercise 1 — Mini-LATS: Tree Search with Reflection vs. linear ReAct.

Task domain: HotpotQA multi-hop questions (distractor setting, bundled
snapshot so it runs in Colab offline).

Protocol (per the spec):
  * ReAct baseline      : one linear Thought -> Action(search) -> Observation
                          -> Thought -> Answer trajectory.
  * Mini-LATS           : generate K=3 DISTINCT first actions, execute all of
                          them, score the resulting states with a value
                          function (LLM grader; heuristic fallback), expand
                          the best branch to an answer.  If the top state
                          value is below a threshold, trigger REFLECTION:
                          feed failed actions + observations back and ask for a
                          "fundamentally different approach", then retry.

Evaluation: exact-match after HotpotQA normalisation (also reports a loose
substring match).  LLM calls are logged to results/hw2_ex1_calls.jsonl so the
comparison can be replayed / token-counted without GPUs.

Usage:
  NVIDIA_API_KEY=... python hw2_ex1.py --limit 20 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_hpq import load_hotpotqa          # noqa: E402
from llm_client import complete             # noqa: E402

CALLS_LOG = os.path.join(_HERE, "..", "results", "hw2_ex1_calls.jsonl")

# ---------------------------------------------------------------- prompt kit
SYS = ("You answer multi-hop questions using ONLY the provided passages. "
       "Be concise.")

REACT_TMPL = """Question: {q}

Passages:
{ctx}

Answer step by step in this exact format:
Thought: <reasoning>
Action: Search[<the key entity or sub-question you looked up>]
Observation: <the supporting sentence(s) from the passages>
Thought: <final reasoning>
Answer: <a short span copied from the passages>"""

GEN_ACTIONS_TMPL = """Question: {q}

Passages:
{ctx}

Decompose this multi-hop question. Propose exactly {k} DIFFERENT first search \
actions (sub-queries) that could start solving it, e.g. look up the bridge \
entity, look up the other entity, or directly compare attributes.

Format:
Action 1: Search[<query>]
Action 2: Search[<query>]
Action 3: Search[<query>]"""

EXECUTE_TMPL = """Question: {q}

Passages:
{ctx}

Sub-query chosen as the first step: {a}

Read the passages and extract what this sub-query yields.
Format:
Observation: <one or two sentences copied/condensed from the passages that \
answer the sub-query, or 'not found'>"""

ANSWER_TMPL = """Question: {q}

Passages:
{ctx}

Reasoning chain so far:
Step {i}. Sub-query: {a}
Observation: {o}
{extra}
Now finish the reasoning and answer.
Format:
Thought: <brief final reasoning>
Answer: <short span from the passages>"""

VALUE_TMPL = """Question: {q}
Gold task context (passages):
{ctx_short}

Candidate reasoning step:
Sub-query: {a}
Observation: {o}

Score how promising this intermediate state is for answering the question:
1 = useless/wrong/not grounded, 5 = clearly retrieves the right bridge fact.
Answer with ONLY the digit."""

REFLECT_TMPL = """Question: {q}

Passages:
{ctx}

These attempts failed because they did not surface the needed bridge fact:
{failures}

Reflection: explain in <=2 sentences why these approaches were wrong \
(e.g. wrong entity resolved first, distractor paragraph picked).
Root Cause Analysis: <one sentence>
Then propose ONE new, FUNDAMENTALLY DIFFERENT first action.
Format:
Action: Search[<new query>]"""


# ------------------------------------------------------------- local helpers
def log_call(tag: str, text: str, client=None, model=None):
    os.makedirs(os.path.dirname(CALLS_LOG), exist_ok=True)
    with open(CALLS_LOG, "a") as f:
        f.write(json.dumps({"tag": tag, "text": text, "model": model,
                            "has_client": client is not None,
                            "ts": time.time()}) + "\n")


def normalize(s: str) -> str:
    """Official HotpotQA exact-match normalisation."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def loose(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    return bool(p) and bool(g) and (p in g or g in p)


def fmt_context(item, max_chars=6000):
    out = []
    for c in item["contexts"]:
        block = f"[{c['title']}] " + ". ".join(c["sentences"][:5])
        out.append(block)
    ctx = "\n".join(out)
    return ctx[:max_chars]


def parse_answer(text: str) -> str:
    m = re.findall(r"Answer:\s*(.+)", text)
    if m:
        a = m[-1].strip().rstrip(".!")
        return a[:80]
    return text.strip().split("\n")[-1][:80]


def parse_actions(text: str, k: int = 3):
    acts = re.findall(r"Search\[(.*?)\]", text)
    acts = [a.strip() for a in acts if a.strip()]
    # de-dup while preserving order, pad to k
    seen, out = set(), []
    for a in acts:
        na = normalize(a)
        if na not in seen:
            seen.add(na)
            out.append(a)
    while len(out) < k:
        out.append(out[0] if out else "main entity")
    return out[:k]


def parse_score(text: str) -> float:
    m = re.search(r"\b([1-5])\b", text)
    return float(m.group(1)) if m else 0.0


def heuristic_value(obs: str) -> float:
    """No-API fallback: reward grounded, informative-looking observations."""
    v = 1.0
    if "not found" in obs.lower():
        return 1.0
    if re.search(r"\d{4}", obs):
        v += 1.0                       # dates often the bridge fact
    if len(obs.split()) > 6:
        v += 1.0
    return min(v, 5.0)


class Runner:
    """Wraps the NVIDIA client; every response is logged for offline replay."""

    def __init__(self, client, model):
        self.client, self.model = client, model

    def ask(self, tag: str, prompt: str, max_tokens=256, temperature=0.7) -> str:
        if self.client is None:
            return ""
        c = complete(self.client, prompt, model=self.model,
                     temperature=temperature, max_tokens=max_tokens,
                     system=SYS, logprobs=False)
        if c.error:
            raise RuntimeError(c.error)
        log_call(tag, c.text, model=self.model)
        return c.text


# ------------------------------------------------------------------ policies
def run_react(R: Runner, item) -> dict:
    ctx = fmt_context(item)
    t0 = time.time()
    traj = R.ask("react_traj", REACT_TMPL.format(q=item["question"], ctx=ctx))
    log_call("react_final", traj)
    # extract the final answer; fall back to whole-trajectory last line
    ans = parse_answer(traj)
    return {"policy": "react", "answer": ans, "trajectory": traj,
            "seconds": time.time() - t0, "calls": 1}


def run_lats(R: Runner, item, K=3, tau=3.0, max_reflect=1) -> dict:
    q, ctx = item["question"], fmt_context(item)
    t0, calls = time.time(), 0
    failures = []
    extra_ctx = ""
    best = None  # (value, action, observation, answer, score_src)

    for attempt in range(max_reflect + 1):
        if attempt == 0:
            gen = R.ask("gen_actions", GEN_ACTIONS_TMPL.format(
                q=q, ctx=ctx, k=K)); calls += 1
            actions = parse_actions(gen, K)
        else:
            fail_txt = "\n".join(f"- tried: {a} -> {o[:120]}"
                                 for a, o, _ in failures)
            ref = R.ask("reflect", REFLECT_TMPL.format(
                q=q, ctx=ctx, failures=fail_txt)); calls += 1
            new_a = parse_actions(ref, 1)[0]
            actions = [new_a]
            extra_ctx = f"Reflection from previous failure:\n{ref}\n"

        scored = []
        for a in actions:
            obs = R.ask("execute", EXECUTE_TMPL.format(
                q=q, ctx=ctx, a=a), max_tokens=160); calls += 1
            if R.client is not None:
                try:
                    sv = R.ask("value", VALUE_TMPL.format(
                        q=q, ctx_short=ctx[:1500], a=a, o=obs),
                        max_tokens=8, temperature=0.0)
                    v = parse_score(sv); src = "llm"
                except Exception:
                    v, src = heuristic_value(obs), "heur"
            else:
                v, src = heuristic_value(obs), "heur"
            scored.append((v, a, obs, src))
            if v < tau:
                failures.append((a, obs, v))

        scored.sort(key=lambda x: -x[0])
        v, a, o, src = scored[0]
        cand = {"value": v, "src": src}
        if v >= tau or attempt == max_reflect:
            ans_txt = R.ask("answer", ANSWER_TMPL.format(
                q=q, ctx=ctx, i=1, a=a, o=o, extra=extra_ctx)); calls += 1
            cand["answer"] = parse_answer(ans_txt)
            cand["reasoning"] = ans_txt
            if best is None or cand["value"] >= best["value"]:
                best = cand
            break
        best = cand  # low-value everywhere -> loop into reflection

    return {"policy": "lats", "answer": best.get("answer", ""),
            "best_value": best.get("value"), "value_src": best.get("src"),
            "n_failures": len(failures), "calls": calls,
            "seconds": time.time() - t0}


def evaluate(item, res):
    res["gold"] = item["gold"]
    res["em"] = em(res.get("answer", ""), item["gold"])
    res["loose"] = loose(res.get("answer", ""), item["gold"])
    return res


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--model", default="meta/llama-3.2-11b-vision-instruct")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API: exercise parsing/scoring on canned outputs")
    args = ap.parse_args()

    items = load_hotpotqa(n=args.limit, seed=42)
    client, model = None, args.model
    if not args.dry_run:
        from llm_client import get_client
        client = get_client(timeout=120)

    R = Runner(client, model)
    rows = []
    for it in items:
        for fn in (run_react, run_lats):
            try:
                res = fn(R, it)
            except Exception as e:  # noqa: BLE001
                res = {"policy": fn.__name__, "error": str(e)[:200]}
            rows.append(evaluate(it, res))
            print(f"{it['id']:8s} {res['policy']:6s} em={res['em']} "
                  f"pred={str(res.get('answer'))[:40]!r} gold={it['gold']!r}",
                  flush=True)

    acc = {}
    for pol in ("react", "lats"):
        sub = [r for r in rows if r["policy"] == pol]
        ok = [r for r in sub if r.get("em")]
        acc[pol] = {"n": len(sub),
                    "em": round(len(ok) / max(len(sub), 1), 3),
                    "loose": round(sum(r.get("loose") for r in sub) / max(len(sub), 1), 3),
                    "mean_calls": round(sum(r.get("calls", 0) for r in sub) / max(len(sub), 1), 2)}
    out = {"summary": acc, "rows": rows}
    dst = os.path.join(_HERE, "..", "results", "hw2_ex1_lats_results.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps(acc, indent=1))


if __name__ == "__main__":
    main()
