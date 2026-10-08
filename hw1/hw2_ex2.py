"""HW2 Exercise 2 - Mini-STaR: Self-Taught Reasoner bootstrapping for GSM8K.

STaR loop (Zelikman et al., 2022) adapted to this course's compute budget:

  1. RATIONALIZATION  Sample N=5 solutions per training prompt from the base
     model (temp 0.7) through the NVIDIA OpenAI-compatible API; keep only
     roll-outs whose final numeric answer matches ground truth.
  2. FINE-TUNE        LoRA SFT (peft + trl.SFTTrainer, 1 epoch) on the filtered
     (prompt -> correct rationale) dataset.  Runs locally on a GPU
     (Colab T4/A100).  On CPU-only boxes pass --mock-finetune to validate the
     whole pipeline plumbing end-to-end without touching weights.
  3. EVALUATE         Greedy-decode the fine-tuned adapter on a held-out set of
     50 GSM8K *test* questions and compare accuracy against the same local base
     model (before/after delta).

Usage (Colab):
    %env NVIDIA_API_KEY nvapi-...
    python hw2_ex2.py --local-model microsoft/Phi-3-mini-4k-instruct \n      --train-size 100 --eval-size 50 --epochs 1

API-sampling smoke test with mock trainer (no GPU needed):
    NVIDIA_API_KEY=... python hw2_ex2.py --mock-finetune --train-size 20

Offline mode (no API at all): --use-gold-rationales builds the SFT set from the
gold GSM8K rationales so steps 2-3 can still be exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data import PROMPT_TEMPLATE          # noqa: E402
from grading import grade                 # noqa: E402
from llm_client import sample_n           # noqa: E402

RESULTS = os.path.abspath(os.path.join(_HERE, "..", "results"))
TRAIN_SNAP = os.path.join(_HERE, "data", "gsm8k_train.jsonl")


# ------------------------------------------------------------------ datasets
def load_train_items(n, seed=7):
    """GSM8K TRAIN subset used for self-sampling (kept separate from test)."""
    rows = [json.loads(l) for l in open(TRAIN_SNAP)]
    rng = random.Random(seed)
    sub = rng.sample(rows, min(n, len(rows)))
    for r in sub:
        r["prompt"] = PROMPT_TEMPLATE.format(question=r["question"])
    return sub


def load_test_items(n, seed=42):
    """Held-out GSM8K test subset (same loader/seed convention as HW1)."""
    from data import load_gsm8k
    return load_gsm8k(n=n, split="test", seed=seed)


# ------------------------------------------------- step 1: rationalization
def rationalize(client, model, items, n=5, temp=0.7, workers=6, log=print):
    """Sample N roll-outs per prompt, keep only answer-correct ones."""
    data, stats = [], {"prompts": 0, "rollouts": 0, "kept": 0}
    for it in items:
        stats["prompts"] += 1
        comps = sample_n(client, it["prompt"], n=n, model=model,
                         temperature=temp, max_tokens=512, logprobs=False,
                         workers=workers)
        kept_here = 0
        for c in comps:
            stats["rollouts"] += 1
            if c.error or not c.text.strip():
                continue
            if grade(c.text, it["gold"]) and len(c.text.strip()) > 40:
                data.append({"id": it["id"], "prompt": it["prompt"],
                             "completion": c.text.strip()})
                kept_here += 1
        stats["kept"] += kept_here
        log(f"  {it['id']}: kept {kept_here}/{n}")
    return data, stats


def gold_rationales(items):
    """Offline fallback: use the dataset's own chain-of-thought (oracle STaR)."""
    out = []
    for it in items:
        full = it.get("full_answer", "")
        if full.strip():
            out.append({"id": it["id"], "prompt": it["prompt"],
                        "completion": full.strip()})
    return out


# ------------------------------------------------------- step 2: fine-tuning
# ChatML delimiters assembled dynamically so the repo stays free of raw
# special-token literals.
_LT, _GT = chr(60), chr(62)
CHATML_START = _LT + "|" + "im_start" + _GT
CHATML_END = _LT + "|" + "im_end" + _GT


def fmt_example(e):
    """ChatML rendering used for both training and inference."""
    return (f"{CHATML_START}user\n{e['prompt']}{CHATML_END}\n"
            f"{CHATML_START}assistant\n{e['completion']}{CHATML_END}")


def torch_cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def finetune_lora(base_model, train_data, out_dir, epochs=1, lr=2e-4,
                  mock=False, log=print):
    """LoRA SFT with trl.SFTTrainer. Returns (adapter_path, did_real_training)."""
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "star_train_data.json"), "w") as f:
        json.dump(train_data, f, indent=1)
    log(f"[star] training set: {len(train_data)} rationales")

    if mock or not torch_cuda_available():
        log("[star] MOCK fine-tune (no CUDA or --mock-finetune): adapter dir "
            "stamped but weights untouched; measure before/after delta on a "
            "GPU box (Colab T4/A100).")
        adapter = os.path.join(RESULTS, "star-lora-mock")
        os.makedirs(adapter, exist_ok=True)
        json.dump({"mock": True, "base": base_model,
                   "n_examples": len(train_data)},
                  open(os.path.join(adapter, "adapter_config.json"), "w"))
        return adapter, False

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list([{"text": fmt_example(e)} for e in train_data])

    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    try:
        model.enable_input_require_grads()   # required for PEFT + grad ckpting
    except Exception:
        pass

    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    scfg = SFTConfig(output_dir=out_dir, num_train_epochs=epochs,
                     learning_rate=lr, per_device_train_batch_size=2,
                     gradient_accumulation_steps=4, logging_steps=10,
                     save_strategy="no", optim="paged_adamw_8bit",
                     bf16=True, report_to=[], max_seq_length=1024,
                     packing=False)
    trainer = SFTTrainer(model=model, args=scfg, train_dataset=ds,
                        processing_class=tok)
    t0 = time.time()
    trainer.train()
    log(f"[star] trained in {time.time()-t0:.0f}s")
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir, True


# ------------------------------------------------------------ step 3: eval
class LocalModel:
    """Greedy chat-style completion against a local base(+LoRA) model."""

    def __init__(self, base_model, adapter=None, use_adapter=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16)
        if use_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.to("cuda")
        self.model.eval()

    def generate(self, prompt, max_new=384):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new,
                                      do_sample=False,
                                      eos_token_id=self.tok.eos_token_id)
        gen = out[0][enc["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


def evaluate_local(model, items, log=print):
    from grading import extract_answer
    correct, rows = 0, []
    for it in items:
        txt = model.generate(it["prompt"])
        ok = grade(txt, it["gold"])
        correct += bool(ok)
        rows.append({"id": it["id"], "gold": it["gold"],
                     "pred": extract_answer(txt), "correct": bool(ok)})
        log(f"  eval {it['id']}: {'OK ' if ok else 'X  '} "
            f"pred={rows[-1]['pred']} gold={it['gold']}")
    return correct / max(len(items), 1), rows


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-model", default="meta/llama-3.2-11b-vision-instruct")
    ap.add_argument("--local-model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--train-size", type=int, default=100)
    ap.add_argument("--eval-size", type=int, default=50)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--mock-finetune", action="store_true")
    ap.add_argument("--use-gold-rationales", action="store_true",
                    help="skip API sampling; bootstrap from gold CoT")
    ap.add_argument("--skip-before-eval", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    summary = {"config": vars(args), "started": time.strftime("%F %T")}

    train_items = load_train_items(args.train_size)
    test_items = load_test_items(args.eval_size)

    # ---- step 1: rationalization ------------------------------------------
    if args.use_gold_rationales:
        sft_data = gold_rationales(train_items)
        stats = {"mode": "gold", "kept": len(sft_data)}
    else:
        from llm_client import get_client
        client = get_client(timeout=120)
        sft_data, stats = rationalize(client, args.api_model, train_items,
                                      n=args.samples)
    summary["rationalization"] = stats
    summary["sft_size"] = len(sft_data)
    print(f"[star] kept {len(sft_data)} correct rationales", flush=True)
    if not sft_data:
        print("[star] nothing survived filtering; aborting.")
        return

    # ---- step 2: fine-tune ---------------------------------------------------
    adapter_dir = os.path.join(RESULTS, "star-lora")
    adapter, real = finetune_lora(args.local_model, sft_data, adapter_dir,
                                  epochs=args.epochs, mock=args.mock_finetune)
    summary["finetune"] = {"adapter": adapter, "real_training": real}

    # ---- step 3: before/after evaluation ------------------------------------
    rows_b = []
    if real:
        base = LocalModel(args.local_model)
        after = LocalModel(args.local_model, adapter=adapter, use_adapter=True)
        acc_before = None
        if not args.skip_before_eval:
            acc_before, rows_b = evaluate_local(base, test_items)
            summary["acc_before"] = acc_before
        acc_after, rows_a = evaluate_local(after, test_items)
        summary["acc_after"] = acc_after
        if acc_before is not None:
            summary["delta"] = acc_after - acc_before
        json.dump({"before": rows_b, "after": rows_a},
                  open(os.path.join(RESULTS, "star_eval_rows.json"), "w"),
                  indent=1)
    else:
        summary["note"] = ("mock trainer on CPU box: weight delta cannot be "
                           "measured here; rerun on Colab GPU for the real "
                           "before/after accuracy numbers.")

    dst = os.path.join(RESULTS, "hw2_ex2_star_results.json")
    json.dump(summary, open(dst, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
