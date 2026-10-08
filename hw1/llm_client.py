"""Shared LLM client helpers for HW1 and the AVR agent (Colab-friendly).

Targets NVIDIA's OpenAI-compatible endpoint (https://integrate.api.nvidia.com/v1)
which serves free-tier Nemotron / Llama models AND returns per-token ``logprobs``.

Key quirks discovered empirically (Oct 2026):
  * ``logprobs=True, top_logprobs=1`` is supported; logprobs live on
    ``choice.message.logprobs`` (fall back to ``choice.logprobs``).
  * ``n > 1`` requires ``temperature > 0`` -> we fan out N independent calls
    concurrently instead of using the ``n`` parameter.
  * Reasoning models (nemotron-3.5) emit chain-of-thought in the plain content
    field unless thinking is disabled via
    ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``.
  * The response object may omit ``usage`` -> always fall back to counting the
    returned logprob tokens.

Security note: the API key is read from the environment only
(``NVIDIA_API_KEY``); nothing is hard-coded in this repo.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import openai

# Make sibling HW1 modules importable no matter the cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"  # fast, logprobs verified


def get_client(timeout: float = 90.0, max_retries: int = 2) -> openai.OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. In Colab run:\n"
            "  %env NVIDIA_API_KEY nvapi-XXXX\n"
            "or export it before launching python."
        )
    return openai.OpenAI(base_url=BASE_URL, api_key=api_key,
                         timeout=timeout, max_retries=max_retries)


@dataclass
class Completion:
    """One model roll-out: text + token-level logprobs + bookkeeping."""
    text: str
    token_logprobs: list = field(default_factory=list)   # list[float]
    n_tokens: int = 0
    latency: float = 0.0
    error: Optional[str] = None

    @property
    def avg_logprob(self) -> float:
        """Mean per-token log P(token_t | context). 0.0 == maximally confident.

        Snell et al. style confidence proxy: exp(avg_logprob) is the geometric
        mean token probability.
        """
        if not self.token_logprobs:
            return 0.0
        return float(np.mean(self.token_logprobs))


def _extract(client, resp, want_logprobs: bool) -> Completion:
    ch = resp.choices[0]
    msg = ch.message
    text = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    lp = getattr(msg, "logprobs", None) or getattr(ch, "logprobs", None)
    toks = []
    if lp is not None and getattr(lp, "content", None):
        toks = [float(t.logprob) for t in lp.content if t.logprob is not None]
    n_tok = len(toks)
    if n_tok == 0:  # no logprobs returned -> use usage metadata if present
        usage = getattr(resp, "usage", None)
        if usage and getattr(usage, "completion_tokens", None):
            n_tok = int(usage.completion_tokens)
        else:
            n_tok = max(len(text.split()), 1)  # crude last-resort estimate
    err = None
    if want_logprobs and not toks:
        err = "no-logprobs-returned"
    full = (reasoning + "\n" + text).strip() if reasoning else text.strip()
    return Completion(text=full, token_logprobs=toks, n_tokens=n_tok)


def complete(client, prompt: str, model: str = DEFAULT_MODEL,
             temperature: float = 0.7, max_tokens: int = 512,
             logprobs: bool = True, system: Optional[str] = None,
             enable_thinking: bool = False, extra_body: Optional[dict] = None,
             max_attempts: int = 4) -> Completion:
    """Single chat completion with retries + exponential backoff."""
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    body = dict(extra_body or {})
    if not enable_thinking:
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens)
    if body:
        kwargs["extra_body"] = body
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 1
    # temperature==0 is greedy; some endpoints reject it explicitly -> clamp
    if temperature == 0:
        kwargs["temperature"] = 0.0

    last_err = None
    for attempt in range(max_attempts):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            c = _extract(client, resp, logprobs)
            c.latency = time.time() - t0
            if logprobs and c.error == "no-logprobs-returned":
                # retry once without logprobs so evaluation can still proceed
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)
                continue
            return c
        except Exception as e:  # noqa: BLE001 - rate limits / transient 5xx
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    return Completion(text="", error=f"{type(last_err).__name__}: {last_err}")


def complete_many(client, prompts, model=DEFAULT_MODEL, temperature=0.7,
                  max_tokens=512, logprobs=True, system=None,
                  enable_thinking=False, workers: int = 6):
    """Fan out many completions concurrently (NVIDIA free tier is serial-ish
    per connection, parallel connections make eval tractable)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(complete, client, p, model, temperature,
                             max_tokens, logprobs, system, enable_thinking)
                   for p in prompts]
        return [f.result() for f in futures]


def sample_n(client, prompt: str, n: int, **kw) -> list:
    """Best-of-N style sampling: n independent calls (endpoint forbids n>1
    together with temperature 0, and parallel calls are faster anyway)."""
    return complete_many(client, [prompt] * n, workers=min(n, 6), **kw)
