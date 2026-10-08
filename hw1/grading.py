"""Answer extraction + grading utilities shared by HW1 Ex1 and the AVR agent.

Deterministic, regex-based verification of math answers (no grading LLM needed
for GSM8K-style data): we normalise to a canonical numeric/fraction string.
"""

from __future__ import annotations

import re

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")

# Phrases that usually immediately precede the final answer in CoT output.
_FINAL_MARKERS = [
    r"final answer\s*(?:is|:|=)?",
    r"answer\s*(?:is|:|=)",
    r"the answer is",
    r"therefore,? the result is",
    r"we get\s*:?",
    r"result\s*(?:is|:|=)",
    r"####\s*",
]


def _norm(num: str) -> str:
    num = num.replace(",", "").strip()
    # integer-valued float -> int  ("42.0" -> "42")
    if "." in num:
        try:
            f = float(num)
            if abs(f - round(f)) < 1e-9:
                num = str(int(round(f)))
        except ValueError:
            pass
    return num


def extract_answer(text: str) -> str | None:
    """Pull the final numerical answer out of a model roll-out.

    Strategy (in order):
      1. GSM8K-style ``#### 42`` marker.
      2. Last sentence containing an explicit 'answer/final/therefore' marker.
      3. Fallback: the LAST number appearing anywhere in the text.
    Returns a normalised numeric string or None.
    """
    if not text:
        return None
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return _norm(m.group(1))

    for marker in _FINAL_MARKERS:
        ms = list(re.finditer(marker + r"\s*\$?(-?[\d,]+(?:\.\d+)?(?:/\d+)?)",
                              text, flags=re.IGNORECASE))
        if ms:
            return _norm(ms[-1].group(1))

    nums = _NUM_RE.findall(text)
    if nums:
        return _norm(nums[-1])
    return None


def answers_equal(a: str | None, b: str | None) -> bool:
    """Canonical numeric equality (handles 42 == 42.0 == '42,')."""
    if a is None or b is None:
        return False
    a, b = _norm(a), _norm(b)
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))
    except ValueError:
        return False


def gold_from_gsm8k(answer_field: str) -> str | None:
    """GSM8K's ``answer`` field ends with '#### <number>'."""
    m = re.search(r"####\s*(-?[\d,\.]+)", answer_field)
    return _norm(m.group(1)) if m else None
