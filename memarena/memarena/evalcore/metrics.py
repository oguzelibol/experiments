"""Deterministic answer-matching metrics.

Every function takes a response string and one or more acceptable gold strings
and returns a score in [0, 1].  They are pure functions of their inputs, so a
third party can recompute any published score exactly.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

from .normalize import numbers_in, tokens

__all__ = [
    "exact_match",
    "contains",
    "token_f1",
    "rouge_l",
    "bleu1",
    "numeric_match",
    "list_recall",
    "token_recall",
]


def _golds(golds: str | Iterable[str]) -> list[str]:
    if isinstance(golds, str):
        return [golds]
    return [g for g in golds if g is not None]


def exact_match(pred: str, golds: str | Iterable[str]) -> float:
    p = tokens(pred)
    return 1.0 if any(p == tokens(g) and p for g in _golds(golds)) else 0.0


def _is_subsequence_contiguous(needle: Sequence[str], hay: Sequence[str]) -> bool:
    n, h = len(needle), len(hay)
    if n == 0 or n > h:
        return False
    first = needle[0]
    for i in range(h - n + 1):
        if hay[i] == first and tuple(hay[i : i + n]) == tuple(needle):
            return True
    return False


def contains(pred: str, golds: str | Iterable[str]) -> float:
    """1 if any gold appears in the response as a contiguous token span."""
    p = tokens(pred)
    return 1.0 if any(_is_subsequence_contiguous(tokens(g), p) for g in _golds(golds)) else 0.0


def _f1(p: Sequence[str], g: Sequence[str]) -> float:
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, golds: str | Iterable[str]) -> float:
    p = tokens(pred)
    return max((_f1(p, tokens(g)) for g in _golds(golds)), default=0.0)


def token_recall(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    return overlap / len(g)


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, golds: str | Iterable[str]) -> float:
    """ROUGE-L F-measure (LCS based), max over golds."""
    p = tokens(pred)
    best = 0.0
    for g in _golds(golds):
        gt = tokens(g)
        lcs = _lcs(p, gt)
        if lcs == 0:
            continue
        precision, recall = lcs / len(p), lcs / len(gt)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def bleu1(pred: str, golds: str | Iterable[str]) -> float:
    """Unigram BLEU with brevity penalty, max over golds."""
    p = tokens(pred)
    if not p:
        return 0.0
    best = 0.0
    pc = Counter(p)
    for g in _golds(golds):
        gt = tokens(g)
        if not gt:
            continue
        clipped = sum((pc & Counter(gt)).values())
        precision = clipped / len(p)
        bp = 1.0 if len(p) >= len(gt) else math.exp(1.0 - len(gt) / len(p))
        best = max(best, precision * bp)
    return best


def numeric_match(pred: str, value: float, tolerance: float = 0.0) -> bool:
    """True if any number in the response is within ``tolerance`` of ``value``."""
    return any(abs(x - value) <= tolerance + 1e-9 for x in numbers_in(pred))


def conflicting_numbers(pred: str, value: float, tolerance: float = 0.0) -> bool:
    """True if the response also commits to a number that is *not* acceptable."""
    return any(abs(x - value) > tolerance + 1e-9 for x in numbers_in(pred))


def list_recall(pred: str, items: Sequence[str]) -> float:
    """Fraction of required list items present in the response (order free)."""
    if not items:
        return 0.0
    p = tokens(pred)
    hit = sum(1 for it in items if _is_subsequence_contiguous(tokens(it), p))
    return hit / len(items)
