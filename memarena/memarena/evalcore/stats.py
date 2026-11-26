"""Statistics for benchmark scores: intervals, clustered errors, paired tests.

Questions that share one conversation history are not independent (LoCoMo has
~1,500 questions over only 10 conversations), so every interval here accepts a
cluster label and falls back to the i.i.d. formula only when each item is its
own cluster.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence

from ..util import derive_seed

__all__ = [
    "mean_se",
    "wilson_interval",
    "cluster_mean_se",
    "bootstrap_ci",
    "paired_bootstrap",
    "mcnemar_exact",
    "noise_floor",
    "normal_cdf",
    "normal_quantile",
]


def mean_se(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, float("inf")
    m = sum(xs) / n
    if n == 1:
        return m, float("inf")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def wilson_interval(k: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion; well behaved near 0 and 1."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cluster_mean_se(xs: Sequence[float], clusters: Sequence[str] | None = None) -> tuple[float, float]:
    """Mean and cluster-robust standard error (linearised ratio estimator)."""
    n = len(xs)
    if n == 0:
        return 0.0, float("inf")
    m = sum(xs) / n
    if clusters is None:
        return mean_se(xs)
    sums: dict[str, float] = {}
    for x, c in zip(xs, clusters):
        sums[c] = sums.get(c, 0.0) + (x - m)
    g = len(sums)
    if g < 2:
        return m, float("inf")
    var = (g / (g - 1)) * sum(s * s for s in sums.values()) / (n * n)
    return m, math.sqrt(var)


def _resample_indices(n: int, clusters: Sequence[str] | None, rng: random.Random) -> list[int]:
    if clusters is None:
        return [rng.randrange(n) for _ in range(n)]
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    names = sorted(groups)
    out: list[int] = []
    for _ in range(len(names)):
        out.extend(groups[names[rng.randrange(len(names))]])
    return out


def bootstrap_ci(
    xs: Sequence[float],
    clusters: Sequence[str] | None = None,
    stat: Callable[[Sequence[float]], float] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | str = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling whole clusters if given."""
    n = len(xs)
    if n == 0:
        return 0.0, 1.0
    stat = stat or (lambda v: sum(v) / len(v) if v else 0.0)
    rng = random.Random(derive_seed("bootstrap", seed))
    vals = []
    for _ in range(n_boot):
        idx = _resample_indices(n, clusters, rng)
        vals.append(stat([xs[i] for i in idx]))
    vals.sort()
    lo = vals[max(0, int(math.floor(alpha / 2 * n_boot)))]
    hi = vals[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return lo, hi


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[str] | None = None,
    n_boot: int = 2000,
    seed: int | str = 0,
) -> dict:
    """Paired comparison of two systems on the same items.

    Returns the mean difference (a - b), a 95% interval and the bootstrap
    probability that a is better.  Use this, not two separate intervals, to
    decide whether a change helped.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    n = len(a)
    if n == 0:
        return {"diff": 0.0, "ci": (0.0, 0.0), "p_better": 0.5, "n": 0}
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(derive_seed("paired", seed))
    diffs = []
    better = 0.0
    for _ in range(n_boot):
        idx = _resample_indices(n, clusters, rng)
        v = sum(d[i] for i in idx) / len(idx)
        diffs.append(v)
        better += 1.0 if v > 0 else 0.5 if v == 0 else 0.0
    diffs.sort()
    return {
        "diff": sum(d) / n,
        "ci": (diffs[int(0.025 * n_boot)], diffs[min(n_boot - 1, int(0.975 * n_boot))]),
        "p_better": better / n_boot,
        "n": n,
    }


def mcnemar_exact(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact two-sided McNemar p-value for paired binary outcomes."""
    n01 = sum(1 for x, y in zip(a, b) if x < 0.5 <= y)
    n10 = sum(1 for x, y in zip(a, b) if y < 0.5 <= x)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def noise_floor(n: int, p: float = 0.5, design_effect: float = 1.0) -> float:
    """Half-width of a 95% interval for a pass rate ``p`` on ``n`` items."""
    if n <= 0:
        return 1.0
    return 1.959964 * math.sqrt(design_effect * p * (1 - p) / n)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )
