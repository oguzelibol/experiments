"""How far a claimed score may sit from a measured one, and what to claim.

A unit states what it expects to score before it is graded.  The gap between
the claim and the measurement is charged with an asymmetric power loss:

    loss = over  * (claim - truth) ** power    if claim > truth
           under * (truth - claim) ** power    otherwise

Two instances are used.  Between peers, over-claiming costs more than
under-claiming (``PEER_LOSS``); against the outside benchmark, over-estimating
costs less (``EXTERNAL_LOSS``).

Under an asymmetric loss the best single number to report is not the mean of
one's belief.  ``best_report`` returns the report that minimises expected loss
for a normal belief; for power 2 that is an expectile.  Units publish both the
mean and the report, so shading a report is transparent and never changes the
underlying measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .stats import normal_quantile

__all__ = ["AsymmetricLoss", "PEER_LOSS", "EXTERNAL_LOSS"]

# Standard-normal quantiles used as a fixed quadrature grid.
_GRID = [normal_quantile((i + 0.5) / 41) for i in range(41)]


@dataclass(frozen=True)
class AsymmetricLoss:
    over: float
    under: float
    power: float = 2.0

    def __post_init__(self):
        if self.over < 0 or self.under < 0 or self.power < 1:
            raise ValueError("loss coefficients must be non-negative and power >= 1")

    def __call__(self, claim: float, truth: float) -> float:
        d = claim - truth
        return (self.over if d > 0 else self.under) * abs(d) ** self.power

    def to_dict(self) -> dict:
        return {"over": self.over, "under": self.under, "power": self.power}

    @staticmethod
    def from_dict(d: Mapping) -> "AsymmetricLoss":
        return AsymmetricLoss(float(d["over"]), float(d["under"]), float(d.get("power", 2.0)))

    def expected(self, claim: float, mean: float, sd: float) -> float:
        """Expected loss of ``claim`` when the truth is N(mean, sd), clipped to [0, 1]."""
        if sd <= 0:
            return self(claim, min(1.0, max(0.0, mean)))
        return sum(self(claim, min(1.0, max(0.0, mean + sd * z))) for z in _GRID) / len(_GRID)

    def best_report(self, mean: float, sd: float, shade: float = 1.0) -> float:
        """Report minimising expected loss; ``shade`` in [0, 1] interpolates from the mean."""
        mean = min(1.0, max(0.0, mean))
        if sd <= 0 or shade <= 0:
            return mean
        lo, hi = 0.0, 1.0
        # expected loss is convex in the claim: ternary search
        for _ in range(60):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            if self.expected(m1, mean, sd) <= self.expected(m2, mean, sd):
                hi = m2
            else:
                lo = m1
        best = (lo + hi) / 2
        return min(1.0, max(0.0, mean + shade * (best - mean)))


PEER_LOSS = AsymmetricLoss(over=1.0, under=0.25)
EXTERNAL_LOSS = AsymmetricLoss(over=0.25, under=1.0)


def binomial_sd(p: float, n: float) -> float:
    if n <= 0:
        return 0.5
    p = min(1.0, max(0.0, p))
    return math.sqrt(max(p * (1 - p), 1e-4) / n)
