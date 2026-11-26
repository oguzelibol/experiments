"""The single evaluation module a unit owns.

One instance, one standard, one grading path.  ``evaluate`` takes answers and
labelled instances and nothing else: there is no parameter through which the
identity of whoever produced the answers could influence the result.  The unit
calls it on its own answers to form its beliefs about itself and on other
units' answers to examine them.

Every evaluation comes with a ``receipt``: the evaluator fingerprint (grading
code hash + judges + standard), a hash of the answers, a commitment to the
answer keys, a Merkle root over the per-item grading records, and the scores.
Once the examiner reveals the keys, anyone can call ``verify_receipt`` and
recompute the result bit for bit.  A grade that was nudged up or down for a
particular examinee does not survive that check.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

from ..benchmark.schema import QUESTION_TYPES, Answer, Instance
from ..util import canonical_json, hash_obj, sha256_hex
from .criteria import CriteriaFamily
from .grader import GradeReport, Grader, answers_hash
from .judges import Judge
from .standard import CAPABILITIES, Standard

__all__ = [
    "EvaluationModule",
    "Evaluation",
    "CapabilityTable",
    "code_hash",
    "keys_commitment",
    "recompute_receipt",
    "verify_receipt",
    "expected_score",
]

_CODE_FILES = (
    "normalize.py", "metrics.py", "judges.py", "criteria.py", "grader.py",
    "standard.py", "calibration.py", "module.py",
)


@lru_cache(maxsize=1)
def code_hash() -> str:
    """Hash of the grading source code; part of every evaluator fingerprint."""
    here = os.path.dirname(os.path.abspath(__file__))
    parts = []
    for name in _CODE_FILES:
        with open(os.path.join(here, name), "rb") as fh:
            parts.append(name + ":" + sha256_hex(fh.read()))
    return sha256_hex("|".join(parts))[:24]


def keys_commitment(instances: Sequence[Instance], salt: str) -> str:
    return sha256_hex("keys:" + salt + ":" + canonical_json([i.key.to_dict() for i in instances]))


@dataclass(frozen=True)
class CapabilityTable:
    """Mean score per capability and criterion for one set of graded answers."""

    cells: Mapping[str, Mapping[str, float]]  # capability -> criterion id -> mean
    counts: Mapping[str, int]  # capability -> number of items

    @staticmethod
    def from_report(report: GradeReport) -> "CapabilityTable":
        sums: dict[str, dict[str, list[float]]] = {}
        counts: dict[str, int] = {}
        for it in report.items:
            cap = "abstention" if it.is_abstention else it.question_type
            counts[cap] = counts.get(cap, 0) + 1
            row = sums.setdefault(cap, {})
            for c in report.family:
                included, s = it.score(c)
                if included:
                    acc = row.setdefault(c.id, [0.0, 0.0])
                    acc[0] += s
                    acc[1] += 1.0
        cells = {cap: {cid: a[0] / a[1] for cid, a in row.items()} for cap, row in sums.items()}
        return CapabilityTable(cells, counts)

    def to_dict(self) -> dict:
        return {"cells": {k: dict(v) for k, v in self.cells.items()}, "counts": dict(self.counts)}

    @staticmethod
    def from_dict(d: Mapping) -> "CapabilityTable":
        return CapabilityTable({k: dict(v) for k, v in d["cells"].items()}, {k: int(v) for k, v in d["counts"].items()})

    def capability_scores(self, criterion_weights: Mapping[str, float]) -> dict[str, tuple[float, int]]:
        """Score per capability under a criterion mixture: capability -> (score, n)."""
        out: dict[str, tuple[float, int]] = {}
        for cap, row in self.cells.items():
            num = den = 0.0
            for cid, w in criterion_weights.items():
                if cid in row:
                    num += w * row[cid]
                    den += w
            if den > 0:
                out[cap] = (num / den, int(self.counts.get(cap, 0)))
        return out

    def merged(self, other: "CapabilityTable") -> "CapabilityTable":
        cells: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        for cap in set(self.cells) | set(other.cells):
            n1, n2 = self.counts.get(cap, 0), other.counts.get(cap, 0)
            counts[cap] = n1 + n2
            row: dict[str, float] = {}
            r1, r2 = self.cells.get(cap, {}), other.cells.get(cap, {})
            for cid in set(r1) | set(r2):
                if cid in r1 and cid in r2 and n1 + n2 > 0:
                    row[cid] = (n1 * r1[cid] + n2 * r2[cid]) / (n1 + n2)
                else:
                    row[cid] = r1.get(cid, r2.get(cid, 0.0))
            cells[cap] = row
        return CapabilityTable(cells, counts)


def expected_score(
    table: CapabilityTable,
    criterion_weights: Mapping[str, float],
    capability_weights: Mapping[str, float],
) -> float:
    """Score this profile should earn on an exam with the given standard.

    Mirrors how ``GradeReport.mixture`` aggregates: per criterion, a micro
    average weights each capability by its share of the exam, a macro average
    gives each question type equal weight; unanswerable items are folded in or
    left out according to the criterion's abstention policy.
    """
    total_mass = sum(max(0.0, float(v)) for v in capability_weights.values())
    if total_mass <= 0:
        return 0.0
    # the abstention weight is a share of the exam, so it only means something after normalising
    abst_share = max(0.0, float(capability_weights.get("abstention", 0.0))) / total_mass
    type_mass = sum(max(0.0, float(capability_weights.get(t, 0.0))) for t in QUESTION_TYPES)
    type_mix = (
        {t: capability_weights.get(t, 0.0) / type_mass for t in QUESTION_TYPES}
        if type_mass > 0 else {t: 1.0 / len(QUESTION_TYPES) for t in QUESTION_TYPES}
    )
    total = wsum = 0.0
    for cid, w in criterion_weights.items():
        if w <= 0:
            continue
        aggregation = cid.rsplit("|", 1)[1]
        abst_cell = table.cells.get("abstention", {}).get(cid)
        per_type: dict[str, float] = {}
        for t in QUESTION_TYPES:
            if type_mix[t] <= 0:
                continue
            cell = table.cells.get(t, {}).get(cid)
            if cell is None:
                continue  # no evidence about this type under this criterion
            if abst_cell is None or abst_share <= 0:
                per_type[t] = cell
            else:
                per_type[t] = (1 - abst_share) * cell + abst_share * abst_cell
        if not per_type:
            continue
        if aggregation == "macro":
            pred = sum(per_type.values()) / len(per_type)
        else:
            mass = sum(type_mix[t] for t in per_type)
            pred = sum(type_mix[t] * v for t, v in per_type.items()) / mass
        total += w * pred
        wsum += w
    return total / wsum if wsum > 0 else 0.0


@dataclass(frozen=True)
class Evaluation:
    report: GradeReport
    score: float
    capability: Mapping[str, tuple[float, int]]
    table: CapabilityTable
    receipt: Mapping


class EvaluationModule:
    def __init__(
        self,
        standard: Standard,
        family: CriteriaFamily | None = None,
        judges: Mapping[str, Judge] | None = None,
    ):
        self.family = family or CriteriaFamily.default()
        unknown = set(standard.criterion_weights) - set(self.family.ids)
        if unknown:
            raise ValueError(f"standard uses criteria outside the family: {sorted(unknown)}")
        self.grader = Grader(self.family, judges)
        self.standard = standard

    def with_standard(self, standard: Standard) -> "EvaluationModule":
        """A new module version with a different standard (same grading code and judges)."""
        return EvaluationModule(standard, self.family, self.grader.judges)

    def fingerprint(self) -> str:
        return hash_obj(
            {"code": code_hash(), "grader": self.grader.spec(), "standard": self.standard.fingerprint()}
        )[:24]

    def grade(self, answers: Mapping[str, Answer], instances: Sequence[Instance]) -> GradeReport:
        return self.grader.grade(answers, instances)

    def evaluate(self, answers: Mapping[str, Answer], instances: Sequence[Instance], salt: str) -> Evaluation:
        """Grade ``answers`` and issue a receipt.  Identical for self and others."""
        report = self.grade(answers, instances)
        table = CapabilityTable.from_report(report)
        weights = self.standard.criterion_weights
        score = report.mixture(weights)
        capability = table.capability_scores(weights)
        receipt = {
            "evaluator": self.fingerprint(),
            "standard": self.standard.to_dict(),
            "subject": answers_hash({i.probe_id: answers[i.probe_id] for i in instances if i.probe_id in answers}),
            "keys_commit": keys_commitment(instances, salt),
            "n_items": report.n_items,
            "not_answered": report.n_not_answered,
            "items_root": report.items_root(),
            "score": score,
            "capability": {c: [s, n] for c, (s, n) in sorted(capability.items())},
        }
        return Evaluation(report, score, capability, table, receipt)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def recompute_receipt(
    receipt: Mapping,
    answers: Mapping[str, Answer],
    instances: Sequence[Instance],
    salt: str,
    family: CriteriaFamily | None = None,
    judges: Mapping[str, Judge] | None = None,
    tolerance: float = 1e-6,
) -> tuple[bool, str, dict]:
    """Recompute an evaluation from keys and answers and compare it with its receipt.

    Returns (ok, reason, recomputed receipt).  Every field of the receipt is
    compared; a number that is NaN or infinite never compares equal, so it
    cannot be used to switch a check off.  Callers should use the recomputed
    values afterwards, not the claimed ones.
    """
    try:
        standard = Standard.from_dict(receipt["standard"])
        module = EvaluationModule(standard, family, judges)
    except (KeyError, ValueError, TypeError) as exc:
        return False, f"malformed receipt: {exc}", {}
    if keys_commitment(instances, salt) != receipt.get("keys_commit"):
        return False, "revealed keys do not match the commitment", {}
    fresh = dict(module.evaluate(answers, instances, salt).receipt)
    if fresh["evaluator"] != receipt.get("evaluator"):
        return False, "evaluator fingerprint mismatch (different grading code, judges or standard)", fresh
    if fresh["subject"] != receipt.get("subject"):
        return False, "answers differ from the ones that were graded", fresh
    if fresh["items_root"] != receipt.get("items_root"):
        return False, "per-item grading records differ", fresh
    if receipt.get("n_items") != fresh["n_items"] or receipt.get("not_answered") != fresh["not_answered"]:
        return False, "item counts differ", fresh
    claimed_score = receipt.get("score")
    if not _finite(claimed_score) or abs(fresh["score"] - float(claimed_score)) > tolerance:
        return False, f"score differs: recomputed {fresh['score']:.6f}, claimed {claimed_score!r}", fresh
    claimed_cap = receipt.get("capability")
    if not isinstance(claimed_cap, Mapping) or set(claimed_cap) != set(fresh["capability"]):
        return False, "receipt does not list exactly the abilities that were examined", fresh
    for cap, (s, n) in fresh["capability"].items():
        got = claimed_cap[cap]
        if (
            not isinstance(got, (list, tuple)) or len(got) != 2 or not _finite(got[0])
            or abs(float(got[0]) - s) > tolerance or got[1] != n
        ):
            return False, f"capability score differs for {cap}", fresh
    return True, "ok", fresh


def verify_receipt(
    receipt: Mapping,
    answers: Mapping[str, Answer],
    instances: Sequence[Instance],
    salt: str,
    family: CriteriaFamily | None = None,
    judges: Mapping[str, Judge] | None = None,
    tolerance: float = 1e-6,
) -> tuple[bool, str]:
    """(ok, reason): does this receipt recompute from these keys and answers?"""
    ok, why, _fresh = recompute_receipt(receipt, answers, instances, salt, family, judges, tolerance)
    return ok, why
