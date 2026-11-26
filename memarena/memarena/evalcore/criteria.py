"""The space of scoring policies an evaluator might be using.

A ``Criterion`` is a complete recipe for turning (question, key, response)
triples into one benchmark score: an item scorer, a policy for abstention
(unanswerable) questions, and an aggregation rule.  ``CriteriaFamily`` is the
finite set of such recipes this module can compute, with a prior over them.

The family plays two roles.  It is the unit's own definition of "score" (a
weighted mixture of criteria), and it is the hypothesis space used to infer
which recipe an external evaluator applies when only its final numbers are
observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..util import hash_obj

STRING_SCORERS = ("em", "contains", "f1", "f1_t50", "rouge_l", "bleu1")
# judge_<mode> for each mode in judges.JUDGE_MODES
JUDGE_SCORERS = ("judge_lenient", "judge_strict", "judge_generous", "judge_graded")
SCORERS = STRING_SCORERS + JUDGE_SCORERS
ABSTENTION_POLICIES = ("detect", "exclude", "literal")
AGGREGATIONS = ("micro", "macro")

# Priors reflect how often each choice appears in published long-term-memory
# evaluations (see docs/BENCHMARKS.md): LongMemEval's official binary judge is
# the default for this setting; "generous" judges are what several vendors use
# on LoCoMo; rubric-style partial credit is used by BEAM; token-F1 is LoCoMo's
# original metric.  Unanswerable items are scored by most LongMemEval reports
# but skipped by most LoCoMo reports.  Most headlines are a single overall
# (micro) average; some are task-averaged (macro).
SCORER_PRIOR = {
    "judge_lenient": 0.30,
    "judge_generous": 0.12,
    "judge_strict": 0.10,
    "judge_graded": 0.08,
    "f1": 0.12,
    "contains": 0.08,
    "em": 0.06,
    "rouge_l": 0.05,
    "bleu1": 0.05,
    "f1_t50": 0.04,
}
ABSTENTION_PRIOR = {"detect": 0.55, "exclude": 0.35, "literal": 0.10}
AGGREGATION_PRIOR = {"micro": 0.70, "macro": 0.30}


@dataclass(frozen=True)
class Criterion:
    scorer: str
    abstention: str
    aggregation: str

    def __post_init__(self):
        if self.scorer not in SCORERS:
            raise ValueError(f"unknown scorer {self.scorer!r}")
        if self.abstention not in ABSTENTION_POLICIES:
            raise ValueError(f"unknown abstention policy {self.abstention!r}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {self.aggregation!r}")
        if self.scorer in JUDGE_SCORERS and self.abstention == "literal":
            raise ValueError("a judge has no 'literal' abstention policy")

    @property
    def id(self) -> str:
        return f"{self.scorer}|{self.abstention}|{self.aggregation}"

    @staticmethod
    def from_id(cid: str) -> "Criterion":
        scorer, abstention, aggregation = cid.split("|")
        return Criterion(scorer, abstention, aggregation)


def aggregate(
    criterion: Criterion,
    rows: Iterable[tuple[str, bool, float]],
) -> float:
    """Combine item scores. ``rows`` yields (question_type, included, score)."""
    if criterion.aggregation == "micro":
        total, n = 0.0, 0
        for _qtype, included, score in rows:
            if included:
                total += score
                n += 1
        return total / n if n else 0.0
    sums: dict[str, list[float]] = {}
    for qtype, included, score in rows:
        if included:
            acc = sums.setdefault(qtype, [0.0, 0.0])
            acc[0] += score
            acc[1] += 1.0
    if not sums:
        return 0.0
    return sum(s / n for s, n in sums.values()) / len(sums)


class CriteriaFamily:
    def __init__(self, criteria: Sequence[Criterion], prior: Mapping[str, float], name: str):
        if not criteria:
            raise ValueError("empty criteria family")
        ids = [c.id for c in criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate criteria")
        self.name = name
        self.criteria: tuple[Criterion, ...] = tuple(criteria)
        self.ids: tuple[str, ...] = tuple(ids)
        total = sum(max(0.0, prior.get(i, 0.0)) for i in ids)
        if total <= 0:
            raise ValueError("prior has no mass on the family")
        self.prior: dict[str, float] = {i: max(0.0, prior.get(i, 0.0)) / total for i in ids}
        self._by_id = {c.id: c for c in self.criteria}

    def __len__(self) -> int:
        return len(self.criteria)

    def __iter__(self):
        return iter(self.criteria)

    def get(self, cid: str) -> Criterion:
        return self._by_id[cid]

    def scorers(self) -> tuple[str, ...]:
        return tuple(s for s in SCORERS if any(c.scorer == s for c in self.criteria))

    def spec(self) -> dict:
        return {"name": self.name, "ids": list(self.ids), "prior_hash": hash_obj(self.prior)}

    @staticmethod
    def default(prior: str = "literature") -> "CriteriaFamily":
        criteria: list[Criterion] = []
        weights: dict[str, float] = {}
        for scorer in SCORERS:
            for abstention in ABSTENTION_POLICIES:
                if scorer in JUDGE_SCORERS and abstention == "literal":
                    continue
                for aggregation in AGGREGATIONS:
                    c = Criterion(scorer, abstention, aggregation)
                    criteria.append(c)
                    if prior == "uniform":
                        weights[c.id] = 1.0
                    elif prior == "literature":
                        weights[c.id] = (
                            SCORER_PRIOR[scorer]
                            * ABSTENTION_PRIOR[abstention]
                            * AGGREGATION_PRIOR[aggregation]
                        )
                    else:
                        raise ValueError(f"unknown prior {prior!r}")
        return CriteriaFamily(criteria, weights, name=f"default-v1:{prior}")

    @staticmethod
    def single(cid: str) -> "CriteriaFamily":
        c = Criterion.from_id(cid)
        return CriteriaFamily([c], {c.id: 1.0}, name=f"single:{cid}")
