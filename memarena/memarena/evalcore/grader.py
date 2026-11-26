"""The one grading path.

``Grader.grade`` is the only function in this project that turns answers into
scores.  Self-evaluation, peer evaluation, the simulated external evaluator and
audit re-computation all call it; the caller's identity is not an input.

Per item it computes every base scorer once, then derives the score under each
criterion in the family.  Rows whose answer never arrived (missing, error,
timeout, refusal) are scored zero in the headline, because that is what an
outside evaluator does with a missing answer, but they are counted and reported
separately so that plumbing failures are never mistaken for wrong answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..benchmark.schema import Answer, AnswerKey, Instance
from ..util import hash_obj, merkle_root
from . import metrics
from .criteria import JUDGE_SCORERS, STRING_SCORERS, CriteriaFamily, Criterion, aggregate
from .judges import HeuristicJudge, Judge
from .normalize import is_clean_abstention, tokens

__all__ = ["ItemGrade", "GradeReport", "Grader", "default_judges", "answers_hash"]

_NOT_ANSWERED = ("missing", "error", "timeout", "refused")


def default_judges() -> dict[str, Judge]:
    """Deterministic judges for every judge scorer (offline default)."""
    return {s: HeuristicJudge(s.split("_", 1)[1]) for s in JUDGE_SCORERS}


def answers_hash(answers: Mapping[str, Answer]) -> str:
    """Hash of exactly what gets graded: probe id -> (text, status)."""
    return hash_obj({pid: [a.text, a.status] for pid, a in sorted(answers.items())})


def _string_scores(response: str, golds: Sequence[str], key: AnswerKey) -> dict[str, float]:
    meta = key.meta or {}
    f1 = metrics.token_f1(response, golds)
    if "list" in meta:
        # order-free handling for list answers: exactly the required items, any order
        recall = metrics.list_recall(response, meta["list"])
        want = sorted(t for item in meta["list"] for t in tokens(item) if t != "and")
        got = sorted(t for t in tokens(response) if t != "and")
        em = 1.0 if want == got else 0.0
        contains = 1.0 if recall >= 1.0 else 0.0
    else:
        em = metrics.exact_match(response, golds)
        contains = metrics.contains(response, golds)
    return {
        "em": em,
        "contains": contains,
        "f1": f1,
        "f1_t50": 1.0 if f1 >= 0.5 else 0.0,
        "rouge_l": metrics.rouge_l(response, golds),
        "bleu1": metrics.bleu1(response, golds),
    }


@dataclass(frozen=True)
class ItemGrade:
    probe_id: str
    haystack_id: str
    question_type: str
    is_abstention: bool
    status: str  # answer status, or "missing"
    abstained: bool
    base: Mapping[str, float]  # scorer -> score treating the item as answerable
    literal: Mapping[str, float]  # string scorer -> score against the explanation text
    judge_errors: tuple[str, ...] = ()

    def score(self, criterion: Criterion) -> tuple[bool, float]:
        """(included, score) of this item under ``criterion``."""
        if self.is_abstention:
            if criterion.abstention == "exclude":
                return False, 0.0
            if criterion.abstention == "literal":
                return True, self.literal.get(criterion.scorer, 0.0)
            # detect
            if criterion.scorer in JUDGE_SCORERS:
                return True, self.base[criterion.scorer]
            return True, 1.0 if self.abstained else 0.0
        return True, self.base[criterion.scorer]

    def record(self) -> list:
        """Canonical, hashable record of this item's grading."""
        return [
            self.probe_id,
            self.status,
            int(self.abstained),
            {k: self.base[k] for k in sorted(self.base)},
            {k: self.literal[k] for k in sorted(self.literal)},
        ]


@dataclass
class GradeReport:
    family: CriteriaFamily
    items: list[ItemGrade]
    scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.scores:
            self.scores = {c.id: self._aggregate(c) for c in self.family}

    def _aggregate(self, criterion: Criterion) -> float:
        return aggregate(criterion, ((it.question_type, *it.score(criterion)) for it in self.items))

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_not_answered(self) -> int:
        return sum(1 for it in self.items if it.status in _NOT_ANSWERED)

    @property
    def n_truncated(self) -> int:
        return sum(1 for it in self.items if it.status == "truncated")

    @property
    def n_judge_errors(self) -> int:
        return sum(1 for it in self.items if it.judge_errors)

    def score(self, criterion_id: str) -> float:
        return self.scores[criterion_id]

    def mixture(self, weights: Mapping[str, float]) -> float:
        total = sum(weights.get(cid, 0.0) for cid in self.scores)
        if total <= 0:
            raise ValueError("weights have no mass on this family")
        return sum(weights.get(cid, 0.0) * s for cid, s in self.scores.items()) / total

    def item_scores(self, criterion_id: str) -> list[tuple[ItemGrade, bool, float]]:
        c = self.family.get(criterion_id)
        return [(it, *it.score(c)) for it in self.items]

    def per_type(self, criterion_id: str) -> dict[str, tuple[float, int]]:
        acc: dict[str, list[float]] = {}
        for it, included, s in self.item_scores(criterion_id):
            if included:
                a = acc.setdefault(it.question_type, [0.0, 0.0])
                a[0] += s
                a[1] += 1
        return {t: (a[0] / a[1], int(a[1])) for t, a in sorted(acc.items())}

    def answered_only(self) -> "GradeReport":
        """Same report restricted to rows that actually produced an answer."""
        return GradeReport(self.family, [it for it in self.items if it.status not in _NOT_ANSWERED])

    def items_root(self) -> str:
        return merkle_root([hash_obj(it.record()) for it in self.items])

    def summary(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_not_answered": self.n_not_answered,
            "n_truncated": self.n_truncated,
            "n_judge_errors": self.n_judge_errors,
            "scores": dict(self.scores),
        }


class Grader:
    def __init__(self, family: CriteriaFamily, judges: Mapping[str, Judge] | None = None):
        self.family = family
        self.scorers = family.scorers()
        judges = dict(judges) if judges is not None else default_judges()
        self.judges: dict[str, Judge] = {}
        for s in self.scorers:
            if s in JUDGE_SCORERS:
                if s not in judges:
                    raise ValueError(f"no judge supplied for scorer {s!r}")
                self.judges[s] = judges[s]

    def spec(self) -> dict:
        return {
            "family": self.family.spec(),
            "judges": {s: j.spec() for s, j in sorted(self.judges.items())},
        }

    def grade_item(self, instance: Instance, answer: Answer | None) -> ItemGrade:
        key, probe = instance.key, instance.probe
        if answer is None:
            status, text = "missing", ""
        else:
            status, text = answer.status, answer.text or ""
        if status in _NOT_ANSWERED:
            text = ""  # nothing to grade; never score an error message as an answer
        golds = (key.answer,) + tuple(key.aliases)
        abstained = is_clean_abstention(text)
        base: dict[str, float] = {}
        literal: dict[str, float] = {}
        errors: list[str] = []
        need_string = any(s in STRING_SCORERS for s in self.scorers)
        if need_string:
            if key.is_abstention:
                zeros = {s: 0.0 for s in STRING_SCORERS}
                base.update(zeros)
                literal.update(_string_scores(text, (key.answer,), key))
            else:
                base.update(_string_scores(text, golds, key))
        for s, judge in self.judges.items():
            if status in _NOT_ANSWERED:
                base[s] = 0.0
                continue
            verdict = judge.judge(probe.question, key, text)
            if verdict.status != "ok":
                errors.append(s)
                base[s] = 0.0
            else:
                base[s] = float(verdict.score)
        base = {s: base.get(s, 0.0) for s in self.scorers}
        return ItemGrade(
            probe_id=probe.probe_id,
            haystack_id=probe.haystack_id,
            question_type=key.question_type,
            is_abstention=key.is_abstention,
            status=status,
            abstained=abstained,
            base=base,
            literal={s: literal.get(s, 0.0) for s in STRING_SCORERS if s in self.scorers} if key.is_abstention else {},
            judge_errors=tuple(errors),
        )

    def grade(self, answers: Mapping[str, Answer], instances: Sequence[Instance]) -> GradeReport:
        """Grade ``answers`` (probe id -> Answer) against labelled ``instances``."""
        return GradeReport(self.family, [self.grade_item(inst, answers.get(inst.probe_id)) for inst in instances])
