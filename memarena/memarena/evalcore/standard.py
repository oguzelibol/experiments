"""A unit's evaluation standard: what it believes good performance is.

Three ingredients:

* ``criterion_weights``  how a response is scored (a mixture over the criteria
  family: which judge or string metric, how unanswerable questions count, how
  item scores are averaged),
* ``capability_weights`` which abilities its exams probe.  A unit examines
  others on the abilities it believes itself to be good at,
* ``exam``               how hard its exams are (history length, paraphrase
  distance, distractor rates).

A standard is published on the unit's feed and hashed into every receipt, so
anyone can re-run an evaluation exactly as the examiner claims to have run it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..benchmark.schema import QUESTION_TYPES, AnswerKey
from ..benchmark.synthetic import GenConfig
from ..util import hash_obj

__all__ = ["CAPABILITIES", "capability_of", "Standard", "normalise", "canonical_weights", "MIN_SESSIONS", "MAX_SESSIONS"]

# Seven abilities: LongMemEval's six question types plus knowing what was never said.
CAPABILITIES: tuple[str, ...] = QUESTION_TYPES + ("abstention",)

# Bounds on the length of an exam history that a published standard may ask for.
MIN_SESSIONS, MAX_SESSIONS = 4, 64

_EXAM_FIELDS = (
    "sessions_per_stream", "hard_paraphrase_rate", "third_party_rate", "negation_rate",
    "update_rate", "echo_rate", "relative_date_rate",
)


def capability_of(key: AnswerKey) -> str:
    return "abstention" if key.is_abstention else key.question_type


def normalise(weights: Mapping[str, float]) -> dict[str, float]:
    clean = {k: float(v) for k, v in weights.items() if v and v > 0}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("weights must have positive mass")
    return {k: v / total for k, v in sorted(clean.items())}


def canonical_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Weights in the one form that survives publication unchanged.

    Posts store numbers to six decimals.  A standard must hash the same before
    and after it is published, otherwise an honest examiner's receipt would
    name a different standard from the one on its feed.  So weights are
    normalised once, rounded to six decimals, and never re-normalised when they
    already sum to one within rounding error.
    """
    clean = {}
    for k, v in weights.items():
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("weights must be finite")
        if v > 0:
            clean[str(k)] = v
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("weights must have positive mass")
    if abs(total - 1.0) > 1e-4:
        clean = {k: v / total for k, v in clean.items()}
    out = {k: round(v, 6) for k, v in sorted(clean.items())}
    out = {k: v for k, v in out.items() if v > 0}
    if not out:
        raise ValueError("weights vanish at six decimals")
    return out


@dataclass(frozen=True)
class Standard:
    criterion_weights: Mapping[str, float]
    capability_weights: Mapping[str, float] = field(
        default_factory=lambda: {c: 1.0 / len(CAPABILITIES) for c in CAPABILITIES}
    )
    exam: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "criterion_weights", canonical_weights(self.criterion_weights))
        caps = canonical_weights(self.capability_weights)
        unknown = set(caps) - set(CAPABILITIES)
        if unknown:
            raise ValueError(f"unknown capabilities: {sorted(unknown)}")
        object.__setattr__(self, "capability_weights", caps)
        if not isinstance(self.exam, Mapping):
            raise ValueError("exam settings must be a mapping")
        exam = {}
        for k in _EXAM_FIELDS:
            if k in self.exam:
                v = self.exam[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                    raise ValueError(f"exam setting {k} must be a finite number")
                if k == "sessions_per_stream":
                    # Below the minimum the generator cannot place every statement its keys rely on;
                    # above the maximum a verifier would be made to build an arbitrarily large exam.
                    if int(v) != v or not MIN_SESSIONS <= int(v) <= MAX_SESSIONS:
                        raise ValueError(f"sessions_per_stream must be a whole number from {MIN_SESSIONS} to {MAX_SESSIONS}")
                    exam[k] = int(v)
                else:
                    if not 0.0 <= float(v) <= 1.0:
                        raise ValueError(f"exam setting {k} must lie between 0 and 1")
                    exam[k] = round(float(v), 6)
        object.__setattr__(self, "exam", exam)

    def to_dict(self) -> dict:
        return {
            "criterion_weights": dict(self.criterion_weights),
            "capability_weights": dict(self.capability_weights),
            "exam": dict(self.exam),
        }

    @staticmethod
    def from_dict(d: Mapping) -> "Standard":
        if not isinstance(d, Mapping) or not isinstance(d.get("criterion_weights"), Mapping):
            raise ValueError("a standard needs criterion weights")
        if not isinstance(d.get("capability_weights") or {}, Mapping):
            raise ValueError("capability weights must be a mapping")
        return Standard(
            criterion_weights=d["criterion_weights"],
            capability_weights=d.get("capability_weights") or {c: 1.0 for c in CAPABILITIES},
            exam=d.get("exam", {}),
        )

    def fingerprint(self) -> str:
        return hash_obj(self.to_dict())[:24]

    def gen_config(self, n_streams: int, questions_per_stream: int, balanced: bool = False) -> GenConfig:
        """Generator settings for an exam set by this standard.

        ``balanced=True`` ignores the capability focus and covers every ability
        evenly; a unit uses that to measure its own full profile.
        """
        caps = {c: 1.0 / len(CAPABILITIES) for c in CAPABILITIES} if balanced else dict(self.capability_weights)
        abst = caps.get("abstention", 0.0)
        type_mix = {t: caps.get(t, 0.0) for t in QUESTION_TYPES}
        if sum(type_mix.values()) <= 0:
            # an abstention-only focus still needs question types to hang the items on
            type_mix = {t: 1.0 for t in QUESTION_TYPES}
        kwargs = dict(self.exam)
        if "sessions_per_stream" in kwargs:
            kwargs["sessions_per_stream"] = int(kwargs["sessions_per_stream"])
        return GenConfig(
            n_streams=n_streams, questions_per_stream=questions_per_stream,
            type_mix=type_mix, abstention_rate=min(1.0, abst), **kwargs,
        )
