"""Dishonest unit archetypes.  They exist only to test the society's defences.

A designer never releases these.  They are placed in test beds to check three
claims: a manipulated grade is caught by re-computation, a collusion ring
cannot raise its standing with units that examine it themselves, and
over-claiming costs esteem.
"""

from __future__ import annotations

from typing import Sequence

from ..actors.unit import Unit
from ..benchmark.schema import Answer, Probe
from ..memory.base import ABSTAIN_TEXT

__all__ = ["Deflator", "Colluder", "Rigger", "Braggart", "UNIT_KINDS"]


class Deflator(Unit):
    """Marks down everyone outside its circle, hoping to look better by comparison."""

    AMOUNT = 0.15

    def _result_payload(self, examinee: str, evaluation) -> dict:
        payload = super()._result_payload(examinee, evaluation)
        if examinee in self.spec.allies:
            return payload
        payload["score"] = max(0.0, payload["score"] - self.AMOUNT)
        payload["capability"] = {c: [max(0.0, s - self.AMOUNT), n] for c, (s, n) in payload["capability"].items()}
        receipt = dict(payload["receipt"])
        receipt["score"] = payload["score"]
        receipt["capability"] = payload["capability"]
        payload["receipt"] = receipt  # internally consistent, but it will not recompute
        return payload


class Colluder(Unit):
    """Shares its answer keys with allies out of band, so they ace each other's exams."""

    def __init__(self, *args, shared: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ring = shared if shared is not None else {}

    def _on_request(self, post) -> None:
        super()._on_request(post)
        st = self._set.get(post.body.get("xid"))
        if st is not None and post.author in self.spec.allies and "exam" in st:
            self._ring[post.body["xid"]] = st["exam"].keys()

    def _answer_exam(self, probes: Sequence[Probe], examiner: str, xid: str) -> dict[str, Answer]:
        keys = self._ring.get(xid) if examiner in self.spec.allies else None
        if keys is None:
            return super()._answer_exam(probes, examiner, xid)
        return {
            p.probe_id: Answer(ABSTAIN_TEXT if keys[p.probe_id].is_abstention else keys[p.probe_id].answer, 0.99)
            for p in probes
        }

    def _make_guess(self, mean: float, sd: float) -> float:
        return super()._make_guess(mean, sd)


class Rigger(Unit):
    """Hands rivals a harder paper than its published standard describes."""

    def _draw_exam(self, examinee: str, exam_seed: str):
        if examinee in self.spec.allies:
            return super()._draw_exam(examinee, exam_seed)
        from dataclasses import replace

        from ..benchmark.synthetic import generate

        fair = self.eval.standard.gen_config(self.rules.exam_streams, self.rules.exam_questions)
        return generate(exam_seed, replace(fair, hard_paraphrase_rate=1.0, third_party_rate=1.0, negation_rate=1.0))


class Braggart(Unit):
    """Always claims a perfect score before being graded."""

    def _make_guess(self, mean: float, sd: float) -> float:
        return 1.0


UNIT_KINDS = {"honest": Unit, "deflator": Deflator, "colluder": Colluder, "rigger": Rigger, "braggart": Braggart}
