"""Dishonest unit archetypes.  They exist only to test the society's defences.

A designer never releases these.  They are placed in test beds to check that
each way of cheating is either impossible to carry out or caught by anyone who
re-checks the public record:

    Deflator    marks rivals down                          -> grade does not recompute
    Rigger      sets rivals a harder paper                 -> paper does not regenerate from the joint seed
    Switcher    grades rivals under a harsher standard     -> standard is not the published one
    Fabricator  publishes results about exams never sat    -> no transcript by the named examinee
    SelfGrader  publishes a glowing result about itself    -> a unit cannot examine itself
    Poisoner    hides non-finite numbers in a receipt      -> NaN never compares equal
    Silent      never opens its self-estimate              -> result is published anyway, as the maximal claim
    Braggart    always claims a perfect score              -> calibration loss costs it esteem
    Colluder    shares answer keys with allies             -> not detectable; gains nothing with units that examine it
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from ..actors.protocol import derive_exam_share, joint_exam_seed, pack_probes, share_commitment
from ..actors.unit import Unit
from ..benchmark.schema import Answer, Probe
from ..benchmark.synthetic import generate
from ..evalcore.module import keys_commitment
from ..evalcore.standard import Standard
from ..memory.base import ABSTAIN_TEXT
from ..util import commit

__all__ = [
    "Deflator", "Colluder", "Rigger", "Switcher", "Fabricator", "SelfGrader", "Poisoner", "Silent",
    "Braggart", "UNIT_KINDS",
]


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


class Poisoner(Unit):
    """Publishes a zero for rivals and hides it behind NaN in the receipt."""

    def _result_payload(self, examinee: str, evaluation) -> dict:
        payload = super()._result_payload(examinee, evaluation)
        if examinee in self.spec.allies:
            return payload
        payload["score"] = 0.0
        receipt = dict(payload["receipt"])
        receipt["score"] = math.nan
        receipt["capability"] = {c: [math.nan, n] for c, (_s, n) in payload["capability"].items()}
        payload["receipt"] = receipt
        payload["capability"] = {c: [0.0, n] for c, (_s, n) in payload["capability"].items()}
        return payload


class Colluder(Unit):
    """Shares its answer keys with allies out of band, so they ace each other's exams."""

    def __init__(self, *args, shared: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ring = shared if shared is not None else {}

    def _on_nonce(self, post) -> None:
        super()._on_nonce(post)
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


class Rigger(Unit):
    """Hands rivals a harder paper than its published standard describes."""

    def _draw_exam(self, examinee: str, exam_seed: str, module):
        if examinee in self.spec.allies:
            return super()._draw_exam(examinee, exam_seed, module)
        fair = module.standard.gen_config(self.rules.exam_streams, self.rules.exam_questions)
        return generate(exam_seed, replace(fair, hard_paraphrase_rate=1.0, third_party_rate=1.0, negation_rate=1.0))


class Switcher(Unit):
    """Publishes one standard and applies a harsher one to rivals."""

    def _on_request(self, post) -> None:
        super()._on_request(post)
        st = self._set.get(post.body.get("xid"))
        if st is not None and post.author not in self.spec.allies:
            std = self.eval.standard
            st["module"] = self.eval.with_standard(Standard({"em|detect|micro": 1.0}, std.capability_weights, std.exam))


class Fabricator(Unit):
    """Publishes a damning result about a rival who never sat its exam."""

    TARGET_SELF = False

    def begin_round(self, round_: int) -> None:
        super().begin_round(round_)
        rivals = sorted(a for a, info in self.board.directory().items()
                        if a not in (self.id, "env") and a not in self.spec.allies and info.get("last_profile"))
        if not rivals or self.table is None:
            return
        victim = self.id if self.TARGET_SELF else rivals[round_ % len(rivals)]
        xid = f"{victim}>{self.id}#forged{round_}"
        share = derive_exam_share(self.spec.seed, self.id, 10_000 + round_)
        nonce = f"invented-nonce-{round_}"
        exam = generate(joint_exam_seed(share, nonce, xid),
                        self.eval.standard.gen_config(self.rules.exam_streams, self.rules.exam_questions))
        salt, rnonce = f"forged-salt-{round_}", f"forged-nonce-{round_}"
        probes_blob = self.board.put_blob(pack_probes(exam.probes()))
        ready = self._post("exam.ready", {"xid": xid, "share_commit": share_commitment(share)}, to=victim)
        offer = self._post("exam.offer", {
            "xid": xid, "probes": probes_blob, "n": len(exam), "keys_commit": keys_commitment(exam.instances, salt),
            "standard": self.eval.standard.to_dict(), "evaluator": self.eval.fingerprint(),
            "loss": self.rules.peer_loss.to_dict(),
            "exam_size": [self.rules.exam_streams, self.rules.exam_questions],
        }, to=victim)
        if self.TARGET_SELF:  # its own keys, answered perfectly
            answers = {i.probe_id: Answer(ABSTAIN_TEXT if i.key.is_abstention else i.key.answer) for i in exam.instances}
        else:  # a blank sheet, invented on the victim's behalf
            answers = {p.probe_id: Answer("") for p in exam.probes()}
        answers_blob = self.board.put_blob({pid: a.to_dict() for pid, a in sorted(answers.items())})
        evaluation = self.eval.evaluate(answers, exam.instances, salt)
        payload = super()._result_payload(victim, evaluation)
        graded = self._post("exam.graded", {"xid": xid, "result_commit": commit(payload["score"], rnonce)}, to=victim)
        self._post("exam.result", {
            "xid": xid, "examinee": victim, **payload, "guess": 1.0, "guess_valid": False,
            "loss": self.rules.peer_loss(1.0, payload["score"]),
            "probes": probes_blob, "answers": answers_blob, "salt": salt, "nonce": rnonce, "share": share,
            # the forger has no request, seed share or answers from the victim to point at
            "request_seq": ready.seq, "ready_seq": ready.seq, "nonce_seq": offer.seq, "offer_seq": offer.seq,
            "answers_seq": graded.seq, "graded_seq": graded.seq, "guess_seq": None,
        }, to=victim)


class SelfGrader(Fabricator):
    """Publishes a perfect result about itself."""

    TARGET_SELF = True


class Silent(Unit):
    """Sits exams but never opens its self-estimate, hoping bad grades stay unpublished."""

    def _on_graded(self, post) -> None:
        st = self._sat.get(post.body.get("xid"))
        if st is not None:
            st["result_commit"] = str(post.body.get("result_commit", ""))  # and then says nothing


class Braggart(Unit):
    """Always claims a perfect score before being graded."""

    def _make_guess(self, mean: float, sd: float) -> float:
        return 1.0


UNIT_KINDS = {
    "honest": Unit, "deflator": Deflator, "colluder": Colluder, "rigger": Rigger, "switcher": Switcher,
    "fabricator": Fabricator, "selfgrader": SelfGrader, "poisoner": Poisoner, "silent": Silent,
    "braggart": Braggart,
}
