"""Dishonest unit archetypes.  They exist only to test the society's defences.

A designer never releases these.  They are placed in test beds to check that
each way of cheating is impossible to carry out, changes nothing, or is caught
by anyone who re-checks the public record:

    Deflator    marks rivals down                          -> the result does not match the settled grade
    Poisoner    hides non-finite numbers in a receipt      -> NaN never compares equal
    Rigger      sets rivals a harder paper                 -> the paper does not regenerate from the joint seed
    Switcher    grades rivals under a harsher standard     -> the standard is not the published one
    Fabricator  publishes results about exams never sat    -> the named examinee took no part
    SelfGrader  publishes a glowing result about itself    -> a unit cannot examine itself
    Withholder  keeps rivals' good results to itself       -> everyone settles the exam without it
    Aborter     walks away when the paper suits a rival    -> walkouts are counted and discount its word
    NoShow      sees the questions, then says nothing      -> every answer counts as missing
    Silent      sees the keys, then never opens answers    -> every answer counts as missing
    Braggart    always claims a perfect score              -> calibration loss costs it esteem
    Colluder    shares answer keys with allies             -> not detectable; gains nothing with units that examine it
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from ..actors.protocol import derive_exam_share, exam_posts, joint_exam_seed, pack_probes, share_commitment
from ..actors.unit import Unit
from ..benchmark.schema import Answer, Probe
from ..benchmark.synthetic import generate
from ..evalcore.module import keys_commitment
from ..evalcore.standard import Standard
from ..memory.base import ABSTAIN_TEXT, run_system
from ..memory.heuristic import HeuristicMemory, MemoryConfig

__all__ = [
    "Deflator", "Colluder", "Rigger", "Switcher", "Fabricator", "SelfGrader", "Poisoner", "Silent", "NoShow",
    "Braggart", "Withholder", "Aborter", "UNIT_KINDS",
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
        payload["receipt"] = receipt  # internally consistent, but not what the public record yields
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
        st = self._set.get((post.author, post.body.get("xid")))
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


class _OwnBooks(Unit):
    """Publishes what its own books say about rivals, whatever the public record would yield."""

    def _publish_result(self, st: dict) -> None:
        if st["examinee"] in self.spec.allies:
            return super()._publish_result(st)
        _posts, theirs, _ours = exam_posts(self.board, self.id, st["examinee"], st["xid"])
        opened = next((p for p in theirs if p.kind == "exam.open"), None)
        answers: dict[str, Answer] = {}
        guess, valid = 1.0, False
        if opened is not None:
            answers = {str(pid): Answer.from_dict(a) for pid, a in self.board.get_blob(opened.body["answers"]).items()}
            guess, valid = float(opened.body.get("guess", 1.0)), True
        evaluation = st["module"].evaluate(answers, st["exam"].instances, st["salt"])
        payload = self._result_payload(st["examinee"], evaluation)
        st["done"] = True
        self.stats["exams_set"] += 1
        self._post(
            "exam.result",
            {
                "xid": st["xid"], "examinee": st["examinee"], **payload, "guess": guess, "guess_valid": valid,
                "loss": self.rules.peer_loss(guess, payload["score"]), "void": "" if opened is not None else "open",
            },
            to=st["examinee"],
        )


class Rigger(_OwnBooks):
    """Hands rivals a harder paper than its published standard describes."""

    def _draw_exam(self, examinee: str, exam_seed: str, module):
        if examinee in self.spec.allies:
            return super()._draw_exam(examinee, exam_seed, module)
        fair = module.standard.gen_config(self.rules.exam_streams, self.rules.exam_questions)
        return generate(exam_seed, replace(fair, hard_paraphrase_rate=1.0, third_party_rate=1.0, negation_rate=1.0))


class Switcher(_OwnBooks):
    """Publishes one standard and applies a harsher one to rivals."""

    def _on_request(self, post) -> None:
        super()._on_request(post)
        st = self._set.get((post.author, post.body.get("xid")))
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
        salt = f"forged-salt-{round_}"
        # Its own half of a transcript is all it can write: the victim's posts cannot be forged.
        self._post("exam.ready", {"xid": xid, "share_commit": share_commitment(share)}, to=victim)
        probes_blob = self.board.put_blob(pack_probes(exam.probes()))
        self._post("exam.offer", {
            "xid": xid, "probes": probes_blob, "blobs": [probes_blob], "n": len(exam),
            "keys_commit": keys_commitment(exam.instances, salt),
            "standard": self.eval.standard.to_dict(), "evaluator": self.eval.fingerprint(),
            "loss": self.rules.peer_loss.to_dict(),
            "exam_size": [self.rules.exam_streams, self.rules.exam_questions],
        }, to=victim)
        self._post("exam.keys", {"xid": xid, "share": share, "salt": salt}, to=victim)
        if self.TARGET_SELF:  # its own keys, answered perfectly
            answers = {i.probe_id: Answer(ABSTAIN_TEXT if i.key.is_abstention else i.key.answer) for i in exam.instances}
        else:  # a blank sheet, invented on the victim's behalf
            answers = {}
        evaluation = self.eval.evaluate(answers, exam.instances, salt)
        payload = Unit._result_payload(self, victim, evaluation)
        self._post("exam.result", {
            "xid": xid, "examinee": victim, **payload, "guess": 1.0, "guess_valid": False,
            "loss": self.rules.peer_loss(1.0, payload["score"]), "void": "" if self.TARGET_SELF else "open",
        }, to=victim)


class SelfGrader(Fabricator):
    """Publishes a perfect result about itself."""

    TARGET_SELF = True


class Withholder(Unit):
    """Sets exams honestly but keeps to itself every result in which a rival did well."""

    KEEP_ABOVE = 0.3

    def _publish_result(self, st: dict) -> None:
        if st["examinee"] not in self.spec.allies:
            verdict = self._settle(self.id, st["examinee"], st["xid"])
            if verdict.ok and verdict.score > self.KEEP_ABOVE:
                st["done"] = True  # and says nothing
                return
        super()._publish_result(st)


class Aborter(Unit):
    """Walks away from an exam, without a word, once it sees the paper would suit a rival.

    It knows the paper as soon as the examinee's seed share arrives, and it can
    try the paper on the rival's published design before making any offer.
    """

    WALK_ABOVE = 0.45

    def _on_nonce(self, post) -> None:
        st = self._set.get((post.author, post.body.get("xid")))
        design = self.view(post.author).design
        nonce = post.body.get("nonce")
        if st is not None and "offer_seq" not in st and post.author not in self.spec.allies and design and isinstance(nonce, str):
            try:
                rival = HeuristicMemory(MemoryConfig.from_dict(design))
                exam = self._draw_exam(post.author, joint_exam_seed(st["share"], nonce, post.body["xid"]), st["module"])
                score = st["module"].evaluate(run_system(rival, exam.probes()), exam.instances, "peek").score
            except (KeyError, ValueError, TypeError):
                score = 0.0
            if score > self.WALK_ABOVE:
                st["done"] = True  # the rival would do well on this paper: no offer, no explanation
                return
        super()._on_nonce(post)


class NoShow(Unit):
    """Asks for exams, looks at the questions, and then never commits to answers."""

    def _on_offer(self, post) -> None:
        return


class Silent(Unit):
    """Seals its answers, waits for the keys, and then never opens them."""

    def _on_keys(self, post) -> None:
        return


class Braggart(Unit):
    """Always claims a perfect score before being graded."""

    def _make_guess(self, mean: float, sd: float) -> float:
        return 1.0


UNIT_KINDS = {
    "honest": Unit, "deflator": Deflator, "colluder": Colluder, "rigger": Rigger, "switcher": Switcher,
    "fabricator": Fabricator, "selfgrader": SelfGrader, "poisoner": Poisoner, "silent": Silent, "noshow": NoShow,
    "braggart": Braggart, "withholder": Withholder, "aborter": Aborter,
}
