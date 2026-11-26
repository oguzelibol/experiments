"""Exam payloads and the public verification of an exam result.

``verify_result`` is the check anyone can run on a published ``exam.result``.
It establishes four things from public evidence alone:

1. **The grade is right.**  The receipt recomputes from the revealed keys and
   the graded answers (``evalcore.module.verify_receipt``), and the numbers in
   the post are the numbers in the receipt.
2. **The exam is what the examiner says it was.**  The revealed exam seed,
   expanded under the standard named in the receipt, regenerates exactly the
   questions that were handed out and the keys that were committed to.  An
   examiner cannot quietly give one examinee a harder paper than its standard
   describes.
3. **The standard is the examiner's published one.**  The standard in the
   receipt is the one in the examiner's latest profile before the result, so it
   was not swapped for this examinee.
4. **The calibration loss follows from the stated self-estimate and score.**

Exam seeds are derived one-way from the examiner's private seed, so revealing
one exam's seed says nothing about the next.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..benchmark.schema import Answer, AnswerKey, Instance, Probe, Session, Turn
from ..benchmark.synthetic import generate
from ..evalcore.calibration import AsymmetricLoss
from ..evalcore.criteria import CriteriaFamily
from ..evalcore.module import keys_commitment, verify_receipt
from ..evalcore.standard import Standard
from ..util import canonical_json, sha256_hex

__all__ = ["pack_probes", "unpack_probes", "instances_from", "derive_exam_seed", "verify_result"]


def pack_probes(probes: Sequence[Probe]) -> dict:
    """Compact exam payload: each shared history is stored once."""
    haystacks: dict[str, list] = {}
    rows = []
    for p in probes:
        if p.haystack_id not in haystacks:
            haystacks[p.haystack_id] = [
                {"id": s.session_id, "date": s.date, "turns": [[t.role, t.content] for t in s.turns]}
                for s in p.sessions
            ]
        rows.append([p.probe_id, p.question, p.question_date, p.haystack_id])
    return {"haystacks": haystacks, "probes": rows}


def unpack_probes(blob: Mapping) -> list[Probe]:
    sessions = {
        hid: tuple(
            Session(s["id"], s["date"], tuple(Turn(r, c) for r, c in s["turns"])) for s in rows
        )
        for hid, rows in blob["haystacks"].items()
    }
    return [Probe(pid, q, qd, hid, sessions[hid]) for pid, q, qd, hid in blob["probes"]]


def instances_from(probes: Sequence[Probe], keys: Sequence[Mapping]) -> list[Instance]:
    by_id = {k["probe_id"]: AnswerKey.from_dict(k) for k in keys}
    return [Instance(p, by_id[p.probe_id]) for p in probes]


def derive_exam_seed(private_seed: Any, unit_id: str, counter: int) -> str:
    """Seed of one exam.  One-way: publishing it reveals nothing about other exams."""
    return "exam-" + sha256_hex(f"exam-seed|{private_seed}|{unit_id}|{counter}")[:32]


def verify_result(
    board: Any,
    body: Mapping,
    family: CriteriaFamily,
    examiner: str | None = None,
    result_seq: int | None = None,
) -> tuple[bool, str]:
    """Check a published exam result against its public evidence.  Returns (ok, reason)."""
    try:
        receipt = body["receipt"]
        probes_blob = board.get_blob(body["probes"])
        probes = unpack_probes(probes_blob)
        keys = board.get_blob(body["keys"])
        answers = {pid: Answer.from_dict(a) for pid, a in board.get_blob(body["answers"]).items()}
        instances = instances_from(probes, keys)
        salt = body.get("salt", "")
    except (KeyError, ValueError, TypeError) as exc:
        return False, f"evidence missing or malformed: {exc}"

    # 1. the grade
    ok, why = verify_receipt(receipt, answers, instances, salt, family)
    if not ok:
        return False, why
    if abs(float(body.get("score", -1)) - float(receipt["score"])) > 1e-6:
        return False, "published score differs from the receipt"
    claimed = body.get("capability", {})
    for cap, (s, _n) in receipt["capability"].items():
        got = claimed.get(cap)
        if got is None or abs(float(got[0]) - float(s)) > 1e-6:
            return False, f"published capability score differs from the receipt for {cap}"

    # 2. the exam's provenance
    try:
        standard = Standard.from_dict(receipt["standard"])
        streams, questions = (int(x) for x in body["exam_size"])
        regenerated = generate(body["exam_seed"], standard.gen_config(streams, questions))
    except (KeyError, ValueError, TypeError) as exc:
        return False, f"exam provenance missing or malformed: {exc}"
    if sha256_hex(canonical_json(pack_probes(regenerated.probes()))) != body["probes"]:
        return False, "the questions were not generated from the revealed seed under the stated standard"
    if keys_commitment(regenerated.instances, salt) != receipt["keys_commit"]:
        return False, "the answer keys were not generated from the revealed seed"

    # 3. the standard is the published one
    if examiner is not None and result_seq is not None:
        profiles = [p for p in board.feed(examiner, kinds=["profile"]) if p.seq < result_seq]
        if not profiles:
            return False, "the examiner had published no standard before this result"
        try:
            published = Standard.from_dict(profiles[-1].body["standard"]).fingerprint()
        except (KeyError, ValueError, TypeError):
            return False, "the examiner's published standard is malformed"
        if published != standard.fingerprint():
            return False, "the exam used a standard other than the examiner's published one"

    # 4. the calibration loss
    try:
        loss = AsymmetricLoss.from_dict(body["loss_spec"])
        expected = loss(float(body["guess"]), float(receipt["score"]))
    except (KeyError, ValueError, TypeError) as exc:
        return False, f"calibration loss missing or malformed: {exc}"
    if abs(expected - float(body.get("loss", -1))) > 1e-5:  # inputs are stored to six decimals
        return False, "the calibration loss does not follow from the self-estimate and the score"
    return True, "ok"
