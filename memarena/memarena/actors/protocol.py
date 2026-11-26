"""Exam payloads and the public verification of an exam result.

An exam is eight posts on the public chain:

    examinee  exam.request   I would like to sit your exam
    examiner  exam.ready     commitment to my share of the exam seed
    examinee  exam.nonce     my share of the exam seed
    examiner  exam.offer     the questions, a commitment to the keys, my standard
    examinee  exam.answers   my answers, a commitment to my self-estimate, the offer I am answering
    examiner  exam.graded    commitment to the grade
    examinee  exam.guess     my self-estimate, opened
    examiner  exam.result    score, receipt, revealed keys, revealed seed share

``check_result`` is the check anyone can run on a published ``exam.result``.
From public evidence alone it establishes:

1. **The exam really happened, between these two units, in this order.**  Each
   post in the transcript is authored by the right party, carries the same exam
   id and appears in the right order.  Nothing can be published about a unit
   that never answered; the graded answers are the ones the examinee posted;
   the offer is the one the examinee said it was answering; the grade was fixed
   before the self-estimate was opened; a unit cannot examine itself.
2. **Neither side chose the paper.**  The exam seed is a hash of the examiner's
   share (committed before it saw the examinee's) and the examinee's share.
   Expanded under the standard named in the receipt, it regenerates exactly the
   questions handed out and the keys committed to.
3. **The grade is right.**  The receipt recomputes from the revealed keys and
   the posted answers, every number in the post equals the recomputed one, and
   nothing in it is NaN or infinite.
4. **The standard is the examiner's published one** at the time it committed
   to the exam, not one picked for this examinee.
5. **The calibration loss** follows from the opened self-estimate, the score
   and the loss the offer announced.  If the examinee never opened its
   self-estimate, the result may be published once the round has ended, with
   the estimate recorded as the maximal claim.

The numbers a verifier should use afterwards are the ones it recomputed, which
``check_result`` returns, not the ones in the post.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..benchmark.schema import Answer, AnswerKey, Instance, Probe, Session, Turn
from ..benchmark.synthetic import generate
from ..evalcore.calibration import AsymmetricLoss
from ..evalcore.criteria import CriteriaFamily
from ..evalcore.module import keys_commitment, recompute_receipt
from ..evalcore.standard import Standard
from ..util import canonical_json, sha256_hex, verify_commit

__all__ = [
    "pack_probes", "unpack_probes", "instances_from", "derive_exam_share", "share_commitment",
    "joint_exam_seed", "Verdict", "check_result", "verify_result", "CLOCK_AUTHOR",
    "MAX_EXAM_STREAMS", "MAX_EXAM_STREAM_QUESTIONS",
]

CLOCK_AUTHOR = "env"

# The largest exam a verifier is prepared to regenerate: the cost of a check is
# set by the party being checked, so it has to be bounded.
MAX_EXAM_STREAMS = 8
MAX_EXAM_STREAM_QUESTIONS = 48


# Bounds on an exam payload a reader is prepared to unpack.
MAX_EXAM_QUESTIONS = 4096
MAX_EXAM_HISTORIES = 512


def pack_probes(probes: Sequence[Probe]) -> dict:
    """Compact exam payload: each shared history is stored once."""
    haystacks: dict[str, list] = {}
    packed_from: dict[str, tuple] = {}
    rows = []
    for p in probes:
        if p.haystack_id not in haystacks:
            haystacks[p.haystack_id] = [
                {"id": s.session_id, "date": s.date, "turns": [[t.role, t.content] for t in s.turns]}
                for s in p.sessions
            ]
            packed_from[p.haystack_id] = p.sessions
        elif packed_from[p.haystack_id] is not p.sessions and packed_from[p.haystack_id] != p.sessions:
            raise ValueError(f"two different histories share the id {p.haystack_id}")
        rows.append([p.probe_id, p.question, p.question_date, p.haystack_id])
    return {"haystacks": haystacks, "probes": rows}


def unpack_probes(blob: Mapping) -> list[Probe]:
    """Rebuild probes from a packed payload.  Raises ``ValueError`` on anything malformed.

    The payload comes from another unit, so every shape is checked here rather
    than trusted: a reader must be able to refuse a bad exam, not crash on it.
    """
    if not isinstance(blob, Mapping) or not isinstance(blob.get("haystacks"), Mapping) or not isinstance(blob.get("probes"), list):
        raise ValueError("malformed exam payload")
    if len(blob["probes"]) > MAX_EXAM_QUESTIONS or len(blob["haystacks"]) > MAX_EXAM_HISTORIES:
        raise ValueError("exam payload is larger than any exam")
    sessions: dict[str, tuple] = {}
    for hid, rows in blob["haystacks"].items():
        if not isinstance(hid, str) or not isinstance(rows, list):
            raise ValueError("malformed history in exam payload")
        built = []
        for srow in rows:
            if (not isinstance(srow, Mapping) or not isinstance(srow.get("id"), str) or not isinstance(srow.get("date"), str)
                    or not isinstance(srow.get("turns"), list)):
                raise ValueError("malformed session in exam payload")
            turns = []
            for turn in srow["turns"]:
                if (not isinstance(turn, (list, tuple)) or len(turn) != 2
                        or not isinstance(turn[0], str) or not isinstance(turn[1], str)):
                    raise ValueError("malformed turn in exam payload")
                turns.append(Turn(turn[0], turn[1]))
            built.append(Session(srow["id"], srow["date"], tuple(turns)))
        sessions[hid] = tuple(built)
    probes: list[Probe] = []
    seen: set[str] = set()
    for row in blob["probes"]:
        if not isinstance(row, (list, tuple)) or len(row) != 4 or not all(isinstance(x, str) for x in row):
            raise ValueError("malformed question in exam payload")
        pid, question, qdate, hid = row
        if pid in seen or hid not in sessions:
            raise ValueError("exam payload repeats a question id or names a missing history")
        seen.add(pid)
        probes.append(Probe(pid, question, qdate, hid, sessions[hid]))
    return probes


def instances_from(probes: Sequence[Probe], keys: Sequence[Mapping]) -> list[Instance]:
    by_id = {k["probe_id"]: AnswerKey.from_dict(k) for k in keys}
    return [Instance(p, by_id[p.probe_id]) for p in probes]


def derive_exam_share(private_seed: Any, unit_id: str, counter: int) -> str:
    """The examiner's share of one exam's seed.  One-way in the private seed."""
    return sha256_hex(f"exam-share|{private_seed}|{unit_id}|{counter}")[:40]


def share_commitment(share: str) -> str:
    return sha256_hex("exam-share-commit|" + str(share))


def joint_exam_seed(share: str, nonce: str, xid: str) -> str:
    """Seed of one exam: fixed only once both parties have contributed."""
    return "exam-" + sha256_hex(f"exam-seed|{share}|{nonce}|{xid}")[:32]


def _num(x: Any) -> float | None:
    """A finite real number, or None.  NaN and infinity never pass a check."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _close(a: Any, b: Any, tol: float) -> bool:
    a, b = _num(a), _num(b)
    return a is not None and b is not None and abs(a - b) <= tol


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = "ok"
    # Recomputed by the verifier; use these, never the numbers in the post.
    xid: str = ""
    examiner: str = ""
    examinee: str = ""
    score: float = 0.0
    n_items: int = 0
    capability: Mapping = field(default_factory=dict)  # ability -> (score, n)
    guess: float = 1.0
    guess_valid: bool = False
    loss: float = 0.0


def _fail(reason: str) -> Verdict:
    return Verdict(False, reason)


def check_result(board: Any, post: Any, family: CriteriaFamily, clock_author: str = CLOCK_AUTHOR) -> Verdict:
    """Verify one ``exam.result`` post against the public chain and blobs.

    Never raises: everything examined here was written by other units, and
    evidence that cannot even be read is evidence that does not hold.
    """
    try:
        return _check_result(board, post, family, clock_author)
    except Exception as exc:  # noqa: BLE001 - untrusted input must end in a verdict, not a crash
        return _fail(f"the evidence could not be read: {type(exc).__name__}: {str(exc)[:120]}")


def _check_result(board: Any, post: Any, family: CriteriaFamily, clock_author: str) -> Verdict:
    body = post.body
    examiner, result_seq = post.author, post.seq
    if post.kind != "exam.result" or not isinstance(body, Mapping):
        return _fail("not an exam result")
    xid, examinee = body.get("xid"), body.get("examinee")
    if not isinstance(xid, str) or not isinstance(examinee, str) or not xid or not examinee:
        return _fail("the result names no exam or no examinee")
    if examinee == examiner:
        return _fail("a unit cannot examine itself")
    receipt = body.get("receipt")
    if not isinstance(receipt, Mapping):
        return _fail("the result carries no receipt")

    # ---- 1. the transcript: who posted what, and in which order
    def post_at(name: str):
        seq = body.get(name)
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            return None
        got = board.read(since_seq=seq - 1, limit=1)
        return got[0] if got and got[0].seq == seq else None

    request, ready, nonce_post = post_at("request_seq"), post_at("ready_seq"), post_at("nonce_seq")
    offer, answers_post, graded = post_at("offer_seq"), post_at("answers_seq"), post_at("graded_seq")
    if None in (request, ready, nonce_post, offer, answers_post, graded):
        return _fail("the result does not point at a complete exam transcript")
    if not (request.seq < ready.seq < nonce_post.seq < offer.seq < answers_post.seq < graded.seq < result_seq):
        return _fail("the exam transcript is out of order")
    if any(p.body.get("xid") != xid for p in (request, ready, nonce_post, offer, answers_post, graded)):
        return _fail("the transcript posts belong to a different exam")
    if request.kind != "exam.request" or request.author != examinee or request.to != examiner:
        return _fail("the named examinee did not request this exam")
    if ready.kind != "exam.ready" or ready.author != examiner or ready.to != examinee:
        return _fail("the examiner did not commit to its share of the exam seed")
    if nonce_post.kind != "exam.nonce" or nonce_post.author != examinee or nonce_post.to != examiner:
        return _fail("the named examinee did not contribute to the exam seed")
    if offer.kind != "exam.offer" or offer.author != examiner or offer.to != examinee:
        return _fail("the result points at an offer that is not its own")
    if answers_post.kind != "exam.answers" or answers_post.author != examinee or answers_post.to != examiner:
        return _fail("the graded answers were not posted by the named examinee")
    if answers_post.body.get("offer_seq") != offer.seq:
        return _fail("the examinee answered a different offer from the one the result cites")
    if answers_post.body.get("answers") != body.get("answers") or not isinstance(body.get("answers"), str):
        return _fail("the graded answers are not the ones the named examinee posted")
    if offer.body.get("probes") != body.get("probes") or not isinstance(body.get("probes"), str):
        return _fail("the questions in the result are not the ones that were offered")
    if graded.kind != "exam.graded" or graded.author != examiner:
        return _fail("the examiner did not commit to the grade")

    # ---- 2. the paper: nobody chose it
    share, nonce = body.get("share"), nonce_post.body.get("nonce")
    if not isinstance(share, str) or not isinstance(nonce, str) or not share or not nonce:
        return _fail("the exam seed shares are missing")
    if ready.body.get("share_commit") != share_commitment(share):
        return _fail("the revealed seed share is not the one the examiner committed to")
    try:
        standard = Standard.from_dict(receipt["standard"])
        streams, questions = (int(x) for x in offer.body["exam_size"])
        if not (1 <= streams <= MAX_EXAM_STREAMS and 1 <= questions <= MAX_EXAM_STREAM_QUESTIONS):
            return _fail("the exam size is out of range")
        regenerated = generate(joint_exam_seed(share, nonce, xid), standard.gen_config(streams, questions))
        salt = body["salt"]
        if not isinstance(salt, str):
            return _fail("the key salt is malformed")
    except (KeyError, ValueError, TypeError) as exc:
        return _fail(f"exam provenance missing or malformed: {exc}")
    if sha256_hex(canonical_json(pack_probes(regenerated.probes()))) != body["probes"]:
        return _fail("the questions were not generated from the joint seed under the stated standard")
    if keys_commitment(regenerated.instances, salt) != receipt.get("keys_commit") or offer.body.get("keys_commit") != receipt.get("keys_commit"):
        return _fail("the answer keys are not the ones generated from the joint seed and committed to in the offer")
    try:
        if Standard.from_dict(offer.body["standard"]).fingerprint() != standard.fingerprint():
            return _fail("the receipt names a different standard from the offer")
    except (KeyError, ValueError, TypeError):
        return _fail("the offer's standard is malformed")

    # ---- 3. the grade: recomputed from the examinee's own posted answers and the regenerated keys
    try:
        answers = {str(pid): Answer.from_dict(a) for pid, a in board.get_blob(body["answers"]).items()}
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        return _fail(f"posted answers missing or malformed: {exc}")
    ok, why, fresh = recompute_receipt(receipt, answers, regenerated.instances, salt, family)
    if not ok:
        return _fail(why)
    if not _close(body.get("score"), fresh["score"], 1e-6):
        return _fail("the published score differs from the recomputed one")
    claimed = body.get("capability")
    if not isinstance(claimed, Mapping) or set(claimed) != set(fresh["capability"]):
        return _fail("the published ability table does not list exactly the abilities examined")
    for cap, (s, n) in fresh["capability"].items():
        got = claimed[cap]
        if not isinstance(got, (list, tuple)) or len(got) != 2 or not _close(got[0], s, 1e-6) or got[1] != n:
            return _fail(f"the published ability score differs from the recomputed one for {cap}")
    if not verify_commit(str(graded.body.get("result_commit", "")), body.get("score"), str(body.get("nonce", ""))):
        return _fail("the published score is not the one committed to before the self-estimate was opened")

    # ---- 4. the standard is the one the examiner had published when it committed to the exam
    profiles = [p for p in board.feed(examiner, kinds=["profile"]) if p.seq < ready.seq]
    if not profiles:
        return _fail("the examiner had published no standard before committing to this exam")
    try:
        published = Standard.from_dict(profiles[-1].body["standard"]).fingerprint()
    except (KeyError, ValueError, TypeError):
        return _fail("the examiner's published standard is malformed")
    if published != standard.fingerprint():
        return _fail("the exam used a standard other than the examiner's published one")

    # ---- 5. the self-estimate and the calibration loss
    reveals = [
        p for p in board.feed(examinee, kinds=["exam.guess"])
        if p.body.get("xid") == xid and graded.seq < p.seq < result_seq
    ]
    quoted = _num(body.get("guess"))
    if quoted is None or not 0.0 <= quoted <= 1.0:
        return _fail("the result quotes no usable self-estimate")
    guess_valid = False
    if reveals:
        guess_post = reveals[0]
        if body.get("guess_seq") != guess_post.seq:
            return _fail("the result ignores the self-estimate the examinee opened")
        revealed = _num(guess_post.body.get("guess"))
        guess_valid = revealed is not None and verify_commit(
            str(answers_post.body.get("guess_commit", "")), guess_post.body.get("guess"), str(guess_post.body.get("nonce", ""))
        )
        if guess_valid:
            if body.get("guess_valid") is not True or abs(quoted - min(1.0, max(0.0, revealed))) > 1e-6:
                return _fail("the result misquotes the examinee's self-estimate")
        elif body.get("guess_valid") is not False or abs(quoted - 1.0) > 1e-6:
            return _fail("a self-estimate that did not open must be recorded as the maximal claim")
    else:
        if body.get("guess_seq") is not None or body.get("guess_valid") is not False or abs(quoted - 1.0) > 1e-6:
            return _fail("no self-estimate was opened, so it must be recorded as the maximal claim")
        ended = any(graded.seq < p.seq < result_seq for p in board.feed(clock_author, kinds=["clock"]))
        if not ended:
            return _fail("the result was published before the examinee had until the end of the round to open its self-estimate")
    try:
        loss_spec = AsymmetricLoss.from_dict(offer.body["loss"])
    except (KeyError, ValueError, TypeError) as exc:
        return _fail(f"the offer announced no usable loss: {exc}")
    loss = loss_spec(quoted, fresh["score"])
    if not _close(body.get("loss"), loss, 1e-5):  # inputs are stored to six decimals
        return _fail("the calibration loss does not follow from the self-estimate, the score and the announced loss")

    return Verdict(
        True, "ok", xid=xid, examiner=examiner, examinee=examinee, score=fresh["score"],
        n_items=int(fresh["n_items"]), capability={c: (float(s), int(n)) for c, (s, n) in fresh["capability"].items()},
        guess=quoted, guess_valid=guess_valid, loss=loss,
    )


def verify_result(board: Any, post: Any, family: CriteriaFamily) -> tuple[bool, str]:
    """(ok, reason) for one ``exam.result`` post.  See ``check_result``."""
    verdict = check_result(board, post, family)
    return verdict.ok, verdict.reason
