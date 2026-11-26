"""Exam payloads and the public settlement of an exam.

An exam is eight posts on the public chain:

    examinee  exam.request   I would like to sit your exam
    examiner  exam.ready     commitment to my share of the exam seed
    examinee  exam.nonce     my share of the exam seed
    examiner  exam.offer     the questions, a commitment to the keys, my standard
    examinee  exam.sealed    commitments to my answers and to my self-estimate
    examiner  exam.keys      my seed share and the key salt: the keys are now public
    examinee  exam.open      my answers and my self-estimate, opened
    examiner  exam.result    the score, the receipt and the calibration loss

The order is the point.  The examinee is bound to its answers before anyone
can know the keys, and the examiner gives the keys away before it can know the
answers.  Once the answers are opened the grade is a function of public data:
``settle`` computes it for anyone, with or without the examiner's result post.
So an examiner has no room to shade a grade, and nothing to gain by keeping a
result to itself.

``settle`` establishes, from public evidence alone:

1. **The exam happened, between these two units, in this order.**  Each post is
   by the right party, addressed to the other, and carries the exam id.  The
   answers were sealed before the keys were revealed.  A unit cannot examine
   itself.  Nothing can be settled about a unit that never took part.
2. **Neither side chose the paper.**  The exam seed is a hash of the examiner's
   share (committed before it saw the examinee's) and the examinee's share.
   Expanded under the standard in the offer, it regenerates exactly the
   questions handed out and the keys committed to.
3. **The standard is the examiner's published one** at the moment the exam was
   requested, which the examinee chooses, so it cannot be one picked for this
   examinee after the fact.
4. **The grade** is recomputed from the opened answers and the regenerated keys.
5. **The calibration loss** follows from the opened self-estimate, the score and
   the loss announced in the offer.  A self-estimate that does not open counts
   as the maximal claim.

An examinee that goes silent after it has seen the questions, or after the
keys are public, does not make the exam disappear: once the round has ended the
exam settles with every answer missing (``void`` is ``"sealed"`` or ``"open"``).

``check_result`` settles the exam a published ``exam.result`` is about and
compares every number in the post with the settled one.  The numbers to use
afterwards are the settled ones it returns, never the ones in the post.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..benchmark.schema import Answer, AnswerKey, Instance, Probe, Session, Turn
from ..benchmark.synthetic import generate
from ..evalcore.calibration import AsymmetricLoss
from ..evalcore.criteria import CriteriaFamily
from ..evalcore.module import EvaluationModule, compare_receipt, keys_commitment
from ..evalcore.standard import Standard
from ..util import canonical_json, sha256_hex, verify_commit

__all__ = [
    "pack_probes", "unpack_probes", "instances_from", "derive_exam_share", "share_commitment",
    "joint_exam_seed", "Verdict", "ExamPosts", "exam_posts", "settle", "called_off", "check_result", "verify_result",
    "CLOCK_AUTHOR", "MAX_EXAM_STREAMS", "MAX_EXAM_STREAM_QUESTIONS",
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
    # Settled by the verifier; use these, never the numbers in a post.
    xid: str = ""
    examiner: str = ""
    examinee: str = ""
    score: float = 0.0
    n_items: int = 0
    capability: Mapping = field(default_factory=dict)  # ability -> (score, n)
    guess: float = 1.0
    guess_valid: bool = False
    loss: float = 0.0
    void: str = ""  # "" for a sat exam; "sealed" or "open" when the examinee went silent at that step
    receipt: Mapping = field(default_factory=dict)
    pending: bool = False  # not ok because the exam is still running, not because anything is wrong


def _fail(reason: str, pending: bool = False) -> Verdict:
    return Verdict(False, reason, pending=pending)


@dataclass(frozen=True)
class ExamPosts:
    """The posts of one exam, as the chain shows them.  Any may be missing."""

    request: Any = None
    ready: Any = None
    nonce: Any = None
    offer: Any = None
    sealed: Any = None
    keys: Any = None


def exam_posts(board: Any, examiner: str, examinee: str, xid: str) -> tuple[ExamPosts, list, list]:
    """Find the exam ``xid`` between two units on the chain.

    Of each kind, the first post that follows the previous step counts; later
    posts with the same exam id change nothing.  Returns the posts found and
    both parties' posts for this exam (examinee's, examiner's), oldest first.
    """
    def about(author: str, to: str, kinds: Sequence[str]) -> list:
        return [
            p for p in board.feed(author, kinds=list(kinds))
            if isinstance(p.body, Mapping) and p.body.get("xid") == xid and p.to == to
        ]

    theirs = about(examinee, examiner, ("exam.request", "exam.nonce", "exam.sealed", "exam.open"))
    ours = about(examiner, examinee, ("exam.ready", "exam.offer", "exam.keys"))

    def first(posts: list, kind: str, after: Any, accept=None) -> Any:
        if after is None:
            return None
        for p in posts:
            if p.kind == kind and p.seq > after.seq and (accept is None or accept(p)):
                return p
        return None

    request = next((p for p in theirs if p.kind == "exam.request"), None)
    ready = first(ours, "exam.ready", request)
    nonce = first(theirs, "exam.nonce", ready)
    offer = first(ours, "exam.offer", nonce)
    keys = first(ours, "exam.keys", offer)
    sealed = first(theirs, "exam.sealed", offer, lambda p: p.body.get("offer_seq") == offer.seq)
    return ExamPosts(request, ready, nonce, offer, sealed, keys), theirs, ours


def settle(
    board: Any, family: CriteriaFamily, examiner: str, examinee: str, xid: str,
    cutoff: int | None = None, clock_author: str = CLOCK_AUTHOR,
) -> Verdict:
    """What exam ``xid`` between two units established, judged from the chain before ``cutoff``.

    ``cutoff`` is a position on the chain (a post's ``seq``); by default the
    whole chain is used.  Never raises: everything examined here was written
    by other units, and evidence that cannot be read is evidence that does not
    hold.
    """
    try:
        return _settle(board, family, examiner, examinee, xid, cutoff, clock_author)
    except Exception as exc:  # noqa: BLE001 - untrusted input must end in a verdict, not a crash
        return _fail(f"the evidence could not be read: {type(exc).__name__}: {str(exc)[:120]}")


def _settle(board, family, examiner, examinee, xid, cutoff, clock_author) -> Verdict:
    if not isinstance(xid, str) or not isinstance(examinee, str) or not isinstance(examiner, str) or not xid or not examinee:
        return _fail("no exam or no examinee is named")
    if examinee == examiner:
        return _fail("a unit cannot examine itself")
    if cutoff is None:
        cutoff = board.head()[0] + 1
    posts, theirs, _ours = exam_posts(board, examiner, examinee, xid)
    request, ready, nonce_post, offer, sealed, keys = posts.request, posts.ready, posts.nonce, posts.offer, posts.sealed, posts.keys

    # ---- 1. the transcript: who posted what, and in which order
    if request is None:
        return _fail("the named examinee did not request this exam")
    if ready is None:
        return _fail("the examiner did not commit to its share of the exam seed")
    if nonce_post is None:
        return _fail("the named examinee did not contribute to the exam seed")
    if offer is None or offer.seq >= cutoff:
        return _fail("the examiner made no offer for this exam")
    if keys is None or keys.seq >= cutoff:
        return _fail("the examiner has not revealed the keys", pending=True)
    clocks = [p.seq for p in board.feed(clock_author, kinds=["clock"])]

    def round_ended(after: int, before: int) -> bool:
        return any(after < c < before for c in clocks)

    void = ""
    opened = None
    if sealed is None or sealed.seq > keys.seq:
        # The examinee saw the questions and did not commit to answers.
        if not round_ended(offer.seq, keys.seq):
            return _fail("the keys were revealed before the examinee had until the end of the round to seal its answers")
        void = "sealed"
    else:
        commitment = str(sealed.body.get("answers_commit", ""))
        for p in theirs:
            if (p.kind == "exam.open" and keys.seq < p.seq < cutoff and isinstance(p.body.get("answers"), str)
                    and p.body["answers"] in (p.body.get("blobs") or ())  # the sheet was on the board when it was opened
                    and verify_commit(commitment, p.body["answers"], str(p.body.get("answers_nonce", "")))):
                opened = p
                break
        if opened is None:
            if not round_ended(keys.seq, cutoff):
                return _fail("the examinee still has until the end of the round to open its answers", pending=True)
            void = "open"

    # ---- 2. the paper: nobody chose it
    share, salt, nonce = keys.body.get("share"), keys.body.get("salt"), nonce_post.body.get("nonce")
    if not isinstance(share, str) or not isinstance(nonce, str) or not isinstance(salt, str) or not share or not nonce:
        return _fail("the exam seed shares or the key salt are missing")
    if ready.body.get("share_commit") != share_commitment(share):
        return _fail("the revealed seed share is not the one the examiner committed to")
    try:
        standard = Standard.from_dict(offer.body["standard"])
        streams, questions = (int(x) for x in offer.body["exam_size"])
    except (KeyError, ValueError, TypeError) as exc:
        return _fail(f"the offer is malformed: {exc}")
    if not (1 <= streams <= MAX_EXAM_STREAMS and 1 <= questions <= MAX_EXAM_STREAM_QUESTIONS):
        return _fail("the exam size is out of range")
    try:
        regenerated = generate(joint_exam_seed(share, nonce, xid), standard.gen_config(streams, questions))
    except ValueError as exc:
        return _fail(f"no exam can be drawn from the joint seed under the stated standard: {exc}")
    probes_blob = offer.body.get("probes")
    if not isinstance(probes_blob, str) or probes_blob not in (offer.body.get("blobs") or ()):
        return _fail("the offer does not vouch for its questions being on the board")
    if sha256_hex(canonical_json(pack_probes(regenerated.probes()))) != probes_blob:
        return _fail("the questions were not generated from the joint seed under the stated standard")
    if keys_commitment(regenerated.instances, salt) != offer.body.get("keys_commit"):
        return _fail("the answer keys are not the ones generated from the joint seed and committed to in the offer")

    # ---- 3. the standard is the one the examiner had published when the exam was asked for
    profiles = [p for p in board.feed(examiner, kinds=["profile"]) if p.seq < request.seq]
    if not profiles:
        return _fail("the examiner had published no standard before this exam was requested")
    try:
        published = Standard.from_dict(profiles[-1].body["standard"]).fingerprint()
    except (KeyError, ValueError, TypeError):
        return _fail("the examiner's published standard is malformed")
    if published != standard.fingerprint():
        return _fail("the exam used a standard other than the examiner's published one")

    # ---- 4. the grade: from the examinee's own opened answers and the regenerated keys
    answers: dict[str, Answer] = {}
    if opened is not None:
        try:
            blob = board.get_blob(opened.body["answers"])
            if not isinstance(blob, Mapping):
                raise ValueError("not a mapping")
            answers = {str(pid): Answer.from_dict(a) for pid, a in blob.items()}
        except (KeyError, ValueError, TypeError, AttributeError):
            # it opened something that is not an answer sheet: every answer is missing
            answers = {}
    evaluation = EvaluationModule(standard, family).evaluate(answers, regenerated.instances, salt)

    # ---- 5. the self-estimate and the calibration loss
    guess, guess_valid = 1.0, False
    if opened is not None:
        revealed = _num(opened.body.get("guess"))
        if revealed is not None and verify_commit(
            str(sealed.body.get("guess_commit", "")), opened.body.get("guess"), str(opened.body.get("guess_nonce", ""))
        ):
            guess, guess_valid = min(1.0, max(0.0, revealed)), True
    try:
        loss_spec = AsymmetricLoss.from_dict(offer.body["loss"])
    except (KeyError, ValueError, TypeError) as exc:
        return _fail(f"the offer announced no usable loss: {exc}")
    loss = loss_spec(guess, evaluation.score)

    return Verdict(
        True, "ok", xid=xid, examiner=examiner, examinee=examinee, score=float(evaluation.score),
        n_items=int(evaluation.receipt["n_items"]),
        capability={c: (float(s), int(n)) for c, (s, n) in evaluation.capability.items()},
        guess=guess, guess_valid=guess_valid, loss=float(loss), void=void, receipt=dict(evaluation.receipt),
    )


def called_off(board: Any, examiner: str, examinee: str, xid: str, exam_size: Sequence[int]) -> bool:
    """Did the examiner show that this exam's joint seed cannot be expanded under its published standard?

    An examiner may not look for another seed when one fails, so it calls the
    exam off and reveals its share; anyone can then check the claim here.  An
    exam left unfinished without such proof was simply walked away from.
    """
    try:
        posts, _theirs, _ours = exam_posts(board, examiner, examinee, xid)
        if posts.nonce is None:
            return False
        for p in board.feed(examiner, kinds=["exam.decline"]):
            if not isinstance(p.body, Mapping) or p.body.get("xid") != xid or p.to != examinee or p.seq < posts.nonce.seq:
                continue
            share, nonce = p.body.get("share"), posts.nonce.body.get("nonce")
            if not isinstance(share, str) or not isinstance(nonce, str) or posts.ready.body.get("share_commit") != share_commitment(share):
                continue
            profiles = [q for q in board.feed(examiner, kinds=["profile"]) if q.seq < posts.request.seq]
            if not profiles:
                continue
            standard = Standard.from_dict(profiles[-1].body["standard"])
            streams, questions = (int(x) for x in exam_size)
            try:
                generate(joint_exam_seed(share, nonce, xid), standard.gen_config(streams, questions))
            except ValueError:
                return True
        return False
    except Exception:  # noqa: BLE001 - unreadable evidence proves nothing
        return False


def check_result(board: Any, post: Any, family: CriteriaFamily, clock_author: str = CLOCK_AUTHOR) -> Verdict:
    """Verify one ``exam.result`` post: settle its exam and compare every number.  Never raises."""
    try:
        return _check_result(board, post, family, clock_author)
    except Exception as exc:  # noqa: BLE001 - untrusted input must end in a verdict, not a crash
        return _fail(f"the evidence could not be read: {type(exc).__name__}: {str(exc)[:120]}")


def _check_result(board: Any, post: Any, family: CriteriaFamily, clock_author: str) -> Verdict:
    body = post.body
    if post.kind != "exam.result" or not isinstance(body, Mapping):
        return _fail("not an exam result")
    settled = settle(board, family, post.author, body.get("examinee"), body.get("xid"), post.seq, clock_author)
    if not settled.ok:
        return settled
    if (body.get("void") or "") != settled.void:
        return _fail("the result misstates whether the examinee sealed and opened its answers")
    ok, why = compare_receipt(body.get("receipt"), settled.receipt)
    if not ok:
        return _fail(why)
    if not _close(body.get("score"), settled.score, 1e-6):
        return _fail("the published score differs from the settled one")
    claimed = body.get("capability")
    if not isinstance(claimed, Mapping) or set(claimed) != set(settled.capability):
        return _fail("the published ability table does not list exactly the abilities examined")
    for cap, (s, n) in settled.capability.items():
        got = claimed[cap]
        if not isinstance(got, (list, tuple)) or len(got) != 2 or not _close(got[0], s, 1e-6) or got[1] != n:
            return _fail(f"the published ability score differs from the settled one for {cap}")
    if body.get("guess_valid") is not settled.guess_valid or not _close(body.get("guess"), settled.guess, 1e-6):
        return _fail("the result misquotes the examinee's self-estimate")
    if not _close(body.get("loss"), settled.loss, 1e-5):  # inputs are stored to six decimals
        return _fail("the calibration loss does not follow from the self-estimate, the score and the announced loss")
    return settled


def verify_result(board: Any, post: Any, family: CriteriaFamily) -> tuple[bool, str]:
    """(ok, reason) for one ``exam.result`` post.  See ``check_result``."""
    verdict = check_result(board, post, family)
    return verdict.ok, verdict.reason
