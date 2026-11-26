"""Exam payloads, the one reading of an exam that everybody uses, and its settlement.

An exam is eight posts on the public chain:

    examinee  exam.request   I would like to sit your exam
    examiner  exam.ready     commitment to my share of the exam seed; the size of the exam and the loss
    examinee  exam.nonce     my share of the exam seed
    examiner  exam.offer     the questions, a commitment to the keys, my standard
    examinee  exam.sealed    commitments to my answers and to my self-estimate
    examiner  exam.keys      my seed share and the key salt: the keys are now public
    examinee  exam.open      my answers and my self-estimate, opened
    examiner  exam.result    the score, the receipt and the calibration loss

The order is the point.  The terms are fixed before either seed share is
known.  The examinee is bound to its answers before anyone can know the keys,
and the examiner gives the keys away before it can know the answers.  Once the
answers are opened the grade is a function of public data: ``settle`` computes
it for anyone, with or without the examiner's result post.

**One reading.**  ``read_exam`` decides, from the chain alone, which posts make
up an exam and whose move it is.  Units act on that reading and on nothing
else, and ``settle`` judges by it, so a post that a unit acts on is always the
post a verifier counts.  Of each kind the first post that follows the previous
step and is well formed counts; anything else with the same exam id is noise.

**Deadlines on the chain.**  A party called on to act has the rest of that
round and all of the next: until the second clock post after the post that
called on it.  What it posts later does not count.  So every outcome becomes
final at a definite position on the chain, and no later post can change it:

* an examinee that has seen the questions and seals nothing, or has seen the
  keys and opens nothing, is settled with every answer missing;
* an examiner that knows the paper (it knows it once the examinee's seed share
  arrives) and then fails to offer, to reveal the keys in time, or to offer a
  paper that is what the joint seed gives, has *abandoned* or *spoiled* the
  exam.  Nothing is settled about the examinee, and it counts against the
  examiner.

``settle`` establishes, from public evidence alone:

1. the exam happened between these two units, in this order, on the terms the
   examiner committed to before it could know the paper;
2. neither side chose the paper: the questions and keys are exactly what the
   joint seed gives under the standard in the offer;
3. that standard is the one the examiner had published before the exam was
   requested, a moment the examinee picks;
4. the grade, recomputed from the opened answers;
5. the calibration loss.  A self-estimate that does not open is charged the
   worst loss any claim could have incurred for that score, so that spoiling
   one's own opening never pays.

``check_result`` settles the exam a published ``exam.result`` is about and
compares every number in the post with the settled one.  The numbers to use
afterwards are the settled ones it returns, never the ones in the post.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..benchmark.schema import Answer, AnswerKey, Instance, Probe, Session, Turn
from ..benchmark.synthetic import generate
from ..evalcore.calibration import AsymmetricLoss
from ..evalcore.criteria import CriteriaFamily
from ..evalcore.grader import answers_hash
from ..evalcore.module import EvaluationModule, compare_receipt, keys_commitment
from ..evalcore.standard import Standard
from ..util import canonical_json, hash_obj, sha256_hex, verify_commit

__all__ = [
    "pack_probes", "unpack_probes", "instances_from", "derive_exam_share", "share_commitment",
    "joint_exam_seed", "Verdict", "Reading", "ExamIndex", "read_exam", "check_paper", "settle", "called_off",
    "check_result", "verify_result", "round_at", "draw_paper", "CLOCK_AUTHOR", "MAX_EXAM_STREAMS",
    "MAX_EXAM_STREAM_QUESTIONS", "MAX_ID_CHARS", "GRACE_CLOCKS",
]

CLOCK_AUTHOR = "env"

# The largest exam a verifier is prepared to regenerate: the cost of a check is
# set by the party being checked, so it has to be bounded.
MAX_EXAM_STREAMS = 8
MAX_EXAM_STREAM_QUESTIONS = 48
# Exam ids and seed shares are repeated in every post of an exam; long ones are not ids.
MAX_ID_CHARS = 128
# The largest answer sheet a reader will grade (canonical JSON, in characters).
MAX_SHEET_CHARS = 400_000
# A party called on to act has until this many clock posts have followed the call.
GRACE_CLOCKS = 2


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


def derive_exam_share(private_seed: Any, unit_id: str, examinee: str, xid: str) -> str:
    """The examiner's share of one exam's seed.  One-way in the private seed, and never the same for two exams."""
    return sha256_hex(f"exam-share|{private_seed}|{unit_id}|{examinee}|{xid}")[:40]


def share_commitment(share: str, examinee: str, xid: str) -> str:
    """Commitment to a seed share, bound to the exam it is for: it cannot be shown again for another."""
    return sha256_hex(f"exam-share-commit|{share}|{examinee}|{xid}")


def joint_exam_seed(share: str, nonce: str, xid: str) -> str:
    """Seed of one exam: fixed only once both parties have contributed."""
    return "exam-" + sha256_hex(f"exam-seed|{share}|{nonce}|{xid}")[:32]


def _num(x: Any) -> float | None:
    """A finite real number, or None.  NaN, infinity and numbers too large to hold never pass a check."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        x = float(x)
    except OverflowError:
        return None
    return x if math.isfinite(x) else None


def _close(a: Any, b: Any, tol: float) -> bool:
    a, b = _num(a), _num(b)
    return a is not None and b is not None and abs(a - b) <= tol


def _text(x: Any, limit: int = MAX_ID_CHARS) -> bool:
    """A non-empty string, short enough to be an id, that can be hashed."""
    if not isinstance(x, str) or not 0 < len(x) <= limit:
        return False
    try:
        x.encode("utf-8")
    except UnicodeError:
        return False
    return True


def _opens(commitment: Any, value: Any, nonce: Any) -> bool:
    """Whether (value, nonce) opens a commitment.  Anything that cannot even be checked does not open."""
    try:
        return isinstance(commitment, str) and isinstance(nonce, str) and verify_commit(commitment, value, nonce)
    except Exception:  # noqa: BLE001 - an opening that cannot be hashed is no opening
        return False


def _terms(body: Mapping) -> tuple[int, int, AsymmetricLoss] | None:
    """The size of the exam and the loss, as the examiner committed to them, or None if they are not usable."""
    size = body.get("exam_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    if any(isinstance(x, bool) or not isinstance(x, int) for x in size):
        return None
    streams, questions = size
    if not (1 <= streams <= MAX_EXAM_STREAMS and 1 <= questions <= MAX_EXAM_STREAM_QUESTIONS):
        return None
    try:
        loss = AsymmetricLoss.from_dict(body.get("loss"))
    except Exception:  # noqa: BLE001
        return None
    return streams, questions, loss


def round_at(clocks: Sequence[tuple[int, int]], seq: int) -> int:
    """The round in force at a position on the chain: that of the last clock post up to and including it."""
    current = 0
    for position, number in clocks:
        if position > seq:
            break
        current = number
    return current


# --------------------------------------------------------------------------- the reading

_EXAMINEE_KINDS = ("exam.request", "exam.nonce", "exam.sealed", "exam.open")
_EXAMINER_KINDS = ("exam.ready", "exam.offer", "exam.keys", "exam.decline")


class ExamIndex:
    """Every exam post on a board, grouped by exam, read in one pass.

    A reader that looks at many exams builds this once instead of scanning
    feeds per exam.  ``board.read`` is the only access used.
    """

    def __init__(self, board: Any, clock_author: str = CLOCK_AUTHOR):
        self.clocks: list[tuple[int, int]] = []  # (position, round number)
        self.profiles: dict[str, list] = {}
        self.exams: dict[tuple[str, str, str], tuple[list, list]] = {}  # (examiner, examinee, id) -> (theirs, ours)
        wanted = list(_EXAMINEE_KINDS + _EXAMINER_KINDS) + ["clock", "profile"]
        for post in board.read(0, kinds=wanted):
            if post.kind == "clock":
                if post.author == clock_author:
                    number = post.body.get("round") if isinstance(post.body, Mapping) else None
                    self.clocks.append((post.seq, number if isinstance(number, int) and not isinstance(number, bool) else len(self.clocks) + 1))
                continue
            if post.kind == "profile":
                self.profiles.setdefault(post.author, []).append(post)
                continue
            xid = post.body.get("xid") if isinstance(post.body, Mapping) else None
            if not _text(xid) or not _text(post.to, 256):
                continue
            if post.kind in _EXAMINEE_KINDS:
                self.exams.setdefault((post.to, post.author, xid), ([], []))[0].append(post)
            else:
                self.exams.setdefault((post.author, post.to, xid), ([], []))[1].append(post)

    def posts(self, examiner: str, examinee: str, xid: str) -> tuple[list, list]:
        return self.exams.get((examiner, examinee, xid), ([], []))


def _from_board(board: Any, examiner: str, examinee: str, xid: str, clock_author: str):
    def about(author: str, to: str, kinds: Sequence[str]) -> list:
        return [
            p for p in board.feed(author, kinds=list(kinds))
            if isinstance(p.body, Mapping) and p.body.get("xid") == xid and p.to == to
        ]

    clocks = []
    for post in board.feed(clock_author, kinds=["clock"]):
        number = post.body.get("round") if isinstance(post.body, Mapping) else None
        clocks.append((post.seq, number if isinstance(number, int) and not isinstance(number, bool) else len(clocks) + 1))
    return about(examinee, examiner, _EXAMINEE_KINDS), about(examiner, examinee, _EXAMINER_KINDS), clocks


@dataclass(frozen=True)
class Reading:
    """What the chain says about one exam: its posts, whose move it is, and whether it is over."""

    examiner: str = ""
    examinee: str = ""
    xid: str = ""
    stage: str = "none"
    reason: str = ""
    fault: str = ""  # "examiner" when the exam is over because of something the examiner did or failed to do
    final_seq: int | None = None  # the position at which the outcome became final; None while it can still change
    void: str = ""  # "sealed" or "open" when the examinee went silent at that step
    terms: Any = None  # (streams, questions per stream, loss), from exam.ready
    request: Any = None
    ready: Any = None
    nonce: Any = None
    offer: Any = None
    sealed: Any = None
    keys: Any = None
    opened: Any = None
    decline: Any = None

    @property
    def settled(self) -> bool:
        """The examinee's part is over and a grade can be computed (if the paper holds)."""
        return self.stage in ("opened", "void-sealed", "void-open")

    @property
    def over(self) -> bool:
        return self.final_seq is not None


def read_exam(
    board: Any, examiner: str, examinee: str, xid: str,
    index: ExamIndex | None = None, clock_author: str = CLOCK_AUTHOR,
) -> Reading:
    """Read exam ``xid`` between two units off the chain.  Never raises.

    Stages, and whose move each is:

        none, requested          examiner: answer the request (exam.ready or exam.decline)
        declined                 over; nobody is bound
        ready                    examinee: send its seed share, if the terms suit it
        seeded                   examiner: make the offer            (it now knows the paper)
        offered                  examinee: seal its answers
        unsealed                 examiner: the examinee ran out of time; reveal the keys
        sealed                   examiner: reveal the keys
        keyed                    examinee: open its answers
        opened                   over: to be graded
        void-sealed, void-open   over: the examinee went silent; every answer is missing
        called-off               over: the examiner showed that no paper could be drawn
        abandoned, spoiled       over: the examiner's doing
    """
    try:
        return _read_exam(board, examiner, examinee, xid, index, clock_author)
    except Exception as exc:  # noqa: BLE001 - whatever is on the chain, reading it must end in a reading
        return Reading(examiner=str(examiner), examinee=str(examinee), xid=str(xid), stage="none",
                       reason=f"the exam could not be read: {type(exc).__name__}: {str(exc)[:120]}")


def _read_exam(board, examiner, examinee, xid, index, clock_author) -> Reading:
    base = dict(examiner=examiner, examinee=examinee, xid=xid)
    if not _text(xid) or not _text(examiner, 256) or not _text(examinee, 256):
        return Reading(**{k: str(v) for k, v in base.items()}, reason="no exam or no examinee is named")
    if examinee == examiner:
        return Reading(**base, reason="a unit cannot examine itself")
    if index is not None:
        (theirs, ours), clocks = index.posts(examiner, examinee, xid), index.clocks
    else:
        theirs, ours, clocks = _from_board(board, examiner, examinee, xid, clock_author)

    def first(posts: list, kind: str, after: int, accept=None) -> Any:
        for p in posts:
            if p.kind == kind and p.seq > after and (accept is None or accept(p)):
                return p
        return None

    def deadline(after: int) -> int | None:
        """The position of the clock post that ends the time allowed after a call made at ``after``."""
        later = [position for position, _n in clocks if position > after]
        return later[GRACE_CLOCKS - 1] if len(later) >= GRACE_CLOCKS else None

    request = first(theirs, "exam.request", 0)
    if request is None:
        return Reading(**base, reason="the named examinee did not request this exam")
    base["request"] = request
    ready = first(ours, "exam.ready", request.seq, lambda p: isinstance(p.body.get("share_commit"), str) and _terms(p.body) is not None)
    decline = first(ours, "exam.decline", request.seq)
    if ready is None or (decline is not None and decline.seq < ready.seq):
        if decline is not None:
            return Reading(**base, stage="declined", decline=decline, final_seq=decline.seq, reason="the examiner declined")
        return Reading(**base, stage="requested", reason="the examiner has not answered the request")
    base.update(ready=ready, terms=_terms(ready.body))
    nonce = first(theirs, "exam.nonce", ready.seq, lambda p: _text(p.body.get("nonce")))
    if nonce is None:
        return Reading(**base, stage="ready", reason="the named examinee did not contribute to the exam seed")
    base["nonce"] = nonce

    # From here on the examiner knows the paper, and owes the exam.
    offer = first(ours, "exam.offer", nonce.seq)
    if offer is None:
        due = deadline(nonce.seq)
        shown = first(ours, "exam.decline", nonce.seq, lambda p: isinstance(p.body.get("share"), str))
        if shown is not None and (due is None or shown.seq < due):
            return Reading(**base, stage="called-off", decline=shown, final_seq=shown.seq,
                           reason="the examiner says no paper can be drawn for this seed")
        if due is None:
            return Reading(**base, stage="seeded", reason="the examiner has not made its offer yet")
        return Reading(**base, stage="abandoned", fault="examiner", final_seq=due, reason="the examiner made no offer for this exam")
    base["offer"] = offer

    seal_due = deadline(offer.seq)
    keys = first(ours, "exam.keys", offer.seq)
    sealed = first(
        theirs, "exam.sealed", offer.seq,
        lambda p: p.body.get("offer_seq") == offer.seq and isinstance(p.body.get("answers_commit"), str)
        and isinstance(p.body.get("guess_commit"), str),
    )
    if sealed is not None and ((seal_due is not None and sealed.seq > seal_due) or (keys is not None and sealed.seq > keys.seq)):
        sealed = None  # too late to count
    if sealed is None:
        if keys is not None:
            base["keys"] = keys
            if seal_due is None or keys.seq < seal_due:
                return Reading(**base, stage="spoiled", fault="examiner", final_seq=keys.seq,
                               reason="the keys were revealed before the examinee had sealed its answers or run out of time")
            return Reading(**base, stage="void-sealed", void="sealed", final_seq=keys.seq,
                           reason="the examinee saw the questions and sealed nothing in time")
        if seal_due is None:
            return Reading(**base, stage="offered", reason="the examinee has not sealed its answers yet")
        late = deadline(seal_due)
        if late is None:
            return Reading(**base, stage="unsealed", reason="the examinee sealed nothing in time; the keys are due")
        return Reading(**base, stage="abandoned", fault="examiner", final_seq=late,
                       reason="the examiner never revealed the keys of an exam it had offered")
    base["sealed"] = sealed

    keys_due = deadline(sealed.seq)
    if keys is None or (keys_due is not None and keys.seq > keys_due):
        if keys_due is None:
            return Reading(**base, stage="sealed", reason="the examiner has not revealed the keys yet")
        return Reading(**base, stage="abandoned", fault="examiner", final_seq=keys_due,
                       reason="the examiner did not reveal the keys in time after the answers were sealed")
    base["keys"] = keys

    open_due = deadline(keys.seq)
    commitment = sealed.body.get("answers_commit")
    opened = first(
        theirs, "exam.open", keys.seq,
        lambda p: (open_due is None or p.seq < open_due) and isinstance(p.body.get("answers"), str)
        and p.body["answers"] in (p.body.get("blobs") or ())  # the sheet was on the board when it was opened
        and _opens(commitment, p.body["answers"], p.body.get("answers_nonce")),
    )
    if opened is not None:
        return Reading(**base, stage="opened", opened=opened, final_seq=opened.seq, reason="the answers were opened")
    if open_due is None:
        return Reading(**base, stage="keyed", reason="the examinee still has time to open its answers")
    return Reading(**base, stage="void-open", void="open", final_seq=open_due,
                   reason="the examinee sealed answers, saw the keys and opened nothing in time")


# --------------------------------------------------------------------------- work done once
#
# Drawing a paper from a seed and grading a sheet against it are pure functions
# of public data, and one exam is checked by its examiner, its examinee, every
# reader of its result and every auditor.  In one process that is the same
# computation several times over.  The two tables below remember results under
# a key made of *everything* the computation reads, so a reader that is shown
# different data gets a different computation.  They change how often the work
# is done and never what a reader concludes; a test runs a society with and
# without them and compares the chains.

MEMO_ENTRIES = 512
_MEMO_LOCK = threading.Lock()
_PAPERS: "OrderedDict[tuple, _Paper]" = OrderedDict()
_GRADES: "OrderedDict[tuple, Any]" = OrderedDict()
_memo_on = True


def _memo(table: "OrderedDict[tuple, Any]", key: tuple, compute: Callable[[], Any]) -> Any:
    if not _memo_on:
        return compute()
    with _MEMO_LOCK:
        if key in table:
            table.move_to_end(key)
            return table[key]
    value = compute()  # an exception is never remembered
    with _MEMO_LOCK:
        table[key] = value
        while len(table) > MEMO_ENTRIES:
            table.popitem(last=False)
    return value


def use_memo(on: bool) -> None:
    """Switch the tables on or off (and empty them).  For tests that compare the two."""
    global _memo_on
    with _MEMO_LOCK:
        _memo_on = bool(on)
        _PAPERS.clear()
        _GRADES.clear()


class _Paper:
    """A drawn exam with the hashes readers compare it by."""

    def __init__(self, key: tuple, exam: Any):
        self.key = key
        self.exam = exam
        self.probes_address = sha256_hex(canonical_json(pack_probes(exam.probes())))
        self._commitments: dict[str, str] = {}

    def keys_commitment(self, salt: str) -> str:
        if salt not in self._commitments:
            if len(self._commitments) > 16:
                self._commitments.clear()
            self._commitments[salt] = keys_commitment(self.exam.instances, salt)
        return self._commitments[salt]


def _paper(seed: str, cfg: Any) -> _Paper:
    """The exam ``seed`` gives under generator settings ``cfg``.  Raises ``ValueError`` if there is none."""
    key = (str(seed), hash_obj(cfg.to_dict()))
    return _memo(_PAPERS, key, lambda: _Paper(key, generate(seed, cfg)))


def draw_paper(seed: str, cfg: Any):
    """The exam ``seed`` gives under ``cfg``: the same object ``settle`` regenerates.  Treat it as read-only."""
    return _paper(seed, cfg).exam


def _grade(module: EvaluationModule, paper: _Paper, answers: Mapping[str, Answer], salt: str):
    """``module.evaluate`` of ``answers`` on a drawn paper.

    The key holds the evaluator's fingerprint (grading code, judges, criteria
    family, standard), the paper's seed and settings, the key salt and a hash
    of every answer that will be read.
    """
    ids = {i.probe_id for i in paper.exam.instances}
    key = (module.fingerprint(), paper.key, str(salt), answers_hash({p: a for p, a in answers.items() if p in ids}))
    return _memo(_GRADES, key, lambda: module.evaluate(answers, paper.exam.instances, salt))


# --------------------------------------------------------------------------- the paper

def check_paper(board: Any, reading: Reading, index: ExamIndex | None = None):
    """Is the paper what the joint seed gives under the examiner's published standard?

    Needs the keys to be public.  Returns (ok, reason, standard, salt, the
    regenerated paper).  Every reason for failure here lies with the examiner.
    Never raises.
    """
    try:
        return _check_paper(board, reading, index)
    except Exception as exc:  # noqa: BLE001
        return False, f"the examiner's posts could not be read: {type(exc).__name__}: {str(exc)[:120]}", None, None, None


def _check_paper(board, reading, index):
    ready, nonce_post, offer, keys, request = reading.ready, reading.nonce, reading.offer, reading.keys, reading.request
    if keys is None or offer is None:
        return False, "the keys are not public", None, None, None
    share, salt, nonce = keys.body.get("share"), keys.body.get("salt"), nonce_post.body.get("nonce")
    if not _text(share) or not _text(salt):
        return False, "the exam seed share or the key salt is missing", None, None, None
    if ready.body.get("share_commit") != share_commitment(share, reading.examinee, reading.xid):
        return False, "the revealed seed share is not the one the examiner committed to", None, None, None
    try:
        standard = Standard.from_dict(offer.body["standard"])
    except (KeyError, ValueError, TypeError) as exc:
        return False, f"the offer's standard is malformed: {exc}", None, None, None
    profiles = index.profiles.get(reading.examiner, []) if index is not None else board.feed(reading.examiner, kinds=["profile"])
    profiles = [p for p in profiles if p.seq < request.seq]
    if not profiles:
        return False, "the examiner had published no standard before this exam was requested", None, None, None
    try:
        published = Standard.from_dict(profiles[-1].body["standard"]).fingerprint()
    except (KeyError, ValueError, TypeError):
        return False, "the examiner's published standard is malformed", None, None, None
    if published != standard.fingerprint():
        return False, "the exam used a standard other than the examiner's published one", None, None, None
    streams, questions, _loss = reading.terms
    try:
        paper = _paper(joint_exam_seed(share, nonce, reading.xid), standard.gen_config(streams, questions))
    except ValueError as exc:
        return False, f"no exam can be drawn from the joint seed under the stated standard: {exc}", None, None, None
    probes_blob = offer.body.get("probes")
    if not isinstance(probes_blob, str) or probes_blob not in (offer.body.get("blobs") or ()):
        return False, "the offer does not vouch for its questions being on the board", None, None, None
    if paper.probes_address != probes_blob:
        return False, "the questions were not generated from the joint seed under the stated standard", None, None, None
    if paper.keys_commitment(salt) != offer.body.get("keys_commit"):
        return False, "the answer keys are not the ones generated from the joint seed and committed to in the offer", None, None, None
    return True, "ok", standard, salt, paper


def called_off(board: Any, examiner: str, examinee: str, xid: str, index: ExamIndex | None = None) -> bool:
    """Did the examiner show that this exam's joint seed cannot be expanded under its published standard?

    An examiner may not look for another seed when one fails, so it calls the
    exam off and reveals its share; anyone can then check the claim here.
    """
    try:
        reading = read_exam(board, examiner, examinee, xid, index)
        if reading.stage != "called-off":
            return False
        share, nonce = reading.decline.body.get("share"), reading.nonce.body.get("nonce")
        if not _text(share) or reading.ready.body.get("share_commit") != share_commitment(share, examinee, xid):
            return False
        profiles = index.profiles.get(examiner, []) if index is not None else board.feed(examiner, kinds=["profile"])
        profiles = [p for p in profiles if p.seq < reading.request.seq]
        if not profiles:
            return False
        standard = Standard.from_dict(profiles[-1].body["standard"])
        streams, questions, _loss = reading.terms
        try:
            generate(joint_exam_seed(share, nonce, xid), standard.gen_config(streams, questions))
        except ValueError:
            return True
        return False
    except Exception:  # noqa: BLE001 - unreadable evidence proves nothing
        return False


# --------------------------------------------------------------------------- settlement

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
    paper: str = ""  # hash of the joint seed: a label for the exam that neither party could choose
    pending: bool = False  # not ok because the exam is still running, not because anything is wrong
    fault: str = ""  # "examiner" when nothing was settled because of the examiner
    stage: str = ""
    final_seq: int | None = None  # where on the chain this outcome became final


def _fail(reason: str, pending: bool = False, **more: Any) -> Verdict:
    return Verdict(False, reason, pending=pending, **more)


def settle(
    board: Any, family: CriteriaFamily, examiner: str, examinee: str, xid: str,
    index: ExamIndex | None = None, clock_author: str = CLOCK_AUTHOR,
) -> Verdict:
    """What exam ``xid`` between two units established, judged from the chain.

    Once a verdict is not ``pending`` it is final: nothing posted later changes
    it, so two readers of the same chain agree whenever both have an answer.
    Never raises: everything examined here was written by other units, and
    evidence that cannot be read is evidence that does not hold.
    """
    try:
        return _settle(board, family, examiner, examinee, xid, index, clock_author)
    except Exception as exc:  # noqa: BLE001 - untrusted input must end in a verdict, not a crash
        return _fail(f"the evidence could not be read: {type(exc).__name__}: {str(exc)[:120]}")


def _settle(board, family, examiner, examinee, xid, index, clock_author) -> Verdict:
    reading = read_exam(board, examiner, examinee, xid, index, clock_author)
    common = dict(xid=reading.xid, examiner=reading.examiner, examinee=reading.examinee, stage=reading.stage,
                  final_seq=reading.final_seq)
    if not reading.settled:
        return _fail(reading.reason, pending=not reading.over and reading.stage not in ("none",), fault=reading.fault, **common)
    ok, why, standard, salt, paper = check_paper(board, reading, index)
    if not ok:
        return _fail(why, fault="examiner", **dict(common, stage="spoiled"))

    # ---- the grade: from the examinee's own opened answers and the regenerated keys
    answers: dict[str, Answer] = {}
    opened, sealed = reading.opened, reading.sealed
    if opened is not None:
        try:
            blob = board.get_blob(opened.body["answers"])
            if not isinstance(blob, Mapping) or len(canonical_json(blob)) > MAX_SHEET_CHARS:
                raise ValueError("not an answer sheet")
            answers = {str(pid): Answer.from_dict(a) for pid, a in blob.items()}
        except Exception:  # noqa: BLE001 - it opened something that is not an answer sheet: every answer is missing
            answers = {}
    evaluation = _grade(EvaluationModule(standard, family), paper, answers, salt)
    score = float(evaluation.score)

    # ---- the self-estimate and the calibration loss
    _streams, _questions, loss_spec = reading.terms
    guess, guess_valid = 1.0, False
    if opened is not None:
        revealed = _num(opened.body.get("guess"))
        if revealed is not None and _opens(sealed.body.get("guess_commit"), opened.body.get("guess"), opened.body.get("guess_nonce")):
            guess, guess_valid = min(1.0, max(0.0, revealed)), True
    # An estimate that does not open is charged the worst any claim could have cost for this score.
    loss = loss_spec(guess, score) if guess_valid else max(loss_spec(0.0, score), loss_spec(1.0, score))
    if not math.isfinite(loss):
        return _fail("the loss the examiner announced does not give a finite number", fault="examiner", **dict(common, stage="spoiled"))

    share, nonce = reading.keys.body["share"], reading.nonce.body["nonce"]
    return Verdict(
        True, "ok", score=score, n_items=int(evaluation.receipt["n_items"]),
        capability={c: (float(s), int(n)) for c, (s, n) in evaluation.capability.items()},
        guess=guess, guess_valid=guess_valid, loss=float(loss), void=reading.void, receipt=dict(evaluation.receipt),
        paper=sha256_hex("paper|" + joint_exam_seed(share, nonce, reading.xid)), **common,
    )


def compare_result(body: Any, settled: Verdict) -> tuple[bool, str]:
    """Does a result post say what the settled exam says?  Every number is compared."""
    if not isinstance(body, Mapping):
        return False, "not an exam result"
    if (body.get("void") or "") != settled.void:
        return False, "the result misstates whether the examinee sealed and opened its answers"
    ok, why = compare_receipt(body.get("receipt"), settled.receipt)
    if not ok:
        return False, why
    if not _close(body.get("score"), settled.score, 1e-6):
        return False, "the published score differs from the settled one"
    claimed = body.get("capability")
    if not isinstance(claimed, Mapping) or set(claimed) != set(settled.capability):
        return False, "the published ability table does not list exactly the abilities examined"
    for cap, (s, n) in settled.capability.items():
        got = claimed[cap]
        if not isinstance(got, (list, tuple)) or len(got) != 2 or not _close(got[0], s, 1e-6) or got[1] != n:
            return False, f"the published ability score differs from the settled one for {cap}"
    if body.get("guess_valid") is not settled.guess_valid or not _close(body.get("guess"), settled.guess, 1e-6):
        return False, "the result misquotes the examinee's self-estimate"
    if not _close(body.get("loss"), settled.loss, 1e-5):  # inputs are stored to six decimals
        return False, "the calibration loss does not follow from the self-estimate, the score and the announced loss"
    return True, "ok"


def check_result(
    board: Any, post: Any, family: CriteriaFamily, clock_author: str = CLOCK_AUTHOR,
    index: ExamIndex | None = None, settled: Verdict | None = None,
) -> Verdict:
    """Verify one ``exam.result`` post: settle its exam and compare every number.  Never raises.

    A result is only good if the exam's outcome was already final when it was
    posted.  ``settled`` lets a caller that has settled the exam pass the
    verdict in instead of having it computed again.
    """
    try:
        body = post.body
        if post.kind != "exam.result" or not isinstance(body, Mapping):
            return _fail("not an exam result")
        if settled is None:
            settled = settle(board, family, post.author, body.get("examinee"), body.get("xid"), index, clock_author)
        if not settled.ok:
            return settled
        if settled.final_seq is None or settled.final_seq >= post.seq:
            return _fail("the result was published before the exam was over")
        ok, why = compare_result(body, settled)
        return settled if ok else _fail(why)
    except Exception as exc:  # noqa: BLE001 - untrusted input must end in a verdict, not a crash
        return _fail(f"the evidence could not be read: {type(exc).__name__}: {str(exc)[:120]}")


def verify_result(board: Any, post: Any, family: CriteriaFamily) -> tuple[bool, str]:
    """(ok, reason) for one ``exam.result`` post.  See ``check_result``."""
    verdict = check_result(board, post, family)
    return verdict.ok, verdict.reason
