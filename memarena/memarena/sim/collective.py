"""What the society knows as a whole, computed from public posts only.

Three things are derived from the board:

* **Reputation.**  An examiner's signals should count in proportion to how
  well the examiner itself performs in the eyes of other reputable examiners.
  That is a fixed point, computed here by damped power iteration over the
  matrix of grades (EigenTrust-style).  Examiners with a verified audit failure
  get zero.
* **Consensus standing.**  For each ability, each unit's score averaged over
  the examiners who measured it, weighted by examiner reputation and evidence.
* **Champions and a collective estimate.**  For each kind of question the
  society's best unit answers it.  The champion is chosen on one random half of
  the signals and its expected score is read from the other half, so the
  estimate is not inflated by picking the luckiest measurement.

Nothing here reads a unit's private state.  An outside observer with a copy of
the board can reproduce every number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..actors.protocol import CLOCK_AUTHOR, called_off, check_result, exam_posts, settle, verify_result
from ..evalcore.calibration import EXTERNAL_LOSS, AsymmetricLoss
from ..evalcore.criteria import CriteriaFamily
from ..util import sha256_hex

__all__ = [
    "Signal", "Audit", "gather_signals", "audit_record", "audit_board", "confirmed_cheats", "reputation", "consensus",
    "CLASS_CAPABILITIES", "select_champions", "collective_estimate", "verify_result_post",
]

# Question classes observable from the question text -> the abilities they draw on.
CLASS_CAPABILITIES: dict[str, dict[str, float]] = {
    "temporal": {"temporal-reasoning": 1.0},
    "aggregate": {"multi-session": 1.0},
    "assistant": {"single-session-assistant": 1.0},
    "preference": {"single-session-preference": 1.0},
    "fact": {"single-session-user": 0.5, "knowledge-update": 0.5},
}


@dataclass(frozen=True)
class Signal:
    xid: str
    examiner: str
    examinee: str
    score: float
    n: int
    capability: Mapping[str, tuple[float, int]]
    round: int
    loss: float
    seq: int


def verify_result_post(board: Any, post: Any, family: CriteriaFamily) -> tuple[bool, str]:
    """Check one ``exam.result`` post against its public evidence."""
    return verify_result(board, post, family)


def _signal(post: Any) -> Signal | None:
    b = post.body
    try:
        return Signal(
            xid=b["xid"], examiner=post.author, examinee=b["examinee"], score=float(b["score"]),
            n=int(b["receipt"]["n_items"]),
            capability={c: (float(s), int(n)) for c, (s, n) in b["capability"].items()},
            round=post.round, loss=float(b.get("loss", 0.0)), seq=post.seq,
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def gather_signals(board: Any) -> list[Signal]:
    """Every published result, as posted and NOT verified.  Only for counting activity."""
    return [s for s in (_signal(p) for p in board.read(0, kinds=["exam.result"])) if s is not None]


@dataclass
class Audit:
    """What the public record establishes, re-derived by the reader."""

    signals: list  # one per settled exam, with the numbers settled here
    cheats: set  # examiners that published a result which does not verify
    bad: set  # the seqs of those results
    settled_without_result: int  # exams settled from the record although no valid result was published
    walkouts: dict  # examiner -> exams it left unfinished after the paper was fixed
    completion: dict  # examiner -> share of the exams it started that were settled


def audit_record(
    board: Any, family: CriteriaFamily, cache: dict | None = None,
    exam_size: Sequence[int] | None = None, clock_author: str = CLOCK_AUTHOR,
) -> Audit:
    """Verify and settle everything on the board before anything is built on it.

    * Every published result is checked; one that does not verify marks its
      examiner as a cheat and is discarded.
    * Every exam whose keys are public is settled here whether or not its
      examiner published a (valid) result, so keeping a result back, or
      publishing a false one, does not change what the society learns.
    * An exam counts once, however many results are posted about it.
    * An exam the examiner left unfinished after the paper was fixed, without
      showing that no paper could be drawn (``exam_size`` is needed to check
      that), is a walkout.  So is one it spoiled, by revealing keys that settle
      nothing.  An examiner has until the end of the following round.

    ``cache`` lets repeated calls skip what was already settled.
    """
    cache = cache if cache is not None else {}
    signals: list[Signal] = []
    cheats: set[str] = set()
    bad: set[int] = set()
    seen: set[tuple[str, str, str]] = set()

    def add(verdict, post) -> None:
        seen.add((verdict.examiner, verdict.examinee, verdict.xid))
        signals.append(
            Signal(
                xid=verdict.xid, examiner=verdict.examiner, examinee=verdict.examinee, score=verdict.score,
                n=verdict.n_items, capability=dict(verdict.capability), round=post.round, loss=verdict.loss, seq=post.seq,
            )
        )

    for post in board.read(0, kinds=["exam.result"]):
        verdict = cache.get(post.seq)
        if verdict is None:
            verdict = cache[post.seq] = check_result(board, post, family, clock_author)
        if not verdict.ok:
            cheats.add(post.author)
            bad.add(post.seq)
        elif (verdict.examiner, verdict.examinee, verdict.xid) not in seen:
            add(verdict, post)

    settled_without_result = 0
    for post in board.read(0, kinds=["exam.keys"]):
        xid = post.body.get("xid") if isinstance(post.body, Mapping) else None
        key = (post.author, post.to, xid)
        if key in seen or not isinstance(xid, str) or not isinstance(post.to, str):
            continue
        verdict = cache.get(("settled",) + key)
        if verdict is None:
            verdict = settle(board, family, post.author, post.to, xid, clock_author=clock_author)
            if verdict.ok or not verdict.pending:
                cache[("settled",) + key] = verdict
        if verdict.ok:
            add(verdict, post)
            settled_without_result += 1

    # Exams an examiner started and then left unfinished or spoiled, once it knew the paper.
    walkouts: dict[str, int] = {}
    clocks = [p.seq for p in board.feed(clock_author, kinds=["clock"])]
    counted: set[tuple[str, str, str]] = set()
    for post in board.read(0, kinds=["exam.nonce"]):
        xid = post.body.get("xid") if isinstance(post.body, Mapping) else None
        key = (post.to, post.author, xid)
        if key in seen or key in counted or not isinstance(xid, str) or not isinstance(post.to, str):
            continue
        if sum(1 for c in clocks if c > post.seq) < 2:
            continue  # the examiner has the rest of that round and all of the next to finish
        posts, _theirs, _ours = exam_posts(board, post.to, post.author, xid)
        if posts.nonce is None or posts.nonce.seq != post.seq:
            continue  # not the seed share of an exam that was properly started
        counted.add(key)
        if posts.keys is not None:
            # keys are out but nothing settles: every reason for that lies with the examiner
            spoiled = cache.get(("settled",) + key)
            if spoiled is None or spoiled.ok or spoiled.pending:
                continue
        elif exam_size is not None and called_off(board, post.to, post.author, xid, exam_size):
            continue
        walkouts[post.to] = walkouts.get(post.to, 0) + 1

    done: dict[str, int] = {}
    for s_ in signals:
        done[s_.examiner] = done.get(s_.examiner, 0) + 1
    completion = {e: done.get(e, 0) / (done.get(e, 0) + w) for e, w in walkouts.items()}
    return Audit(signals, cheats, bad, settled_without_result, walkouts, completion)


def audit_board(board: Any, family: CriteriaFamily, cache: dict | None = None) -> tuple[list[Signal], set[str], set[int]]:
    """(signals, examiners with a result that does not verify, the seqs of those results).  See ``audit_record``."""
    audit = audit_record(board, family, cache)
    return audit.signals, audit.cheats, audit.bad


def confirmed_cheats(board: Any, family: CriteriaFamily) -> tuple[set[str], set[int]]:
    """Examiners accused on the board whose accused result really fails verification.

    Accusations are never taken on trust: each one is recomputed here.
    """
    cheats: set[str] = set()
    bad_results: set[int] = set()
    checked: dict[int, bool] = {}
    for post in board.read(0, kinds=["audit.fail"]):
        try:
            seq = int(post.body["result_seq"])
            target = board.read(since_seq=seq - 1, limit=1)[0]
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if target.kind != "exam.result":
            continue
        if seq not in checked:
            checked[seq] = verify_result_post(board, target, family)[0]
        if not checked[seq]:
            cheats.add(target.author)
            bad_results.add(seq)
    return cheats, bad_results


def reputation(
    signals: Sequence[Signal], units: Sequence[str], banned: set[str] = frozenset(),
    damping: float = 0.85, iterations: int = 60, seeds: Sequence[str] | None = None,
) -> dict[str, float]:
    """Reputation as examiner-weighted performance (damped power iteration).

    An examiner passes its reputation on to the units it examined in
    proportion to the *level* of the scores it gave them: a unit that scored
    0.05 receives a twentieth of what a unit that scored 1.0 would.  Whatever an
    examiner does not pass on returns to the seeds.  (Normalising each
    examiner's row instead would cancel the level of its grades, and a unit
    examined only by examiners who saw nobody else would look excellent however
    badly it did.)

    ``seeds`` are the units whose judgement is trusted to begin with (for a
    designer: the units it built).  Reputation restarts from the seeds only, so
    a ring of outsiders praising each other gains nothing unless seeded units
    measure them as good first-hand.  A unit's opinion of itself is ignored.
    """
    units = [u for u in units]
    live = [u for u in units if u not in banned]
    if not live:
        return {u: 0.0 for u in units}
    live_set = set(live)
    seed_set = [u for u in (seeds if seeds is not None else live) if u in live_set] or live
    pooled: dict[tuple[str, str], list[float]] = {}
    for s in signals:
        if s.examiner == s.examinee or s.examiner not in live_set or s.examinee not in live_set or s.n <= 0:
            continue
        acc = pooled.setdefault((s.examiner, s.examinee), [0.0, 0.0])
        acc[0] += min(1.0, max(0.0, s.score)) * s.n
        acc[1] += s.n
    examined: dict[str, int] = {}
    for (i, _j) in pooled:
        examined[i] = examined.get(i, 0) + 1
    # share of examiner i's reputation that flows to examinee j
    flow = {(i, j): (a[0] / a[1]) / examined[i] for (i, j), a in pooled.items()}
    restart = {u: (1.0 / len(seed_set) if u in seed_set else 0.0) for u in live}
    rep = dict(restart)
    for _ in range(iterations):
        nxt = {u: (1.0 - damping) * restart[u] for u in live}
        passed: dict[str, float] = {}
        for (i, j), share in flow.items():
            amount = damping * rep[i] * share
            nxt[j] += amount
            passed[i] = passed.get(i, 0.0) + amount
        returned = sum(damping * rep[i] - passed.get(i, 0.0) for i in live)
        for u in live:
            nxt[u] += returned * restart[u]
        total = sum(nxt.values())
        rep = {u: v / total for u, v in nxt.items()}
    out = {u: 0.0 for u in units}
    out.update(rep)
    return out


def consensus(
    signals: Sequence[Signal], rep: Mapping[str, float], current_since: Mapping[str, int] | None = None,
    debias: bool = True, iterations: int = 12,
) -> dict[str, dict[str, tuple[float, float]]]:
    """capability -> unit -> (reputation-weighted score, evidence in items).

    Examiners differ in leniency (their standards differ), and units are not all
    measured by the same examiners.  With ``debias`` an additive model
    ``score = examinee level + examiner offset`` is fitted by alternating
    weighted means, so standings are on the scale of an average examiner.
    """
    obs = []  # (examiner, examinee, capability, score, weight, n)
    for s in signals:
        w = rep.get(s.examiner, 0.0)
        if w <= 0:
            continue
        if current_since is not None and s.round < current_since.get(s.examinee, 0):
            continue  # measured an earlier version of the examinee
        for cap, (score, n) in s.capability.items():
            if n > 0:
                obs.append((s.examiner, s.examinee, cap, score, w * n, n))
    if not obs:
        return {}
    offset: dict[str, float] = {}
    level: dict[tuple[str, str], float] = {}
    for _ in range(iterations if debias else 1):
        acc: dict[tuple[str, str], list[float]] = {}
        for (j, i, cap, score, w, _n) in obs:
            a = acc.setdefault((cap, i), [0.0, 0.0])
            a[0] += w * (score - offset.get(j, 0.0))
            a[1] += w
        level = {k: a[0] / a[1] for k, a in acc.items()}
        if not debias:
            break
        acc_j: dict[str, list[float]] = {}
        for (j, i, cap, score, w, _n) in obs:
            a = acc_j.setdefault(j, [0.0, 0.0])
            a[0] += w * (score - level[(cap, i)])
            a[1] += w
        mean_w = sum(a[1] for a in acc_j.values()) / len(acc_j)
        offset = {j: a[0] / (a[1] + 0.5 * mean_w) for j, a in acc_j.items()}  # shrunk towards zero
        centre = sum(offset[j] * acc_j[j][1] for j in offset) / sum(a[1] for a in acc_j.values())
        offset = {j: b - centre for j, b in offset.items()}
    evidence: dict[tuple[str, str], float] = {}
    for (_j, i, cap, _score, _w, n) in obs:
        evidence[(cap, i)] = evidence.get((cap, i), 0.0) + n
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for (cap, i), v in level.items():
        out.setdefault(cap, {})[i] = (min(1.0, max(0.0, v)), evidence[(cap, i)])
    return out


def _class_score(cons: Mapping, unit: str, cls: str, prior: float, strength: float) -> tuple[float, float]:
    num = den = evidence = 0.0
    for cap, w in CLASS_CAPABILITIES[cls].items():
        got = cons.get(cap, {}).get(unit)
        if got is None:
            continue
        num += w * got[0] * got[1]
        den += w * got[1]
        evidence += w * got[1]
    score = (num + strength * prior) / (den + strength)
    return score, evidence


def select_champions(
    signals: Sequence[Signal], rep: Mapping[str, float], units: Sequence[str],
    current_since: Mapping[str, int] | None = None, prior_strength: float = 8.0,
) -> dict[str, dict]:
    """Pick, per question class, the unit the society measures as best.

    Signals are split in two by a hash of the exam id.  Selection uses one half
    and the reported expectation uses the other.
    """
    half_a = [s for s in signals if int(sha256_hex(s.xid)[:2], 16) % 2 == 0]
    half_b = [s for s in signals if int(sha256_hex(s.xid)[:2], 16) % 2 == 1]
    cons_a = consensus(half_a, rep, current_since)
    cons_b = consensus(half_b, rep, current_since)
    cons_all = consensus(signals, rep, current_since)
    out: dict[str, dict] = {}
    for cls in CLASS_CAPABILITIES:
        all_scores = [
            v[0] for cap in CLASS_CAPABILITIES[cls] for v in cons_all.get(cap, {}).values()
        ]
        prior = sum(all_scores) / len(all_scores) if all_scores else 0.3
        ranked = sorted(
            ((u, *_class_score(cons_a, u, cls, prior, prior_strength)) for u in units if rep.get(u, 0.0) > 0),
            key=lambda t: (-t[1], t[0]),
        )
        if not ranked:
            continue
        champion = ranked[0][0]
        est, evidence = _class_score(cons_b, champion, cls, prior, prior_strength)
        abst = cons_all.get("abstention", {}).get(champion)
        out[cls] = {
            "unit": champion, "selection_score": ranked[0][1], "expected": est,
            "evidence": evidence, "abstention": abst[0] if abst else None,
        }
    return out


def collective_estimate(
    champions: Mapping[str, Mapping], class_counts: Mapping[str, int],
    abstention_share: float = 0.06, loss: AsymmetricLoss = EXTERNAL_LOSS, shade: float = 1.0,
) -> dict:
    """The society's own expectation of its score on an outside benchmark."""
    total = sum(class_counts.values())
    if total <= 0 or not champions:
        return {"mean": 0.0, "sd": 0.5, "report": 0.0}
    mean = var = 0.0
    for cls, count in class_counts.items():
        champ = champions.get(cls)
        share = count / total
        if champ is None:
            continue
        p = champ["expected"]
        if champ.get("abstention") is not None:
            p = (1 - abstention_share) * p + abstention_share * champ["abstention"]
        mean += share * p
        n_eff = max(1.0, champ["evidence"])
        var += share ** 2 * max(p * (1 - p), 0.01) / n_eff
    var += max(mean * (1 - mean), 0.01) / total  # finite outside benchmark
    sd = math.sqrt(var)
    return {"mean": mean, "sd": sd, "report": loss.best_report(mean, sd, shade)}
