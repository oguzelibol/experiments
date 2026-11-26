"""What the society knows as a whole, computed from public posts only.

Three things are derived from the board:

* **Reputation.**  An examiner's signals count in proportion to how well the
  examiner itself performs in the eyes of the trusted units, measured by them
  first-hand.  Examiners with a result that does not verify get zero.
* **Consensus standing.**  For each ability, each unit's score averaged over
  the examiners who measured it, weighted by examiner reputation and evidence,
  with examiner leniency fitted and removed.
* **Champions and a collective estimate.**  For each kind of question the
  society's best units answer it.  Champions are chosen on one half of the
  signals and their expected score is read from the other half, in both
  directions, so the estimate is not inflated by picking the luckiest
  measurement and no signal is wasted.

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
    "Signal", "Audit", "gather_signals", "audit_record", "audit_board", "confirmed_cheats", "published_since",
    "reputation", "consensus",
    "CLASS_CAPABILITIES", "select_champions", "champion_for", "collective_estimate", "verify_result_post",
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
    paper: str = ""  # hash of the exam's joint seed, which neither party chose


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
                paper=verdict.paper,
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


def published_since(board: Any, units: Sequence[str]) -> dict[str, int]:
    """For each unit, the round from which its current design has been in use, as the board shows it.

    A unit's profile carries a version number that it raises when it adopts a
    new design.  Signals about an earlier version say nothing about the unit as
    it is now.  The round is that of the first profile carrying the latest
    version; a unit's private state is never consulted, so a reader with a copy
    of the board gets the same answer.
    """
    out: dict[str, int] = {}
    for unit in units:
        profiles = board.feed(unit, kinds=["profile"])
        since = 0
        if profiles:
            latest = profiles[-1].body.get("version") if isinstance(profiles[-1].body, Mapping) else None
            for post in profiles:
                if isinstance(post.body, Mapping) and post.body.get("version") == latest:
                    since = int(post.round)
                    break
        out[unit] = since
    return out


def reputation(
    signals: Sequence[Signal], units: Sequence[str], banned: set[str] = frozenset(),
    seeds: Sequence[str] | None = None, prior_strength: float = 12.0,
) -> dict[str, float]:
    """How well each unit performs in the eyes of the trusted units, measured first-hand.

    ``seeds`` are the units whose judgement is trusted to begin with (for a
    designer: the units it built; by default every unit).  A unit's reputation
    is the mean score the seeds gave it on their own exams, weighted by the
    number of questions and pulled towards a prior while evidence is thin: the
    seeds' average for another seed, zero for anyone else, who has to earn it.

    Nothing is passed on at second hand.  What one outsider says about another
    moves nothing, so a ring of units praising each other gains exactly
    nothing, and neither does a unit for sitting many exams or for being some
    examiner's only examinee.  A unit's opinion of itself is ignored, and a
    unit caught publishing a false result has no reputation.

    Returned values are scaled to sum to one over the units.
    """
    units = list(units)
    live = {u for u in units if u not in banned}
    if not live:
        return {u: 0.0 for u in units}
    seed_set = {u for u in (seeds if seeds is not None else units) if u in live} or set(live)
    got: dict[str, list[float]] = {}
    for s in signals:
        if s.examiner == s.examinee or s.examiner not in seed_set or s.examinee not in live or s.n <= 0:
            continue
        score = s.score if isinstance(s.score, (int, float)) and s.score == s.score else 0.0
        acc = got.setdefault(s.examinee, [0.0, 0.0])
        acc[0] += min(1.0, max(0.0, score)) * s.n
        acc[1] += s.n
    total_n = sum(a[1] for a in got.values())
    mean = sum(a[0] for a in got.values()) / total_n if total_n > 0 else 0.3
    level = {}
    for u in units:
        if u not in live:
            level[u] = 0.0
            continue
        num, den = got.get(u, (0.0, 0.0))
        prior = mean if u in seed_set else 0.0
        level[u] = (num + prior_strength * prior) / (den + prior_strength)
    total = sum(level.values())
    if total <= 0:
        return {u: (1.0 / len(seed_set) if u in seed_set else 0.0) for u in units}
    return {u: v / total for u, v in level.items()}


# Pseudo-items that hold an examiner's leniency estimate at zero until it has graded enough.
_OFFSET_PRIOR_ITEMS = 8.0


def consensus(
    signals: Sequence[Signal], rep: Mapping[str, float], current_since: Mapping[str, int] | None = None,
    debias: bool = True, iterations: int = 500, max_items: int | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """capability -> unit -> (reputation-weighted score, effective evidence in items).

    Examiners differ in leniency (their standards differ), and units are not all
    measured by the same examiners.  With ``debias`` an additive model
    ``score = examinee level + examiner offset`` is fitted by alternating
    weighted means until it stops moving, so standings are on the scale of an
    average examiner.  An examiner's offset is held towards zero by a few
    pseudo-items of its own weight, which fade as it grades more.

    A signal weighs its examiner's reputation times its number of questions.
    ``max_items`` caps the questions one exam can count for (pass the world's
    exam size), so volume cannot stand in for reputation.  The evidence
    returned is an effective sample size consistent with those weights: items
    graded by an examiner of little reputation add little.
    """
    obs = []  # (examiner, examinee, capability, score, weight, n)
    for s in signals:
        r = rep.get(s.examiner, 0.0)
        if r <= 0:
            continue
        if current_since is not None and s.round < current_since.get(s.examinee, 0):
            continue  # measured an earlier version of the examinee
        scale = min(1.0, max_items / s.n) if max_items and s.n > max_items else 1.0
        for cap, (score, n) in s.capability.items():
            if n > 0:
                obs.append((s.examiner, s.examinee, cap, score, r * n * scale, n * scale))
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
        fresh = {j: a[0] / (a[1] + rep.get(j, 0.0) * _OFFSET_PRIOR_ITEMS) for j, a in acc_j.items()}
        centre = sum(fresh[j] * acc_j[j][1] for j in fresh) / sum(a[1] for a in acc_j.values())
        fresh = {j: b - centre for j, b in fresh.items()}
        moved = max(abs(fresh[j] - offset.get(j, 0.0)) for j in fresh)
        offset = fresh
        if moved < 1e-10:
            # one more pass of levels under the settled offsets
            acc = {}
            for (j, i, cap, score, w, _n) in obs:
                a = acc.setdefault((cap, i), [0.0, 0.0])
                a[0] += w * (score - offset.get(j, 0.0))
                a[1] += w
            level = {k: a[0] / a[1] for k, a in acc.items()}
            break
    sum_w: dict[tuple[str, str], float] = {}
    sum_w2n: dict[tuple[str, str], float] = {}
    for (_j, i, cap, _score, w, n) in obs:
        sum_w[(cap, i)] = sum_w.get((cap, i), 0.0) + w
        sum_w2n[(cap, i)] = sum_w2n.get((cap, i), 0.0) + w * w / n
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for (cap, i), v in level.items():
        out.setdefault(cap, {})[i] = (min(1.0, max(0.0, v)), sum_w[(cap, i)] ** 2 / sum_w2n[(cap, i)])
    return out


def _class_score(cons: Mapping, unit: str, cls: str, priors: Mapping[str, float], strength: float) -> tuple[float, float]:
    """A unit's score on a class of questions and the evidence behind it.

    Each ability the class draws on is estimated on its own (pulled towards
    its own prior while evidence is thin) and the abilities are then mixed in
    the class's proportions, so a unit examined mostly on one of two abilities
    is not credited with the other.  The evidence is the number of questions
    that would give the mixture the same precision.
    """
    score = inverse = 0.0
    for cap, w in CLASS_CAPABILITIES[cls].items():
        level, n = cons.get(cap, {}).get(unit, (0.0, 0.0))
        score += w * (level * n + strength * priors[cap]) / (n + strength)
        inverse += w * w / (n + strength)
    return score, max(0.0, 1.0 / inverse - strength)


def _half(signal: Signal) -> int:
    """Which half of the evidence a signal falls in: decided by the exam's joint seed, which neither party chose."""
    return int(sha256_hex("half|" + (signal.paper or signal.xid))[:2], 16) % 2


def _regressed(cons: Mapping, cap: str, unit: str, fallback: float) -> float:
    """A unit's level on one ability, regressed towards the mean of all units measured on it.

    The best of several noisy measurements overstates the unit behind it.  How
    much to take back depends on how much of the spread between units is real:
    with ``t`` the variance of true levels (the observed spread minus what
    measurement noise alone would produce) and ``v`` the noise variance of this
    unit's own measurement, the expected true level is
    ``mean + t / (t + v) * (measured - mean)``.  This is the standard
    empirical-Bayes estimate, and it stays right for a unit picked because it
    looked best.
    """
    rows = cons.get(cap, {})
    total = sum(n for _l, n in rows.values())
    if total <= 0:
        return fallback
    mean = sum(l * n for l, n in rows.values()) / total
    got = rows.get(unit)
    if got is None or got[1] <= 0 or len(rows) < 3:
        return mean
    spread = sum(n * (l - mean) ** 2 for l, n in rows.values()) / total
    noise = sum(n * (max(mean * (1 - mean), 0.01) / n) for _l, n in rows.values()) / total
    real = max(0.0, spread - noise)
    own = max(mean * (1 - mean), 0.01) / got[1]
    return mean + real / (real + own) * (got[0] - mean)


def _levels(cons: Mapping, cap: str, among: Sequence[str] | None = None) -> float | None:
    rows = [(v[0], v[1]) for u, v in cons.get(cap, {}).items() if among is None or u in among]
    weight = sum(n for _l, n in rows)
    return sum(l * n for l, n in rows) / weight if weight > 0 else None


def select_champions(
    signals: Sequence[Signal], rep: Mapping[str, float], units: Sequence[str],
    current_since: Mapping[str, int] | None = None, prior_strength: float = 4.0, max_items: int | None = None,
    reading_strength: float = 0.5,
) -> dict[str, dict]:
    """Pick, per question class, the units the society measures as best, and what to expect of them.

    The signals are split in two by a hash of each exam's joint seed.  One
    champion is chosen on the first half and its expected score is read from
    the second; another is chosen on the second half and read from the first.
    Each reading is independent of the choice it belongs to, so it is not
    inflated by having picked the luckiest measurement, and every signal is
    used once for choosing and once for estimating.  The class is then served
    by both champions, half of the questions each (they are usually the same
    unit), and the expectation is the mean of the two readings.

    A reading is taken as measured.  Pulling it towards the average unit
    would bias it low, because the champion was chosen for being better than
    average, and the outside loss charges most for an estimate that is too
    low.  Where a champion has little or no evidence in its estimation half,
    the reading leans instead on the champion's own score in the selection
    half, regressed towards the mean by as much as the spread between units
    and the noise of its measurement call for (see ``_regressed``).
    ``reading_strength`` is the weight of that fallback in questions;
    ``prior_strength`` steadies the ranking, where bias does not matter and
    noise does.
    """
    halves = ([s for s in signals if _half(s) == 0], [s for s in signals if _half(s) == 1])
    cons = [consensus(h, rep, current_since, max_items=max_items) for h in halves]
    cons_all = consensus(signals, rep, current_since, max_items=max_items)
    eligible = [u for u in units if rep.get(u, 0.0) > 0]
    out: dict[str, dict] = {}
    if not eligible:
        return out
    for cls, caps in CLASS_CAPABILITIES.items():
        everyone = {cap: _levels(cons_all, cap) for cap in caps}
        parts = []
        for chosen_on, read_from in ((0, 1), (1, 0)):
            wide = {cap: _levels(cons[chosen_on], cap) for cap in caps}
            wide = {cap: v if v is not None else (everyone[cap] if everyone[cap] is not None else 0.3) for cap, v in wide.items()}
            ranked = sorted(
                ((u, _class_score(cons[chosen_on], u, cls, wide, prior_strength)[0]) for u in eligible),
                key=lambda t: (-t[1], t[0]),
            )
            champion = ranked[0][0]
            near = {cap: _regressed(cons[chosen_on], cap, champion, wide[cap]) for cap in caps}
            expected, evidence = _class_score(cons[read_from], champion, cls, near, reading_strength)
            parts.append({"unit": champion, "selection_score": ranked[0][1], "expected": expected, "evidence": evidence})
        absts = [cons_all.get("abstention", {}).get(part["unit"]) for part in parts]
        absts = [a[0] for a in absts if a]
        out[cls] = {
            "unit": parts[0]["unit"], "units": [part["unit"] for part in parts], "parts": parts,
            "selection_score": parts[0]["selection_score"],
            "expected": sum(part["expected"] for part in parts) / len(parts),
            "evidence": sum(part["evidence"] for part in parts),
            "abstention": sum(absts) / len(absts) if absts else None,
        }
    return out


def champion_for(champions: Mapping[str, Mapping], cls: str, probe_id: str) -> str | None:
    """Which of a class's champions answers this question: each takes half, by a hash of the question id."""
    info = champions.get(cls)
    if not info:
        return None
    units = info.get("units") or [info["unit"]]
    return units[int(sha256_hex("serve|" + probe_id)[:2], 16) % len(units)]


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
        answerable = 1.0
        p = champ["expected"]
        if champ.get("abstention") is not None:
            answerable = 1 - abstention_share
            p = answerable * p + abstention_share * champ["abstention"]
        mean += share * p
        parts = champ.get("parts") or [{"expected": champ["expected"], "evidence": champ["evidence"]}]
        spread = sum(max(q["expected"] * (1 - q["expected"]), 0.01) / max(1.0, q["evidence"]) for q in parts) / len(parts) ** 2
        var += (share * answerable) ** 2 * spread
    var += max(mean * (1 - mean), 0.01) / total  # finite outside benchmark
    sd = math.sqrt(var)
    return {"mean": mean, "sd": sd, "report": loss.best_report(mean, sd, shade)}
