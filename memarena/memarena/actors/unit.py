"""A unit: one self-improving memory module in a society of peers.

What a unit does, every round:

* **Knows itself.**  It runs its memory system on a fresh, private, labelled
  hold-out and grades the answers with its single evaluation module.  The result
  is its believed competence profile.
* **Examines others on what it believes it is good at.**  Its exams focus on its
  strongest abilities and are graded by the same module under its own published
  standard.  Each result is a public signal with a receipt and the revealed keys.
* **Decides whom it finds high-performing.**  Esteem for a peer comes first from
  how that peer did on the unit's own exams, tempered by how well the peer knew
  its own level beforehand and zeroed if the peer was caught issuing a result
  that does not recompute.  Signals relayed by already-esteemed peers fill in
  for peers it has not examined itself.
* **Attends to their feeds.**  It reads only the feeds of the peers it esteems
  most: their standards, their designs, the signals they publish.
* **Strives to be the best in what it does, as they measure it.**  It sits the
  exams of esteemed peers, and it changes its memory system when a change does
  better on its own specialties and under esteemed peers' standards.

All interaction goes through the board, as posts.  ``on_message`` is a pure
message handler, so the same unit runs inside the in-process simulator or
against an HTTP board.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from ..benchmark.schema import QUESTION_TYPES, Answer, AnswerKey, Instance, Probe, Session, Turn
from ..benchmark.synthetic import generate
from ..evalcore.calibration import PEER_LOSS, AsymmetricLoss, binomial_sd
from ..evalcore.criteria import SCORERS, CriteriaFamily
from ..evalcore.module import CapabilityTable, EvaluationModule, expected_score, keys_commitment
from ..evalcore.standard import CAPABILITIES, Standard, normalise
from ..memory.base import MemorySystem, run_system
from ..memory.heuristic import FEATURES, STYLES, HeuristicMemory, MemoryConfig
from ..feeds.board import blob_address
from ..util import ConfigurationError, commit, make_rng, sha256_hex
from .protocol import (
    check_paper, check_result, derive_exam_share, draw_paper, joint_exam_seed, pack_probes, read_exam, settle,
    share_commitment, unpack_probes,
)

__all__ = ["WorldRules", "UnitPolicy", "UnitSpec", "Unit", "PeerView", "pack_probes", "unpack_probes", "mutate_config"]


# ------------------------------------------------------------------ data shapes

@dataclass(frozen=True)
class WorldRules:
    """Facts about the world every unit lives under (not chosen by a designer)."""

    feature_budget: int = 5  # how many capability features one memory system may enable
    exam_streams: int = 2
    exam_questions: int = 12  # per stream
    peer_loss: AsymmetricLoss = PEER_LOSS
    family_prior: str = "literature"

    def to_dict(self) -> dict:
        return {
            "feature_budget": self.feature_budget, "exam_streams": self.exam_streams,
            "exam_questions": self.exam_questions, "peer_loss": self.peer_loss.to_dict(),
            "family_prior": self.family_prior,
        }


@dataclass(frozen=True)
class UnitPolicy:
    """Behavioural settings of a unit: what a designer tunes."""

    attention_k: int = 4  # feeds read per round
    explore: float = 0.25  # chance an exam slot goes to a peer outside the attention list
    exams_per_round: int = 2  # exams it sits per round
    serve_capacity: int = 6  # exams it sets per round
    holdout_streams: int = 8  # size of the private hold-out behind its self-belief
    dev_streams: int = 5  # size of the fresh dev set used to compare designs
    trials_per_round: int = 2  # design changes tried per round
    min_gain: float = 0.015  # improvement on dev needed to adopt a change
    replicate: bool = True  # a change must also win on a second, fresh dev set
    specialty_count: int = 3  # abilities its exams focus on
    specialty_focus: float = 0.5  # weight of own standard vs esteemed peers' standards when striving
    # What esteemed peers contribute to "better": "criteria" = how they score, applied to MY specialties;
    # "full" = how they score and what they examine (pulls a unit towards its peers' specialties);
    # "none" = nothing.
    peer_scope: str = "criteria"
    focused_dev: bool = True  # compare designs on data drawn from my own specialties, not on a balanced set
    # What a unit strives on.  "lead": the abilities where no rival seen in settled results is more than
    # ``lead_tol`` ahead of it; designs are compared on data drawn from those abilities, and a change that
    # lowers any of them under any scoring recipe of the family is refused.  "specialties": expected score
    # on its specialties, as ``peer_scope`` defines it (the earlier rule, kept for comparison).
    strive_on: str = "lead"
    lead_tol: float = 0.10
    # "tune": settings moves of one or two notches, and peers lend their settings, never their features;
    # "stock": feature swaps and wholesale copies of peers' designs as well.
    proposals: str = "tune"
    exam_floor: float = 0.3  # share of every exam I set spread evenly over all question abilities
    niche: float = 0.5  # 0: specialise where I score highest; 1: where I beat the peers I know
    crowd_aversion: float = 0.2  # avoid abilities many known peers already examine
    scan: int = 2  # other units whose latest profile I glance at each round
    half_life: float = 3.0  # rounds after which evidence about a peer counts half
    esteem_sharpness: float = 4.0  # how strongly attention concentrates on the best
    second_hand: float = 0.0  # weight of signals relayed by attended peers (0: trust only what I measured)
    calibration_weight: float = 4.0  # how much a peer's over-claiming lowers esteem for it
    prior_strength: float = 12.0  # pseudo-items pulling thin evidence towards the average
    adopt_designs: bool = True  # try designs published by attended peers
    share_design: bool = True  # publish own design on the feed
    conformity: float = 0.0  # per-round drift of own scoring belief towards esteemed peers'
    audit_rate: float = 0.25  # chance of re-computing a result read on an attended feed
    guess_shade: float = 1.0  # 0: state the mean; 1: state the loss-optimal self-estimate

    def __post_init__(self):
        if self.peer_scope not in ("criteria", "full", "none"):
            raise ValueError(f"unknown peer_scope {self.peer_scope!r}")
        if self.strive_on not in ("lead", "specialties"):
            raise ValueError(f"unknown strive_on {self.strive_on!r}")
        if self.proposals not in ("tune", "stock"):
            raise ValueError(f"unknown proposals {self.proposals!r}")
        if not 0.0 <= self.exam_floor < 1.0:
            raise ValueError("exam_floor must be in [0, 1)")

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: Mapping) -> "UnitPolicy":
        base = UnitPolicy()
        return replace(base, **{k: type(getattr(base, k))(v) for k, v in d.items() if hasattr(base, k)})


@dataclass(frozen=True)
class UnitSpec:
    """Everything needed to build a unit: what a designer releases."""

    unit_id: str
    owner: str
    policy: UnitPolicy
    memory: MemoryConfig
    criterion_weights: Mapping[str, float]
    exam: Mapping = field(default_factory=dict)
    seed: int | str = 0
    kind: str = "honest"  # adversary archetypes exist only in test beds
    allies: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id, "owner": self.owner, "policy": self.policy.to_dict(),
            "memory": self.memory.to_dict(), "criterion_weights": dict(self.criterion_weights),
            "exam": dict(self.exam), "seed": str(self.seed), "kind": self.kind, "allies": list(self.allies),
        }


@dataclass
class PeerView:
    """What one unit knows about another."""

    standard: Standard | None = None
    believed: dict = field(default_factory=dict)  # capability -> [score, n] as the peer claims
    design: dict | None = None
    feed_cursor: int = 0
    first_hand: list = field(default_factory=list)  # (score, n, loss, round) from my exams
    signals: list = field(default_factory=list)  # (capability dict, score, n, round, predicted) they gave me
    relayed: dict = field(default_factory=dict)  # xid -> (relayer, score, n, round)
    integrity: float = 1.0
    esteem: float = 0.0


# ------------------------------------------------------------- design mutation

def mutate_config(cfg: MemoryConfig, rng, budget: int, donor: MemoryConfig | None = None) -> MemoryConfig:
    """One local change to a memory design that respects the feature budget."""
    feats = set(cfg.features)
    if donor is not None:
        gain = sorted(set(donor.features) - feats)
        if gain and rng.random() < 0.7:
            add = rng.choice(gain)
            if len(feats) >= budget:
                droppable = sorted(feats - set(donor.features)) or sorted(feats)
                feats.discard(rng.choice(droppable))
            feats.add(add)
            return replace(cfg, features=frozenset(feats))
        return replace(
            cfg, top_k=donor.top_k, abstain_threshold=donor.abstain_threshold,
            latest_margin=donor.latest_margin, style=donor.style,
        )
    move = rng.choice(["feature", "feature", "top_k", "threshold", "margin", "style"])
    if move == "feature":
        missing = sorted(set(FEATURES) - feats)
        if missing:
            add = rng.choice(missing)
            if len(feats) >= budget:
                feats.discard(rng.choice(sorted(feats)))
            feats.add(add)
        return replace(cfg, features=frozenset(feats))
    if move == "top_k":
        return replace(cfg, top_k=int(min(14, max(2, cfg.top_k + rng.choice([-2, 2])))))
    if move == "threshold":
        return replace(cfg, abstain_threshold=round(min(0.6, max(0.1, cfg.abstain_threshold + rng.choice([-0.05, 0.05]))), 2))
    if move == "margin":
        return replace(cfg, latest_margin=round(min(0.95, max(0.3, cfg.latest_margin + rng.choice([-0.1, 0.1]))), 2))
    return replace(cfg, style=rng.choice([s for s in STYLES if s != cfg.style]))


def clamp_config(d: Mapping, budget: int) -> MemoryConfig | None:
    """Validate a design proposed from outside (a brain, a feed); None if unusable."""
    try:
        feats = [f for f in d.get("features", ()) if f in FEATURES][:budget]
        style = d.get("style", "terse")
        return MemoryConfig(
            features=frozenset(feats),
            top_k=int(min(14, max(2, int(d.get("top_k", 6))))),
            abstain_threshold=float(min(0.6, max(0.1, float(d.get("abstain_threshold", 0.3))))),
            latest_margin=float(min(0.95, max(0.3, float(d.get("latest_margin", 0.6))))),
            style=style if style in STYLES else "terse",
        )
    except (TypeError, ValueError):
        return None


def mutate_settings(cfg: MemoryConfig, rng, donor: MemoryConfig | None = None, max_step: int = 2) -> MemoryConfig:
    """One change to a design's settings that leaves its feature set alone.

    A donor lends all four settings.  Otherwise one setting moves by one or
    up to ``max_step`` notches, within the ranges ``clamp_config`` allows.
    """
    if donor is not None:
        return replace(cfg, top_k=donor.top_k, abstain_threshold=donor.abstain_threshold,
                       latest_margin=donor.latest_margin, style=donor.style)
    move = rng.choice(["top_k", "threshold", "margin", "style"])
    step = rng.choice(list(range(1, max(1, max_step) + 1)))
    if move == "top_k":
        return replace(cfg, top_k=int(min(14, max(2, cfg.top_k + step * rng.choice([-2, 2])))))
    if move == "threshold":
        return replace(cfg, abstain_threshold=round(min(0.6, max(0.1, cfg.abstain_threshold + step * rng.choice([-0.05, 0.05]))), 2))
    if move == "margin":
        return replace(cfg, latest_margin=round(min(0.95, max(0.3, cfg.latest_margin + step * rng.choice([-0.1, 0.1]))), 2))
    return replace(cfg, style=rng.choice([x for x in STYLES if x != cfg.style]))


# ------------------------------------------------------------------------ unit

class Unit:
    def __init__(
        self,
        spec: UnitSpec,
        board: Any,
        rules: WorldRules | None = None,
        brain: Any = None,
        memory_factory: Callable[[MemoryConfig], MemorySystem] | None = None,
        family: CriteriaFamily | None = None,
    ):
        self.spec = spec
        self.id = spec.unit_id
        self.board = board
        self.rules = rules or WorldRules()
        self.policy = spec.policy
        self.brain = brain
        self._factory = memory_factory or (lambda cfg: HeuristicMemory(cfg))
        self.memory_cfg = spec.memory
        self.memory = self._factory(self.memory_cfg)
        family = family or CriteriaFamily.default(self.rules.family_prior)
        self.eval = EvaluationModule(Standard(spec.criterion_weights, exam=spec.exam), family)
        self.version = 0  # bumps whenever the memory design changes
        self.version_round = 0
        self.table: CapabilityTable | None = None
        # The design my latest profile describes: what others' evidence about me is about, and so what I
        # answer with when the society calls on me.  A change I adopt is served once it is published.
        self.served_cfg = self.memory_cfg
        # Settled results I know of, by examiner: (examinee, ability table, round).  Mine come from exams
        # I settled; others' from results on attended feeds that I verified myself.
        self._results: dict[str, list] = {}
        self.self_score = 0.0
        self._profile_version = -1
        self._published: tuple | None = None
        self.known: dict[str, PeerView] = {}
        self.attention: list[tuple[str, float]] = []  # (peer, weight), strongest first
        self._set: dict[tuple[str, str], dict] = {}  # exams I set, by (examinee, exam id)
        self._standards: list[tuple[int, EvaluationModule]] = []  # my published standards, by position on the chain
        # Exam ids I have already answered, as examiner.  Read back from the chain, so that a unit
        # started again with the same secret never serves an id, and so a seed share, a second time.
        self._answered: set[tuple[str, str]] = set()
        try:
            for post in self.board.feed(self.id, kinds=["exam.ready", "exam.decline"]):
                if isinstance(post.body, Mapping) and isinstance(post.body.get("xid"), str) and isinstance(post.to, str):
                    self._answered.add((post.to, post.body["xid"]))
        except Exception:  # noqa: BLE001 - an unreadable feed is treated as an empty one
            pass
        self._sat: dict[str, dict] = {}  # exams I sit, by xid
        self._served = 0
        self._exam_counter = 0
        self._request_counter = 0
        self._inbox_cursor = 0
        self._plan: dict = {}  # advice from the brain for the coming round
        self.round = 0
        self.log: list[dict] = []  # adopted changes, audits: for analysis
        self.stats = {"exams_set": 0, "exams_sat": 0, "audits": 0, "audit_fails": 0, "adoptions": 0, "trials": 0,
                      "exams_called_off": 0, "results_settled_myself": 0, "walkouts": 0, "brain_calls": 0, "brain_errors": 0}

    # ------------------------------------------------------------ utilities
    def _rng(self, *parts):
        return make_rng("unit", self.spec.seed, self.id, *parts)

    def _post(self, kind: str, body: Mapping, to: str | None = None):
        return self.board.post(self.id, kind, body, to=to, round=self.round)

    def view(self, peer: str) -> PeerView:
        v = self.known.get(peer)
        if v is None:
            v = self.known[peer] = PeerView()
        return v

    # -------------------------------------------------------- self-knowledge
    def _refresh_profile(self) -> None:
        """Measure myself on a fresh hold-out with my own evaluation module."""
        if self._profile_version == self.version and self.table is not None:
            return
        std = self.eval.standard
        holdout = generate(
            f"{self.spec.seed}:{self.id}:holdout:{self.version}",
            std.gen_config(self.policy.holdout_streams, 12, balanced=True),
        )
        answers = run_system(self.memory, holdout.probes())
        ev = self.eval.evaluate(answers, holdout.instances, salt=f"self:{self.version}")
        self.table = ev.table
        self.self_score = ev.score
        self._profile_version = self.version

    def believed(self) -> dict[str, tuple[float, int]]:
        assert self.table is not None
        return self.table.capability_scores(self.eval.standard.criterion_weights)

    def _update_specialties(self) -> None:
        """Examine others on the abilities I believe I am good at.

        "Good at" blends my absolute level with my advantage over the peers I
        know and a dislike of crowded abilities, so a society does not pile
        onto whatever is easiest to measure.
        """
        scores = self.believed()
        pol = self.policy
        peers = [v for v in self.known.values() if v.believed]
        niche: dict[str, float] = {}
        for c in CAPABILITIES:
            mine = scores.get(c, (0.0, 0))[0]
            ref = crowd = 0.0
            if peers:
                ref = sum(float(v.believed.get(c, (0.0, 0))[0]) for v in peers) / len(peers)
                with_std = [v for v in peers if v.standard is not None]
                if with_std:
                    # an ability a peer carries only at floor level is not one it examines
                    floor = pol.exam_floor / len(QUESTION_TYPES) + 1e-6
                    crowd = sum(1.0 for v in with_std if v.standard.capability_weights.get(c, 0.0) > floor) / len(with_std)
            niche[c] = mine - pol.niche * ref - pol.crowd_aversion * crowd
        able = [c for c in CAPABILITIES if scores.get(c, (0.0, 0))[0] >= 0.2] or list(scores)
        ranked = sorted(able, key=lambda c: (-niche[c], CAPABILITIES.index(c)))
        top = ranked[: max(1, pol.specialty_count)]
        weights = {c: max(0.05, scores.get(c, (0.0, 0))[0]) for c in top}
        if pol.exam_floor > 0:
            # a little of every ability in every exam, so that a unit that alone holds one can be measured on it
            total = sum(weights.values())
            weights = {c: (1 - pol.exam_floor) * w / total for c, w in weights.items()}
            for c in QUESTION_TYPES:
                weights[c] = weights.get(c, 0.0) + pol.exam_floor / len(QUESTION_TYPES)
        std = self.eval.standard
        new = Standard(std.criterion_weights, weights, std.exam)
        if new.fingerprint() != std.fingerprint():
            self.eval = self.eval.with_standard(new)

    def _publish_profile(self) -> None:
        std = self.eval.standard
        body = {
            "standard": std.to_dict(),
            "evaluator": self.eval.fingerprint(),
            "believed": {c: [s, n] for c, (s, n) in sorted(self.believed().items())},
            "self_score": self.self_score,
            "version": self.version,
            "attending": [p for p, _w in self.attention],
            "owner": self.spec.owner,
        }
        if self.policy.share_design:
            body["design"] = self.memory_cfg.to_dict()
        key = (std.fingerprint(), self.version, tuple(body["attending"]))
        if key != self._published:
            post = self._post("profile", body)
            self._published = key
            self.served_cfg = self.memory_cfg
            self._standards.append((post.seq, self.eval))

    # ---------------------------------------------------------------- feeds
    def _read_feeds(self) -> None:
        """Read the feeds I attend to: profiles, signals about others, audit reports."""
        rng = self._rng("audit", self.round)
        for peer, _w in self.attention[: self.policy.attention_k]:
            v = self.view(peer)
            posts = self.board.feed(peer, since=v.feed_cursor)
            for post in posts:
                v.feed_cursor = max(v.feed_cursor, post.author_seq)
                if not isinstance(post.body, Mapping):
                    continue
                if post.kind == "profile":
                    self._absorb_profile(peer, post.body)
                elif post.kind == "exam.result":
                    examinee = post.body.get("examinee")
                    if not examinee or examinee == self.id:
                        continue
                    relies = self.policy.second_hand > 0 or (self.policy.strive_on == "lead" and self.policy.trials_per_round > 0)
                    if not relies and rng.random() >= self.policy.audit_rate:
                        continue  # not relying on others' results, and not auditing this one
                    self.stats["audits"] += 1
                    verdict = self._check(post)
                    if not verdict.ok:
                        self._flag(post.author, post.seq, verdict.reason)
                        continue
                    # a relayed signal is only ever used after I have recomputed it myself
                    self.view(examinee).relayed[verdict.xid] = (peer, verdict.score, verdict.n_items, self.round)
                    self._note_result(post.author, verdict)
                elif post.kind == "audit.fail":
                    self._check_accusation(post)
        # A glance at a few other units' latest profile: cheap discovery, not attention.
        if self.policy.scan > 0:
            attended = {p for p, _w in self.attention}
            directory = self.board.directory()
            others = sorted(a for a, info in directory.items() if a != self.id and a not in attended and info.get("last_profile"))
            self._rng("scan", self.round).shuffle(others)
            for peer in others[: self.policy.scan]:
                last = directory[peer]["last_profile"]
                posts = self.board.feed(peer, since=last - 1, kinds=["profile"], limit=1)
                if posts:
                    self._absorb_profile(peer, posts[0].body)

    def _absorb_profile(self, peer: str, body: Mapping) -> None:
        """Take in a peer's profile.  It is the peer's own text: anything odd in it is simply not used."""
        v = self.view(peer)
        try:
            standard = Standard.from_dict(body["standard"])
            believed = {}
            for cap, row in (body.get("believed") or {}).items():
                score, n = row
                if cap in CAPABILITIES and isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                    believed[cap] = [float(score), int(n)]
            # a published design is only ever read as a design: unknown fields and odd values do not survive
            design = clamp_config(body.get("design"), self.rules.feature_budget) if isinstance(body.get("design"), Mapping) else None
            design = design.to_dict() if design is not None else None
        except Exception:  # noqa: BLE001
            return
        v.standard, v.believed, v.design = standard, believed, design

    # --------------------------------------------------------------- audits
    def _check(self, post):
        """Verify a published exam result from public evidence (see ``protocol.check_result``)."""
        return check_result(self.board, post, self.eval.family)

    def _verify_result(self, post) -> tuple[bool, str]:
        verdict = self._check(post)
        return verdict.ok, verdict.reason

    def _audit_result(self, post) -> bool:
        """Verify a result seen on a feed; publish an audit failure if it does not hold."""
        self.stats["audits"] += 1
        ok, why = self._verify_result(post)
        if not ok:
            self._flag(post.author, post.seq, why)
        return ok

    def _flag(self, examiner: str, result_seq: int, why: str) -> None:
        self.stats["audit_fails"] += 1
        self.view(examiner).integrity = 0.0
        self.log.append({"round": self.round, "event": "audit_fail", "examiner": examiner, "why": why})
        self._post("audit.fail", {"examiner": examiner, "result_seq": result_seq, "reason": why})

    def _check_accusation(self, post) -> None:
        """Never take an accusation on trust: recompute it, transcript and all."""
        try:
            seq = int(post.body["result_seq"])
            target = self.board.read(since_seq=seq - 1, limit=1)[0]
        except (KeyError, ValueError, TypeError, IndexError):
            return
        if target.kind != "exam.result" or target.author != post.body.get("examiner"):
            self.view(post.author).integrity = 0.0  # points at something that is not a result
            return
        ok, _why = self._verify_result(target)
        if ok:
            self.view(post.author).integrity = 0.0  # false accusation
        else:
            self.view(target.author).integrity = 0.0

    # ----------------------------------------------------------- prediction
    def _predict(self, standard: Standard, n_items: int, examiner: str | None = None) -> tuple[float, float]:
        """What I expect to score under ``standard``: (mean, sd)."""
        assert self.table is not None
        mean = expected_score(self.table, standard.criterion_weights, standard.capability_weights)
        n_own = max(1.0, sum(self.table.counts.get(c, 0) * w for c, w in standard.capability_weights.items()) * 3)
        if examiner is not None:
            past = [(score - pred) for (_c, score, _n, rnd, pred) in self.view(examiner).signals
                    if pred is not None and rnd >= self.version_round]
            if past:
                mean += sum(past) / (len(past) + 2.0)  # learned offset to this examiner, shrunk
        mean = min(1.0, max(0.0, mean))
        sd = math.sqrt(binomial_sd(mean, n_items) ** 2 + binomial_sd(mean, n_own) ** 2)
        return mean, sd

    # ----------------------------------------------------- examining others
    def _result_payload(self, examinee: str, evaluation) -> dict:
        """The result I publish.  One code path for everyone, myself included."""
        return {
            "score": evaluation.score,
            "capability": {c: [s, n] for c, (s, n) in sorted(evaluation.capability.items())},
            "receipt": dict(evaluation.receipt),
        }

    def _draw_exam(self, examinee: str, exam_seed: str, module: EvaluationModule):
        """The exam for one examinee: drawn from my published standard, whoever is asking."""
        return draw_paper(exam_seed, module.standard.gen_config(self.rules.exam_streams, self.rules.exam_questions))

    def _answer_exam(self, probes: Sequence[Probe], examiner: str, xid: str) -> dict[str, Answer]:
        return run_system(self.memory, probes)

    def _make_guess(self, mean: float, sd: float) -> float:
        return self.rules.peer_loss.best_report(mean, sd, self.policy.guess_shade)

    def on_message(self, post) -> None:
        """Handle one post addressed to me.  Whatever a peer sends, it must not be able to stop me."""
        try:
            if not isinstance(post.body, Mapping):
                return
            if post.kind == "exam.result":
                self._on_result(post)
            elif post.kind in ("exam.request", "exam.nonce", "exam.sealed", "exam.open"):
                self._examiner_step(post.author, post.body.get("xid"))
            elif post.kind in ("exam.ready", "exam.offer", "exam.keys", "exam.decline"):
                self._examinee_step(post.author, post.body.get("xid"))
        except Exception as exc:  # noqa: BLE001
            self.log.append({"round": self.round, "event": "bad_message", "kind": post.kind,
                             "from": post.author, "why": f"{type(exc).__name__}: {str(exc)[:200]}"})

    def _read(self, examiner: str, examinee: str, xid: str):
        """The one reading of an exam that everybody uses (see ``protocol.read_exam``)."""
        return read_exam(self.board, examiner, examinee, xid)

    def _settle(self, examiner: str, examinee: str, xid: str):
        """What an exam established, computed the way anyone would (see ``protocol.settle``)."""
        return settle(self.board, self.eval.family, examiner, examinee, xid)

    # -- as examiner
    def _examiner_step(self, examinee: str, xid: Any) -> None:
        """Do whatever the chain says is my move, as examiner, in this exam.  Nothing else prompts me.

        I never act on a post directly.  I read the exam off the chain the way
        a verifier will, and act only if that reading says it is my turn, so
        what I respond to is exactly what counts.
        """
        if examinee == self.id or not isinstance(xid, str):
            return
        key = (examinee, xid)
        reading = self._read(self.id, examinee, xid)
        st = self._set.get(key)
        if reading.stage == "requested":
            if key in self._answered:
                return  # one answer per exam id, whatever is asked again
            self._answered.add(key)
            # The exam is set under the standard I had published when it was asked for: the examinee
            # chose the moment, so I cannot have chosen the standard.
            module = next((m for seq, m in reversed(self._standards) if seq < reading.request.seq), None)
            if self._served >= self.policy.serve_capacity or self.table is None or module is None:
                self._post("exam.decline", {"xid": xid}, to=examinee)
                return
            self._served += 1
            share = derive_exam_share(self.spec.seed, self.id, examinee, xid)
            self._set[key] = {"examinee": examinee, "xid": xid, "share": share, "module": module, "round": self.round}
            # The size of the exam and the loss are fixed here, before either seed share is known.
            self._post("exam.ready", {
                "xid": xid, "share_commit": share_commitment(share, examinee, xid),
                "exam_size": [self.rules.exam_streams, self.rules.exam_questions], "loss": self.rules.peer_loss.to_dict(),
            }, to=examinee)
        elif st is None or st.get("done"):
            return
        elif reading.stage == "seeded":
            self._make_offer(st, reading)
        elif reading.stage in ("sealed", "unsealed"):
            # sealed: the examinee is bound, so the keys can go out.  unsealed: it ran out of time.
            self._reveal_keys(st)
            if reading.stage == "unsealed":
                self._publish_result(st)
        elif reading.settled:
            self._publish_result(st)
        elif reading.over:
            st["done"] = True

    def _make_offer(self, st: dict, reading) -> None:
        xid, examinee, module = st["xid"], st["examinee"], st["module"]
        try:
            exam = self._draw_exam(examinee, joint_exam_seed(st["share"], reading.nonce.body["nonce"], xid), module)
        except ValueError as exc:
            # This seed cannot be expanded into the exam my standard describes.  I may not try another
            # seed, since that would be choosing the paper, so the exam is called off, and I show the
            # share so anyone can see that it could not be.
            st["done"] = True
            self.stats["exams_called_off"] += 1
            self.log.append({"round": self.round, "event": "exam_called_off", "examinee": examinee, "why": str(exc)[:200]})
            self._post("exam.decline", {"xid": xid, "reason": "no exam could be drawn for this seed", "share": st["share"]},
                       to=examinee)
            return
        st["salt"] = sha256_hex(f"{self.spec.seed}:{self.id}:salt:{examinee}:{xid}")[:20]
        st["exam"] = exam
        probes_blob = self.board.put_blob(pack_probes(exam.probes()))
        self._post(
            "exam.offer",
            {
                "xid": xid, "probes": probes_blob, "blobs": [probes_blob], "n": len(exam),
                "keys_commit": keys_commitment(exam.instances, st["salt"]),
                "standard": module.standard.to_dict(), "evaluator": module.fingerprint(),
            },
            to=examinee,
        )

    def _reveal_keys(self, st: dict) -> None:
        """Make the keys public.  From here on anyone can grade this exam, and I can no longer affect it."""
        if "salt" not in st or st.get("keys_out"):
            return
        st["keys_out"] = True
        self._post("exam.keys", {"xid": st["xid"], "share": st["share"], "salt": st["salt"]}, to=st["examinee"])

    def _publish_result(self, st: dict) -> None:
        """Publish the outcome of an exam I set: the public signal about the examinee.

        The numbers are not mine to choose.  They are the ones the public
        record yields for anyone who settles the exam.
        """
        verdict = self._settle(self.id, st["examinee"], st["xid"])
        if not verdict.ok:
            if not verdict.pending:
                st["done"] = True
                self.log.append({"round": self.round, "event": "exam_unsettled", "examinee": st["examinee"], "why": verdict.reason})
            return
        payload = self._result_payload(st["examinee"], verdict)
        st["done"] = True
        st.pop("exam", None)  # the keys are public; the paper is spent
        self.stats["exams_set"] += 1
        self._post(
            "exam.result",
            {
                "xid": st["xid"], "examinee": st["examinee"], **payload,
                "guess": verdict.guess, "guess_valid": verdict.guess_valid, "loss": verdict.loss, "void": verdict.void,
            },
            to=st["examinee"],
        )
        self.view(st["examinee"]).first_hand.append((float(verdict.score), int(verdict.n_items), float(verdict.loss), self.round))
        self._note_result(self.id, verdict)

    # -- as examinee
    def _examinee_step(self, examiner: str, xid: Any) -> None:
        """Do whatever the chain says is my move, as examinee, in this exam."""
        st = self._sat.get(xid) if isinstance(xid, str) else None
        if st is None or st.get("examiner") != examiner or st.get("done"):
            return
        reading = self._read(examiner, self.id, xid)
        if reading.stage == "ready" and "nonce" not in st:
            streams, questions, loss = reading.terms
            # I only sit exams on the terms this world runs on, and they are fixed before I contribute.
            if (streams, questions) != (self.rules.exam_streams, self.rules.exam_questions) or loss != self.rules.peer_loss:
                st["done"] = True
                return
            st["nonce"] = sha256_hex(f"{self.spec.seed}:{self.id}:exam-nonce:{examiner}:{xid}")[:32]
            self._post("exam.nonce", {"xid": xid, "nonce": st["nonce"]}, to=examiner)
        elif reading.stage == "offered" and "sheet" not in st:
            self._seal_answers(st, reading)
        elif reading.stage == "keyed" and "sheet" in st and "opened_round" not in st:
            self._open_answers(st, reading)
        elif reading.over and not reading.settled:
            if reading.fault == "examiner":
                self._examiner_failed(st, reading.reason)
            else:
                st["done"] = True  # declined, or called off for a reason anyone can check
                st.pop("sheet", None)

    def _seal_answers(self, st: dict, reading) -> None:
        xid, examiner, offer = reading.xid, reading.examiner, reading.offer
        probes = unpack_probes(self.board.get_blob(offer.body["probes"]))
        streams, questions, _loss = reading.terms
        if len(probes) != streams * questions:
            return  # not the exam that was agreed; I do not sit it, and the examiner cannot settle it
        standard = Standard.from_dict(offer.body["standard"])
        self.view(examiner).standard = standard
        answers = self._answer_exam(probes, examiner, xid)
        mean, sd = self._predict(standard, len(probes), examiner)
        guess = self._make_guess(mean, sd)
        sheet = {pid: a.to_dict() for pid, a in sorted(answers.items())}
        st.update(
            sheet=sheet, guess=guess, predicted=mean,
            answers_nonce=sha256_hex(f"{self.spec.seed}:{self.id}:answers-nonce:{examiner}:{xid}")[:20],
            guess_nonce=sha256_hex(f"{self.spec.seed}:{self.id}:nonce:{examiner}:{xid}")[:20],
        )
        # Commitments only: the answers themselves stay with me until the keys are public.
        self._post(
            "exam.sealed",
            {
                "xid": xid, "offer_seq": offer.seq,
                "answers_commit": commit(blob_address(sheet), st["answers_nonce"]),
                "guess_commit": commit(guess, st["guess_nonce"]),
            },
            to=examiner,
        )

    def _open_answers(self, st: dict, reading) -> None:
        # Open only for keys that really are the keys: otherwise I would be showing my answers for nothing.
        ok, why, _standard, _salt, _exam = check_paper(self.board, reading)
        if not ok:
            self._examiner_failed(st, why)
            return
        st["opened_round"] = self.round
        sheet = self.board.put_blob(st["sheet"])
        self._post(
            "exam.open",
            {
                "xid": reading.xid, "answers": sheet, "blobs": [sheet], "answers_nonce": st["answers_nonce"],
                "guess": st["guess"], "guess_nonce": st["guess_nonce"],
            },
            to=reading.examiner,
        )

    def _examiner_failed(self, st: dict, why: str) -> None:
        """An examiner that knew the paper left my exam unfinished, or spoiled it.

        It may be choosing which exams to finish, so each time halves what its
        word is worth to me.
        """
        st["done"] = True
        st.pop("sheet", None)
        self.stats["walkouts"] += 1
        self.view(st["examiner"]).integrity *= 0.5
        self.log.append({"round": self.round, "event": "walkout", "examiner": st["examiner"], "why": why[:160]})

    def _on_result(self, post) -> None:
        xid = post.body.get("xid")
        if post.body.get("examinee") != self.id:
            return
        st = self._sat.get(xid) if isinstance(xid, str) else None
        self.stats["audits"] += 1
        verdict = self._check(post)
        if not verdict.ok:
            # including a result about an exam I never sat: it has no posts of mine behind it
            self._flag(post.author, post.seq, verdict.reason)
            return
        if st is None or st.get("examiner") != post.author:
            return
        self._record_sat(st, verdict)

    def _record_sat(self, st: dict, verdict) -> None:
        """Note what an exam I sat established.  The numbers are the settled ones, never a post's."""
        if st.get("done"):
            return
        st["done"] = True
        st.pop("sheet", None)
        self.stats["exams_sat"] += 1
        self.view(st["examiner"]).signals.append(
            (dict(verdict.capability), verdict.score, verdict.n_items, self.round, st.get("predicted"))
        )

    def _tend_exams(self) -> None:
        """Once a round, look again at every exam of mine that is still open.

        Deadlines pass with the clock, not with a message, so this is where an
        exam whose other party went quiet gets finished: keys revealed for an
        examinee that sealed nothing, a silent examinee settled, an examiner
        that walked out noted, and a result I was owed but never sent settled
        by myself from the public record.
        """
        for (examinee, xid), st in list(self._set.items()):
            if not st.get("done"):
                try:
                    self._examiner_step(examinee, xid)
                except Exception as exc:  # noqa: BLE001 - one exam must not stop the round
                    st["done"] = True
                    self.log.append({"round": self.round, "event": "exam_dropped", "examinee": examinee, "why": f"{type(exc).__name__}: {str(exc)[:160]}"})
        for xid, st in list(self._sat.items()):
            if st.get("done"):
                continue
            try:
                self._examinee_step(st["examiner"], xid)
                if st.get("done") or "opened_round" not in st:
                    continue
                verdict = self._settle(st["examiner"], self.id, xid)
                if verdict.ok and st["opened_round"] < self.round:
                    # The keys are public, so the grade does not depend on the examiner saying it.
                    self.stats["results_settled_myself"] += 1
                    self._record_sat(st, verdict)
                elif not verdict.ok and not verdict.pending:
                    if verdict.fault == "examiner":
                        self._examiner_failed(st, verdict.reason)
                    st["done"] = True
            except Exception as exc:  # noqa: BLE001
                st["done"] = True
                self.log.append({"round": self.round, "event": "exam_dropped", "examiner": st.get("examiner"), "why": f"{type(exc).__name__}: {str(exc)[:160]}"})

    def _note_result(self, examiner: str, verdict) -> None:
        """Keep a settled result's ability table: what rivals have shown they can do."""
        if verdict.void or not verdict.capability or verdict.examinee in (self.id, examiner):
            return
        self._results.setdefault(examiner, []).append((verdict.examinee, dict(verdict.capability), self.round))

    # ---------------------------------------------------------------- esteem
    def _update_esteem(self) -> None:
        """Whom do I find high-performing?"""
        pol = self.policy
        firsts = [s for v in self.known.values() for (s, _n, _l, _r) in v.first_hand]
        prior = (sum(firsts) / len(firsts)) if firsts else 0.3
        weights_now = dict(self.attention)
        def fresh(r: int) -> float:
            return 0.5 ** (max(0, self.round - r) / max(0.5, pol.half_life))

        for peer, v in self.known.items():
            n1 = sum(n * fresh(r) for (_s, n, _l, r) in v.first_hand)
            s1 = sum(s * n * fresh(r) for (s, n, _l, r) in v.first_hand) / n1 if n1 else 0.0
            n2 = s2 = 0.0
            for (relayer, score, n, r) in v.relayed.values():
                trust = weights_now.get(relayer, 0.0) * self.view(relayer).integrity * fresh(r)
                n2 += trust * n
                s2 += trust * n * score
            n2 *= pol.second_hand
            s2 *= pol.second_hand
            skill = (n1 * s1 + s2 + pol.prior_strength * prior) / (n1 + n2 + pol.prior_strength)
            if v.first_hand:
                lw = sum(fresh(r) for (_s, _n, _l, r) in v.first_hand)
                mean_loss = sum(l * fresh(r) for (_s, _n, l, r) in v.first_hand) / lw
                skill *= math.exp(-pol.calibration_weight * mean_loss)
            v.esteem = skill * v.integrity
        ranked = sorted(
            ((p, v.esteem) for p, v in self.known.items() if p != self.id and v.esteem > 0),
            key=lambda pe: (-pe[1], pe[0]),
        )[: max(1, pol.attention_k)]
        total = sum(e ** pol.esteem_sharpness for _p, e in ranked)
        self.attention = [(p, (e ** pol.esteem_sharpness) / total) for p, e in ranked] if total > 0 else []

    def standing(self) -> dict[str, tuple[float, float]]:
        """How good am I, as measured by the peers I esteem: capability -> (score, evidence)."""
        weights = dict(self.attention)
        acc: dict[str, list[float]] = {}
        for peer, v in self.known.items():
            w = weights.get(peer, 0.0)
            if w <= 0:
                continue
            for (caps, _score, _n, rnd, _pred) in v.signals:
                if rnd < self.version_round:
                    continue  # that measured an older version of me
                for cap, (s, n) in caps.items():
                    a = acc.setdefault(cap, [0.0, 0.0])
                    a[0] += w * n * s
                    a[1] += w * n
        return {c: (a[0] / a[1], a[1]) for c, a in acc.items() if a[1] > 0}

    # --------------------------------------------------------------- striving
    def _objective(self, table: CapabilityTable) -> float:
        """What "better" means to me.

        Being the best in what I do: my own specialties, scored the way I score
        and the way the peers I esteem score.  With ``peer_scope="full"`` their
        choice of abilities counts too, which rewards becoming more like them.
        """
        own = self.eval.standard
        pol = self.policy
        mine = expected_score(table, own.criterion_weights, own.capability_weights)
        if pol.peer_scope == "none":
            return mine
        theirs = wsum = 0.0
        for peer, w in self.attention:
            std = self.view(peer).standard
            if std is not None:
                abilities = std.capability_weights if pol.peer_scope == "full" else own.capability_weights
                theirs += w * expected_score(table, std.criterion_weights, abilities)
                wsum += w
        if wsum <= 0:
            return mine
        return pol.specialty_focus * mine + (1 - pol.specialty_focus) * theirs / wsum

    def _candidates(self, rng) -> list[MemoryConfig]:
        budget = self.rules.feature_budget
        out: list[MemoryConfig] = []
        for d in self._plan.get("candidates", ()):
            cfg = clamp_config(d, budget)
            if cfg is not None:
                out.append(cfg)
        donors = [
            (clamp_config(self.view(p).design, budget), w)
            for p, w in self.attention if self.policy.adopt_designs and self.view(p).design
        ]
        donors = [(d, w) for d, w in donors if d is not None and d.hash() != self.memory_cfg.hash()]
        while len(out) < self.policy.trials_per_round:
            if donors and rng.random() < 0.5:
                donor = rng.choices([d for d, _ in donors], weights=[w for _, w in donors], k=1)[0]
                out.append(donor if rng.random() < 0.4 else mutate_config(self.memory_cfg, rng, budget, donor))
            else:
                out.append(mutate_config(self.memory_cfg, rng, budget))
        return out[: max(self.policy.trials_per_round, len(self._plan.get("candidates", ())))]

    def _strive(self) -> None:
        if self.policy.trials_per_round <= 0:
            return
        if self.policy.strive_on == "lead":
            self._strive_on_leads()
        else:
            self._strive_on_specialties()

    # -- striving where I lead
    #
    # "Be the best in what you do" taken literally.  Striving on my whole set of
    # specialties, or on what my esteemed peers examine, makes units trade a rare
    # ability that only they hold for a common one that scores well, and the
    # society loses its specialists.  So a unit improves only where it is already
    # at the front, compares designs on data drawn from those abilities, and
    # refuses any change that costs one of them under any way of scoring it.

    def _blend(self, fn: Callable[[Mapping[str, float]], float]) -> float:
        """My scoring and my esteemed peers' scoring, mixed as ``specialty_focus`` says."""
        pol = self.policy
        mine = fn(self.eval.standard.criterion_weights)
        if pol.peer_scope == "none":
            return mine
        theirs = wsum = 0.0
        for peer, w in self.attention:
            std = self.view(peer).standard
            if std is not None:
                theirs += w * fn(std.criterion_weights)
                wsum += w
        if wsum <= 0:
            return mine
        return pol.specialty_focus * mine + (1 - pol.specialty_focus) * theirs / wsum

    def _ability_score(self, table: CapabilityTable, ability: str) -> float:
        def under(criteria: Mapping[str, float]) -> float:
            got = table.capability_scores(criteria).get(ability)
            return got[0] if got else 0.0
        return self._blend(under)

    def _best_rival(self, examiner: str) -> dict[str, float]:
        """The best level a unit other than me has shown on each ability in one examiner's settled results.

        Recent results count more; thin evidence is pulled towards that examiner's average for the ability.
        """
        pol = self.policy
        acc: dict[str, dict[str, list[float]]] = {}
        for examinee, caps, rnd in self._results.get(examiner, ()):
            fresh = 0.5 ** (max(0, self.round - rnd) / max(0.5, pol.half_life))
            for cap, (score, n) in caps.items():
                if n > 0:
                    a = acc.setdefault(cap, {}).setdefault(examinee, [0.0, 0.0])
                    a[0] += fresh * n * score
                    a[1] += fresh * n
        out: dict[str, float] = {}
        for cap, by in acc.items():
            mean = sum(a[0] for a in by.values()) / sum(a[1] for a in by.values())
            out[cap] = max((a[0] + 6.0 * mean) / (a[1] + 6.0) for a in by.values())
        return out

    def _sources(self) -> list[tuple[float, Mapping[str, float], dict[str, float]]]:
        """(weight, scoring, best rival per ability): my own exams, then each esteemed examiner's."""
        pol = self.policy
        own = (1.0, self.eval.standard.criterion_weights, self._best_rival(self.id))
        peers = [(p, w) for p, w in self.attention if self.view(p).standard is not None] if pol.peer_scope != "none" else []
        wsum = sum(w for _p, w in peers)
        if wsum <= 0:
            return [own]
        return [(pol.specialty_focus, own[1], own[2])] + [
            ((1 - pol.specialty_focus) * w / wsum, self.view(p).standard.criterion_weights, self._best_rival(p))
            for p, w in peers
        ]

    def lead_abilities(self) -> list[str]:
        """Question abilities I can do where no rival seen is more than ``lead_tol`` ahead of me.

        Every question ability counts, not only my specialties: the one thing I
        alone can do need not be among the abilities I score highest on.  A tie
        counts as leading, so twins do not both walk away from what they share.
        If I lead nowhere: the ability where I am closest to the front.
        """
        if self.table is None:
            return []
        able = [c for c in QUESTION_TYPES if self._ability_score(self.table, c) >= 0.2]
        if not able:
            return []
        sources = self._sources()
        margins: dict[str, float] = {}
        for c in able:
            num = den = 0.0
            for w, criteria, rivals in sources:
                if w > 0 and c in rivals:
                    got = self.table.capability_scores(criteria).get(c)
                    num += w * ((got[0] if got else 0.0) - rivals[c])
                    den += w
            if den > 0:
                margins[c] = num / den
        lead = [c for c in able if c not in margins or margins[c] >= -self.policy.lead_tol]
        if lead:
            return lead
        return [max(margins, key=lambda c: (margins[c], -QUESTION_TYPES.index(c)))]

    def _proposals(self, rng) -> list[MemoryConfig]:
        if self.policy.proposals == "stock":
            return self._candidates(rng)
        budget = self.rules.feature_budget
        out = [cfg for cfg in (clamp_config(d, budget) for d in self._plan.get("candidates", ())) if cfg is not None]
        donors = [
            (clamp_config(self.view(p).design, budget), w)
            for p, w in self.attention if self.policy.adopt_designs and self.view(p).design
        ]
        donors = [(d, w) for d, w in donors if d is not None and d.hash() != self.memory_cfg.hash()]
        while len(out) < self.policy.trials_per_round:
            if donors and rng.random() < 0.5:
                donor = rng.choices([d for d, _ in donors], weights=[w for _, w in donors], k=1)[0]
                out.append(mutate_settings(self.memory_cfg, rng, donor))
            else:
                out.append(mutate_settings(self.memory_cfg, rng))
        return out[: max(self.policy.trials_per_round, len(self._plan.get("candidates", ())))]

    def _strive_on_leads(self) -> None:
        pol = self.policy
        rng = self._rng("strive", self.round)
        lead = self.lead_abilities()
        if not lead:
            return
        abstain = 0.06  # unanswerable questions stay in the comparison at about the generator's rate
        focus = {**{c: (1.0 - abstain) / len(lead) for c in lead}, "abstention": abstain}
        std = self.eval.standard
        cfg_dev = Standard(std.criterion_weights, focus, std.exam).gen_config(pol.dev_streams, 12)
        dev = generate(f"{self.spec.seed}:{self.id}:dev:{self.round}", cfg_dev)
        # The guard: under my scoring, each esteemed peer's, and every single scorer of the family
        # (the recipes an outside evaluator might use), no lead ability may get worse.
        checks = [std.criterion_weights]
        if pol.peer_scope != "none":
            checks += [self.view(p).standard.criterion_weights for p, _w in self.attention if self.view(p).standard is not None]
        checks += [{f"{sc}|detect|micro": 1.0} for sc in SCORERS if f"{sc}|detect|micro" in self.eval.family.ids]

        def table_of(memory: MemorySystem, bench) -> CapabilityTable:
            return CapabilityTable.from_report(self.eval.grade(run_system(memory, bench.probes()), bench.instances))

        def objective(table: CapabilityTable) -> float:
            return self._blend(lambda criteria: expected_score(table, criteria, focus))

        def holds(new: CapabilityTable, old: CapabilityTable) -> bool:
            if any(self._ability_score(new, c) < self._ability_score(old, c) for c in lead):
                return False
            for criteria in checks:
                a, b = new.capability_scores(criteria), old.capability_scores(criteria)
                if any(a.get(c, (0.0, 0))[0] < b.get(c, (0.0, 0))[0] for c in lead):
                    return False
            return True

        base_t = table_of(self.memory, dev)
        base_j = objective(base_t)
        best_cfg, best_j = self.memory_cfg, base_j
        seen = {self.memory_cfg.hash()}
        confirm = None
        for cfg in self._proposals(rng):
            if cfg.hash() in seen or len(cfg.features) > self.rules.feature_budget:
                continue
            seen.add(cfg.hash())
            self.stats["trials"] += 1
            candidate = self._factory(cfg)
            cand_t = table_of(candidate, dev)
            j = objective(cand_t)
            if j <= best_j + pol.min_gain or not holds(cand_t, base_t):
                continue
            if pol.replicate:
                # a win has to repeat, guard and all, on data the comparison has not seen
                if confirm is None:
                    bench = generate(f"{self.spec.seed}:{self.id}:confirm:{self.round}", cfg_dev)
                    confirm = (bench, table_of(self.memory, bench))
                bench, base_c = confirm
                cand_c = table_of(candidate, bench)
                if objective(cand_c) <= objective(base_c) + pol.min_gain / 2 or not holds(cand_c, base_c):
                    continue
            best_cfg, best_j = cfg, j
        if best_cfg.hash() != self.memory_cfg.hash():
            self._adopt(best_cfg, best_j - base_j, lead=lead)

    def _adopt(self, cfg: MemoryConfig, gain: float, **note: Any) -> None:
        self.log.append({"round": self.round, "event": "adopt", "from": self.memory_cfg.to_dict(),
                         "to": cfg.to_dict(), "dev_gain": gain, **note})
        self.memory_cfg = cfg
        self.memory = self._factory(cfg)
        self.version += 1
        self.version_round = self.round + 1
        self.stats["adoptions"] += 1

    # -- striving on my specialties (the earlier rule)

    def _strive_on_specialties(self) -> None:
        rng = self._rng("strive", self.round)
        # Under "full" scope every ability matters, so the comparison needs a balanced set;
        # otherwise the set is drawn from my own specialties, where the comparison is decided.
        balanced = self.policy.peer_scope == "full" or not self.policy.focused_dev
        dev = generate(
            f"{self.spec.seed}:{self.id}:dev:{self.round}",
            self.eval.standard.gen_config(self.policy.dev_streams, 12, balanced=balanced),
        )

        def measure(memory: MemorySystem, bench) -> float:
            report = self.eval.grade(run_system(memory, bench.probes()), bench.instances)
            return self._objective(CapabilityTable.from_report(report))

        base_j = measure(self.memory, dev)
        best_cfg, best_j = self.memory_cfg, base_j
        seen = {self.memory_cfg.hash()}
        confirm = None
        base_confirm = 0.0
        for cfg in self._candidates(rng):
            if cfg.hash() in seen or len(cfg.features) > self.rules.feature_budget:
                continue
            seen.add(cfg.hash())
            self.stats["trials"] += 1
            candidate = self._factory(cfg)
            j = measure(candidate, dev)
            if j <= best_j + self.policy.min_gain:
                continue
            if self.policy.replicate:
                # A win has to repeat on data the comparison has not seen.
                if confirm is None:
                    confirm = generate(
                        f"{self.spec.seed}:{self.id}:confirm:{self.round}",
                        self.eval.standard.gen_config(self.policy.dev_streams, 12, balanced=balanced),
                    )
                    base_confirm = measure(self.memory, confirm)
                if measure(candidate, confirm) <= base_confirm + self.policy.min_gain / 2:
                    continue
            best_cfg, best_j = cfg, j
        if best_cfg.hash() != self.memory_cfg.hash():
            self._adopt(best_cfg, best_j - base_j)

    def _conform(self) -> None:
        """Optionally let my belief about scoring drift towards esteemed peers' beliefs."""
        rate = self.policy.conformity
        if rate <= 0 or not self.attention:
            return
        mix: dict[str, float] = {}
        wsum = 0.0
        for peer, w in self.attention:
            std = self.view(peer).standard
            if std is None:
                continue
            for cid, cw in std.criterion_weights.items():
                mix[cid] = mix.get(cid, 0.0) + w * cw
            wsum += w
        if wsum <= 0:
            return
        own = self.eval.standard
        blended = {cid: (1 - rate) * own.criterion_weights.get(cid, 0.0) + rate * mix.get(cid, 0.0) / wsum
                   for cid in set(own.criterion_weights) | set(mix)}
        blended = {c: w for c, w in blended.items() if w >= 0.02}
        new = Standard(normalise(blended), own.capability_weights, own.exam)
        if new.fingerprint() != own.fingerprint():
            self.eval = self.eval.with_standard(new)
            self._profile_version = -1  # my self-belief is stated under my standard

    def _rescore(self) -> None:
        """Adopt the scoring belief my brain proposed, if it proposed a usable one.

        Only the mixture over scoring recipes changes.  It is published with my
        next profile and applies to exams requested after that.
        """
        weights = self._plan.get("scoring")
        if not weights:
            return
        own = self.eval.standard
        try:
            wanted = Standard(weights, own.capability_weights, own.exam).criterion_weights
            # A belief about scoring moves at most half-way in one round, so one odd proposal (or one
            # piece of text a model was talked into following) cannot replace the standard outright.
            step = {c: 0.5 * own.criterion_weights.get(c, 0.0) + 0.5 * wanted.get(c, 0.0)
                    for c in set(own.criterion_weights) | set(wanted)}
            new = Standard({c: w for c, w in step.items() if w >= 0.02}, own.capability_weights, own.exam)
        except ValueError:
            return
        if new.fingerprint() != own.fingerprint():
            self.eval = self.eval.with_standard(new)
            self._profile_version = -1  # my self-belief is stated under my standard
            self.log.append({"round": self.round, "event": "rescore", "criteria": dict(new.criterion_weights)})

    # ------------------------------------------------------------ round hooks
    def begin_round(self, round_: int) -> None:
        self.round = round_
        self._served = 0
        self._tend_exams()
        self._refresh_profile()
        self._update_specialties()
        self._read_feeds()
        self._publish_profile()
        self._request_exams()

    def _choose_examiners(self) -> list[str]:
        rng = self._rng("examiners", self.round)
        directory = [a for a, info in self.board.directory().items() if a != self.id and info.get("last_profile")]
        if not directory:
            return []
        chosen: list[str] = []
        for peer in self._plan.get("sit_with", ()):
            if peer in directory and peer not in chosen and self.view(peer).integrity > 0:
                chosen.append(peer)
        pool = [(p, w) for p, w in self.attention if p in directory]
        for _ in range(self.policy.exams_per_round - len(chosen)):
            options = [(p, w) for p, w in pool if p not in chosen]
            explore = rng.random() < self.policy.explore or not options
            if explore:
                rest = sorted(p for p in directory if p not in chosen and self.view(p).integrity > 0)
                if rest:
                    chosen.append(rng.choice(rest))
            else:
                chosen.append(rng.choices([p for p, _ in options], weights=[w for _, w in options], k=1)[0])
        return chosen[: max(self.policy.exams_per_round, 0)]

    def _request_exams(self) -> None:
        for peer in self._choose_examiners():
            self._request_counter += 1
            xid = f"{self.id}>{peer}#{self._request_counter}"
            self._sat[xid] = {"examiner": peer, "round": self.round}
            self._post("exam.request", {"xid": xid}, to=peer)

    def end_round(self) -> None:
        self._update_esteem()
        if self.brain is not None:
            self._plan = self._ask_brain()
        self._strive()
        self._conform()
        self._rescore()
        self._plan = {k: v for k, v in self._plan.items() if k == "sit_with"}

    # ------------------------------------------------------------------ brain
    def summary(self) -> dict:
        """What I know, in a form a brain (or a person) can read."""
        believed = self.believed() if self.table is not None else {}
        return {
            "unit": self.id,
            "round": self.round,
            "feature_budget": self.rules.feature_budget,
            "design": self.memory_cfg.to_dict(),
            "standard": self.eval.standard.to_dict(),
            "believed": {c: [round(s, 3), n] for c, (s, n) in believed.items()},
            "standing": {c: [round(s, 3), round(n, 1)] for c, (s, n) in self.standing().items()},
            "attending": [
                {
                    "unit": p, "weight": round(w, 3), "esteem": round(self.view(p).esteem, 3),
                    "standard": self.view(p).standard.to_dict() if self.view(p).standard else None,
                    "design": self.view(p).design, "believed": self.view(p).believed,
                }
                for p, w in self.attention
            ],
            "recent_changes": self.log[-5:],
        }

    def _ask_brain(self) -> dict:
        self.stats["brain_calls"] += 1
        try:
            plan = self.brain.decide(self.summary())
        except ConfigurationError:
            raise  # no credentials or no such model: running on silently would hide it
        except Exception as exc:  # noqa: BLE001 - a failing brain must never stop the unit
            self.stats["brain_errors"] += 1
            self.log.append({"round": self.round, "event": "brain_error", "why": f"{type(exc).__name__}: {exc}"})
            return {}
        if not isinstance(plan, Mapping):
            return {}
        out: dict = {}
        if isinstance(plan.get("candidates"), list):
            out["candidates"] = [c for c in plan["candidates"] if isinstance(c, Mapping)][:4]
        if isinstance(plan.get("sit_with"), list):
            out["sit_with"] = [p for p in plan["sit_with"] if isinstance(p, str)][: self.policy.exams_per_round]
        if isinstance(plan.get("scoring"), list):
            weights: dict[str, float] = {}
            for entry in plan["scoring"][:6]:
                cid, w = (entry.get("criterion"), entry.get("weight")) if isinstance(entry, Mapping) else (None, None)
                if (cid in self.eval.family.ids and isinstance(w, (int, float)) and not isinstance(w, bool)
                        and math.isfinite(float(w)) and w > 0):
                    weights[cid] = weights.get(cid, 0.0) + float(w)
            if weights:
                out["scoring"] = weights
        note = plan.get("note")
        if isinstance(note, str) and note.strip():
            self._post("note", {"text": note.strip()[:600]})
        return out
