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

from ..benchmark.schema import Answer, AnswerKey, Instance, Probe, Session, Turn
from ..benchmark.synthetic import generate
from ..evalcore.calibration import PEER_LOSS, AsymmetricLoss, binomial_sd
from ..evalcore.criteria import CriteriaFamily
from ..evalcore.module import CapabilityTable, EvaluationModule, expected_score, verify_receipt
from ..evalcore.standard import CAPABILITIES, Standard, normalise
from ..memory.base import MemorySystem, run_system
from ..memory.heuristic import FEATURES, STYLES, HeuristicMemory, MemoryConfig
from ..util import commit, make_rng, sha256_hex, verify_commit
from .protocol import (
    check_result, derive_exam_share, joint_exam_seed, pack_probes, share_commitment, unpack_probes,
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
        self.self_score = 0.0
        self._profile_version = -1
        self._published: tuple | None = None
        self.known: dict[str, PeerView] = {}
        self.attention: list[tuple[str, float]] = []  # (peer, weight), strongest first
        self._set: dict[str, dict] = {}  # exams I set, by xid
        self._sat: dict[str, dict] = {}  # exams I sit, by xid
        self._served = 0
        self._exam_counter = 0
        self._request_counter = 0
        self._inbox_cursor = 0
        self._plan: dict = {}  # advice from the brain for the coming round
        self.round = 0
        self.log: list[dict] = []  # adopted changes, audits: for analysis
        self.stats = {"exams_set": 0, "exams_sat": 0, "audits": 0, "audit_fails": 0, "adoptions": 0, "trials": 0}

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
                    crowd = sum(1.0 for v in with_std if v.standard.capability_weights.get(c, 0.0) > 0) / len(with_std)
            niche[c] = mine - pol.niche * ref - pol.crowd_aversion * crowd
        able = [c for c in CAPABILITIES if scores.get(c, (0.0, 0))[0] >= 0.2] or list(scores)
        ranked = sorted(able, key=lambda c: (-niche[c], CAPABILITIES.index(c)))
        top = ranked[: max(1, pol.specialty_count)]
        weights = {c: max(0.05, scores.get(c, (0.0, 0))[0]) for c in top}
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
            self._post("profile", body)
            self._published = key

    # ---------------------------------------------------------------- feeds
    def _read_feeds(self) -> None:
        """Read the feeds I attend to: profiles, signals about others, audit reports."""
        rng = self._rng("audit", self.round)
        for peer, _w in self.attention[: self.policy.attention_k]:
            v = self.view(peer)
            posts = self.board.feed(peer, since=v.feed_cursor)
            for post in posts:
                v.feed_cursor = max(v.feed_cursor, post.author_seq)
                if post.kind == "profile":
                    self._absorb_profile(peer, post.body)
                elif post.kind == "exam.result":
                    examinee = post.body.get("examinee")
                    if not examinee or examinee == self.id:
                        continue
                    if self.policy.second_hand <= 0 and rng.random() >= self.policy.audit_rate:
                        continue  # not relying on relayed signals, and not auditing this one
                    self.stats["audits"] += 1
                    verdict = self._check(post)
                    if not verdict.ok:
                        self._flag(post.author, post.seq, verdict.reason)
                        continue
                    # a relayed signal is only ever used after I have recomputed it myself
                    self.view(examinee).relayed[verdict.xid] = (peer, verdict.score, verdict.n_items, post.round)
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
        v = self.view(peer)
        try:
            v.standard = Standard.from_dict(body["standard"])
        except (KeyError, ValueError):
            return
        v.believed = dict(body.get("believed", {}))
        v.design = body.get("design")

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
        return generate(exam_seed, module.standard.gen_config(self.rules.exam_streams, self.rules.exam_questions))

    def _answer_exam(self, probes: Sequence[Probe], examiner: str, xid: str) -> dict[str, Answer]:
        return run_system(self.memory, probes)

    def _make_guess(self, mean: float, sd: float) -> float:
        return self.rules.peer_loss.best_report(mean, sd, self.policy.guess_shade)

    def on_message(self, post) -> None:
        """Handle one post addressed to me.  A malformed post is ignored, never fatal."""
        handler = {
            "exam.request": self._on_request, "exam.ready": self._on_ready, "exam.nonce": self._on_nonce,
            "exam.offer": self._on_offer, "exam.answers": self._on_answers, "exam.graded": self._on_graded,
            "exam.guess": self._on_guess, "exam.result": self._on_result,
        }.get(post.kind)
        if handler is None or not isinstance(post.body, Mapping):
            return
        try:
            handler(post)
        except Exception as exc:  # noqa: BLE001 - whatever a peer sends, it must not be able to stop me
            self.log.append({"round": self.round, "event": "bad_message", "kind": post.kind,
                             "from": post.author, "why": f"{type(exc).__name__}: {str(exc)[:200]}"})

    # -- as examiner
    def _on_request(self, post) -> None:
        xid = post.body.get("xid")
        if not isinstance(xid, str) or not xid or xid in self._set or post.author == self.id:
            return
        if self._served >= self.policy.serve_capacity or self.table is None:
            self._post("exam.decline", {"xid": xid}, to=post.author)
            return
        self._served += 1
        self._exam_counter += 1
        share = derive_exam_share(self.spec.seed, self.id, self._exam_counter)
        # The standard in force now is frozen for this exam, even if I change my mind before grading.
        self._set[xid] = {
            "examinee": post.author, "share": share, "module": self.eval, "request_seq": post.seq,
            "round": self.round,
        }
        ready = self._post("exam.ready", {"xid": xid, "share_commit": share_commitment(share)}, to=post.author)
        self._set[xid]["ready_seq"] = ready.seq

    def _on_nonce(self, post) -> None:
        xid = post.body.get("xid")
        st = self._set.get(xid)
        nonce = post.body.get("nonce")
        if st is None or st["examinee"] != post.author or "exam" in st or not isinstance(nonce, str) or not nonce:
            return
        module: EvaluationModule = st["module"]
        exam = self._draw_exam(post.author, joint_exam_seed(st["share"], nonce, xid), module)
        salt = sha256_hex(f"{self.spec.seed}:{self.id}:salt:{xid}")[:20]
        probes_blob = self.board.put_blob(pack_probes(exam.probes()))
        from ..evalcore.module import keys_commitment

        st.update(exam=exam, salt=salt, probes=probes_blob, nonce_seq=post.seq)
        offer = self._post(
            "exam.offer",
            {
                "xid": xid, "probes": probes_blob, "n": len(exam),
                "keys_commit": keys_commitment(exam.instances, salt),
                "standard": module.standard.to_dict(), "evaluator": module.fingerprint(),
                "loss": self.rules.peer_loss.to_dict(),
                "exam_size": [self.rules.exam_streams, self.rules.exam_questions],
            },
            to=post.author,
        )
        st["offer_seq"] = offer.seq

    def _on_answers(self, post) -> None:
        xid = post.body.get("xid")
        st = self._set.get(xid)
        if st is None or st["examinee"] != post.author or "exam" not in st or "evaluation" in st:
            return
        if post.body.get("offer_seq") != st["offer_seq"]:
            return  # not an answer to the paper I set
        answers = {str(pid): Answer.from_dict(a) for pid, a in self.board.get_blob(post.body["answers"]).items()}
        evaluation = st["module"].evaluate(answers, st["exam"].instances, st["salt"])
        payload = self._result_payload(post.author, evaluation)
        nonce = sha256_hex(f"{self.spec.seed}:{self.id}:rnonce:{xid}")[:20]
        st.update(evaluation=evaluation, payload=payload, nonce=nonce, answers_seq=post.seq,
                  answers_blob=post.body["answers"], guess_commit=str(post.body.get("guess_commit", "")))
        graded = self._post("exam.graded", {"xid": xid, "result_commit": commit(payload["score"], nonce)}, to=post.author)
        st["graded_seq"] = graded.seq
        st["graded_round"] = self.round

    def _on_guess(self, post) -> None:
        xid = post.body.get("xid")
        st = self._set.get(xid)
        if st is None or st["examinee"] != post.author or "payload" not in st or st.get("done"):
            return
        guess = post.body.get("guess")
        valid = (
            isinstance(guess, (int, float)) and not isinstance(guess, bool) and math.isfinite(float(guess))
            and verify_commit(st["guess_commit"], guess, str(post.body.get("nonce", "")))
        )
        self._publish_result(xid, st, min(1.0, max(0.0, float(guess))) if valid else 1.0, bool(valid), post.seq)

    def _publish_result(self, xid: str, st: dict, guess: float, valid: bool, guess_seq: int | None) -> None:
        """Publish the outcome of an exam: the public signal about the examinee."""
        payload = st["payload"]
        loss = self.rules.peer_loss(guess, payload["score"])
        st["done"] = True
        self.stats["exams_set"] += 1
        self._post(
            "exam.result",
            {
                "xid": xid, "examinee": st["examinee"], **payload,
                "guess": guess, "guess_valid": valid, "loss": loss,
                "probes": st["probes"], "answers": st["answers_blob"],
                "salt": st["salt"], "nonce": st["nonce"], "share": st["share"],
                "request_seq": st["request_seq"], "ready_seq": st["ready_seq"], "nonce_seq": st["nonce_seq"],
                "offer_seq": st["offer_seq"], "answers_seq": st["answers_seq"], "graded_seq": st["graded_seq"],
                "guess_seq": guess_seq,
            },
            to=st["examinee"],
        )
        n = int(payload["receipt"]["n_items"])
        self.view(st["examinee"]).first_hand.append((float(payload["score"]), n, float(loss), self.round))
        st.pop("exam", None)  # the seed share is public now; the paper is spent

    def _close_unopened(self) -> None:
        """Exams whose examinee never opened its self-estimate are published once the round is over.

        Otherwise an examinee could make a bad grade disappear by going silent.
        The unopened estimate counts as the maximal claim.
        """
        for xid, st in list(self._set.items()):
            if "payload" in st and not st.get("done") and st.get("graded_round", self.round) < self.round:
                self._publish_result(xid, st, 1.0, False, None)

    # -- as examinee
    def _on_ready(self, post) -> None:
        xid = post.body.get("xid")
        st = self._sat.get(xid)
        if st is None or st.get("examiner") != post.author or "nonce" in st:
            return
        st["nonce"] = sha256_hex(f"{self.spec.seed}:{self.id}:exam-nonce:{xid}")[:32]
        self._post("exam.nonce", {"xid": xid, "nonce": st["nonce"]}, to=post.author)

    def _on_offer(self, post) -> None:
        xid = post.body.get("xid")
        st = self._sat.get(xid)
        if st is None or st.get("examiner") != post.author or "nonce" not in st or "answers" in st:
            return
        # I only sit exams on the terms this world runs on.
        if post.body.get("loss") != self.rules.peer_loss.to_dict():
            return
        if list(post.body.get("exam_size", ())) != [self.rules.exam_streams, self.rules.exam_questions]:
            return
        probes = unpack_probes(self.board.get_blob(post.body["probes"]))
        if len(probes) != self.rules.exam_streams * self.rules.exam_questions:
            return  # not the size of exam this world sits
        standard = Standard.from_dict(post.body["standard"])
        self.view(post.author).standard = standard
        answers = self._answer_exam(probes, post.author, xid)
        mean, sd = self._predict(standard, len(probes), post.author)
        guess = self._make_guess(mean, sd)
        guess_nonce = sha256_hex(f"{self.spec.seed}:{self.id}:nonce:{xid}")[:20]
        answers_blob = self.board.put_blob({pid: a.to_dict() for pid, a in sorted(answers.items())})
        st.update(answers=answers, guess=guess, guess_nonce=guess_nonce, predicted=mean, offer_seq=post.seq)
        self._post(
            "exam.answers",
            {"xid": xid, "answers": answers_blob, "guess_commit": commit(guess, guess_nonce), "offer_seq": post.seq},
            to=post.author,
        )

    def _on_graded(self, post) -> None:
        xid = post.body.get("xid")
        st = self._sat.get(xid)
        if st is None or st.get("examiner") != post.author or "guess" not in st or "result_commit" in st:
            return
        st["result_commit"] = str(post.body.get("result_commit", ""))
        self._post("exam.guess", {"xid": xid, "guess": st["guess"], "nonce": st["guess_nonce"]}, to=post.author)

    def _on_result(self, post) -> None:
        xid = post.body.get("xid")
        if post.body.get("examinee") != self.id:
            return
        st = self._sat.get(xid)
        self.stats["audits"] += 1
        verdict = self._check(post)
        if not verdict.ok:
            # including a result about an exam I never sat: it has no transcript of mine to point at
            self._flag(post.author, post.seq, verdict.reason)
            return
        if st is None or st.get("examiner") != post.author or st.get("done"):
            return
        st["done"] = True
        self.stats["exams_sat"] += 1
        # record what I recomputed, not what the post says
        self.view(post.author).signals.append(
            (dict(verdict.capability), verdict.score, verdict.n_items, self.round, st.get("predicted"))
        )

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
            self.log.append({
                "round": self.round, "event": "adopt", "from": self.memory_cfg.to_dict(),
                "to": best_cfg.to_dict(), "dev_gain": best_j - base_j,
            })
            self.memory_cfg = best_cfg
            self.memory = self._factory(best_cfg)
            self.version += 1
            self.version_round = self.round + 1
            self.stats["adoptions"] += 1

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

    # ------------------------------------------------------------ round hooks
    def begin_round(self, round_: int) -> None:
        self.round = round_
        self._served = 0
        self._close_unopened()
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
            self._sat[xid] = {"examiner": peer}
            self._post("exam.request", {"xid": xid}, to=peer)

    def end_round(self) -> None:
        self._update_esteem()
        if self.brain is not None:
            self._plan = self._ask_brain()
        self._strive()
        self._conform()
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
        try:
            plan = self.brain.decide(self.summary())
        except Exception as exc:  # noqa: BLE001 - a failing brain must never stop the unit
            self.log.append({"round": self.round, "event": "brain_error", "why": f"{type(exc).__name__}: {exc}"})
            return {}
        if not isinstance(plan, Mapping):
            return {}
        out: dict = {}
        if isinstance(plan.get("candidates"), list):
            out["candidates"] = [c for c in plan["candidates"] if isinstance(c, Mapping)][:4]
        if isinstance(plan.get("sit_with"), list):
            out["sit_with"] = [p for p in plan["sit_with"] if isinstance(p, str)][: self.policy.exams_per_round]
        note = plan.get("note")
        if isinstance(note, str) and note.strip():
            self._post("note", {"text": note.strip()[:600]})
        return out
