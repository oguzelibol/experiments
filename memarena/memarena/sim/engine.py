"""Deterministic simulation of a society of units, with an outside check.

Inside the simulation, units only ever see each other: exams, signals, feeds.
Outside it sits an ``ExternalEvaluator`` with a hidden benchmark and a hidden
scoring criterion.  Units never interact with it while the society runs.  It is
used afterwards to answer the questions a designer (or a sceptic) cares about:

* Did the society get good at the real task?  (outside score of the collective)
* Did it know how good it was?  (its own estimate vs the outside score, charged
  with the asymmetric loss)
* Were its internal signals honest?  (does esteem track true skill; were
  manipulated grades caught)

Every run is a pure function of (scenario, unit specs).  The result carries the
board's head hash so two parties can confirm they simulated the same history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..actors.unit import Unit, UnitPolicy, UnitSpec, WorldRules
from ..benchmark.schema import Answer
from ..benchmark.synthetic import GenConfig, generate
from ..evalcore.calibration import EXTERNAL_LOSS, AsymmetricLoss
from ..evalcore.criteria import JUDGE_SCORERS, CriteriaFamily, Criterion, aggregate
from ..evalcore.grader import GradeReport, Grader
from ..evalcore.module import CapabilityTable, expected_score
from ..evalcore.standard import CAPABILITIES
from ..feeds.board import AuthorBoard, Board
from ..memory.base import run_system
from ..memory.heuristic import FEATURES, MemoryConfig, observable_class
from ..util import make_rng, sha256_hex
from .adversaries import UNIT_KINDS, Colluder
from .collective import (
    CLASS_CAPABILITIES, audit_board, audit_record, champion_for, collective_estimate, gather_signals, published_since,
    reputation, select_champions,
)

__all__ = [
    "ExternalConfig", "ExternalEvaluator", "Scenario", "Simulation", "SimResult",
    "spearman", "random_population", "unit_secret",
]


@dataclass(frozen=True)
class ExternalConfig:
    """The outside benchmark.  Hidden from units and from the designer's search."""

    criterion_id: str = "judge_lenient|detect|micro"
    n_streams: int = 10
    questions_per_stream: int = 12
    gen: Mapping = field(default_factory=dict)  # difficulty overrides for the hidden data
    judge_noise: float = 0.01  # chance a binary judge verdict flips (LLM-judge variance)
    loss: AsymmetricLoss = EXTERNAL_LOSS
    penalty_weight: float = 5.0

    def to_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id, "n_streams": self.n_streams,
            "questions_per_stream": self.questions_per_stream, "gen": dict(self.gen),
            "judge_noise": self.judge_noise, "loss": self.loss.to_dict(), "penalty_weight": self.penalty_weight,
        }


class ExternalEvaluator:
    def __init__(self, cfg: ExternalConfig, seed: int | str):
        self.cfg = cfg
        self.seed = seed
        gen_kwargs = dict(cfg.gen)
        if "sessions_per_stream" in gen_kwargs:
            gen_kwargs["sessions_per_stream"] = int(gen_kwargs["sessions_per_stream"])
        self.benchmark = generate(
            f"external:{seed}", GenConfig(n_streams=cfg.n_streams, questions_per_stream=cfg.questions_per_stream, **gen_kwargs)
        )
        self.criterion = Criterion.from_id(cfg.criterion_id)
        self._grader = Grader(CriteriaFamily.default())

    def probes(self):
        return self.benchmark.probes()

    def grade(self, answers: Mapping[str, Answer]) -> GradeReport:
        return self._grader.grade(answers, self.benchmark.instances)

    def score(self, answers: Mapping[str, Answer], report: GradeReport | None = None) -> float:
        """Score under the hidden criterion, including judge noise."""
        report = report or self.grade(answers)
        noisy = self.criterion.scorer in JUDGE_SCORERS and self.cfg.judge_noise > 0
        rows = []
        for it in report.items:
            included, s = it.score(self.criterion)
            # noise models a judge misreading an answer; a row that never arrived has nothing to misread
            if included and noisy and s in (0.0, 1.0) and it.status in ("ok", "truncated"):
                text = answers[it.probe_id].text if it.probe_id in answers else ""
                if make_rng("judge-noise", self.seed, it.probe_id, sha256_hex(text)).random() < self.cfg.judge_noise:
                    s = 1.0 - s
            rows.append((it.question_type, included, s))
        return aggregate(self.criterion, rows)


@dataclass(frozen=True)
class Scenario:
    seed: int | str = 0
    rounds: int = 10
    rules: WorldRules = field(default_factory=WorldRules)
    external: ExternalConfig = field(default_factory=ExternalConfig)

    def to_dict(self) -> dict:
        return {"seed": str(self.seed), "rounds": self.rounds, "rules": self.rules.to_dict(), "external": self.external.to_dict()}


@dataclass
class SimResult:
    scenario: dict
    value: float  # collective outside score minus weighted estimate loss
    metrics: dict
    trace: list
    units: dict
    by_owner: dict
    board_head: tuple

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario, "value": self.value, "metrics": self.metrics, "trace": self.trace,
            "units": self.units, "by_owner": self.by_owner, "board_head": list(self.board_head),
        }


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation; None when undefined (too few points or no variation)."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (vx * vy) ** 0.5


def unit_secret(world_seed: int | str, index: int | str) -> str:
    """A unit's private seed.  One-way in the world seed: a unit that knows its own secret
    learns nothing about other units' secrets, the world, or the outside benchmark."""
    return sha256_hex(f"unit-secret|{world_seed}|{index}")[:40]


def random_population(
    n: int, seed: int | str, rules: WorldRules | None = None, policy: UnitPolicy | None = None,
    owner: str = "designer", prefix: str = "u",
) -> list[UnitSpec]:
    """Units with random designs and random beliefs about scoring (drawn from the prior)."""
    rules = rules or WorldRules()
    policy = policy or UnitPolicy()
    family = CriteriaFamily.default(rules.family_prior)
    ids = list(family.ids)
    weights = [family.prior[i] for i in ids]
    specs = []
    for k in range(n):
        rng = make_rng("population", seed, k)
        feats = frozenset(rng.sample(list(FEATURES), rules.feature_budget))
        cfg = MemoryConfig(features=feats, top_k=rng.choice([4, 6, 8, 10]), style=rng.choice(["terse", "sentence"]))
        criterion = rng.choices(ids, weights=weights, k=1)[0]
        specs.append(
            UnitSpec(
                unit_id=f"{prefix}{k:02d}", owner=owner, policy=policy, memory=cfg,
                criterion_weights={criterion: 1.0}, seed=unit_secret(seed, k),
            )
        )
    return specs


class Simulation:
    def __init__(
        self,
        scenario: Scenario,
        specs: Sequence[UnitSpec],
        board: Any = None,
        brain_factory: Callable[[UnitSpec], Any] | None = None,
        on_round: Callable[["Simulation", int], None] | None = None,
        focus_owner: str | None = None,
        estimate_shade: float | Mapping[str, float] = 1.0,
        board_for: Callable[[str], Any] | None = None,
    ):
        if len({s.unit_id for s in specs}) != len(specs):
            raise ValueError("unit ids must be unique")
        self.scenario = scenario
        self.board = board if board is not None else Board()
        self.external = ExternalEvaluator(scenario.external, scenario.seed)
        self.family = CriteriaFamily.default(scenario.rules.family_prior)
        self._ring: dict = {}
        self.units: dict[str, Unit] = {}
        self._brain_factory = brain_factory
        # Over HTTP each unit holds its own authenticated connection to the board.
        self._board_for = board_for
        for spec in specs:
            self.add_unit(spec)
        self._cursor = 0
        self._verified: dict[int, bool] = {}
        self._outside_published: set[tuple[str, int]] = set()
        self._truth: dict[str, dict] = {}
        self.trace: list[dict] = []
        self.round = 0
        self._on_round = on_round
        # The owner whose collective the headline value describes (default: everyone).
        self.focus_owner = focus_owner
        self.estimate_shade = estimate_shade

    # ------------------------------------------------------------- population
    def add_unit(self, spec: UnitSpec) -> Unit:
        cls = UNIT_KINDS.get(spec.kind)
        if cls is None:
            raise ValueError(f"unknown unit kind {spec.kind!r}")
        brain = self._brain_factory(spec) if self._brain_factory else None
        kwargs: dict = {"rules": self.scenario.rules, "brain": brain, "family": self.family}
        if cls is Colluder:
            kwargs["shared"] = self._ring
        if self._board_for:
            unit_board = self._board_for(spec.unit_id)
        elif isinstance(self.board, Board):
            unit_board = AuthorBoard(self.board, spec.unit_id)  # a unit can post only under its own name
        else:
            unit_board = self.board
        unit = cls(spec, unit_board, **kwargs)
        self.units[spec.unit_id] = unit
        return unit

    def remove_unit(self, unit_id: str) -> None:
        self.units.pop(unit_id, None)

    # ------------------------------------------------------------------ truth
    def truth(self, unit: Unit) -> dict:
        """A unit's real performance on the hidden outside benchmark (never shown to units)."""
        key = unit.memory.config_hash()
        hit = self._truth.get(key)
        if hit is None:
            # A fresh system of the same design answers the hidden questions, not the unit's own
            # object: nothing a unit holds is ever shown them.
            answers = run_system(unit._factory(unit.memory_cfg), self.external.probes())
            report = self.external.grade(answers)
            hit = {
                "answers": answers,
                "score": self.external.score(answers, report),
                "table": CapabilityTable.from_report(report),
            }
            self._truth[key] = hit
        return hit

    # ------------------------------------------------------------------- loop
    def _pump(self) -> None:
        while True:
            posts = self.board.read(since_seq=self._cursor)
            if not posts:
                return
            for post in posts:
                self._cursor = post.seq
                target = self.units.get(post.to) if post.to else None
                if target is not None:
                    target.on_message(post)

    def step(self) -> None:
        self.round += 1
        r = self.round
        self.board.post("env", "clock", {"round": r}, round=r)
        order = sorted(self.units)
        make_rng("order", self.scenario.seed, r).shuffle(order)
        for uid in order:
            self.units[uid].begin_round(r)
        self._pump()
        for uid in order:
            self.units[uid].end_round()
        scores = [self.truth(u)["score"] for u in self.units.values()]
        self.trace.append({
            "round": r, "mean_true": sum(scores) / len(scores), "max_true": max(scores),
            "posts": self.board.head()[0], "units": len(self.units),
        })
        if self._on_round:
            self._on_round(self, r)

    def run(self) -> SimResult:
        while self.round < self.scenario.rounds:
            self.step()
        return self.result()

    # ---------------------------------------------------------------- results
    def _shade_for(self, owner: str | None) -> float:
        """How far an owner shades its estimate: its own setting when one was given per owner."""
        if isinstance(self.estimate_shade, Mapping):
            return float(self.estimate_shade.get(owner, 1.0))
        return float(self.estimate_shade)

    def _current_since(self) -> dict[str, int]:
        # read from the board, not from the units: the society's accounts use the public record only
        return published_since(self.board, sorted(self.units))

    def collective(self, members: Sequence[str] | None = None, shade: float = 1.0) -> dict:
        """Champions, composite answers, the society's estimate and the outside outcome."""
        members = sorted(members if members is not None else self.units)
        # Every result is re-verified here; nothing unverified reaches the society's accounts.
        rules = self.scenario.rules
        audit = audit_record(self.board, self.family, self._verified)
        signals, cheats = audit.signals, audit.cheats
        rep = reputation(signals, sorted(self.units), banned=cheats, seeds=members)
        # An examiner that walks away from exams once it knows the paper publishes a selected sample.
        rep = {u: r * audit.completion.get(u, 1.0) ** 2 for u, r in rep.items()}
        exam_items = rules.exam_streams * rules.exam_questions
        champions = select_champions(signals, rep, members, self._current_since(), max_items=exam_items)
        probes = self.external.probes()
        class_counts: dict[str, int] = {}
        composite: dict[str, Answer] = {}
        fallback = max(members, key=lambda u: (rep.get(u, 0.0), u)) if members else None
        for p in probes:
            cls = observable_class(p.question)
            class_counts[cls] = class_counts.get(cls, 0) + 1
            champ = champion_for(champions, cls, p.probe_id) or fallback
            if champ is not None:
                composite[p.probe_id] = self.truth(self.units[champ])["answers"][p.probe_id]
        estimate = collective_estimate(champions, class_counts, loss=self.scenario.external.loss, shade=shade)
        outside = self.external.score(composite)
        loss = self.scenario.external.loss(estimate["report"], outside)
        # how much better the truly best unit per class would have been
        regret = []
        for cls, info in champions.items():
            caps = CLASS_CAPABILITIES[cls]
            crit = {self.scenario.external.criterion_id: 1.0}

            def true_class(uid: str) -> float:
                scores = self.truth(self.units[uid])["table"].capability_scores(crit)
                return sum(w * scores.get(c, (0.0, 0))[0] for c, w in caps.items())

            best = max(true_class(u) for u in members)
            regret.append(best - sum(true_class(u) for u in info["units"]) / len(info["units"]))
        return {
            "champions": {c: list(i["units"]) for c, i in champions.items()},
            "estimate_mean": estimate["mean"], "estimate_sd": estimate["sd"], "estimate_report": estimate["report"],
            "outside_score": outside, "loss": loss,
            "value": outside - self.scenario.external.penalty_weight * loss,
            "champion_regret": sum(regret) / len(regret) if regret else None,
            "reputation": rep, "cheats": sorted(cheats), "n_signals": len(signals),
            "mean_peer_loss": sum(s.loss for s in signals) / len(signals) if signals else None,
            "results_posted": len(gather_signals(self.board)),
            "settled_without_result": audit.settled_without_result,
            "walkouts": dict(sorted(audit.walkouts.items())),
        }

    def _publish_outside(self, by_owner: Mapping[str, Mapping]) -> None:
        """The outside evaluator publishes its verdict on the shared board, once per owner and round.

        Each owner's estimate is a function of the posts that precede this one
        (the society's accounts are computed from the public record alone), so
        the chain itself shows that the estimate was fixed before the outside
        score was known.
        """
        round_ = len(self.trace)
        for owner, info in by_owner.items():
            if (owner, round_) in self._outside_published:
                continue
            self._outside_published.add((owner, round_))
            self.board.post("env", "outside.score", {
                "owner": owner, "estimate": info["estimate_report"], "score": info["outside_score"],
                "loss": info["loss"], "value": info["value"],
            }, round=round_)

    def result(self) -> SimResult:
        units = self.units
        focus = (
            [uid for uid, u in units.items() if u.spec.owner == self.focus_owner]
            if self.focus_owner is not None else None
        )
        coll = self.collective(focus, shade=self._shade_for(self.focus_owner))
        true_scores = {uid: self.truth(u)["score"] for uid, u in units.items()}
        rep = coll["reputation"]
        honest = [uid for uid, u in units.items() if u.spec.kind == "honest"]

        # Does each unit's esteem track how peers would really do on its own exam?
        validities = []
        for uid in honest:
            u = units[uid]
            std = u.eval.standard
            xs, ys = [], []
            for peer, view in u.known.items():
                if peer not in units or not view.first_hand:
                    continue
                xs.append(view.esteem)
                ys.append(expected_score(self.truth(units[peer])["table"], std.criterion_weights, std.capability_weights))
            rho = spearman(xs, ys)
            if rho is not None:
                validities.append(rho)
        ids = sorted(units)
        rep_validity = spearman([rep.get(u, 0.0) for u in ids], [true_scores[u] for u in ids])

        attention_to_bad = []
        bad_kinds = {uid for uid, u in units.items() if u.spec.kind != "honest"}
        for uid in honest:
            attention_to_bad.append(sum(w for p, w in units[uid].attention if p in bad_kinds))
        totals = {k: sum(u.stats[k] for u in units.values()) for k in ("exams_set", "audits", "audit_fails", "adoptions", "trials")}
        metrics = {
            "outside_score": coll["outside_score"],
            "estimate_report": coll["estimate_report"],
            "estimate_mean": coll["estimate_mean"],
            "estimate_error": coll["estimate_report"] - coll["outside_score"],
            "estimate_loss": coll["loss"],
            "champion_regret": coll["champion_regret"],
            "champions": coll["champions"],
            "mean_true_first": self.trace[0]["mean_true"] if self.trace else None,
            "mean_true_last": self.trace[-1]["mean_true"] if self.trace else None,
            "best_single_true": max(true_scores.values()) if true_scores else None,
            "esteem_validity": sum(validities) / len(validities) if validities else None,
            "reputation_validity": rep_validity,
            "exchanges": coll["n_signals"],  # exams whose published result verifies
            "results_posted": coll["results_posted"],
            "settled_without_result": coll["settled_without_result"],
            "walkouts": sum(coll["walkouts"].values()),
            "mean_peer_loss": coll["mean_peer_loss"],
            "cheats_confirmed": coll["cheats"],
            "attention_to_dishonest": sum(attention_to_bad) / len(attention_to_bad) if attention_to_bad and bad_kinds else None,
            **totals,
        }
        unit_info = {
            uid: {
                "owner": u.spec.owner, "kind": u.spec.kind, "true_score": true_scores[uid],
                "reputation": rep.get(uid, 0.0), "design": u.memory_cfg.to_dict(),
                "standard": u.eval.standard.to_dict(), "self_score": u.self_score,
                "attending": [p for p, _ in u.attention], "version": u.version, "stats": dict(u.stats),
            }
            for uid, u in units.items()
        }
        by_owner: dict[str, dict] = {}
        for owner in sorted({u.spec.owner for u in units.values()}):
            members = [uid for uid, u in units.items() if u.spec.owner == owner]
            c = (
                coll if sorted(members) == sorted(focus if focus is not None else units)
                else self.collective(members, shade=self._shade_for(owner))
            )
            by_owner[owner] = {
                "units": len(members), "value": c["value"], "outside_score": c["outside_score"],
                "estimate_report": c["estimate_report"], "loss": c["loss"],
            }
        self._publish_outside(by_owner)
        return SimResult(
            scenario=self.scenario.to_dict(), value=coll["value"], metrics=metrics, trace=self.trace,
            units=unit_info, by_owner=by_owner, board_head=self.board.head(),
        )
