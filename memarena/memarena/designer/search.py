"""The designer: builds societies, tests them, keeps the best, shows its work.

Method, in the order a sceptical reader would want it:

1. **Search** on *training* worlds.  Designs are compared on the same worlds
   (common random numbers).  Cheap short runs screen a population; survivors get
   longer runs on more worlds; the best seed the next generation through
   mutation and crossover.
2. **Validation** on fresh worlds picks one design among the finalists, so the
   pick is not simply the luckiest training score.
3. **Pre-registration.**  The chosen design and the evaluation protocol are
   written to the notebook *before* any held-out world exists.  Held-out seeds
   are derived from the hash of that entry and, when a randomness beacon is
   used, from beacon randomness that did not exist when the entry was written
   and published.  Without a beacon the seeds depend only on the notebook,
   which its author controls and could re-roll; verification says so.
4. **Held-out evaluation** gives the claimed value with a confidence interval,
   next to baselines and ablations run on the same worlds.
5. **Refinement** may start a further round from what the held-out arms
   showed.  That spends those worlds as test data, so the round pre-registers
   again and is judged on new worlds.
6. **Everything is in the notebook**, hash-chained, including what did not
   work.  ``Designer.verify_notebook`` recomputes every report from the logged
   entries, checks each commitment, and re-runs experiments, comparing values
   and simulated histories (board head hashes).

The designer never sees a world's outside criterion: it only receives the value
a design achieved there.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from ..evalcore.module import code_hash
from ..evalcore.stats import t_quantile_975
from ..sim.engine import Simulation
from ..util import canonical_json, make_rng
from .beacon import Beacon, BeaconError, is_randomness, public_beacons
from .design import Design, build_seed, clamp_design, crossover, mutate_design, sample_scenario
from .notebook import Notebook

__all__ = [
    "evaluate_design", "Designer", "mean_ci", "ABLATIONS", "heldout_world_seeds", "heldout_report",
    "refinement_candidates",
]

_METRIC_KEYS = (
    "outside_score", "estimate_report", "estimate_error", "estimate_loss", "champion_regret",
    "mean_true_first", "mean_true_last", "best_single_true", "esteem_validity",
    "reputation_validity", "exchanges", "mean_peer_loss", "adoptions", "audit_fails",
    "results_posted", "settled_without_result", "walkouts",
)

# The notebook stores floats to six decimals, so a recorded number and its
# recomputation may differ by that much and no more.
_TOLERANCE = 1e-6

# A value is a score in [0, 1] minus a bounded penalty.  Verification refuses
# recorded values far outside that range before it does arithmetic on them.
_MAX_VALUE = 1e6


def evaluate_design(job: Mapping) -> dict:
    """Run one design in one world.  A pure function of ``job``; safe to run in a worker."""
    design = Design.from_dict(job["design"])
    scenario = sample_scenario(job["scenario"], rounds=int(job["rounds"]))
    # units are built from a one-way hash of the world's name: the hidden evaluator cannot be derived from it
    specs = design.build(seed=build_seed(job["scenario"]), rules=scenario.rules)
    result = Simulation(scenario, specs, estimate_shade=design.shade).run()
    return {
        "value": result.value,
        "metrics": {k: result.metrics.get(k) for k in _METRIC_KEYS},
        "board_head": result.board_head[1],
    }


def _map(fn: Callable[[Mapping], dict], jobs: Sequence[Mapping], workers: int) -> list[dict]:
    """Apply ``fn`` to every job, in worker processes when that can help."""
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, jobs))
    return [fn(j) for j in jobs]


def mean_ci(xs: Sequence[float]) -> tuple[float, float, float]:
    """Mean and 95% t-interval, with the exact Student t quantile for ``n - 1`` degrees of freedom.

    One observation says nothing about spread, so for ``n == 1`` the interval
    collapses to the point; that is the absence of an interval, not certainty.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(xs) / n
    if n == 1:
        return m, m, m
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    half = t_quantile_975(n - 1) * sd / math.sqrt(n)
    return m, m - half, m + half


def _mean_metrics(results: Sequence[Mapping]) -> dict:
    out = {}
    for k in _METRIC_KEYS:
        vals = [r["metrics"][k] for r in results if isinstance(r["metrics"].get(k), (int, float))]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


# Ablations of a chosen design.  Each changes exactly one idea, so the held-out
# comparison says what that idea is worth.  "flipped" arms toggle a switch
# whichever way the chosen design has it.
ABLATIONS: dict[str, Any] = {
    # no self-improvement at all
    "striving_off": lambda d: replace(d, policy=replace(d.policy, trials_per_round=0)),
    # strive on own specialties under own standard only, ignoring how esteemed peers score
    "peer_standards_ignored": lambda d: replace(d, policy=replace(d.policy, peer_scope="none")),
    # strive towards esteemed peers' whole standards, their specialties included, instead of only their
    # way of scoring within the unit's own niche (or the reverse, if the design already does that)
    "peer_scope_flipped": lambda d: replace(d, policy=replace(
        d.policy, **({"peer_scope": "criteria", "focused_dev": True} if d.policy.peer_scope == "full"
                     else {"peer_scope": "full", "focused_dev": False}))),
    # sit exams of random units instead of the ones it esteems
    "attention_random": lambda d: replace(d, policy=replace(d.policy, explore=1.0)),
    # specialise where the absolute score is highest, ignoring peers and crowding
    "niche_off": lambda d: replace(d, policy=replace(d.policy, niche=0.0, crowd_aversion=0.0)),
    # copy designs published by esteemed peers (or stop doing so)
    "design_copying_flipped": lambda d: replace(d, policy=replace(d.policy, adopt_designs=not d.policy.adopt_designs)),
    # accept a design change on one development set instead of two (or the reverse)
    "replication_flipped": lambda d: replace(d, policy=replace(d.policy, replicate=not d.policy.replicate)),
    # start from random designs instead of a covering set of specialists (or the reverse)
    "initial_designs_flipped": lambda d: replace(d, init="random" if d.init == "covering" else "covering"),
    # report the mean of the society's estimate instead of the loss-optimal number
    "estimate_unshaded": lambda d: replace(d, shade=0.0),
}


# --------------------------------------------------------- held-out protocol
# These functions define the protocol.  The designer follows them and the
# verifier recomputes with them, so both read a notebook the same way.


def heldout_world_seeds(prereg_hash: str, randomness: str | None, n_worlds: int) -> list[str]:
    """Seeds of the held-out worlds that belong to one pre-registration entry.

    They contain the hash of that entry, so they cannot exist before the design
    is committed, and the beacon randomness when there is one, so that the
    committed notebook does not determine them either.  Without randomness the
    notebook's author can compute them in advance, and re-roll them by changing
    anything earlier in the notebook.
    """
    stem = f"heldout:{prereg_hash[:20]}" + (f":{randomness[:20]}" if randomness is not None else "")
    return [f"{stem}:{k}" for k in range(n_worlds)]


def _arms(design: Design, baselines: Mapping[str, Design] | None) -> list[tuple[str, Design]]:
    """The held-out arms in protocol order: the chosen design, then baselines and ablations by name."""
    arms = [("chosen", design)]
    arms += [(f"baseline:{name}", baseline) for name, baseline in sorted((baselines or {}).items())]
    arms += [(f"ablation:{name}", ABLATIONS[name](design)) for name in sorted(ABLATIONS)]
    return arms


def _registered_arms(protocol: Mapping) -> list[tuple[str, str, Design | None]]:
    """The arms a pre-registration commits to, in the order they must be logged: name, design hash, design.

    Held-out eval entries carry no arm name; their position in this order is
    what names them.  Ablations are recomputed from the pre-registered design,
    so an arm called ``ablation:x`` is that ablation and nothing weaker.  The
    design is None for baselines, which a pre-registration fixes by hash only.
    Raises ``KeyError`` for an ablation this code does not define.
    """
    design = Design.from_dict(protocol["design"])
    arms: list[tuple[str, str, Design | None]] = [("chosen", protocol["design_hash"], design)]
    arms += [(f"baseline:{name}", design_hash, None) for name, design_hash in sorted(protocol["baselines"].items())]
    for name in protocol["ablations"]:
        ablated = ABLATIONS[name](design)
        arms.append((f"ablation:{name}", ablated.hash(), ablated))
    return arms


def heldout_report(arms: Sequence[tuple[str, Mapping]]) -> dict:
    """The held-out report as a pure function of the arms' logged eval entries, chosen arm first.

    Every number in a report follows from the "eval" entries as the notebook
    stores them, so anyone holding the notebook can recompute the report and
    compare.  Differences are paired by world: all arms ran on the same worlds.
    """
    chosen = arms[0][1]
    mean, lo, hi = mean_ci(chosen["values"])
    report: dict = {
        "design_hash": chosen["design_hash"], "worlds": len(chosen["scenarios"]), "rounds": chosen["rounds"],
        "value_mean": mean, "value_ci95": [lo, hi], "metrics_mean": chosen["metrics_mean"], "arms": {},
    }
    for name, entry in arms:
        arm_mean, arm_lo, arm_hi = mean_ci(entry["values"])
        diff, diff_lo, diff_hi = mean_ci([a - b for a, b in zip(chosen["values"], entry["values"])])
        report["arms"][name] = {
            "design_hash": entry["design_hash"], "mean": arm_mean, "ci95": [arm_lo, arm_hi],
            "chosen_minus_arm": diff, "chosen_minus_arm_ci95": [diff_lo, diff_hi], "metrics_mean": entry["metrics_mean"],
        }
    return report


def refinement_candidates(chosen: Design, report: Mapping) -> list[tuple[tuple[str, ...], Design]]:
    """Designs worth validating after a held-out round, each with the ablation changes it applies.

    The rule: keep the chosen design; add every ablation whose arm scored
    higher than the chosen arm on the held-out worlds; add every pair of those
    changes applied together.  Candidates that turn out to be the same design
    (same design hash) are kept once, under the simplest recipe.
    """
    arms = report["arms"]
    winners = [
        name for name in ABLATIONS
        if f"ablation:{name}" in arms and arms[f"ablation:{name}"]["mean"] > arms["chosen"]["mean"]
    ]
    recipes: list[tuple[str, ...]] = [(), *((name,) for name in winners), *itertools.combinations(winners, 2)]
    candidates: dict[str, tuple[tuple[str, ...], Design]] = {}
    for recipe in recipes:
        design = chosen
        for name in recipe:
            design = ABLATIONS[name](design)
        candidates.setdefault(design.hash(), (recipe, design))
    return list(candidates.values())


@dataclass
class _Round:
    """One pre-registration and the notebook entries that carry out its protocol."""

    prereg: dict
    beacon: dict | None = None
    worlds: dict | None = None
    evals: list[dict] = field(default_factory=list)  # held-out "eval" entries, in notebook order
    report: dict | None = None
    surplus: list[dict] = field(default_factory=list)  # a second beacon, worlds or report entry


_PROTOCOL_KINDS = ("beacon", "heldout_worlds", "heldout_report")


def _rounds(entries: Sequence[Mapping]) -> tuple[list[_Round], list[dict]]:
    """Group held-out protocol entries under the latest pre-registration that precedes them.

    Returns the rounds, and the protocol entries that precede every
    pre-registration: those belong to no commitment at all.
    """
    rounds: list[_Round] = []
    orphans: list[dict] = []
    for entry in entries:
        kind = entry["kind"]
        heldout_eval = kind == "eval" and entry["body"].get("stage") == "heldout"
        if kind == "preregistration":
            rounds.append(_Round(prereg=entry))
        elif kind in _PROTOCOL_KINDS or heldout_eval:
            if not rounds:
                orphans.append(entry)
            elif heldout_eval:
                rounds[-1].evals.append(entry)
            elif kind == "beacon" and rounds[-1].beacon is None:
                rounds[-1].beacon = entry
            elif kind == "heldout_worlds" and rounds[-1].worlds is None:
                rounds[-1].worlds = entry
            elif kind == "heldout_report" and rounds[-1].report is None:
                rounds[-1].report = entry
            else:
                rounds[-1].surplus.append(entry)
    return rounds, orphans


# Publishes the notebook where its author cannot change it (for instance a git
# commit and push) and may return a receipt saying where.  Raises on failure.
Anchor = Callable[[Notebook], Optional[Mapping]]


class Designer:
    """Carries out the method described at the top of this module, logging every step in one notebook."""

    def __init__(self, designer_id: str, notebook: Notebook, workers: int = 4, seed: int | str = 0, proposer: Any = None):
        self.id = designer_id
        self.notebook = notebook
        self.workers = max(1, workers)
        self.seed = seed
        self.proposer = proposer
        self._cache: dict[tuple, dict] = {}
        self.sims_run = 0
        if not notebook.entries:
            notebook.add("header", {"designer": designer_id, "seed": str(seed), "evaluator_code": code_hash()})
            notebook.add("round", {
                "round": 1, "informed_by": None, "spent_worlds": [],
                "note": "Search and validation use training and validation worlds only; no held-out world exists yet.",
            })

    # ----------------------------------------------------------- experiments
    def _run(self, jobs: Sequence[Mapping]) -> list[dict]:
        todo = [j for j in jobs if (json.dumps(j["design"], sort_keys=True), j["scenario"], j["rounds"]) not in self._cache]
        if todo:
            results = _map(evaluate_design, todo, self.workers)
            self.sims_run += len(todo)
            for j, r in zip(todo, results):
                self._cache[(json.dumps(j["design"], sort_keys=True), j["scenario"], j["rounds"])] = r
        return [self._cache[(json.dumps(j["design"], sort_keys=True), j["scenario"], j["rounds"])] for j in jobs]

    def assess(self, designs: Sequence[Design], scenarios: Sequence[str], rounds: int, stage: str, generation: int = -1) -> list[dict]:
        """Evaluate designs on the same worlds; log one notebook entry per design."""
        jobs = [{"design": d.to_dict(), "scenario": s, "rounds": rounds} for d in designs for s in scenarios]
        results = self._run(jobs)
        out = []
        for i, d in enumerate(designs):
            chunk = results[i * len(scenarios): (i + 1) * len(scenarios)]
            values = [r["value"] for r in chunk]
            entry = {
                "stage": stage, "generation": generation, "design_hash": d.hash(), "design": d.to_dict(),
                "scenarios": list(scenarios), "rounds": rounds, "values": values,
                "mean": sum(values) / len(values), "board_heads": [r["board_head"] for r in chunk],
                "metrics_mean": _mean_metrics(chunk),
            }
            self.notebook.add("eval", entry)
            out.append(entry)
        return out

    # ---------------------------------------------------------------- search
    def initial_population(self, size: int) -> list[Design]:
        rng = make_rng("designer", self.seed, "init")
        base = Design()
        seeds = [
            base,
            replace(base, policy=replace(base.policy, specialty_focus=0.8, niche=1.0, crowd_aversion=0.3)),
            replace(base, init="covering"),
            replace(base, criterion_plan="official"),
        ]
        pop = seeds[:size]
        while len(pop) < size:
            pop.append(mutate_design(rng.choice(seeds), rng, n_moves=3))
        return pop

    def search(
        self, generations: int = 5, population: int = 10,
        rung1: tuple[int, int] = (2, 6), rung2: tuple[int, int] = (4, 10), elites: int = 3,
    ) -> list[dict]:
        """Evolve designs on training worlds.  Returns the finalists' rung-2 entries."""
        rng = make_rng("designer", self.seed, "search")
        train = [f"train:{self.seed}:{k}" for k in range(max(rung1[0], rung2[0]))]
        pop = self.initial_population(population)
        best: dict[str, dict] = {}
        for g in range(generations):
            unique = list({d.hash(): d for d in pop}.values())
            r1 = self.assess(unique, train[: rung1[0]], rung1[1], "rung1", g)
            order = sorted(range(len(unique)), key=lambda i: -r1[i]["mean"])
            survivors = [unique[i] for i in order[: max(elites, len(unique) // 2)]]
            r2 = self.assess(survivors, train[: rung2[0]], rung2[1], "rung2", g)
            for e in r2:
                best[e["design_hash"]] = e
            ranked = sorted(best.values(), key=lambda e: -e["mean"])
            top = [Design.from_dict(e["design"]) for e in ranked[:elites]]
            self.notebook.add("generation", {
                "generation": g, "evaluated": len(unique), "survivors": len(survivors),
                "best_hash": ranked[0]["design_hash"], "best_mean": ranked[0]["mean"],
                "elite_means": [e["mean"] for e in ranked[:elites]], "sims_so_far": self.sims_run,
            })
            proposed: list[Design] = []
            if self.proposer is not None:
                try:
                    proposed = [clamp_design(d) for d in self.proposer.propose(ranked[:5]) if isinstance(d, Design)][:2]
                except Exception as exc:  # noqa: BLE001 - a failing proposer must not stop the search
                    self.notebook.add("note", {"proposer_error": f"{type(exc).__name__}: {exc}"})
            pop = list(top) + proposed
            while len(pop) < population - 1:
                if len(top) >= 2 and rng.random() < 0.3:
                    a, b = rng.sample(top, 2)
                    pop.append(mutate_design(crossover(a, b, rng), rng, n_moves=1))
                else:
                    pop.append(mutate_design(rng.choice(top), rng, n_moves=rng.choice([1, 2, 3])))
            pop.append(mutate_design(Design(), rng, n_moves=5))  # an immigrant, to keep exploring
        return sorted(best.values(), key=lambda e: -e["mean"])[: max(elites, 3)]

    def _select(self, designs: Sequence[Design], worlds: Sequence[str], rounds: int) -> Design:
        """Pick the design with the best mean on ``worlds`` and log the choice; a tie goes to the earlier design."""
        entries = self.assess(designs, worlds, rounds, "validation")
        pick = max(range(len(designs)), key=lambda i: entries[i]["mean"])
        self.notebook.add("selection", {
            "candidates": [{"design_hash": e["design_hash"], "validation_mean": e["mean"]} for e in entries],
            "chosen": entries[pick]["design_hash"],
        })
        return designs[pick]

    def validate(self, finalists: Sequence[Mapping], n_worlds: int = 8, rounds: int = 10) -> Design:
        """Choose among finalists on worlds the search never touched."""
        worlds = [f"validate:{self.seed}:{k}" for k in range(n_worlds)]
        return self._select([Design.from_dict(e["design"]) for e in finalists], worlds, rounds)

    # ------------------------------------------------------------- held out
    def preregister(
        self, design: Design, n_worlds: int = 16, rounds: int = 10, baselines: Mapping[str, Design] | None = None,
        beacon: Beacon | None = None, anchor: Anchor | None = None,
    ) -> dict:
        """Commit to the design and the protocol, then let its held-out worlds come into existence.

        The order of the steps is what makes the commitment worth something:

        1. The "preregistration" entry is written first.  It holds the design,
           the arms and, with a beacon, the name of a round the beacon has not
           produced yet.
        2. ``anchor(notebook)`` publishes the notebook where its author cannot
           change it, and may return a receipt saying where.  If it raises,
           nothing further is written: the worlds must not be derived from a
           pre-registration that is not public.
        3. Only then is the beacon round revealed and logged in a "beacon" entry.
        4. The world seeds follow from the hash of the pre-registration entry
           and that randomness; a "heldout_worlds" entry records them.

        With a beacon and an anchor nobody, the designer included, could have
        known the worlds when the design was fixed, and others can check that.
        Without a beacon the worlds follow from the notebook alone.

        A pre-registration is evaluated before the next one is made: abandoning
        one would be a way to discard an unfavourable draw of worlds.

        Returns the body of the "heldout_worlds" entry.
        """
        history, _ = _rounds(self.notebook.entries)
        if history and history[-1].report is None:
            raise ValueError(f"the pre-registration at entry {history[-1].prereg['index']} has not been evaluated")
        entry = self.notebook.add("preregistration", {
            "design_hash": design.hash(), "design": design.to_dict(), "n_worlds": n_worlds, "rounds": rounds,
            "primary_metric": "mean value over the held-out worlds",
            "baselines": {name: baseline.hash() for name, baseline in (baselines or {}).items()},
            "ablations": sorted(ABLATIONS),
            "beacon": None if beacon is None else dict(beacon.commit()),
            "world_seeds": "heldout:<hash of this entry, 20 hex>:<beacon randomness, 20 hex, if a round is committed>:<k>",
        })
        receipt = None if anchor is None else dict(anchor(self.notebook) or {})
        randomness = None
        if beacon is not None:
            commitment = entry["body"]["beacon"]
            randomness = beacon.reveal(commitment)
            if not is_randomness(randomness):
                raise BeaconError(f"the beacon did not return hex randomness for round {commitment['round']}")
            self.notebook.add("beacon", {"source": commitment["source"], "round": commitment["round"], "randomness": randomness})
        return self.notebook.add("heldout_worlds", {
            "anchor": receipt, "worlds": heldout_world_seeds(entry["hash"], randomness, n_worlds),
        })["body"]

    def heldout(self, design: Design, baselines: Mapping[str, Design] | None = None) -> dict:
        """Evaluate the pre-registered design, its baselines and its ablations on its held-out worlds.

        A held-out number means something only for the design that was
        committed before its worlds existed.  So the design, the arms and the
        worlds are those of the latest pre-registration, read from the
        notebook; a different design or other baselines are refused, and a
        pre-registration is evaluated once.

        The report is computed from the eval entries as the notebook stores
        them, because that is what a verifier recomputes it from.
        """
        history, _ = _rounds(self.notebook.entries)
        if not history or history[-1].worlds is None:
            raise ValueError("pre-register a design before its held-out evaluation")
        current = history[-1]
        protocol = current.prereg["body"]
        if current.evals or current.report is not None:
            raise ValueError(f"the pre-registration at entry {current.prereg['index']} has already been evaluated")
        arms = _arms(design, baselines)
        registered = [(name, design_hash) for name, design_hash, _ in _registered_arms(protocol)]
        if [(name, arm.hash()) for name, arm in arms] != registered:
            raise ValueError(
                f"the design and baselines differ from those pre-registered at entry {current.prereg['index']} "
                f"(design {protocol['design_hash']})"
            )
        first = len(self.notebook.entries)
        self.assess([arm for _, arm in arms], current.worlds["body"]["worlds"], protocol["rounds"], "heldout")
        logged = [entry["body"] for entry in self.notebook.entries[first:]]
        report = heldout_report([(name, body) for (name, _), body in zip(arms, logged)])
        self.notebook.add("heldout_report", report)
        return report

    def refine(
        self, previous_report: Mapping, n_validation_worlds: int = 8, n_worlds: int = 16, rounds: int = 10,
        baselines: Mapping[str, Design] | None = None, beacon: Beacon | None = None, anchor: Anchor | None = None,
    ) -> tuple[Design, dict]:
        """Run one more round that starts from what the previous held-out round showed.

        The candidates are those of ``refinement_candidates``.  They are
        compared on fresh validation worlds, the best is pre-registered with a
        new beacon round, and it is evaluated on new held-out worlds.

        Using the previous held-out arms to pick candidates spends those worlds
        as test data: a design chosen with their help would look better on them
        than it is.  The "round" entry written here says so, and the claim of
        this round rests only on the worlds derived after its pre-registration.

        Returns the design this round pre-registered and its held-out report.
        """
        history, _ = _rounds(self.notebook.entries)
        previous = history[-1] if history else None
        if previous is None or previous.report is None or _differences(previous.report["body"], previous_report):
            raise ValueError("refine starts from the latest held-out report in the notebook")
        number = len(history) + 1
        candidates = refinement_candidates(Design.from_dict(previous.prereg["body"]["design"]), previous_report)
        self.notebook.add("round", {
            "round": number,
            "informed_by": {"preregistration": previous.prereg["index"], "heldout_report": previous.report["index"]},
            "spent_worlds": previous.worlds["body"]["worlds"],
            "candidates": [{"design_hash": design.hash(), "ablations_applied": list(recipe)} for recipe, design in candidates],
            "note": (
                f"The held-out worlds of round {number - 1} informed this round's candidates and are spent as test "
                "data: no later claim may rest on them. This round's claim rests only on worlds derived after its "
                "own pre-registration."
            ),
        })
        worlds = [f"validate{number}:{self.seed}:{k}" for k in range(n_validation_worlds)]
        pick = self._select([design for _, design in candidates], worlds, rounds)
        self.preregister(pick, n_worlds, rounds, baselines, beacon, anchor)
        return pick, self.heldout(pick, baselines)

    # ---------------------------------------------------------- verification
    @staticmethod
    def verify_notebook(
        path: str, sample: int | None = 6, seed: int | str | None = None, workers: int = 4,
        submission: Mapping | None = None, check_beacon: bool = False,
        beacons: Mapping[str, Beacon] | None = None, anchored: Sequence[str] = (),
    ) -> dict:
        """Check a notebook, and the submission that rests on it, the way a rival designer would.

        Nothing the notebook says about itself is taken on trust:

        - the hash chain is intact and the entries verification reads are well formed;
        - each pre-registered design hashes to its stated hash, is the design
          the latest selection chose, and was evaluated;
        - a committed beacon round is revealed by the next entry, and the
          held-out worlds re-derive from the hash of the pre-registration entry
          and that randomness;
        - every held-out eval entry uses exactly those worlds, no earlier entry
          uses any of them, and the held-out entries are the pre-registered
          arms, the pre-registered design first;
        - each held-out report is recomputed from its eval entries;
        - ``submission`` (the parsed submission.json) must state the last
          round's design and the recomputed numbers;
        - every world of each round's chosen design is re-run, and one world
          from each of ``sample`` other eval entries, picked with a seed nobody
          can predict unless ``seed`` fixes it; ``sample=None`` re-runs
          everything.

        ``check_beacon`` also asks the public beacon (``beacons``, by default
        drand) for each committed round and compares the randomness; a beacon
        that cannot be reached is then a failure, reported as "beacon not
        checked".  ``anchored`` lists copies of the notebook that were
        published at pre-registration; the notebook must extend each of them.

        The result names every check with its status and its failures, each
        with the index of the entry concerned.  ``ok`` is true only if no check
        failed.  ``preregistrations`` lists each commitment with its beacon
        round and the anchor receipt the designer recorded, ``rerun_cells``
        what was re-run as ``[entry, world]`` pairs, and passing
        ``sample_seed`` back as ``seed`` repeats the sample.
        """
        return _verify(path, sample, seed, workers, submission, check_beacon, beacons, anchored)


# -------------------------------------------------------------- verification
_CHECKS = {
    "chain": "the notebook is an unbroken hash chain",
    "entries": "eval and pre-registration entries are well formed; an eval entry's design hashes to its "
               "design_hash and its mean is the mean of its values",
    "preregistration": "each pre-registered design hashes to its design_hash, is the design the latest selection "
                       "chose, and was evaluated",
    "beacon": "a committed beacon round is revealed by the entry that follows the pre-registration",
    "beacon_randomness": "the recorded randomness is what the public beacon published for the committed round",
    "anchor": "the notebook extends the copies published at pre-registration",
    "heldout_worlds": "held-out worlds follow from the hash of the pre-registration entry and the beacon randomness",
    "heldout_isolation": "held-out evaluations use exactly the pre-registered worlds, and no earlier entry uses them",
    "heldout_arms": "the held-out entries are the pre-registered arms, the pre-registered design first",
    "heldout_report": "each held-out report recomputes from its eval entries",
    "submission": "the submission states the last round's design and the recomputed held-out numbers",
    "rerun_chosen": "every held-out world of each chosen design reproduces: value, metrics and simulated history",
    "rerun_sample": "one world from each of a random sample of the other eval entries reproduces",
}

# Which check a held-out protocol entry in the wrong place counts against.
_PLACEMENT_CHECK = {
    "eval": "heldout_isolation", "beacon": "beacon", "heldout_worlds": "heldout_worlds",
    "heldout_report": "heldout_report",
}

# Fields verification computes with, by entry kind.
_FIELDS: dict[str, dict[str, Any]] = {
    "eval": {
        "stage": str, "design_hash": str, "design": dict, "scenarios": list, "rounds": int, "values": list,
        "mean": (int, float), "board_heads": list, "metrics_mean": dict,
    },
    "preregistration": {
        "design_hash": str, "design": dict, "n_worlds": int, "rounds": int, "baselines": dict, "ablations": list,
    },
}

_SUBMISSION_FIELDS = ("design", "design_hash", "claimed_value", "claimed_value_ci95", "notebook_head")


class _Verdict:
    """What verification found, check by check, so that a rejection names the rule that was broken."""

    def __init__(self) -> None:
        self.checks = {name: {"rule": rule, "status": "not run", "failures": []} for name, rule in _CHECKS.items()}
        self.warnings: list[str] = []

    def ran(self, check: str) -> None:
        if self.checks[check]["status"] == "not run":
            self.checks[check]["status"] = "passed"

    def fail(self, check: str, reason: str, entry: int | None = None) -> None:
        self.checks[check]["status"] = "failed"
        self.checks[check]["failures"].append({"entry": entry, "reason": reason})

    def result(self, summary: Mapping) -> dict:
        """``checks`` maps each check to its rule, status and failures; ``problems`` lists the failures as text."""
        problems = [
            f"{name}: " + (f"entry {failure['entry']}: " if failure["entry"] is not None else "") + failure["reason"]
            for name, check in self.checks.items() for failure in check["failures"]
        ]
        return {**summary, "ok": not problems, "checks": self.checks, "problems": problems, "warnings": self.warnings}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_value(value: Any) -> bool:
    return _is_number(value) and abs(value) <= _MAX_VALUE


def _close(recorded: float, expected: float) -> bool:
    """Whether a recorded number agrees with its recomputation; one too large to compare does not."""
    try:
        return abs(recorded - expected) <= _TOLERANCE
    except OverflowError:
        return False


def _differences(recorded: Any, expected: Any, where: str = "") -> list[str]:
    """Where ``recorded`` departs from ``expected``; numbers may differ by the storage rounding."""
    if isinstance(recorded, Mapping) and isinstance(expected, Mapping):
        found = []
        for key in sorted(set(recorded) | set(expected)):
            inner = f"{where}.{key}" if where else str(key)
            if key not in recorded:
                found.append(f"{inner} is missing")
            elif key not in expected:
                found.append(f"{inner} is not among the recomputed fields")
            else:
                found += _differences(recorded[key], expected[key], inner)
        return found
    if isinstance(recorded, list) and isinstance(expected, (list, tuple)) and len(recorded) == len(expected):
        return [d for i, pair in enumerate(zip(recorded, expected)) for d in _differences(*pair, f"{where}[{i}]")]
    if _is_number(recorded) and _is_number(expected):
        shown = f"{expected:.6f}" if isinstance(expected, float) else repr(expected)
        return [] if _close(recorded, expected) else [f"{where} is {recorded!r}, recomputed {shown}"]
    return [] if recorded == expected else [f"{where} is {recorded!r}, expected {expected!r}"]


def _briefly(differences: Sequence[str], limit: int = 3) -> str:
    more = len(differences) - limit
    return "; ".join(differences[:limit]) + (f" (and {more} more)" if more > 0 else "")


# What reading a design from data that does not describe one, or hashing one
# with a setting that is not a finite number, can raise.
_NOT_A_DESIGN = (AttributeError, TypeError, ValueError, OverflowError)


def _design(described: Any) -> Design | None:
    """The design a dict describes, or None if it does not describe one."""
    try:
        return Design.from_dict(described)
    except _NOT_A_DESIGN:
        return None


def _design_hash(described: Any) -> str | None:
    """Hash of the design a dict describes, or None if it describes none that can be hashed."""
    try:
        return Design.from_dict(described).hash()
    except _NOT_A_DESIGN:
        return None


def _flaw(entry: Mapping) -> str | None:
    """Why verification cannot compute with an entry of a kind it reads, or None if it can."""
    kind, body = entry["kind"], entry["body"]
    for name, expected in _FIELDS.get(kind, {}).items():
        if not isinstance(body.get(name), expected) or isinstance(body.get(name), bool):
            return f"field {name!r} is missing or of the wrong type"
    if kind in _FIELDS and body["rounds"] < 1:
        return "rounds must be positive"
    if kind == "eval":
        cells = (body["scenarios"], body["values"], body["board_heads"])
        if not body["scenarios"] or len({len(cell) for cell in cells}) != 1:
            return "scenarios, values and board_heads must be equally long and not empty"
        if not all(isinstance(s, str) for s in body["scenarios"] + body["board_heads"]):
            return "scenarios and board_heads must be strings"
        if not all(_is_value(v) for v in [body["mean"], *body["values"]]):
            return "mean and values must be numbers a world can produce"
    if kind == "preregistration":
        commitment = body.get("beacon")
        if commitment is not None and not (
            isinstance(commitment, dict) and isinstance(commitment.get("source"), str)
            and isinstance(commitment.get("round"), int) and not isinstance(commitment.get("round"), bool)
        ):
            return "a beacon commitment must name a source and a round"
        if body["n_worlds"] < 1:
            return "n_worlds must be positive"
        if not all(isinstance(h, str) for h in body["baselines"].values()) or not all(isinstance(a, str) for a in body["ablations"]):
            return "baselines must map names to design hashes and ablations must be names"
    return None


def _readable(nb: Notebook, verdict: _Verdict) -> list[dict]:
    """The entries verification can compute with; the others are reported and left out.

    An eval entry must also agree with itself: its design hashes to its
    ``design_hash`` (so the hash names the design that a re-run simulates) and
    its ``mean`` is the mean of its ``values``.  The stored mean was rounded
    after averaging and the stored values before, hence twice the tolerance.
    """
    verdict.ran("entries")
    readable = []
    for entry in nb.entries:
        flaw = _flaw(entry)
        if flaw is not None:
            verdict.fail("entries", flaw, entry["index"])
            continue
        readable.append(entry)
        if entry["kind"] == "eval":
            body = entry["body"]
            if _design_hash(body["design"]) != body["design_hash"]:
                verdict.fail("entries", f"the design does not hash to the stated {body['design_hash']}", entry["index"])
            if abs(body["mean"] - sum(body["values"]) / len(body["values"])) > 2 * _TOLERANCE:
                verdict.fail("entries", "the mean is not the mean of the values", entry["index"])
    return readable


def _check_commitment(rnd: _Round, last: bool, selections: Sequence[Mapping], verdict: _Verdict) -> None:
    """The pre-registered design is what it says, is what validation chose, and was not abandoned.

    Selecting one design and pre-registering another, or pre-registering
    several times and evaluating only a lucky draw of worlds, would both let
    held-out worlds pick the design.  Only the last pre-registration may still
    be waiting for its evaluation.
    """
    at, protocol = rnd.prereg["index"], rnd.prereg["body"]
    verdict.ran("preregistration")
    if _design_hash(protocol["design"]) != protocol["design_hash"]:
        verdict.fail("preregistration", f"the design does not hash to the stated {protocol['design_hash']}", at)
    earlier = [s for s in selections if s["index"] < at]
    if not earlier:
        verdict.fail("preregistration", "no selection entry precedes it", at)
    elif earlier[-1]["body"].get("chosen") != protocol["design_hash"]:
        verdict.fail(
            "preregistration",
            f"commits to design {protocol['design_hash']}, but the latest selection (entry {earlier[-1]['index']}) "
            f"chose {earlier[-1]['body'].get('chosen')}", at,
        )
    if rnd.report is None and not last:
        verdict.fail(
            "preregistration",
            "was abandoned without a held-out report, which is how an unfavourable draw of worlds would be discarded", at,
        )
    for entry in rnd.surplus:
        verdict.fail(
            _PLACEMENT_CHECK[entry["kind"]], f"is a second {entry['kind']} entry for the pre-registration at entry {at}",
            entry["index"],
        )


def _check_reveal(number: int, rnd: _Round, verdict: _Verdict) -> tuple[bool, str | None]:
    """A committed beacon round must be revealed by the entry that follows the pre-registration.

    The commitment was fixed with the pre-registration entry; the reveal must
    be for that source and round, or the designer could have looked at several
    rounds and kept one.  A pre-registration without a commitment is allowed,
    with a warning: nothing but the notebook decided its worlds.

    Returns whether the held-out worlds can be re-derived, and the randomness
    to derive them with (None when no round was committed).
    """
    at, commitment = rnd.prereg["index"], rnd.prereg["body"].get("beacon")
    if commitment is None:
        if rnd.beacon is not None:
            verdict.fail("beacon", f"reveals randomness, but the pre-registration at entry {at} committed to no beacon round",
                         rnd.beacon["index"])
        verdict.warnings.append(
            f"round {number}: no randomness beacon was used, so the held-out worlds depend only on the notebook head, "
            "which the designer controls: the notebook could have been re-rolled until the worlds suited the design"
        )
        return True, None
    source, committed = commitment["source"], commitment["round"]
    verdict.ran("beacon")
    if rnd.beacon is None:
        verdict.fail("beacon", f"commits to {source} round {committed}, but no beacon entry reveals it", at)
        return False, None
    where, revealed = rnd.beacon["index"], rnd.beacon["body"]
    if where != at + 1:
        verdict.fail("beacon", f"must directly follow the pre-registration at entry {at}", where)
    if (revealed.get("source"), revealed.get("round")) != (source, committed):
        verdict.fail(
            "beacon",
            f"reveals {revealed.get('source')} round {revealed.get('round')}, but entry {at} committed to "
            f"{source} round {committed}", where,
        )
    if not is_randomness(revealed.get("randomness")):
        verdict.fail("beacon", "does not record hex randomness", where)
        return False, None
    return True, revealed["randomness"]


def _check_public_randomness(
    number: int, rnd: _Round, randomness: str, verdict: _Verdict, beacons: Mapping[str, Beacon], check_beacon: bool,
) -> None:
    """Compare the randomness a notebook records with what the beacon itself published.

    Nothing in a notebook can vouch for the randomness it records: its author
    could have written down a value that gives agreeable worlds.  The beacon is
    looked up by source name among those the verifier accepts, never by an
    address from the notebook.  Asking it needs the network, so it happens only
    on request; then a beacon that cannot be asked leaves the randomness
    unchecked, and that is a failure.
    """
    commitment, where = rnd.prereg["body"]["beacon"], rnd.beacon["index"]
    source, committed = commitment["source"], commitment["round"]
    if source not in beacons:
        verdict.warnings.append(
            f"round {number}: {source!r} is not a public randomness beacon this verifier knows, so the held-out "
            "worlds are only as unpredictable as that source"
        )
    if not check_beacon:
        verdict.warnings.append(
            f"round {number}: beacon not checked: the randomness recorded for {source} round {committed} was not "
            "compared with the beacon itself (--check-beacon)"
        )
        return
    verdict.ran("beacon_randomness")
    if source not in beacons:
        verdict.fail("beacon_randomness", f"beacon not checked: {source!r} is not a beacon this verifier can ask", where)
        return
    try:
        published = beacons[source].reveal(commitment)
    except BeaconError as exc:
        verdict.fail("beacon_randomness", f"beacon not checked: {exc}", where)
        return
    if published != randomness:
        verdict.fail(
            "beacon_randomness",
            f"records randomness {randomness[:16]}... for {source} round {committed}, but the beacon published "
            f"{published[:16]}...", where,
        )


def _check_worlds(rnd: _Round, derivable: bool, randomness: str | None, verdict: _Verdict) -> list[str] | None:
    """Re-derive the round's held-out worlds from the chain and compare them with the recorded ones.

    The worlds are whatever follows from the hash of the pre-registration
    entry and the revealed randomness; the "heldout_worlds" entry only records
    them, directly after the entries they follow from.  Returns the worlds
    every held-out entry of the round must use, or None if the notebook does
    not establish them.
    """
    at, n_worlds = rnd.prereg["index"], rnd.prereg["body"]["n_worlds"]
    if rnd.worlds is None:
        if rnd.evals or rnd.report is not None:
            verdict.fail("heldout_worlds", "no heldout_worlds entry records the worlds of this pre-registration", at)
        return None
    verdict.ran("heldout_worlds")
    where, recorded = rnd.worlds["index"], rnd.worlds["body"].get("worlds")
    if where != (rnd.beacon or rnd.prereg)["index"] + 1:
        verdict.fail("heldout_worlds", f"must directly follow the pre-registration at entry {at} and its beacon entry", where)
    if not derivable:
        verdict.fail("heldout_worlds", "cannot be re-derived without the beacon randomness", where)
        return None
    if not isinstance(recorded, list) or len(recorded) != n_worlds:
        verdict.fail("heldout_worlds", f"does not list the {n_worlds} worlds pre-registered at entry {at}", where)
        return None
    expected = heldout_world_seeds(rnd.prereg["hash"], randomness, n_worlds)
    if recorded != expected:
        basis = f"the hash of entry {at}" + (" and the beacon randomness" if randomness is not None else "")
        verdict.fail("heldout_worlds", f"does not list the worlds that follow from {basis}", where)
    return expected


def _check_arms(rnd: _Round, verdict: _Verdict) -> list[str] | None:
    """The round's held-out entries must be the pre-registered arms in order, the pre-registered design first.

    Evaluating a different design on the committed worlds, or leaving an arm
    out, fails here.  Where the protocol fixes a design in full (the chosen
    design and its ablations) the logged design itself is compared and not
    only its hash: a design hash has 64 bits, few enough that a second design
    with the same hash can be manufactured.

    Returns the arm names in the order of the entries, or None if the entries
    cannot be matched to the arms one to one.
    """
    at, protocol = rnd.prereg["index"], rnd.prereg["body"]
    verdict.ran("heldout_arms")
    try:
        registered = _registered_arms(protocol)
    except KeyError as exc:
        verdict.fail("heldout_arms", f"pre-registers ablation {exc.args[0]!r}, which this code does not define", at)
        return None
    except _NOT_A_DESIGN:
        verdict.fail("heldout_arms", "cannot be established: the pre-registered design is not a design", at)
        return None
    for (name, design_hash, design), entry in zip(registered, rnd.evals):
        logged = entry["body"]
        if logged["design_hash"] != design_hash:
            found = f"is design {logged['design_hash']}"
        elif design is not None and _design(logged["design"]) != design:
            found = "carries that hash but is a different design"
        else:
            continue
        verdict.fail(
            "heldout_arms",
            f"arm {name!r} is pre-registered at entry {at} as design {design_hash}, but this held-out entry {found}",
            entry["index"],
        )
    if len(rnd.evals) != len(registered):
        verdict.fail("heldout_arms", f"pre-registers {len(registered)} arms, but {len(rnd.evals)} held-out evaluations follow", at)
        return None
    return [name for name, _, _ in registered]


def _check_heldout(rnd: _Round, worlds: Sequence[str] | None, evals: Sequence[Mapping], verdict: _Verdict) -> dict | None:
    """Held-out entries use the committed worlds, are the committed arms, and add up to the report.

    - Isolation: every held-out eval entry of the round uses exactly its
      worlds for the pre-registered number of rounds, and no eval entry before
      the pre-registration uses any of them.
    - Arms: see ``_check_arms``.
    - Report: recomputed from those entries and compared field by field.

    Returns the recomputed report, or None if the notebook holds no report
    that can be recomputed.
    """
    at, protocol = rnd.prereg["index"], rnd.prereg["body"]
    if worlds is not None:
        verdict.ran("heldout_isolation")
        for entry in rnd.evals:
            if entry["body"]["scenarios"] != worlds or entry["body"]["rounds"] != protocol["rounds"]:
                verdict.fail("heldout_isolation", f"does not use exactly the worlds and rounds pre-registered at entry {at}",
                             entry["index"])
        for entry in evals:
            if entry["index"] < at and set(entry["body"]["scenarios"]) & set(worlds):
                verdict.fail("heldout_isolation", f"uses held-out worlds of the pre-registration at entry {at} before it was written",
                             entry["index"])
    names = _check_arms(rnd, verdict) if rnd.evals else None
    if rnd.report is None:
        return None
    verdict.ran("heldout_report")
    if names is None:
        verdict.fail("heldout_report", "cannot be recomputed: the held-out entries are not the pre-registered arms",
                     rnd.report["index"])
        return None
    recomputed = heldout_report([(name, entry["body"]) for name, entry in zip(names, rnd.evals)])
    differences = _differences(rnd.report["body"], recomputed)
    if differences:
        verdict.fail("heldout_report", "does not follow from the held-out eval entries: " + _briefly(differences),
                     rnd.report["index"])
    return recomputed


def _check_anchor(nb: Notebook, rounds: Sequence[_Round], anchored: Sequence[str], verdict: _Verdict) -> None:
    """Compare the notebook with copies of it that were published at pre-registration.

    A hash chain cannot show that a notebook was not rewritten as a whole, for
    instance to drop an entry, or to re-roll the held-out worlds once the
    beacon randomness was known.  A copy published before then can.  Each copy
    must be the notebook as it stood right after a pre-registration entry, and
    the last pre-registration, on which the claim rests, must be among them.
    Whether a copy was public early enough can only be established where it
    was published.
    """
    if not anchored:
        for number, rnd in enumerate(rounds, start=1):
            if rnd.worlds is None:
                continue
            at, receipt = rnd.prereg["index"], rnd.worlds["body"].get("anchor")
            if receipt is None:
                verdict.warnings.append(
                    f"round {number}: the pre-registration at entry {at} was not published anywhere, so nothing "
                    "outside the notebook shows that it was not rewritten after its worlds were known"
                )
            else:
                timing = "confirm it was public before the beacon round was produced, " if rnd.beacon is not None else ""
                verdict.warnings.append(
                    f"round {number}: anchor not checked: the designer reports publishing the pre-registration at "
                    f"entry {at} ({json.dumps(receipt, sort_keys=True)}); obtain that copy, {timing}and pass it "
                    "with --anchored"
                )
        return
    verdict.ran("anchor")
    covered = set()
    for path in anchored:
        try:
            copy = Notebook(path)
        except (OSError, ValueError) as exc:
            verdict.fail("anchor", f"{path}: {exc}")
            continue
        if not copy.entries or copy.entries[-1]["kind"] != "preregistration":
            verdict.fail("anchor", f"{path} does not end at a pre-registration entry")
        elif not nb.extends(copy):
            verdict.fail("anchor", f"the notebook does not extend {path}: it was rewritten after that copy was published",
                         len(copy.entries))
        else:
            covered.add(len(copy.entries))
    if rounds and rounds[-1].prereg["index"] not in covered:
        verdict.fail("anchor", "no published copy ends at the last pre-registration", rounds[-1].prereg["index"])


def _check_submission(
    submission: Mapping, nb: Notebook, rounds: Sequence[_Round], reports: Sequence[Mapping | None], verdict: _Verdict,
) -> None:
    """The submission must claim what the notebook supports, for the last pre-registered round.

    Its design must be the one the last pre-registration commits to, and its
    value, interval and arm comparisons are compared with the report
    *recomputed* from that round's eval entries, not with what the notebook's
    report entry states.  Earlier rounds informed later ones, so only the last
    round's worlds are unspent test data and only its numbers may be the
    headline; a per-round record, if given, must match every round.
    """
    verdict.ran("submission")
    missing = [name for name in _SUBMISSION_FIELDS if name not in submission]
    if missing:
        verdict.fail("submission", "lacks " + ", ".join(missing))
    if submission.get("notebook_head") != nb.head:
        verdict.fail("submission", "names a notebook head other than this notebook's: the notebook was extended or "
                                   "replaced after the submission was written")
    if "design" in submission and _design_hash(submission["design"]) != submission.get("design_hash"):
        verdict.fail("submission", "the submitted design does not hash to the submitted design_hash")
    if not rounds:
        verdict.fail("submission", "the notebook holds no pre-registration to support a claim")
        return
    last = rounds[-1].prereg
    if submission.get("design_hash") != last["body"]["design_hash"]:
        verdict.fail(
            "submission",
            f"submits design {submission.get('design_hash')}, but the last pre-registration (entry {last['index']}) "
            f"commits to design {last['body']['design_hash']}",
        )
    elif "design" in submission and _design(submission["design"]) != _design(last["body"]["design"]):
        verdict.fail("submission", f"submits a design that carries the hash of the one pre-registered at entry "
                                   f"{last['index']} but is a different design")
    if reports[-1] is None:
        verdict.fail("submission", f"the pre-registration at entry {last['index']} has no held-out report that can be recomputed")
    else:
        claims = {
            "claimed_value": reports[-1]["value_mean"], "claimed_value_ci95": reports[-1]["value_ci95"],
            "heldout_worlds": reports[-1]["worlds"], "rounds": reports[-1]["rounds"],
            "heldout_metrics": reports[-1]["metrics_mean"], "arms": reports[-1]["arms"],
        }
        stated = {name: submission[name] for name in claims if name in submission}
        differences = _differences(stated, {name: claims[name] for name in stated})
        if differences:
            verdict.fail("submission", "does not state the recomputed held-out numbers: " + _briefly(differences))
    records = submission.get("refinement_rounds")
    if records is None:
        return
    if not isinstance(records, list) or len(records) != len(rounds):
        verdict.fail("submission", f"refinement_rounds must record each of the {len(rounds)} pre-registered rounds")
        return
    for number, (record, rnd, report) in enumerate(zip(records, rounds, reports), start=1):
        if report is None:
            verdict.fail("submission", f"round {number} has no held-out report that can be recomputed")
            continue
        expected = {
            "round": number, "design_hash": rnd.prereg["body"]["design_hash"],
            "claimed_value": report["value_mean"], "claimed_value_ci95": report["value_ci95"],
        }
        stated = {name: record[name] for name in expected if name in record} if isinstance(record, dict) else record
        differences = _differences(stated, expected, f"refinement_rounds[{number - 1}]")
        if differences:
            verdict.fail("submission", _briefly(differences))


def _rerun(job: Mapping) -> dict:
    """Run a recorded experiment again; one that cannot run at all is reported, not raised."""
    try:
        return evaluate_design(job)
    except Exception as exc:  # noqa: BLE001 - whatever stops a recorded experiment from running fails verification
        return {"error": f"{type(exc).__name__}: {exc}"}


def _check_reruns(
    rounds: Sequence[_Round], evals: Sequence[Mapping], sample: int | None, seed: str, workers: int, verdict: _Verdict,
) -> tuple[int, list[list[int]]]:
    """Re-run recorded experiments and compare values and simulated histories.

    Every world of each round's chosen held-out entry is re-run: the claimed
    value is their mean, so one forged cell moves it, and a sample would miss
    it most of the time.  The entry's metrics are recomputed from the same
    runs.  From the remaining eval entries a random sample is drawn and one
    random world of each is re-run.  The sample follows from ``seed``, which
    is unpredictable unless the caller fixed it, so a forger cannot know which
    cells will be looked at.  ``sample=None`` re-runs every world of every
    entry.  Returns the number of simulations run and the cells compared, each
    as ``[entry index, world index]``.
    """
    def job(body: Mapping, k: int) -> dict:
        return {"design": body["design"], "scenario": body["scenarios"][k], "rounds": body["rounds"]}

    runnable = [entry for entry in evals if _design_hash(entry["body"]["design"]) is not None]
    chosen = {rnd.evals[0]["index"] for rnd in rounds if rnd.evals}
    others = [entry for entry in runnable if entry["index"] not in chosen]
    cells = [
        ("rerun_chosen", entry, k)
        for entry in runnable if entry["index"] in chosen for k in range(len(entry["body"]["scenarios"]))
    ]
    if sample is None:
        cells += [("rerun_sample", entry, k) for entry in others for k in range(len(entry["body"]["scenarios"]))]
    else:
        rng = make_rng("verify", seed)
        for entry in rng.sample(others, min(max(sample, 0), len(others))):
            cells.append(("rerun_sample", entry, rng.randrange(len(entry["body"]["scenarios"]))))

    jobs: dict[str, dict] = {}
    for _, entry, k in cells:
        jobs.setdefault(canonical_json(job(entry["body"], k)), job(entry["body"], k))
    results = dict(zip(jobs, _map(_rerun, list(jobs.values()), workers)))

    reproduced: dict[int, list[dict]] = {}
    for check, entry, k in cells:
        verdict.ran(check)
        body, result = entry["body"], results[canonical_json(job(entry["body"], k))]
        if "error" in result:
            verdict.fail(check, f"world {k} could not be re-run: {result['error']}", entry["index"])
        elif not _close(body["values"][k], result["value"]):
            verdict.fail(check, f"world {k}: recorded value {body['values'][k]:.6f}, reproduced {result['value']:.6f}",
                         entry["index"])
        elif result["board_head"] != body["board_heads"][k]:
            verdict.fail(check, f"world {k}: the value matches but the simulated history differs", entry["index"])
        elif check == "rerun_chosen":
            reproduced.setdefault(entry["index"], []).append(result)
    for entry in runnable:
        runs = reproduced.get(entry["index"], [])
        if len(runs) == len(entry["body"]["scenarios"]):
            differences = _differences(entry["body"]["metrics_mean"], _mean_metrics(runs))
            if differences:
                verdict.fail("rerun_chosen", "metrics_mean does not follow from the re-run worlds: " + _briefly(differences),
                             entry["index"])
    return len(jobs), [[entry["index"], k] for _, entry, k in cells]


def _verify(
    path: str, sample: int | None, seed: int | str | None, workers: int, submission: Mapping | None,
    check_beacon: bool, beacons: Mapping[str, Beacon] | None, anchored: Sequence[str],
) -> dict:
    """Run every check of ``Designer.verify_notebook`` and assemble the result."""
    verdict = _Verdict()
    seed = os.urandom(16).hex() if seed is None else str(seed)
    summary = {
        "entries": 0, "chain_ok": False, "preregistrations": [], "rerun": 0, "rerun_cells": [], "sample_seed": seed,
        "evaluator_code_recorded": None, "evaluator_code_now": code_hash(),
    }
    try:
        if not os.path.exists(path):
            raise ValueError(f"there is no notebook at {path}")
        nb = Notebook(path)
    except (OSError, ValueError) as exc:
        verdict.fail("chain", str(exc))
        return verdict.result(summary)
    verdict.ran("chain")
    entries = _readable(nb, verdict)
    evals = [entry for entry in entries if entry["kind"] == "eval"]
    selections = [entry for entry in entries if entry["kind"] == "selection"]
    rounds, orphans = _rounds(entries)
    for entry in orphans:
        verdict.fail(_PLACEMENT_CHECK[entry["kind"]], "no well-formed pre-registration precedes it", entry["index"])
    beacons = public_beacons() if beacons is None else beacons
    reports = []
    for number, rnd in enumerate(rounds, start=1):
        _check_commitment(rnd, number == len(rounds), selections, verdict)
        derivable, randomness = _check_reveal(number, rnd, verdict)
        if randomness is not None:
            _check_public_randomness(number, rnd, randomness, verdict, beacons, check_beacon)
        worlds = _check_worlds(rnd, derivable, randomness, verdict)
        reports.append(_check_heldout(rnd, worlds, evals, verdict))
    _check_anchor(nb, rounds, anchored, verdict)
    if submission is not None:
        _check_submission(submission, nb, rounds, reports, verdict)
    recorded_code = nb.entries[0]["body"].get("evaluator_code") if nb.entries else None
    if recorded_code != summary["evaluator_code_now"]:
        verdict.warnings.append(
            "the grading code differs from the version this notebook was produced with, so re-runs may not reproduce"
        )
    summary.update(entries=len(nb.entries), chain_ok=True, evaluator_code_recorded=recorded_code)
    summary["preregistrations"] = [
        {
            "entry": rnd.prereg["index"], "design_hash": rnd.prereg["body"]["design_hash"],
            "beacon": rnd.prereg["body"].get("beacon"),
            "anchor": rnd.worlds["body"].get("anchor") if rnd.worlds is not None else None,
        }
        for rnd in rounds
    ]
    summary["rerun"], summary["rerun_cells"] = _check_reruns(rounds, evals, sample, seed, workers, verdict)
    return verdict.result(summary)
