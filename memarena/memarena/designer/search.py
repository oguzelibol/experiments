"""The designer: builds societies, tests them, keeps the best, shows its work.

Method, in the order a sceptical reader would want it:

1. **Search** on *training* worlds.  Designs are compared on the same worlds
   (common random numbers).  Cheap short runs screen a population; survivors get
   longer runs on more worlds; the best seed the next generation through
   mutation and crossover.
2. **Validation** on fresh worlds picks one design among the finalists, so the
   pick is not simply the luckiest training score.
3. **Pre-registration.**  The chosen design is written to the notebook *before*
   any held-out world is generated.  Held-out seeds are derived from the
   notebook's head hash at that moment, so they cannot have been tried, or
   chosen, in advance.
4. **Held-out evaluation** gives the claimed value with a confidence interval,
   next to baselines and ablations run on the same worlds.
5. **Everything is in the notebook**, hash-chained, including what did not
   work.  ``verify_notebook`` re-runs recorded experiments and checks that the
   numbers and the simulated histories (board head hashes) are reproduced.

The designer never sees a world's outside criterion: it only receives the value
a design achieved there.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Mapping, Sequence

from ..actors.unit import UnitPolicy
from ..evalcore.module import code_hash
from ..sim.engine import Simulation
from ..util import make_rng
from .design import Design, crossover, mutate_design, sample_scenario
from .notebook import Notebook

__all__ = ["evaluate_design", "Designer", "mean_ci", "ABLATIONS"]

_METRIC_KEYS = (
    "outside_score", "estimate_report", "estimate_error", "estimate_loss", "champion_regret",
    "mean_true_first", "mean_true_last", "best_single_true", "esteem_validity",
    "reputation_validity", "exchanges", "mean_peer_loss", "adoptions", "audit_fails",
)


def evaluate_design(job: Mapping) -> dict:
    """Run one design in one world.  A pure function of ``job``; safe to run in a worker."""
    design = Design.from_dict(job["design"])
    scenario = sample_scenario(job["scenario"], rounds=int(job["rounds"]))
    specs = design.build(seed=job["scenario"], rules=scenario.rules)
    result = Simulation(scenario, specs, estimate_shade=design.shade).run()
    return {
        "value": result.value,
        "metrics": {k: result.metrics.get(k) for k in _METRIC_KEYS},
        "board_head": result.board_head[1],
    }


_T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 12: 2.20,
        16: 2.13, 20: 2.09, 24: 2.07, 32: 2.04}


def mean_ci(xs: Sequence[float]) -> tuple[float, float, float]:
    """Mean and 95% t-interval."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(xs) / n
    if n == 1:
        return m, m, m
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t = _T95.get(n) or next((v for k, v in sorted(_T95.items()) if k >= n), 1.96)
    half = t * sd / math.sqrt(n)
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
    # strive only on own specialties under own standard, ignoring esteemed peers' standards
    "peer_standards_ignored": lambda d: replace(d, policy=replace(d.policy, specialty_focus=1.0)),
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


class Designer:
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

    # ----------------------------------------------------------- experiments
    def _run(self, jobs: Sequence[Mapping]) -> list[dict]:
        todo = [j for j in jobs if (json.dumps(j["design"], sort_keys=True), j["scenario"], j["rounds"]) not in self._cache]
        if todo:
            if self.workers > 1 and len(todo) > 1:
                with ProcessPoolExecutor(max_workers=self.workers) as pool:
                    results = list(pool.map(evaluate_design, todo))
            else:
                results = [evaluate_design(j) for j in todo]
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
                    proposed = [d for d in self.proposer.propose(ranked[:5]) if isinstance(d, Design)][:2]
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

    def validate(self, finalists: Sequence[Mapping], n_worlds: int = 8, rounds: int = 10) -> Design:
        """Choose among finalists on worlds the search never touched."""
        worlds = [f"validate:{self.seed}:{k}" for k in range(n_worlds)]
        designs = [Design.from_dict(e["design"]) for e in finalists]
        entries = self.assess(designs, worlds, rounds, "validation")
        pick = max(range(len(designs)), key=lambda i: entries[i]["mean"])
        self.notebook.add("selection", {
            "candidates": [{"design_hash": e["design_hash"], "validation_mean": e["mean"]} for e in entries],
            "chosen": entries[pick]["design_hash"],
        })
        return designs[pick]

    # ------------------------------------------------------------- held out
    def preregister(self, design: Design, n_worlds: int = 16, rounds: int = 10, baselines: Mapping[str, Design] | None = None) -> dict:
        """Commit to the design and the protocol before any held-out world exists."""
        head = self.notebook.head
        worlds = [f"heldout:{head[:20]}:{k}" for k in range(n_worlds)]
        body = {
            "design_hash": design.hash(), "design": design.to_dict(), "worlds": worlds, "rounds": rounds,
            "derived_from_head": head, "primary_metric": "mean value over the held-out worlds",
            "baselines": {k: v.hash() for k, v in (baselines or {}).items()},
            "ablations": sorted(ABLATIONS),
        }
        self.notebook.add("preregistration", body)
        return body

    def heldout(self, design: Design, prereg: Mapping, baselines: Mapping[str, Design] | None = None) -> dict:
        worlds, rounds = list(prereg["worlds"]), int(prereg["rounds"])
        arms: dict[str, Design] = {"chosen": design}
        arms.update({f"baseline:{k}": v for k, v in (baselines or {}).items()})
        arms.update({f"ablation:{k}": fn(design) for k, fn in ABLATIONS.items()})
        names = list(arms)
        entries = self.assess([arms[n] for n in names], worlds, rounds, "heldout")
        chosen = entries[0]
        mean, lo, hi = mean_ci(chosen["values"])
        report = {
            "design_hash": design.hash(), "worlds": len(worlds), "rounds": rounds,
            "value_mean": mean, "value_ci95": [lo, hi], "metrics_mean": chosen["metrics_mean"], "arms": {},
        }
        for name, e in zip(names, entries):
            diffs = [a - b for a, b in zip(chosen["values"], e["values"])]
            dm, dlo, dhi = mean_ci(diffs)
            report["arms"][name] = {
                "design_hash": e["design_hash"], "mean": e["mean"], "ci95": list(mean_ci(e["values"])[1:]),
                "chosen_minus_arm": dm, "chosen_minus_arm_ci95": [dlo, dhi], "metrics_mean": e["metrics_mean"],
            }
        self.notebook.add("heldout_report", report)
        return report

    # ---------------------------------------------------------- verification
    @staticmethod
    def verify_notebook(path: str, sample: int = 6, seed: int | str = 0, workers: int = 4) -> dict:
        """Re-check a notebook: chain integrity, pre-registration order, and re-run experiments."""
        nb = Notebook(path)  # raises if the chain is broken
        problems: list[str] = []
        prereg = nb.of_kind("preregistration")
        heldout_evals = [e for e in nb.of_kind("eval") if e["body"]["stage"] == "heldout"]
        if prereg:
            p = prereg[-1]
            prev_head = nb.entries[p["index"] - 2]["hash"] if p["index"] > 1 else None
            if p["body"]["derived_from_head"] != prev_head:
                problems.append("held-out worlds were not derived from the notebook head before pre-registration")
            expected = [f"heldout:{p['body']['derived_from_head'][:20]}:{k}" for k in range(len(p["body"]["worlds"]))]
            if expected != p["body"]["worlds"]:
                problems.append("held-out world seeds do not follow the pre-registered derivation")
            for e in heldout_evals:
                if e["index"] < p["index"]:
                    problems.append(f"entry {e['index']} ran a held-out evaluation before pre-registration")
                if e["body"]["scenarios"] != p["body"]["worlds"]:
                    problems.append(f"entry {e['index']} used worlds other than the pre-registered ones")
            for e in nb.of_kind("eval"):
                if e["index"] < p["index"] and any(str(s).startswith("heldout:") for s in e["body"]["scenarios"]):
                    problems.append(f"entry {e['index']} touched held-out worlds during the search")
        evals = nb.of_kind("eval")
        rng = make_rng("verify", seed)
        picks = rng.sample(evals, min(sample, len(evals)))
        chosen_heldout = [e for e in heldout_evals[:1]]
        jobs, owners = [], []
        for e in picks + chosen_heldout:
            b = e["body"]
            k = rng.randrange(len(b["scenarios"]))
            jobs.append({"design": b["design"], "scenario": b["scenarios"][k], "rounds": b["rounds"]})
            owners.append((e["index"], k, b["values"][k], b["board_heads"][k]))
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(evaluate_design, jobs))
        else:
            results = [evaluate_design(j) for j in jobs]
        for (index, k, value, head), r in zip(owners, results):
            # Values are stored to six decimals; the board head hash is the exact check.
            if abs(r["value"] - value) > 1e-6:
                problems.append(f"entry {index}, world {k}: recorded value {value:.6f}, reproduced {r['value']:.6f}")
            elif r["board_head"] != head:
                problems.append(f"entry {index}, world {k}: value matches but the simulated history differs")
        return {
            "entries": len(nb.entries), "chain_ok": True, "rerun": len(jobs), "problems": problems,
            "ok": not problems, "evaluator_code_recorded": nb.entries[0]["body"].get("evaluator_code"),
            "evaluator_code_now": code_hash(),
        }
