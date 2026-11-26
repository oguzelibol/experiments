"""Keeping a live society healthy: release new units, retire idle ones.

The search in ``memarena.designer.search`` picks a design before a society
starts.  ``operate`` is what the designer does while the society runs.  It looks
only at what is public on the board (signals, reputations, published designs),
never at any outside score, and after each epoch it may:

* **release** a new unit aimed at the ability where the society's best is
  weakest, built by varying that ability's current champion's published design,
* **retire** one unit that is not a champion or runner-up anywhere and sits at
  the bottom of the reputation table.

Each action is logged with the evidence that triggered it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..actors.unit import UnitSpec, clamp_config, mutate_config
from ..sim.collective import CLASS_CAPABILITIES, confirmed_cheats, gather_signals, reputation, select_champions
from ..sim.engine import Simulation
from ..util import make_rng
from .design import Design

__all__ = ["operate"]


def _published_design(sim: Simulation, unit_id: str) -> Mapping | None:
    posts = sim.board.feed(unit_id, kinds=["profile"])
    return posts[-1].body.get("design") if posts else None


def operate(
    sim: Simulation,
    design: Design,
    epochs: int,
    epoch_rounds: int = 3,
    owner: str = "designer",
    seed: int | str = 0,
    max_units: int = 12,
    min_units: int = 4,
) -> list[dict]:
    """Run ``sim`` for ``epochs`` x ``epoch_rounds`` rounds, adjusting the population in between."""
    log: list[dict] = []
    released = 0
    born: dict[str, int] = {uid: 0 for uid in sim.units}
    for epoch in range(epochs):
        for _ in range(epoch_rounds):
            sim.step()
        if epoch == epochs - 1:
            break
        members = sorted(uid for uid, u in sim.units.items() if u.spec.owner == owner)
        signals = gather_signals(sim.board)
        cheats, bad = confirmed_cheats(sim.board, sim.family)
        signals = [s for s in signals if s.seq not in bad]
        rep = reputation(signals, sorted(sim.units), banned=cheats, seeds=members)
        since = {uid: u.version_round for uid, u in sim.units.items()}
        champions = select_champions(signals, rep, members, since)
        if not champions:
            continue
        rng = make_rng("operate", seed, epoch)
        weakest = min(champions, key=lambda c: (champions[c]["expected"], c))

        # retire: a unit nobody would miss
        protected = {info["unit"] for info in champions.values()}
        idle = [u for u in members if u not in protected and born.get(u, 0) < sim.round - epoch_rounds + 1]
        if idle and len(members) > min_units:
            victim = min(idle, key=lambda u: (rep.get(u, 0.0), u))
            if rep.get(victim, 0.0) <= sorted(rep.get(u, 0.0) for u in members)[len(members) // 4]:
                sim.remove_unit(victim)
                members.remove(victim)
                log.append({"round": sim.round, "action": "retire", "unit": victim,
                            "reputation": rep.get(victim, 0.0), "reason": "not a champion and bottom-quartile reputation"})

        # release: a challenger for the weakest ability
        if len(members) < max_units:
            champion = champions[weakest]["unit"]
            base = clamp_config(_published_design(sim, champion) or {}, sim.scenario.rules.feature_budget)
            if base is not None:
                released += 1
                uid = f"{owner[:1]}n{released:02d}"
                memory = mutate_config(base, rng, sim.scenario.rules.feature_budget)
                template = design.build(seed=f"{seed}:release:{released}", rules=sim.scenario.rules, owner=owner)[0]
                spec = replace(template, unit_id=uid, memory=memory, seed=f"{seed}:release:{released}")
                sim.add_unit(spec)
                born[uid] = sim.round
                log.append({"round": sim.round, "action": "release", "unit": uid, "aimed_at": weakest,
                            "varies_design_of": champion, "expected_before": champions[weakest]["expected"],
                            "design": memory.to_dict()})
    return log
