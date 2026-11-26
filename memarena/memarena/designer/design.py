"""What a designer decides: the genome of a society, and the worlds it is tested in.

A ``Design`` fixes how many units are released, how they behave (``UnitPolicy``),
how their initial memory designs and scoring beliefs are assigned, and how the
society shades its estimate of its own outside score.  ``Design.build`` turns it
into concrete unit specs.

``sample_scenario`` draws a world.  The outside criterion and the difficulty of
the outside data are drawn at random and are never visible to the units or to
``Design.build``: a design has to work across the range of evaluators it might
meet, because the real one is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from ..actors.unit import UnitPolicy, UnitSpec, WorldRules
from ..evalcore.criteria import CriteriaFamily
from ..memory.heuristic import FEATURES, MemoryConfig
from ..sim.engine import ExternalConfig, Scenario, unit_secret
from ..util import hash_obj, make_rng

__all__ = ["Design", "sample_scenario", "mutate_design", "crossover", "SPECIALIST_TEMPLATES", "CRITERION_PLANS"]

# Starting designs that between them cover every ability (used by init="covering").
SPECIALIST_TEMPLATES: tuple[dict, ...] = (
    {"features": ["temporal", "relative", "third_party", "expansion", "negation"], "top_k": 6},
    {"features": ["aggregate", "expansion", "third_party", "negation", "recency"], "top_k": 10},
    {"features": ["assistant", "exchange", "expansion", "third_party", "negation"], "top_k": 6},
    {"features": ["expansion", "recency", "third_party", "negation", "comparative"], "top_k": 6},
    {"features": ["comparative", "expansion", "temporal", "relative", "third_party"], "top_k": 6},
    {"features": ["aggregate", "assistant", "expansion", "third_party", "temporal"], "top_k": 10},
)

CRITERION_PLANS = ("prior", "official", "strict", "panel", "prior_mixture")
INITS = ("random", "covering")


@dataclass(frozen=True)
class Design:
    n_units: int = 8
    policy: UnitPolicy = field(default_factory=UnitPolicy)
    init: str = "random"
    criterion_plan: str = "prior"
    shade: float = 1.0  # shading of the society's outside estimate (0: mean, 1: loss-optimal)

    def __post_init__(self):
        if self.init not in INITS:
            raise ValueError(f"unknown init {self.init!r}")
        if self.criterion_plan not in CRITERION_PLANS:
            raise ValueError(f"unknown criterion plan {self.criterion_plan!r}")

    def to_dict(self) -> dict:
        return {
            "n_units": self.n_units, "policy": self.policy.to_dict(), "init": self.init,
            "criterion_plan": self.criterion_plan, "shade": self.shade,
        }

    @staticmethod
    def from_dict(d: Mapping) -> "Design":
        return Design(
            n_units=int(d.get("n_units", 8)), policy=UnitPolicy.from_dict(d.get("policy", {})),
            init=d.get("init", "random"), criterion_plan=d.get("criterion_plan", "prior"),
            shade=float(d.get("shade", 1.0)),
        )

    def hash(self) -> str:
        return hash_obj(self.to_dict())[:16]

    def build(self, seed: int | str, rules: WorldRules | None = None, owner: str = "designer", prefix: str = "u") -> list[UnitSpec]:
        rules = rules or WorldRules()
        family = CriteriaFamily.default(rules.family_prior)
        ids = list(family.ids)
        prior = [family.prior[i] for i in ids]
        specs: list[UnitSpec] = []
        for k in range(self.n_units):
            rng = make_rng("build", seed, k)
            if self.init == "covering":
                t = SPECIALIST_TEMPLATES[k % len(SPECIALIST_TEMPLATES)]
                feats = frozenset(t["features"][: rules.feature_budget])
                memory = MemoryConfig(features=feats, top_k=t["top_k"])
            else:
                feats = frozenset(rng.sample(list(FEATURES), rules.feature_budget))
                memory = MemoryConfig(features=feats, top_k=rng.choice([4, 6, 8, 10]), style=rng.choice(["terse", "sentence"]))
            if self.criterion_plan == "official":
                weights = {"judge_lenient|detect|micro": 1.0}
            elif self.criterion_plan == "strict":
                weights = {"judge_strict|detect|micro": 1.0}
            elif self.criterion_plan == "prior_mixture":
                weights = dict(family.prior)
            elif self.criterion_plan == "panel":
                picks = rng.choices(ids, weights=prior, k=3)
                weights = {}
                for c in picks:
                    weights[c] = weights.get(c, 0.0) + 1.0
            else:  # "prior": one criterion per unit, drawn from the prior
                weights = {rng.choices(ids, weights=prior, k=1)[0]: 1.0}
            specs.append(
                UnitSpec(
                    unit_id=f"{prefix}{k:02d}", owner=owner, policy=self.policy, memory=memory,
                    criterion_weights=weights, seed=unit_secret(seed, k),
                )
            )
        return specs


def sample_scenario(seed: int | str, rounds: int = 10, rules: WorldRules | None = None) -> Scenario:
    """A world with a hidden outside evaluator drawn from the range of plausible ones."""
    rng = make_rng("scenario", seed)
    family = CriteriaFamily.default("literature")
    ids = list(family.ids)
    criterion = rng.choices(ids, weights=[family.prior[i] for i in ids], k=1)[0]
    gen = {
        "sessions_per_stream": rng.choice([16, 20, 26]),
        "hard_paraphrase_rate": round(rng.uniform(0.15, 0.5), 3),
        "third_party_rate": round(rng.uniform(0.3, 0.7), 3),
        "negation_rate": round(rng.uniform(0.1, 0.5), 3),
        "update_rate": round(rng.uniform(0.3, 0.7), 3),
        "relative_date_rate": round(rng.uniform(0.2, 0.6), 3),
        "abstention_rate": round(rng.uniform(0.03, 0.12), 3),
    }
    return Scenario(
        seed=seed, rounds=rounds, rules=rules or WorldRules(),
        external=ExternalConfig(criterion_id=criterion, gen=gen, judge_noise=0.01),
    )


# ------------------------------------------------------------------- variation

_POLICY_MOVES: dict[str, tuple] = {
    "attention_k": ("int", 2, 6, 1),
    "explore": ("float", 0.05, 0.6, 0.1),
    "exams_per_round": ("int", 1, 4, 1),
    "trials_per_round": ("int", 0, 4, 1),
    "min_gain": ("float", 0.005, 0.04, 0.005),
    "specialty_count": ("int", 1, 4, 1),
    "specialty_focus": ("float", 0.0, 1.0, 0.15),
    "niche": ("float", 0.0, 1.0, 0.25),
    "crowd_aversion": ("float", 0.0, 0.6, 0.1),
    "esteem_sharpness": ("float", 1.0, 8.0, 1.5),
    "second_hand": ("float", 0.0, 0.6, 0.15),
    "calibration_weight": ("float", 0.0, 8.0, 2.0),
    "conformity": ("float", 0.0, 0.5, 0.1),
    "half_life": ("float", 1.0, 6.0, 1.0),
    "holdout_streams": ("int", 4, 12, 2),
    "dev_streams": ("int", 3, 8, 1),
    "adopt_designs": ("bool",),
    "replicate": ("bool",),
    "focused_dev": ("bool",),
    "peer_scope": ("choice", ("criteria", "full", "none")),
}


def mutate_design(design: Design, rng, n_moves: int = 2) -> Design:
    """A few random local changes to a design."""
    d = design
    for _ in range(n_moves):
        choice = rng.random()
        if choice < 0.1:
            d = replace(d, n_units=int(min(12, max(4, d.n_units + rng.choice([-2, -1, 1, 2])))))
        elif choice < 0.17:
            d = replace(d, init=rng.choice(INITS))
        elif choice < 0.27:
            d = replace(d, criterion_plan=rng.choice(CRITERION_PLANS))
        elif choice < 0.35:
            d = replace(d, shade=round(min(1.5, max(0.0, d.shade + rng.choice([-0.25, 0.25]))), 2))
        else:
            name = rng.choice(sorted(_POLICY_MOVES))
            spec = _POLICY_MOVES[name]
            cur = getattr(d.policy, name)
            if spec[0] == "bool":
                new = not cur
            elif spec[0] == "choice":
                new = rng.choice([v for v in spec[1] if v != cur])
            elif spec[0] == "int":
                new = int(min(spec[2], max(spec[1], cur + rng.choice([-spec[3], spec[3]]))))
            else:
                new = round(min(spec[2], max(spec[1], cur + rng.choice([-spec[3], spec[3]]))), 4)
            d = replace(d, policy=replace(d.policy, **{name: new}))
    return d


def crossover(a: Design, b: Design, rng) -> Design:
    pa, pb = a.policy.to_dict(), b.policy.to_dict()
    policy = UnitPolicy.from_dict({k: (pa[k] if rng.random() < 0.5 else pb[k]) for k in pa})
    return Design(
        n_units=rng.choice([a.n_units, b.n_units]), policy=policy, init=rng.choice([a.init, b.init]),
        criterion_plan=rng.choice([a.criterion_plan, b.criterion_plan]), shade=rng.choice([a.shade, b.shade]),
    )
