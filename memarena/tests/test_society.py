from dataclasses import replace

import pytest

from memarena.actors.unit import Unit, UnitPolicy, UnitSpec, WorldRules, clamp_config, mutate_config, pack_probes, unpack_probes
from memarena.benchmark.synthetic import GenConfig, generate
from memarena.evalcore.criteria import CriteriaFamily
from memarena.feeds.board import Board
from memarena.feeds.http import BoardClient, BoardServer
from memarena.llm.client import MockLLM
from memarena.actors.brain import LLMBrain
from memarena.memory.heuristic import FEATURES, MemoryConfig
from memarena.actors.protocol import instances_from, verify_result
from memarena.sim.collective import confirmed_cheats, consensus, gather_signals, reputation, Signal, verify_result_post
from memarena.sim.engine import ExternalConfig, Scenario, Simulation, random_population, spearman
from memarena.util import make_rng

POLICY = UnitPolicy(specialty_focus=0.8, niche=1.0, crowd_aversion=0.3, holdout_streams=5, dev_streams=3)


def small_world(seed=1, rounds=4, n=5, **kw):
    specs = random_population(n, seed=seed, policy=POLICY)
    scenario = Scenario(seed=seed, rounds=rounds, external=ExternalConfig(n_streams=5))
    return scenario, specs


@pytest.fixture(scope="module")
def honest_run():
    scenario, specs = small_world()
    sim = Simulation(scenario, specs)
    return sim, sim.run()


def test_probe_packing_roundtrip():
    probes = generate("pack", GenConfig(n_streams=2, questions_per_stream=5)).probes()
    assert unpack_probes(pack_probes(probes)) == probes
    assert len(pack_probes(probes)["haystacks"]) == 2  # each history stored once


def test_simulation_is_deterministic(honest_run):
    _sim, first = honest_run
    scenario, specs = small_world()
    second = Simulation(scenario, specs).run()
    assert second.board_head == first.board_head and second.value == first.value


def test_every_exam_follows_commit_then_reveal(honest_run):
    sim, result = honest_run
    assert sim.board.verify() == (True, "ok")
    by_xid = {}
    for post in sim.board.read(0):
        if post.kind.startswith("exam."):
            by_xid.setdefault(post.body["xid"], []).append(post.kind)
    finished = [kinds for kinds in by_xid.values() if "exam.result" in kinds]
    assert finished and result.metrics["exchanges"] == len(finished)
    for kinds in finished:
        assert kinds == ["exam.request", "exam.offer", "exam.answers", "exam.graded", "exam.guess", "exam.result"]


def test_every_published_result_recomputes(honest_run):
    sim, result = honest_run
    family = CriteriaFamily.default()
    results = sim.board.read(0, kinds=["exam.result"])
    assert results and all(verify_result_post(sim.board, p, family)[0] for p in results)
    assert result.metrics["audit_fails"] == 0 and result.metrics["cheats_confirmed"] == []


def test_self_and_peer_evaluation_are_the_same_computation(honest_run):
    """A unit's answers graded through the exam path equal the module's direct evaluation."""
    sim, _ = honest_run
    post = sim.board.read(0, kinds=["exam.result"])[0]
    examiner = sim.units[post.author]
    from memarena.benchmark.schema import Answer
    from memarena.evalcore.module import EvaluationModule
    from memarena.evalcore.standard import Standard

    probes = unpack_probes(sim.board.get_blob(post.body["probes"]))
    instances = instances_from(probes, sim.board.get_blob(post.body["keys"]))
    answers = {pid: Answer.from_dict(a) for pid, a in sim.board.get_blob(post.body["answers"]).items()}
    module = EvaluationModule(Standard.from_dict(post.body["receipt"]["standard"]))
    direct = module.evaluate(answers, instances, post.body["salt"])
    from memarena.util import canonical_json

    # the board stores numbers to six decimals; compare in that canonical form
    assert canonical_json(direct.receipt) == canonical_json(post.body["receipt"])
    assert direct.receipt["evaluator"] == post.body["receipt"]["evaluator"]
    assert direct.receipt["items_root"] == post.body["receipt"]["items_root"]
    assert not hasattr(examiner.eval, "examinee")  # no place to keep an identity


def test_deflated_grades_are_caught_and_cost_all_standing():
    scenario, specs = small_world(seed=2)
    specs = [replace(s, kind="deflator", owner="adversary") if s.unit_id == "u01" else s for s in specs]
    sim = Simulation(scenario, specs, focus_owner="designer")
    result = sim.run()
    assert result.metrics["cheats_confirmed"] == ["u01"]
    assert result.units["u01"]["reputation"] == 0.0
    victims = [u for uid, u in sim.units.items() if uid != "u01" and u.view("u01").signals == [] and "u01" in u.known]
    assert any(u.view("u01").integrity == 0.0 for u in sim.units.values() if u.id != "u01")
    for u in sim.units.values():
        assert all(p != "u01" for p, _w in u.attention) or u.view("u01").integrity > 0 or u.id == "u01"


def test_a_rigged_exam_is_caught_by_its_provenance():
    """A harder paper than the published standard verifies as a grade but not as an exam."""
    scenario, specs = small_world(seed=7)
    specs = [replace(s, kind="rigger", owner="adversary") if s.unit_id == "u02" else s for s in specs]
    sim = Simulation(scenario, specs, focus_owner="designer")
    result = sim.run()
    assert result.metrics["cheats_confirmed"] == ["u02"]
    reasons = {e["why"] for u in sim.units.values() for e in u.log if e.get("event") == "audit_fail"}
    assert any("revealed seed" in r for r in reasons)


def test_result_must_use_the_examiners_published_standard(honest_run):
    sim, _ = honest_run
    family = CriteriaFamily.default()
    post = sim.board.read(0, kinds=["exam.result"])[0]
    assert verify_result(sim.board, post.body, family, examiner=post.author, result_seq=post.seq) == (True, "ok")
    # the same evidence attributed to an examiner who published a different standard is rejected
    other = next(u for u in sim.units if u != post.author
                 and sim.units[u].eval.standard.fingerprint() != sim.units[post.author].eval.standard.fingerprint())
    ok, why = verify_result(sim.board, post.body, family, examiner=other, result_seq=post.seq)
    assert not ok and "published" in why
    # and a doctored calibration loss is rejected
    forged = dict(post.body, loss=float(post.body["loss"]) + 0.05)
    assert not verify_result(sim.board, forged, family, examiner=post.author, result_seq=post.seq)[0]
    # revealing one exam's seed does not reveal the unit's other exams
    from memarena.actors.protocol import derive_exam_seed

    assert derive_exam_seed("s", "u", 1) != derive_exam_seed("s", "u", 2)
    assert "secret-master-seed" not in derive_exam_seed("secret-master-seed", "u", 1)


def test_overclaiming_costs_esteem():
    def esteem_for(kind):
        scenario, specs = small_world(seed=3)
        specs = [replace(s, kind=kind) if s.unit_id == "u01" else s for s in specs]
        sim = Simulation(scenario, specs)
        sim.run()
        vals = [u.view("u01").esteem for u in sim.units.values() if u.id != "u01" and u.view("u01").first_hand]
        return sum(vals) / len(vals)

    assert esteem_for("braggart") < esteem_for("honest") - 0.05


def test_collusion_cannot_raise_first_hand_esteem():
    """Ring members ace each other's exams, but units that examine them see their real level."""
    ring = ("u03", "u04")

    def honest_esteem(kind):
        scenario, specs = small_world(seed=4, rounds=5)
        specs = [replace(s, kind=kind, allies=ring, owner="adversary" if kind != "honest" else s.owner)
                 if s.unit_id in ring else s for s in specs]
        sim = Simulation(scenario, specs, focus_owner="designer" if kind != "honest" else None)
        sim.run()
        out = {}
        for u in sim.units.values():
            if u.id in ring:
                continue
            for r in ring:
                if u.view(r).first_hand:
                    out[(u.id, r)] = max(s for (s, _n, _l, _r) in u.view(r).first_hand)
        mutual = [sim.units[a].view(b).esteem for a in ring for b in ring if a != b and sim.units[a].view(b).first_hand]
        return out, mutual

    colluding, mutual = honest_esteem("colluder")
    assert all(score < 0.999 for score in colluding.values())  # nobody outside the ring saw a leaked key
    assert mutual and max(mutual) > 0.6  # inside the ring, esteem is inflated


def test_reputation_ignores_outsider_rings():
    sig = lambda i, j, s: Signal(f"{i}>{j}", i, j, s, 20, {"multi-session": (s, 20)}, 1, 0.0, 0)
    signals = [sig("a", "b", 0.6), sig("b", "a", 0.6), sig("a", "x", 0.2), sig("b", "y", 0.2),
               sig("x", "y", 1.0), sig("y", "x", 1.0), sig("x", "a", 0.1), sig("y", "b", 0.1)]
    seeded = reputation(signals, ["a", "b", "x", "y"], seeds=["a", "b"])
    assert seeded["a"] > seeded["x"] and seeded["b"] > seeded["y"]
    banned = reputation(signals, ["a", "b", "x", "y"], banned={"x"})
    assert banned["x"] == 0.0


def test_consensus_removes_examiner_leniency():
    """Unit p is only seen by a harsh examiner, q only by a lenient one; p is truly better."""
    def sig(i, j, s, k):
        return Signal(f"{i}>{j}#{k}", i, j, s, 40, {"multi-session": (s, 40)}, 1, 0.0, k)

    signals = [
        sig("harsh", "p", 0.60, 1), sig("lenient", "q", 0.70, 2),
        sig("harsh", "anchor", 0.40, 3), sig("lenient", "anchor", 0.80, 4),
    ]
    rep = {"harsh": 0.5, "lenient": 0.5}
    raw = consensus(signals, rep, debias=False)["multi-session"]
    fair = consensus(signals, rep, debias=True)["multi-session"]
    assert raw["q"][0] > raw["p"][0]
    assert fair["p"][0] > fair["q"][0]


def test_mutation_respects_budget_and_clamps_foreign_designs():
    rng = make_rng("m")
    cfg = MemoryConfig(features=frozenset(list(FEATURES)[:5]))
    for _ in range(200):
        cfg = mutate_config(cfg, rng, budget=5)
        assert len(cfg.features) <= 5 and 2 <= cfg.top_k <= 14 and 0.1 <= cfg.abstain_threshold <= 0.6
    clamped = clamp_config({"features": list(FEATURES) + ["telepathy"], "top_k": 999, "style": "shouting"}, budget=5)
    assert len(clamped.features) == 5 and clamped.top_k == 14 and clamped.style == "terse"
    assert clamp_config({"top_k": "many"}, 5) is None


def test_brain_can_steer_but_not_bypass_validation():
    plans = []

    def responder(req):
        plans.append(1)
        return {"candidates": [{"features": list(FEATURES), "top_k": 500, "abstain_threshold": 0.3,
                                "latest_margin": 0.6, "style": "terse"}],
                "sit_with": ["nobody", "u01"], "note": "x" * 2000}

    scenario, specs = small_world(seed=5, rounds=3, n=4)
    sim = Simulation(scenario, specs, brain_factory=lambda spec: LLMBrain(MockLLM(responder)))
    sim.run()
    assert plans
    for u in sim.units.values():
        assert len(u.memory_cfg.features) <= scenario.rules.feature_budget and u.memory_cfg.top_k <= 14
    notes = sim.board.read(0, kinds=["note"])
    assert notes and all(len(p.body["text"]) <= 600 for p in notes)


def test_units_run_unchanged_over_http():
    board = Board()
    scenario, specs = small_world(seed=6, rounds=2, n=3)
    server = BoardServer(board, tokens={f"tok-{s.unit_id}": s.unit_id for s in specs} | {"tok-env": "env"}).start()
    try:
        http_result = Simulation(
            scenario, specs,
            board=BoardClient(server.url, "tok-env"),  # the environment's own connection
            board_for=lambda uid: BoardClient(server.url, f"tok-{uid}"),  # one authenticated client per unit
        ).run()
    finally:
        server.stop()
    local = Simulation(scenario, specs).run()
    assert http_result.board_head == local.board_head  # identical history, different transport
    assert board.verify() == (True, "ok")


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None and spearman([1, 2], [1, 2]) is None
