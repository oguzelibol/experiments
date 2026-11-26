import dataclasses
import inspect
import json
import math
import random
from dataclasses import replace

import pytest

from memarena.actors.brain import LLMBrain
from memarena.actors.protocol import (
    check_result, derive_exam_share, joint_exam_seed, pack_probes, share_commitment, unpack_probes, verify_result,
)
from memarena.actors.unit import Unit, UnitPolicy, clamp_config, mutate_config
from memarena.benchmark.schema import Answer
from memarena.benchmark.synthetic import GenConfig, generate
from memarena.evalcore.criteria import CriteriaFamily
from memarena.evalcore.module import EvaluationModule
from memarena.evalcore.standard import CAPABILITIES, Standard
from memarena.feeds.board import AuthorBoard, Board
from memarena.feeds.http import BoardClient, BoardServer
from memarena.llm.client import MockLLM
from memarena.memory.heuristic import FEATURES, MemoryConfig
from memarena.sim.collective import Signal, audit_board, consensus, gather_signals, reputation
from memarena.sim.engine import ExternalConfig, Scenario, Simulation, random_population, spearman, unit_secret
from memarena.util import canonical_json, make_rng

POLICY = UnitPolicy(specialty_focus=0.8, niche=1.0, crowd_aversion=0.3, holdout_streams=5, dev_streams=3)
FAMILY = CriteriaFamily.default()
EXAM_KINDS = ["exam.request", "exam.ready", "exam.nonce", "exam.offer", "exam.answers", "exam.graded", "exam.guess", "exam.result"]


def small_world(seed=1, rounds=4, n=5):
    specs = random_population(n, seed=seed, policy=POLICY)
    scenario = Scenario(seed=seed, rounds=rounds, external=ExternalConfig(n_streams=5))
    return scenario, specs


def run_with(kind, unit_id="u02", seed=2, rounds=4, allies=()):
    scenario, specs = small_world(seed=seed, rounds=rounds)
    specs = [replace(s, kind=kind, owner="adversary", allies=tuple(allies)) if s.unit_id == unit_id else s for s in specs]
    sim = Simulation(scenario, specs, focus_owner="designer")
    return sim, sim.run()


def reasons(sim, by_other_than=None):
    return {e["why"] for u in sim.units.values() if u.id != by_other_than for e in u.log if e.get("event") == "audit_fail"}


@pytest.fixture(scope="module")
def honest_run():
    scenario, specs = small_world()
    sim = Simulation(scenario, specs)
    return sim, sim.run()


# ------------------------------------------------------------------ basics

def test_probe_packing_roundtrip():
    probes = generate("pack", GenConfig(n_streams=2, questions_per_stream=5)).probes()
    assert unpack_probes(pack_probes(probes)) == probes
    assert len(pack_probes(probes)["haystacks"]) == 2  # each history stored once


def test_simulation_is_deterministic(honest_run):
    _sim, first = honest_run
    scenario, specs = small_world()
    second = Simulation(scenario, specs).run()
    assert second.board_head == first.board_head and second.value == first.value


def test_honest_units_are_never_convicted():
    """Across several worlds no honest unit fails anyone's verification."""
    for seed in ("h1", "h2", "h3", "h4"):
        scenario, specs = small_world(seed=seed, rounds=5, n=6)
        sim = Simulation(scenario, specs)
        result = sim.run()
        signals, cheats, bad = audit_board(sim.board, FAMILY)
        assert not cheats and not bad, (seed, cheats)
        assert result.metrics["audit_fails"] == 0 and result.metrics["cheats_confirmed"] == []
        assert not sim.board.read(0, kinds=["audit.fail"])
        assert len(signals) == result.metrics["results_posted"] == result.metrics["exchanges"] > 0


def test_standard_fingerprint_survives_publication():
    """The receipt and the profile must name the same standard after the board rounds numbers."""
    rng = random.Random(3)
    for _ in range(1500):
        cw = {c: rng.random() ** rng.choice([1, 3]) for c in rng.sample(list(FAMILY.ids), rng.randint(1, 6))}
        kw = {c: rng.random() for c in rng.sample(list(CAPABILITIES), rng.randint(1, 5))}
        a = Standard(cw, kw, {"sessions_per_stream": 20, "hard_paraphrase_rate": rng.random()})
        b = Standard.from_dict(json.loads(canonical_json(a.to_dict())))
        assert a.fingerprint() == b.fingerprint()
    with pytest.raises(ValueError):
        Standard({"em|detect|micro": float("nan")})


def test_every_exam_follows_the_eight_post_protocol(honest_run):
    sim, result = honest_run
    assert sim.board.verify() == (True, "ok")
    by_xid = {}
    for post in sim.board.read(0):
        if post.kind.startswith("exam."):
            by_xid.setdefault(post.body["xid"], []).append(post.kind)
    finished = [kinds for kinds in by_xid.values() if "exam.result" in kinds]
    assert finished and result.metrics["exchanges"] == len(finished)
    for kinds in finished:
        assert kinds == EXAM_KINDS


def test_every_published_result_verifies(honest_run):
    sim, result = honest_run
    results = sim.board.read(0, kinds=["exam.result"])
    assert results and all(verify_result(sim.board, p, FAMILY) == (True, "ok") for p in results)
    assert result.metrics["audit_fails"] == 0 and result.metrics["cheats_confirmed"] == []


def test_self_and_peer_evaluation_are_the_same_computation(honest_run):
    """A published grade equals the module's direct evaluation of the same answers on the same paper."""
    sim, _ = honest_run
    post = sim.board.read(0, kinds=["exam.result"])[0]
    b = post.body
    nonce = sim.board.read(since_seq=b["nonce_seq"] - 1, limit=1)[0].body["nonce"]
    standard = Standard.from_dict(b["receipt"]["standard"])
    exam = generate(joint_exam_seed(b["share"], nonce, b["xid"]), standard.gen_config(2, 12))
    answers = {pid: Answer.from_dict(a) for pid, a in sim.board.get_blob(b["answers"]).items()}
    direct = EvaluationModule(standard).evaluate(answers, exam.instances, b["salt"])
    assert canonical_json(direct.receipt) == canonical_json(b["receipt"])
    # the grading entry point has no way to be told whose answers these are
    assert set(inspect.signature(EvaluationModule.evaluate).parameters) == {"self", "answers", "instances", "salt"}


def test_neither_side_chooses_the_paper(honest_run):
    sim, _ = honest_run
    post = sim.board.read(0, kinds=["exam.result"])[0]
    b = post.body
    ready = sim.board.read(since_seq=b["ready_seq"] - 1, limit=1)[0]
    nonce = sim.board.read(since_seq=b["nonce_seq"] - 1, limit=1)[0]
    assert ready.seq < nonce.seq  # the examiner was committed before it saw the examinee's share
    assert ready.body["share_commit"] == share_commitment(b["share"])
    assert joint_exam_seed(b["share"], nonce.body["nonce"], b["xid"]) != joint_exam_seed(b["share"], nonce.body["nonce"] + "x", b["xid"])
    assert derive_exam_share("s", "u", 1) != derive_exam_share("s", "u", 2)
    assert "secret-master-seed" not in derive_exam_share("secret-master-seed", "u", 1)


def test_unit_secrets_do_not_reveal_the_world():
    a, b = unit_secret("world-7", 0), unit_secret("world-7", 1)
    assert a != b and "world-7" not in a
    specs = random_population(3, seed="world-7")
    assert all("world-7" not in str(s.seed) for s in specs)


# ------------------------------------------------------- tampering with a result

def _forge(sim, post, **changes):
    """A real result with some fields changed, as if its examiner had published it a moment later.

    Built by hand rather than posted, so that the verifier is tested even on
    content the board itself would refuse (it rejects NaN outright).
    """
    body = dict(post.body)
    body.update(changes)
    return dataclasses.replace(post, seq=sim.board.head()[0] + 1, body=body)


def test_a_result_cannot_be_altered_in_any_field(honest_run):
    scenario, specs = small_world()
    sim = Simulation(scenario, specs)
    sim.run()
    post = sim.board.read(0, kinds=["exam.result"])[0]
    assert check_result(sim.board, post, FAMILY).ok
    b = post.body
    nan_receipt = dict(b["receipt"], score=math.nan, capability={c: [math.nan, n] for c, (_s, n) in b["capability"].items()})
    extra_cap = dict(b["capability"]); extra_cap["abstention" if "abstention" not in extra_cap else "multi-session"] = [1.0, 10 ** 6]
    other_offer = next((p.seq for p in sim.board.read(0, kinds=["exam.offer"]) if p.author == post.author and p.seq != b["offer_seq"]), 1)
    attacks = {
        "lower score": dict(score=b["score"] - 0.05),
        "nan receipt": dict(score=0.0, receipt=nan_receipt),
        "nan score": dict(score=math.nan),
        "extra ability": dict(capability=extra_cap),
        "inflated counts": dict(capability={c: [s, 10 ** 6] for c, (s, _n) in b["capability"].items()}),
        "other loss": dict(loss=b["loss"] + 0.05),
        "deny the self-estimate": dict(guess=1.0, guess_valid=False, guess_seq=None),
        "misquote the self-estimate": dict(guess=min(1.0, b["guess"] + 0.2)),
        "another offer": dict(offer_seq=other_offer),
        "another examinee": dict(examinee=next(u for u in sim.units if u not in (post.author, b["examinee"]))),
        "self exam": dict(examinee=post.author),
        "other share": dict(share=b["share"][::-1]),
        "other answers": dict(answers=sim.board.put_blob({})),
        "other salt": dict(salt="x"),
    }
    for name, changes in attacks.items():
        forged = _forge(sim, post, **changes)
        verdict = check_result(sim.board, forged, FAMILY)
        assert not verdict.ok, name
    # the board itself refuses NaN, before any verifier sees it
    with pytest.raises(ValueError):
        sim.board.post(post.author, "exam.result", dict(b, score=math.nan), to=post.to)
    # the same result posted again verifies, but is counted once
    copy = sim.board.post(post.author, "exam.result", dict(b), to=post.to, round=post.round)
    assert check_result(sim.board, copy, FAMILY).ok
    signals, _cheats, _bad = audit_board(sim.board, FAMILY)
    assert sum(1 for s in signals if s.xid == b["xid"]) == 1
    # and a result attributed to a different examiner fails
    other = next(u for u in sim.units if u not in (post.author, b["examinee"]))
    impostor = sim.board.post(other, "exam.result", dict(b), to=post.to, round=post.round)
    assert not check_result(sim.board, impostor, FAMILY).ok


# ------------------------------------------------------------- dishonest units

def test_deflated_grades_are_caught_and_cost_all_standing():
    sim, result = run_with("deflator", "u01")
    assert result.metrics["cheats_confirmed"] == ["u01"] and result.units["u01"]["reputation"] == 0.0
    assert any(u.view("u01").integrity == 0.0 for u in sim.units.values() if u.id != "u01")
    signals, cheats, _ = audit_board(sim.board, FAMILY)
    assert cheats == {"u01"} and not any(s.examiner == "u01" and s.examinee != "u01" and s.score < 0 for s in signals)


def test_nan_cannot_be_published():
    """A zero hidden behind NaN never reaches the board, let alone anyone's accounts."""
    sim, result = run_with("poisoner", "u01", seed=11)
    assert not [p for p in sim.board.read(0, kinds=["exam.result"]) if p.author == "u01"]
    signals, _cheats, _ = audit_board(sim.board, FAMILY)
    assert all(math.isfinite(s.score) for s in signals) and not any(s.examiner == "u01" for s in signals)
    assert any(e.get("event") == "bad_message" for e in sim.units["u01"].log)


def test_a_rigged_exam_is_caught_by_its_provenance():
    sim, result = run_with("rigger", "u02", seed=7)
    assert result.metrics["cheats_confirmed"] == ["u02"]
    assert any("joint seed" in r for r in reasons(sim))


def test_a_swapped_standard_is_caught():
    sim, result = run_with("switcher", "u02", seed=8)
    assert result.metrics["cheats_confirmed"] == ["u02"]
    assert any("published" in r for r in reasons(sim))


def test_a_fabricated_result_is_rejected_by_everyone():
    sim, result = run_with("fabricator", "u02", seed=9)
    forged = [p for p in sim.board.read(0, kinds=["exam.result"]) if "forged" in p.body["xid"]]
    assert forged
    signals, cheats, bad = audit_board(sim.board, FAMILY)
    assert cheats == {"u02"} and {p.seq for p in forged} <= bad
    assert not any("forged" in s.xid for s in signals)  # nothing forged reaches the accounts
    assert result.metrics["cheats_confirmed"] == ["u02"] and result.units["u02"]["reputation"] == 0.0
    # the victims noticed, although they never took part in those exams
    assert any("request" in r or "transcript" in r for r in reasons(sim, by_other_than="u02"))


def test_a_unit_cannot_publish_a_grade_of_itself():
    sim, result = run_with("selfgrader", "u02", seed=10)
    own = [p for p in sim.board.read(0, kinds=["exam.result"]) if p.author == "u02" and p.body["examinee"] == "u02"]
    assert own and all(not check_result(sim.board, p, FAMILY).ok for p in own)
    signals, cheats, _ = audit_board(sim.board, FAMILY)
    assert "u02" in cheats and not any(s.examiner == s.examinee for s in signals)
    assert result.units["u02"]["reputation"] == 0.0


def test_going_silent_does_not_hide_a_grade():
    """An examinee that never opens its self-estimate still gets its result published."""
    sim, result = run_with("silent", "u03", seed=12, rounds=4)
    about = [p for p in sim.board.read(0, kinds=["exam.result"]) if p.body["examinee"] == "u03"]
    assert about, "results about the silent unit must still be published"
    for p in about:
        assert p.body["guess_valid"] is False and p.body["guess"] == 1.0 and p.body["guess_seq"] is None
        assert check_result(sim.board, p, FAMILY).ok
    assert result.metrics["cheats_confirmed"] == []
    # ...but only once the round is over: a void result rushed out in the same round is rejected
    scenario, specs = small_world(seed=12, rounds=1)
    specs = [replace(s, kind="silent") if s.unit_id == "u03" else s for s in specs]
    sim2 = Simulation(scenario, specs)
    sim2.run()
    assert not [p for p in sim2.board.read(0, kinds=["exam.result"]) if p.body["examinee"] == "u03"]


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
    scenario, specs = small_world(seed=4, rounds=5)
    specs = [replace(s, kind="colluder", allies=ring, owner="adversary") if s.unit_id in ring else s for s in specs]
    sim = Simulation(scenario, specs, focus_owner="designer")
    sim.run()
    seen = [s for u in sim.units.values() if u.id not in ring for r in ring for (s, _n, _l, _r) in u.view(r).first_hand]
    assert seen and all(score < 0.999 for score in seen)  # nobody outside the ring saw a leaked key
    mutual = [sim.units[a].view(b).esteem for a in ring for b in ring if a != b and sim.units[a].view(b).first_hand]
    assert mutual and max(mutual) > 0.6  # inside the ring, esteem is inflated


# ------------------------------------------------------------ accusations

def test_accusations_are_recomputed_not_believed(honest_run):
    scenario, specs = small_world()
    sim = Simulation(scenario, specs)
    sim.run()
    honest = sim.board.read(0, kinds=["exam.result"])[0]
    accuser = next(u for u in sim.units if u not in (honest.author, honest.body["examinee"]))
    bystander = next(u for u in sim.units.values() if u.id not in (accuser, honest.author))
    false = sim.board.post(accuser, "audit.fail", {"examiner": honest.author, "result_seq": honest.seq, "reason": "made up"})
    bystander._check_accusation(false)
    assert bystander.view(accuser).integrity == 0.0 and bystander.view(honest.author).integrity == 1.0
    # an accusation pointing at something that is not a result also costs only the accuser
    odd = sim.board.post(accuser, "audit.fail", {"examiner": honest.author, "result_seq": 1, "reason": "x"})
    bystander._check_accusation(odd)
    assert bystander.view(honest.author).integrity == 1.0
    # a true accusation is upheld
    forged = sim.board.post(honest.author, "exam.result", dict(honest.body, score=honest.body["score"] - 0.1), to=honest.to)
    truthful = sim.board.post(accuser, "audit.fail", {"examiner": honest.author, "result_seq": forged.seq, "reason": "lowered"})
    other = next(u for u in sim.units.values() if u.id not in (accuser, honest.author, bystander.id))
    other._check_accusation(truthful)
    assert other.view(honest.author).integrity == 0.0 and other.view(accuser).integrity == 1.0


def test_malformed_messages_do_not_crash_a_unit(honest_run):
    sim, _ = honest_run
    unit = next(iter(sim.units.values()))
    other = next(u for u in sim.units if u != unit.id)
    for kind in EXAM_KINDS:
        for body in ({}, {"xid": None}, {"xid": 5}, {"xid": "x", "answers": 3, "probes": [], "nonce": {}, "guess": "a"},
                     {"xid": "x", "standard": "nope", "exam_size": "big", "loss": 1}):
            unit.on_message(sim.board.post(other, kind, body, to=unit.id))


# ------------------------------------------------------- collective arithmetic

def test_a_unit_cannot_post_under_another_name():
    board = Board()
    mine = AuthorBoard(board, "u00")
    mine.post("u00", "note", {"text": "hello"})
    with pytest.raises(PermissionError):
        mine.post("u01", "note", {"text": "impersonation"})
    assert [p.author for p in board.read(0)] == ["u00"]


def _sig(i, j, s, k=0, n=20):
    return Signal(f"{i}>{j}#{k}", i, j, s, n, {"multi-session": (s, n)}, 1, 0.0, k)


def test_reputation_respects_the_level_of_grades_and_ignores_rings():
    # A did badly, B and C did well, each examined by different examiners
    signals = [_sig("x", "A", 0.05), _sig("y", "B", 0.95), _sig("z", "C", 0.95),
               _sig("A", "x", 0.5), _sig("B", "y", 0.5), _sig("C", "z", 0.5)]
    rep = reputation(signals, ["A", "B", "C", "x", "y", "z"])
    assert rep["B"] > rep["A"] * 2 and rep["C"] > rep["A"] * 2  # every unit keeps a small restart share
    assert sum(rep.values()) == pytest.approx(1.0)
    # an outsider ring praising itself gains nothing over seeded units
    ring = [_sig("a", "b", 0.6), _sig("b", "a", 0.6), _sig("a", "p", 0.2), _sig("b", "q", 0.2),
            _sig("p", "q", 1.0), _sig("q", "p", 1.0), _sig("p", "a", 0.1), _sig("q", "b", 0.1)]
    seeded = reputation(ring, ["a", "b", "p", "q"], seeds=["a", "b"])
    assert seeded["a"] > seeded["p"] and seeded["b"] > seeded["q"]
    assert reputation(ring, ["a", "b", "p", "q"], banned={"p"})["p"] == 0.0
    # a unit's opinion of itself counts for nothing
    base = reputation([_sig("a", "b", 0.5), _sig("b", "a", 0.5)], ["a", "b"])
    selfish = reputation([_sig("a", "b", 0.5), _sig("b", "a", 0.5), _sig("a", "a", 1.0)], ["a", "b"])
    assert selfish == pytest.approx(base)
    assert reputation([], ["a", "b"]) == {"a": 0.5, "b": 0.5}


def test_consensus_removes_examiner_leniency():
    """Unit p is only seen by a harsh examiner, q only by a lenient one; p is truly better."""
    signals = [_sig("harsh", "p", 0.60, 1, 40), _sig("lenient", "q", 0.70, 2, 40),
               _sig("harsh", "anchor", 0.40, 3, 40), _sig("lenient", "anchor", 0.80, 4, 40)]
    rep = {"harsh": 0.5, "lenient": 0.5}
    raw = consensus(signals, rep, debias=False)["multi-session"]
    fair = consensus(signals, rep, debias=True)["multi-session"]
    assert raw["q"][0] > raw["p"][0] and fair["p"][0] > fair["q"][0]


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
