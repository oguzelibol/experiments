import threading
import time

from memarena.actors.runner import UnitRunner
from memarena.actors.unit import Unit, UnitPolicy
from memarena.designer.design import Design
from memarena.designer.operate import operate
from memarena.feeds.board import Board
from memarena.feeds.http import BoardClient, BoardServer
from memarena.sim.engine import ExternalConfig, Scenario, Simulation

POLICY = UnitPolicy(holdout_streams=4, dev_streams=3)


def test_operate_releases_and_retires_using_public_information_only():
    design = Design(n_units=5, policy=POLICY)
    scenario = Scenario(seed="op", rounds=6, external=ExternalConfig(n_streams=4))
    sim = Simulation(scenario, design.build("op"))
    log = operate(sim, design, epochs=3, epoch_rounds=2, seed="op")
    assert sim.round == 6
    releases = [e for e in log if e["action"] == "release"]
    assert releases and all(e["unit"] in sim.units or any(r["unit"] == e["unit"] for r in log if r["action"] == "retire") for e in releases)
    for e in releases:
        assert len(e["design"]["features"]) <= scenario.rules.feature_budget
    result = sim.result()
    assert 0.0 <= result.metrics["outside_score"] <= 1.0
    assert sim.board.verify() == (True, "ok")


def test_runner_follows_the_clock_over_http():
    board = Board()
    specs = Design(n_units=3, policy=POLICY).build("run")
    server = BoardServer(board, tokens={f"t-{s.unit_id}": s.unit_id for s in specs}).start()
    try:
        runners = [UnitRunner(Unit(s, BoardClient(server.url, f"t-{s.unit_id}")), poll_seconds=0.02) for s in specs]
        stop = threading.Event()
        threads = [threading.Thread(target=r.run, kwargs={"should_stop": stop.is_set}, daemon=True) for r in runners]
        for t in threads:
            t.start()
        for r in (1, 2):
            board.post("env", "clock", {"round": r}, round=r)
            deadline = time.time() + 20
            want = r * 1  # at least one finished exam per round
            while time.time() < deadline and len(board.read(0, kinds=["exam.result"])) < want:
                time.sleep(0.05)
        stop.set()
        for t in threads:
            t.join(timeout=20)
    finally:
        server.stop()
    assert len(board.read(0, kinds=["profile"])) >= 3
    assert len(board.read(0, kinds=["exam.result"])) >= 1
    assert board.verify() == (True, "ok")
