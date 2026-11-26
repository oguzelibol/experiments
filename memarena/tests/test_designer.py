import json

import pytest

from memarena.designer.design import Design, crossover, mutate_design, sample_scenario
from memarena.designer.notebook import Notebook
from memarena.designer.search import Designer, evaluate_design, mean_ci
from memarena.actors.unit import UnitPolicy
from memarena.util import make_rng

TINY = Design(n_units=4, policy=UnitPolicy(holdout_streams=4, dev_streams=3, exams_per_round=1))


def test_design_roundtrip_and_build():
    d = mutate_design(Design(), make_rng("d"), n_moves=6)
    assert Design.from_dict(d.to_dict()).hash() == d.hash()
    specs = Design(n_units=6, init="covering").build("s")
    assert len(specs) == 6 and all(len(s.memory.features) <= 5 for s in specs)
    covered = set().union(*(s.memory.features for s in specs))
    assert {"temporal", "aggregate", "assistant", "comparative", "recency"} <= covered
    assert all(list(s.criterion_weights) == ["judge_lenient|detect|micro"] for s in Design(criterion_plan="official").build("s"))
    child = crossover(Design(), d, make_rng("x"))
    assert isinstance(child, Design)


def test_scenarios_hide_a_varying_evaluator():
    criteria = {sample_scenario(f"w{k}").external.criterion_id for k in range(40)}
    assert len(criteria) >= 8  # the outside evaluator is not one fixed recipe
    assert sample_scenario("w1").to_dict() == sample_scenario("w1").to_dict()


def test_notebook_chain_detects_tampering(tmp_path):
    path = str(tmp_path / "nb.jsonl")
    nb = Notebook(path)
    nb.add("header", {"designer": "x"})
    nb.add("eval", {"mean": 0.5})
    assert Notebook(path).head == nb.head
    lines = (tmp_path / "nb.jsonl").read_text().splitlines()
    forged = json.loads(lines[1])
    forged["body"]["mean"] = 0.9
    (tmp_path / "nb.jsonl").write_text(lines[0] + "\n" + json.dumps(forged) + "\n")
    with pytest.raises(ValueError):
        Notebook(path)


def test_mean_ci():
    m, lo, hi = mean_ci([0.5, 0.7, 0.6, 0.6])
    assert m == pytest.approx(0.6) and lo < 0.6 < hi
    assert mean_ci([0.4]) == (0.4, 0.4, 0.4)


def test_evaluate_design_is_a_pure_function():
    job = {"design": TINY.to_dict(), "scenario": "pure:1", "rounds": 2}
    assert evaluate_design(job) == evaluate_design(job)


def test_full_pipeline_preregisters_and_verifies(tmp_path):
    path = str(tmp_path / "notebook.jsonl")
    designer = Designer("test", Notebook(path), workers=1, seed="t")
    designer.initial_population = lambda size: [TINY, mutate_design(TINY, make_rng("p"), 2)][:size]
    finalists = designer.search(generations=1, population=2, rung1=(1, 2), rung2=(1, 2), elites=1)
    chosen = designer.validate(finalists[:2], n_worlds=1, rounds=2)
    prereg = designer.preregister(chosen, n_worlds=2, rounds=2)
    assert all(w.startswith("heldout:") for w in prereg["worlds"])
    report = designer.heldout(chosen, prereg)
    assert set(report["arms"]) >= {"chosen", "ablation:striving_off", "ablation:estimate_unshaded", "ablation:attention_random"}
    assert report["arms"]["chosen"]["chosen_minus_arm"] == 0.0
    check = Designer.verify_notebook(path, sample=3, workers=1)
    assert check["ok"], check["problems"]

    # a notebook that evaluated held-out worlds before committing to a design is rejected
    lines = (tmp_path / "notebook.jsonl").read_text().splitlines()
    entries = [json.loads(l) for l in lines]
    pre = next(i for i, e in enumerate(entries) if e["kind"] == "preregistration")
    held = next(i for i, e in enumerate(entries) if e["kind"] == "eval" and e["body"]["stage"] == "heldout")
    reordered = entries[:pre] + [entries[held]] + [entries[pre]] + entries[pre + 1: held] + entries[held + 1:]
    bad = Notebook()
    for e in reordered:
        bad.add(e["kind"], e["body"])
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text("\n".join(json.dumps(e) for e in bad.entries) + "\n")
    assert not Designer.verify_notebook(str(bad_path), sample=0, workers=1)["ok"]
