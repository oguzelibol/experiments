import copy
import importlib.util
import json
import os
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest

from memarena.designer.beacon import BeaconError, DrandBeacon, FakeBeacon
from memarena.designer.design import Design, crossover, mutate_design, sample_scenario
from memarena.designer.notebook import Notebook
from memarena.designer.search import (
    ABLATIONS, Designer, evaluate_design, heldout_report, heldout_world_seeds, mean_ci, refinement_candidates,
)
from memarena.actors.unit import UnitPolicy
from memarena.sim.engine import Simulation
from memarena.util import make_rng

TINY = Design(n_units=4, policy=UnitPolicy(holdout_streams=4, dev_streams=3, exams_per_round=1))
SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_designer(path):
    designer = Designer("test", Notebook(path), workers=1, seed="t")
    designer.initial_population = lambda size: [TINY, mutate_design(TINY, make_rng("p"), 2)][:size]
    return designer


def submission_for(design, report, notebook_head=None, history=None):
    """The claims scripts/run_designer.py puts into submission.json."""
    submission = {
        "design": design.to_dict(), "design_hash": design.hash(), "claimed_value": report["value_mean"],
        "claimed_value_ci95": report["value_ci95"], "arms": report["arms"], "notebook_head": notebook_head,
    }
    if history is not None:
        submission["refinement_rounds"] = [
            {"round": n, "design_hash": d.hash(), "claimed_value": r["value_mean"], "claimed_value_ci95": r["value_ci95"]}
            for n, (d, r) in enumerate(history, start=1)
        ]
    return submission


def write_package(root, entries, submission):
    """Write a package the way a forger would: any entries, the chain re-linked, the submission bound to the new head."""
    notebook = Notebook(os.path.join(root, "notebook.jsonl"))
    for entry in entries:
        notebook.add(entry["kind"], entry["body"])
    with open(os.path.join(root, "submission.json"), "w", encoding="utf-8") as fh:
        json.dump({**submission, "notebook_head": notebook.head}, fh)
    return str(root)


def heldout_entries(entries):
    return [e for e in entries if e["kind"] == "eval" and e["body"]["stage"] == "heldout"]


def failed(checks):
    """Names of the failed checks, from Designer.verify_notebook's "checks" or from the script's status map."""
    statuses = {name: check["status"] if isinstance(check, dict) else check for name, check in checks.items()}
    return {name for name, status in statuses.items() if status == "failed"}


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    """A tiny real contest package: search, validation, pre-registration, held-out run, submission."""
    root = tmp_path_factory.mktemp("honest")
    path = str(root / "notebook.jsonl")
    designer = tiny_designer(path)
    finalists = designer.search(generations=1, population=2, rung1=(1, 2), rung2=(1, 2), elites=1)
    chosen = designer.validate(finalists[:2], n_worlds=1, rounds=2)
    prereg = designer.preregister(chosen, n_worlds=2, rounds=2)
    report = designer.heldout(chosen)
    submission = submission_for(chosen, report, designer.notebook.head)
    (root / "submission.json").write_text(json.dumps(submission))
    return SimpleNamespace(
        root=str(root), path=path, chosen=chosen, prereg=prereg, report=report, submission=submission,
        entries=designer.notebook.entries,
    )


@pytest.fixture
def verify_package(capsys):
    """Run scripts/verify_contest.py on a package: (exit status, its JSON report, what it printed for people)."""
    script = load_script("verify_contest")

    def run(root, *options):
        capsys.readouterr()
        status = script.main([str(root), "--workers", "1", "--sample", "0", *options])
        printed = capsys.readouterr()
        return status, json.loads(printed.out), printed.err

    return run


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


def test_notebook_rejects_what_cannot_be_compared(tmp_path):
    # NaN compares as neither equal nor different, so a forged NaN value would slip through every tolerance check
    path = tmp_path / "nb.jsonl"
    nb = Notebook(str(path))
    nb.add("eval", {"values": [0.5]})
    path.write_text(path.read_text().replace("[0.5]", "[NaN]"))
    with pytest.raises(ValueError, match="not finite"):
        Notebook(str(path))
    path.write_text("{not json\n")
    with pytest.raises(ValueError, match="line 1 is not JSON"):
        Notebook(str(path))
    check = Designer.verify_notebook(str(path), sample=0, workers=1)
    assert not check["ok"] and failed(check["checks"]) == {"chain"}
    assert not Designer.verify_notebook(str(tmp_path / "missing.jsonl"), sample=0, workers=1)["ok"]


def test_mean_ci():
    m, lo, hi = mean_ci([0.5, 0.7, 0.6, 0.6])
    assert m == pytest.approx(0.6) and lo < 0.6 < hi
    assert mean_ci([0.4]) == (0.4, 0.4, 0.4)


def test_mean_ci_uses_the_exact_t_quantile():
    def quantile(n):
        xs = [float(k) for k in range(n)]
        m, lo, hi = mean_ci(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
        assert m - lo == pytest.approx(hi - m)
        return (hi - m) * n ** 0.5 / sd

    assert quantile(13) == pytest.approx(2.1788, abs=1e-4)  # 12 degrees of freedom, not the next larger n of a table
    assert quantile(33) == pytest.approx(2.0369, abs=1e-3)  # 32 degrees of freedom
    assert quantile(2) == pytest.approx(12.7062, abs=1e-4)


def test_evaluate_design_is_a_pure_function():
    job = {"design": TINY.to_dict(), "scenario": "pure:1", "rounds": 2}
    assert evaluate_design(job) == evaluate_design(job)


def test_full_pipeline_preregisters_and_verifies(package, tmp_path):
    assert all(w.startswith("heldout:") for w in package.prereg["worlds"])
    report = package.report
    assert set(report["arms"]) >= {"chosen", "ablation:striving_off", "ablation:estimate_unshaded", "ablation:attention_random"}
    assert report["arms"]["chosen"]["chosen_minus_arm"] == 0.0
    check = Designer.verify_notebook(package.path, sample=3, workers=1)
    assert check["ok"], check["problems"]

    # every held-out world of the chosen design was re-run, and three other experiments were sampled
    chosen_entry = heldout_entries(package.entries)[0]["index"]
    assert [cell for cell in check["rerun_cells"] if cell[0] == chosen_entry] == [[chosen_entry, 0], [chosen_entry, 1]]
    # every world of every other arm of the round too, since the submission compares the design with them
    arms = [e["index"] for e in heldout_entries(package.entries)]
    for index in arms:
        assert sum(1 for cell in check["rerun_cells"] if cell[0] == index) == 2
    others = [cell for cell in check["rerun_cells"] if cell[0] not in arms]
    assert len({tuple(c) for c in others}) >= 3
    assert check["checks"]["rerun_sample"]["status"] == check["checks"]["rerun_arms"]["status"] == "passed"
    # nothing outside the notebook fixed these worlds, and verification says so
    assert any("depend only on the notebook head, which the designer controls" in w for w in check["warnings"])

    # a notebook that evaluated held-out worlds before committing to a design is rejected
    lines = open(package.path).read().splitlines()
    entries = [json.loads(l) for l in lines]
    pre = next(i for i, e in enumerate(entries) if e["kind"] == "preregistration")
    held = next(i for i, e in enumerate(entries) if e["kind"] == "eval" and e["body"]["stage"] == "heldout")
    reordered = entries[:pre] + [entries[held]] + [entries[pre]] + entries[pre + 1: held] + entries[held + 1:]
    bad = Notebook()
    for e in reordered:
        bad.add(e["kind"], e["body"])
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text("\n".join(json.dumps(e) for e in bad.entries) + "\n")
    rejected = Designer.verify_notebook(str(bad_path), sample=0, workers=1)
    assert not rejected["ok"]
    assert f"heldout_isolation: entry {pre + 1}: no well-formed pre-registration precedes it" in rejected["problems"]


def test_verify_contest_accepts_the_honest_package_and_warns_without_a_beacon(package, verify_package):
    status, report, printed = verify_package(package.root)
    assert status == 0 and report["ok"] and report["problems"] == []
    assert {report["checks"][name] for name in ("heldout_report", "submission", "rerun_chosen")} == {"passed"}
    assert report["checks"]["beacon"] == "not run"
    assert "WARNING: round 1: no randomness beacon was used" in printed
    assert "depend only on the notebook head, which the designer controls" in printed
    prereg = next(e["index"] for e in package.entries if e["kind"] == "preregistration")
    assert report["preregistrations"] == [{"entry": prereg, "design_hash": package.chosen.hash(), "beacon": None, "anchor": None}]

    # a verifier who accepts only what could be checked turns those warnings into a rejection
    status, report, printed = verify_package(package.root, "--strict")
    assert status == 1 and failed(report["checks"]) == set()
    assert report["problems"][0].startswith("strict: round 1: no randomness beacon was used")
    assert "FAILED: strict: round 1: no randomness beacon was used" in printed


def test_a_changed_report_is_rejected(package, tmp_path, verify_package):
    """Doctoring 1: only the reported number is raised; the chain is re-linked and the submission repeats it."""
    entries = copy.deepcopy(package.entries)
    report = entries[-1]
    assert report["kind"] == "heldout_report"
    report["body"]["value_mean"] = forged = round(report["body"]["value_mean"] + 0.4, 6)
    root = write_package(tmp_path, entries, {**package.submission, "claimed_value": forged})

    status, verdict, _ = verify_package(root)
    assert status == 1 and failed(verdict["checks"]) == {"heldout_report", "submission"}
    assert any(
        p.startswith(f"heldout_report: entry {report['index']}: does not follow from the held-out eval entries")
        and f"value_mean is {forged!r}, recomputed {package.report['value_mean']:.6f}" in p
        for p in verdict["problems"]
    )


def test_forged_heldout_values_are_rejected(package, tmp_path, verify_package):
    """Doctoring 2: one held-out value of the chosen design is raised and everything derived from it is
    recomputed, so the package is consistent with itself and only a re-run of that very world can tell."""
    entries = copy.deepcopy(package.entries)
    held = heldout_entries(entries)
    chosen = held[0]["body"]
    true_value = chosen["values"][1]
    chosen["values"][1] = forged = round(true_value + 0.25, 6)
    chosen["mean"] = sum(chosen["values"]) / len(chosen["values"])
    report = heldout_report(list(zip(package.report["arms"], [e["body"] for e in held])))
    entries[-1]["body"] = report
    assert report["value_mean"] == pytest.approx(package.report["value_mean"] + 0.125, abs=1e-5)
    root = write_package(tmp_path, entries, submission_for(package.chosen, report))

    status, verdict, _ = verify_package(root)
    assert status == 1 and failed(verdict["checks"]) == {"rerun_chosen"}
    assert verdict["failures"] == [{
        "check": "rerun_chosen", "entry": held[0]["index"],
        "reason": f"world 1: recorded value {forged:.6f}, reproduced {true_value:.6f}",
    }]


def test_a_design_other_than_the_preregistered_one_is_rejected(package, tmp_path, verify_package):
    """Doctoring 4: one design is pre-registered; the arm that did best on the held-out worlds is then presented
    as the chosen design, with a report and a submission that are consistent with that."""
    entries = copy.deepcopy(package.entries)
    held = heldout_entries(entries)
    rivals = [e for e in held[1:] if e["body"]["design_hash"] != package.chosen.hash()]
    best = max(rivals, key=lambda e: e["body"]["mean"])
    held[0]["body"] = copy.deepcopy(best["body"])
    report = heldout_report(list(zip(package.report["arms"], [e["body"] for e in held])))
    entries[-1]["body"] = report
    other = Design.from_dict(best["body"]["design"])
    root = write_package(tmp_path, entries, submission_for(other, report))

    status, verdict, _ = verify_package(root)
    assert status == 1 and failed(verdict["checks"]) == {"heldout_arms", "submission"}
    prereg = next(e["index"] for e in entries if e["kind"] == "preregistration")
    assert (f"heldout_arms: entry {held[0]['index']}: arm 'chosen' is pre-registered at entry {prereg} as design "
            f"{package.chosen.hash()}, but this held-out entry is design {other.hash()}") in verdict["problems"]
    assert any(p.startswith(f"submission: submits design {other.hash()}") for p in verdict["problems"])

    # the designer's own API refuses to produce such a notebook in the first place
    committed = tmp_path / "committed.jsonl"
    worlds = next(i for i, e in enumerate(package.entries) if e["kind"] == "heldout_worlds")
    committed.write_text("".join(json.dumps(e) + "\n" for e in package.entries[: worlds + 1]))
    with pytest.raises(ValueError, match="differ from those pre-registered"):
        Designer("test", Notebook(str(committed)), workers=1, seed="t").heldout(other)


def test_a_colliding_design_hash_does_not_stand_in_for_the_preregistered_design(package, tmp_path, verify_package, monkeypatch):
    """A design hash has 64 bits, so a second design with the hash of the pre-registered one can be manufactured.
    The collision is arranged here by patching the hash.  The twin's held-out run is real and every hash in the
    package agrees, yet it is not the design that was pre-registered."""
    committed, real_hash = package.chosen.hash(), Design.hash
    twin = replace(package.chosen, shade=0.123)
    monkeypatch.setattr(Design, "hash", lambda self: committed if self == twin else real_hash(self))
    assert twin.hash() == committed and twin != package.chosen

    entries = copy.deepcopy(package.entries)
    held = heldout_entries(entries)
    forger = Designer("forger", Notebook(), workers=1)
    forger.assess([twin], package.prereg["worlds"], 2, "heldout")
    held[0]["body"] = forger.notebook.entries[-1]["body"]
    report = heldout_report(list(zip(package.report["arms"], [e["body"] for e in held])))
    entries[-1]["body"] = report
    root = write_package(tmp_path, entries, submission_for(twin, report))

    status, verdict, _ = verify_package(root)
    assert status == 1 and failed(verdict["checks"]) == {"heldout_arms", "submission"}
    assert [f["reason"].split(", but ")[1] for f in verdict["failures"] if f["check"] == "heldout_arms"] == [
        "this held-out entry carries that hash but is a different design"
    ]
    assert any("carries the hash of the one pre-registered" in p for p in verdict["problems"])


def test_the_rerun_sample_is_unpredictable_unless_a_seed_is_given(package, tmp_path):
    # the start of the honest notebook: two experiments, no held-out round
    assert [e["kind"] for e in package.entries[:4]] == ["header", "round", "eval", "eval"]
    path = tmp_path / "search.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in package.entries if e["index"] <= 4))
    first, second = (Designer.verify_notebook(str(path), sample=0, workers=1) for _ in range(2))
    assert first["ok"] and first["sample_seed"] != second["sample_seed"] and len(first["sample_seed"]) == 32
    fixed = [Designer.verify_notebook(str(path), sample=1, seed="again", workers=1) for _ in range(2)]
    assert fixed[0]["ok"] and fixed[0]["sample_seed"] == "again"
    assert fixed[0]["rerun_cells"] == fixed[1]["rerun_cells"] and len(fixed[0]["rerun_cells"]) == 1
    everything = Designer.verify_notebook(str(path), sample=None, workers=1)
    assert everything["ok"] and everything["rerun_cells"] == [[3, 0], [4, 0]]


def test_grinding_the_head_moves_the_worlds_unless_a_beacon_decides_them():
    """Doctoring 3: throw-away notes before the pre-registration re-roll the held-out worlds."""
    def preregistered(note, beacon=None, anchor=None):
        notebook = Notebook()
        designer = Designer("grinder", notebook, workers=1, seed="g", world_nonce="n")
        notebook.add("note", {"note": note})
        return notebook, designer.preregister(TINY, n_worlds=2, rounds=2, beacon=beacon, anchor=anchor)["worlds"]

    # Without a beacon every note gives another set of worlds, and each set can be computed (and its
    # value tried) before anything is committed: the same notebook always gives the same worlds.
    drafts = [preregistered(f"housekeeping {n}")[1] for n in range(6)]
    assert len({tuple(worlds) for worlds in drafts}) == 6
    again, worlds_again = preregistered("housekeeping 3")
    assert worlds_again == drafts[3]
    assert worlds_again == [f"heldout:{again.of_kind('preregistration')[0]['hash'][:20]}:{k}" for k in range(2)]

    # With a beacon the notebook no longer decides.  When the pre-registration is written and published it
    # names a round, and that round's randomness has not been asked for yet.
    one, two = FakeBeacon(seed="one"), FakeBeacon(seed="two")
    public = []

    def anchor(notebook):
        public.append((notebook.entries[-1]["kind"], notebook.entries[-1]["body"]["beacon"], list(one.revealed)))

    notebook_one, worlds_one = preregistered("housekeeping 3", one, anchor)
    notebook_two, worlds_two = preregistered("housekeeping 3", two)
    assert public == [("preregistration", {"source": "fake", "round": 1}, [])] and one.revealed == [1]
    assert [e["kind"] for e in notebook_one.entries[-3:]] == ["preregistration", "beacon", "heldout_worlds"]

    # Two beacons that disagree about round 1 give different worlds for the very same committed notebook,
    # so nothing known at pre-registration determines them and grinding notes has nothing to aim at.
    committed = notebook_one.of_kind("preregistration")[0]
    assert committed == notebook_two.of_kind("preregistration")[0]
    assert worlds_one != worlds_two and drafts[3] not in (worlds_one, worlds_two)
    randomness = notebook_one.of_kind("beacon")[0]["body"]["randomness"]
    assert worlds_one == [f"heldout:{committed['hash'][:20]}:{randomness[:20]}:{k}" for k in range(2)]


def test_preregistration_stops_if_the_notebook_cannot_be_published():
    notebook, beacon = Notebook(), FakeBeacon()
    designer = Designer("test", notebook, workers=1)

    def unreachable(_notebook):
        raise RuntimeError("push rejected")

    with pytest.raises(RuntimeError, match="push rejected"):
        designer.preregister(TINY, n_worlds=2, rounds=2, beacon=beacon, anchor=unreachable)
    assert notebook.entries[-1]["kind"] == "preregistration"
    assert beacon.revealed == [] and not notebook.of_kind("heldout_worlds")
    # the stranded pre-registration cannot be quietly replaced by another one
    with pytest.raises(ValueError, match="has not been evaluated"):
        designer.preregister(TINY, n_worlds=2, rounds=2, beacon=beacon)


@pytest.fixture
def pending(tmp_path):
    """A notebook whose pre-registration, with a beacon and a published copy, awaits its held-out run.
    Checking it needs no simulation."""
    path = str(tmp_path / "notebook.jsonl")
    notebook = Notebook(path)
    designer = Designer("test", notebook, workers=1)
    # a validation result recorded by hand (nothing here is re-run) and the selection it supports
    notebook.add("eval", {"stage": "validation", "design_hash": TINY.hash(), "design": TINY.to_dict(),
                          "scenarios": ["validate:by-hand:0"], "rounds": 2, "values": [0.5], "mean": 0.5,
                          "board_heads": ["0" * 64], "metrics_mean": {}})
    notebook.add("selection", {"candidates": [{"design_hash": TINY.hash(), "validation_mean": 0.5}], "chosen": TINY.hash()})
    published = str(tmp_path / "published.jsonl")

    def anchor(nb):
        shutil.copyfile(nb.path, published)
        return {"kind": "test", "copy": "published.jsonl"}

    designer.preregister(TINY, n_worlds=2, rounds=2, beacon=FakeBeacon(seed="public"), anchor=anchor)
    return SimpleNamespace(path=path, notebook=notebook, published=published, tmp_path=tmp_path)


def test_beacon_randomness_is_checked_against_the_beacon_itself(pending):
    def verify(**options):
        return Designer.verify_notebook(pending.path, sample=0, workers=1, **options)

    # by default nothing is fetched: the notebook passes, with a warning that the beacon was not checked
    unchecked = verify()
    assert unchecked["ok"] and unchecked["checks"]["beacon"]["status"] == "passed"
    assert unchecked["checks"]["beacon_randomness"]["status"] == "not run"
    assert any("beacon not checked" in w for w in unchecked["warnings"])
    assert any("'fake' is not a public randomness beacon" in w for w in unchecked["warnings"])
    assert not any("depend only on the notebook head" in w for w in unchecked["warnings"])

    assert verify(check_beacon=True, beacons={"fake": FakeBeacon(seed="public")})["ok"]

    # the beacon published something else for that round: the recorded randomness was made up
    forged = verify(check_beacon=True, beacons={"fake": FakeBeacon(seed="what the beacon really said")})
    assert failed(forged["checks"]) == {"beacon_randomness"}

    # a beacon that cannot be reached is "not checked", which fails only because the check was asked for
    class Unreachable:
        def reveal(self, commitment):
            raise BeaconError("could not read https://beacon.example/public/1: network is down")

    down = verify(check_beacon=True, beacons={"fake": Unreachable()})
    assert failed(down["checks"]) == {"beacon_randomness"}
    reveal = pending.notebook.of_kind("beacon")[0]["index"]
    assert any(p.startswith(f"beacon_randomness: entry {reveal}: beacon not checked: could not read") for p in down["problems"])
    unknown = verify(check_beacon=True)  # the verifier only asks beacons it knows to be public
    assert any("beacon not checked: 'fake' is not a beacon this verifier can ask" in p for p in unknown["problems"])


def test_a_reveal_that_does_not_match_the_commitment_is_rejected(pending, tmp_path):
    def relinked(change):
        entries = copy.deepcopy(pending.notebook.entries)
        change(entries)
        forged = Notebook(str(tmp_path / "forged.jsonl"))
        for entry in entries:
            forged.add(entry["kind"], entry["body"])
        check = Designer.verify_notebook(forged.path, sample=0, workers=1)
        os.remove(forged.path)
        return check

    def another_round(entries):  # reveal a round other than the committed one
        entries[5]["body"]["round"] = 7

    def chosen_worlds(entries):  # keep the randomness, list worlds that do not follow from it
        entries[6]["body"]["worlds"] = heldout_world_seeds("0" * 64, "f" * 64, 2)

    def no_reveal(entries):  # drop the beacon entry and derive the worlds from the notebook alone
        del entries[5]
        entries[5]["body"]["worlds"] = [f"heldout:{pending.notebook.entries[4]['hash'][:20]}:{k}" for k in range(2)]

    assert [e["kind"] for e in pending.notebook.entries[4:]] == ["preregistration", "beacon", "heldout_worlds"]
    assert failed(relinked(another_round)["checks"]) == {"beacon"}
    assert failed(relinked(chosen_worlds)["checks"]) == {"heldout_worlds"}
    assert failed(relinked(no_reveal)["checks"]) == {"beacon", "heldout_worlds"}


def test_a_notebook_rewritten_after_publication_fails_the_anchor_check(pending, tmp_path):
    worlds = pending.notebook.of_kind("heldout_worlds")[0]["body"]
    assert worlds["anchor"] == {"kind": "test", "copy": "published.jsonl"}
    unchecked = Designer.verify_notebook(pending.path, sample=0, workers=1)
    assert unchecked["checks"]["anchor"]["status"] == "not run"
    assert any("anchor not checked" in w and "published.jsonl" in w for w in unchecked["warnings"])
    assert Designer.verify_notebook(pending.path, sample=0, workers=1, anchored=[pending.published])["ok"]

    # After the randomness is public, an entry is slipped in before the pre-registration and the worlds are
    # re-derived for the new chain.  The result is a valid notebook; only the published copy shows the rewrite.
    entries = copy.deepcopy(pending.notebook.entries)
    rewritten = Notebook(str(tmp_path / "rewritten.jsonl"))
    for entry in entries[:4] + [{"kind": "note", "body": {"note": "housekeeping"}}] + entries[4:6]:
        rewritten.add(entry["kind"], entry["body"])
    prereg, beacon = rewritten.entries[-2:]
    rewritten.add("heldout_worlds", {**worlds, "worlds": heldout_world_seeds(prereg["hash"], beacon["body"]["randomness"], 2)})
    assert Designer.verify_notebook(rewritten.path, sample=0, workers=1)["ok"]
    exposed = Designer.verify_notebook(rewritten.path, sample=0, workers=1, anchored=[pending.published])
    assert failed(exposed["checks"]) == {"anchor"}
    assert any("was rewritten after that copy was published" in p for p in exposed["problems"])


def test_an_abandoned_preregistration_fails_verification(pending):
    notebook = pending.notebook
    assert Designer.verify_notebook(pending.path, sample=0, workers=1)["ok"]  # the last one may still be pending
    first = notebook.of_kind("preregistration")[0]
    again = notebook.add("preregistration", {**first["body"], "beacon": None})
    notebook.add("heldout_worlds", {"anchor": None, "worlds": heldout_world_seeds(again["hash"], None, 2)})
    check = Designer.verify_notebook(pending.path, sample=0, workers=1)
    assert failed(check["checks"]) == {"preregistration"}
    assert f"preregistration: entry {first['index']}: was abandoned without a held-out report, " \
           "which is how an unfavourable draw of worlds would be discarded" in check["problems"]


def test_drand_beacon_commits_ahead_and_waits_for_its_round():
    state = {"now": 0.0, "asked": []}

    def latest():
        return 100 + int(state["now"] // 30)  # one round every 30 seconds

    def fetch(url):
        state["asked"].append(url)
        number = latest() if url.endswith("/latest") else int(url.rsplit("/", 1)[1])
        assert number <= latest(), "a round that does not exist yet was requested"
        return {"round": number, "randomness": f"{number:064x}", "signature": "..."}

    def sleep(seconds):
        state["now"] += seconds

    beacon = DrandBeacon(fetch=fetch, clock=lambda: state["now"], sleep=sleep)
    commitment = beacon.commit()
    assert commitment == {"source": "drand", "round": 102}  # the latest round plus two: not produced yet
    assert beacon.reveal(commitment) == f"{102:064x}"
    assert 60 <= state["now"] < 90  # it waited for the round instead of settling for an earlier one
    assert all(url.startswith("https://api.drand.sh/public/") for url in state["asked"])

    state["now"] = 3000.0  # long after: a verifier asks for the same round and gets the same randomness
    assert beacon.reveal({"source": "drand", "round": 102}) == f"{102:064x}"
    assert state["asked"][-1] == "https://api.drand.sh/public/102"

    stuck = DrandBeacon(fetch=lambda url: {"round": 7, "randomness": "ab" * 32}, clock=lambda: state["now"], sleep=sleep, timeout=90)
    with pytest.raises(BeaconError, match="drand round 9 could not be read within 90 seconds .the latest round is 7."):
        stuck.reveal(stuck.commit())

    def down(url):
        raise OSError("network is unreachable")

    with pytest.raises(BeaconError, match="could not read https://api.drand.sh/public/latest"):
        DrandBeacon(fetch=down).commit()
    with pytest.raises(BeaconError, match="not a commitment to a round of the 'drand' beacon"):
        beacon.reveal({"source": "fake", "round": 1})


def test_git_anchor_commits_and_pushes_the_notebook_or_stops_the_run(tmp_path):
    run_designer = load_script("run_designer")
    commands = []

    def run(command, cwd, capture_output, text):
        commands.append((command, cwd))
        answers = {"rev-parse HEAD": "c0ffee\n", "rev-parse --show-prefix": "contest/\n"}
        return SimpleNamespace(returncode=0, stdout=answers.get(" ".join(command[1:]), ""), stderr="")

    notebook = Notebook(str(tmp_path / "notebook.jsonl"))
    designer = Designer("test", notebook, workers=1)
    worlds = designer.preregister(TINY, n_worlds=2, rounds=2, beacon=FakeBeacon(),
                                  anchor=run_designer.git_anchor(str(tmp_path), run=run))
    committed = notebook.of_kind("preregistration")[0]
    assert [command for command, _ in commands[:3]] == [
        ["git", "add", "notebook.jsonl"],
        ["git", "commit", "-m", f"Pre-register design {TINY.hash()} (notebook head {committed['hash'][:12]})", "--", "notebook.jsonl"],
        ["git", "push"],
    ]
    assert {cwd for _, cwd in commands} == {str(tmp_path)}
    assert worlds["anchor"] == {"kind": "git", "commit": "c0ffee", "path": "contest/notebook.jsonl"}

    def rejected(command, **_):
        return SimpleNamespace(returncode=int(command[1] == "push"), stdout="", stderr="remote: permission denied\n")

    stranded, beacon = Notebook(str(tmp_path / "other" / "notebook.jsonl")), FakeBeacon()
    with pytest.raises(run_designer.AnchorFailed, match="`git push` failed in .*: remote: permission denied"):
        Designer("test", stranded, workers=1).preregister(
            TINY, n_worlds=2, rounds=2, beacon=beacon, anchor=run_designer.git_anchor(str(tmp_path / "other"), run=rejected))
    assert beacon.revealed == [] and not stranded.of_kind("heldout_worlds")


def test_a_push_that_outlasts_the_beacon_round_stops_the_run(tmp_path):
    run_designer = load_script("run_designer")
    state = {"now": 0.0}

    def fetch(url):
        number = 100 + int(state["now"] // 30) if url.endswith("/latest") else int(url.rsplit("/", 1)[1])
        return {"round": number, "randomness": f"{number:064x}"}

    def push(seconds):
        def anchor(notebook):
            state["now"] += seconds
            return {"kind": "test"}
        return anchor

    def preregister(name, seconds):
        notebook = Notebook(str(tmp_path / name / "notebook.jsonl"))
        beacon = DrandBeacon(fetch=fetch, clock=lambda: state["now"], sleep=lambda s: state.update(now=state["now"] + s))
        anchor = run_designer.published_before_round(push(seconds), beacon)
        return notebook, lambda: Designer("test", notebook, workers=1).preregister(TINY, 2, 2, beacon=beacon, anchor=anchor)

    notebook, run = preregister("quick", seconds=10)  # public well before the committed round
    assert run()["anchor"] == {"kind": "test"}
    assert notebook.of_kind("beacon")[0]["body"]["round"] == notebook.of_kind("preregistration")[0]["body"]["beacon"]["round"]

    notebook, run = preregister("slow", seconds=90)  # the committed round came and went during the push
    with pytest.raises(run_designer.AnchorFailed, match="was produced before the notebook had been published"):
        run()
    assert notebook.entries[-1]["kind"] == "preregistration"  # no randomness was read, no worlds were derived


def test_refinement_candidates_follow_the_winning_ablations():
    def report(**winners):
        means = {**{name: 0.4 for name in ABLATIONS}, **winners}
        return {"arms": {"chosen": {"mean": 0.5}, **{f"ablation:{name}": {"mean": mean} for name, mean in means.items()}}}

    assert [(recipe, design.hash()) for recipe, design in refinement_candidates(TINY, report())] == [((), TINY.hash())]

    # an arm that only ties the chosen design is not a winner
    candidates = refinement_candidates(TINY, report(striving_off=0.6, estimate_unshaded=0.7, niche_off=0.5))
    assert [recipe for recipe, _ in candidates] == [
        (), ("striving_off",), ("estimate_unshaded",), ("striving_off", "estimate_unshaded"),
    ]
    both = candidates[-1][1]
    assert both.policy.trials_per_round == 0 and both.shade == 0.0
    assert both == ABLATIONS["estimate_unshaded"](ABLATIONS["striving_off"](TINY))

    # changes that leave the design as it is do not make new candidates
    unshaded = replace(TINY, shade=0.0)
    assert [recipe for recipe, _ in refinement_candidates(unshaded, report(striving_off=0.6, estimate_unshaded=0.7))] == [
        (), ("striving_off",),
    ]
    # every winning ablation alone and every pair of them, without repeating a design two recipes both reach
    everything = refinement_candidates(TINY, report(**{name: 0.9 for name in ABLATIONS}))
    n = len(ABLATIONS)
    assert 1 + n <= len(everything) <= 1 + n + n * (n - 1) // 2
    assert len({design.hash() for _recipe, design in everything}) == len(everything)
    assert {recipe for recipe, _design in everything if len(recipe) == 1} == {(name,) for name in ABLATIONS}


def test_two_round_refinement_with_a_beacon_verifies(tmp_path):
    path = str(tmp_path / "notebook.jsonl")
    beacon, published = FakeBeacon(seed="public"), []

    def anchor(notebook):
        published.append(str(tmp_path / f"published-{len(notebook.entries)}.jsonl"))
        shutil.copyfile(notebook.path, published[-1])
        return {"kind": "test", "copy": os.path.basename(published[-1])}

    designer = Designer("test", Notebook(path), workers=1, seed="t")
    finalists = designer.assess([TINY, replace(TINY, shade=0.5)], ["train:t:0"], 1, "rung2", 0)
    first = designer.validate(finalists, n_worlds=1, rounds=1)
    designer.preregister(first, n_worlds=2, rounds=1, beacon=beacon, anchor=anchor)
    first_report = designer.heldout(first)
    second, second_report = designer.refine(first_report, n_validation_worlds=1, n_worlds=2, rounds=1, beacon=beacon, anchor=anchor)

    notebook = designer.notebook
    commitments, reveals, worlds = (notebook.of_kind(kind) for kind in ("preregistration", "beacon", "heldout_worlds"))
    assert [e["body"]["design_hash"] for e in commitments] == [first.hash(), second.hash()]
    assert [e["body"]["beacon"]["round"] for e in commitments] == [e["body"]["round"] for e in reveals] == [1, 2]
    assert not set(worlds[0]["body"]["worlds"]) & set(worlds[1]["body"]["worlds"])  # new worlds for the new round

    # the second round says what it was built on, and that those worlds are no longer test data
    markers = notebook.of_kind("round")
    assert [m["body"]["round"] for m in markers] == [1, 2] and markers[0]["body"]["spent_worlds"] == []
    assert markers[1]["body"]["spent_worlds"] == worlds[0]["body"]["worlds"]
    assert "spent as test data" in markers[1]["body"]["note"]
    expected = refinement_candidates(first, first_report)
    assert markers[1]["body"]["candidates"] == [
        {"design_hash": design.hash(), "ablations_applied": list(recipe)} for recipe, design in expected
    ]
    assert second.hash() in {design.hash() for _, design in expected}

    # its candidates were compared on validation worlds no earlier round used, and the pick was pre-registered
    validation = [e for e in notebook.of_kind("eval") if e["body"]["stage"] == "validation" and e["index"] > markers[1]["index"]]
    assert [e["body"]["design_hash"] for e in validation] == [design.hash() for _, design in expected]
    assert all(len(e["body"]["scenarios"]) == 1 and e["body"]["scenarios"][0].startswith("validate2:t:") for e in validation)
    assert notebook.of_kind("selection")[-1]["body"]["chosen"] == second.hash()
    assert second_report["design_hash"] == second.hash() and second_report["arms"]["chosen"]["chosen_minus_arm"] == 0.0

    history = [(first, first_report), (second, second_report)]
    check = Designer.verify_notebook(
        path, sample=2, workers=1, submission=submission_for(second, second_report, notebook.head, history),
        check_beacon=True, beacons={"fake": FakeBeacon(seed="public")}, anchored=published,
    )
    assert check["ok"], check["problems"]
    assert len(check["preregistrations"]) == 2 and failed(check["checks"]) == set()
    assert {c["status"] for c in check["checks"].values()} == {"passed"}  # every check ran
    chosen_entries = [heldout_entries(notebook.entries)[k]["index"] for k in (0, len(second_report["arms"]))]
    assert [cell for cell in check["rerun_cells"] if cell[0] in chosen_entries] == [[e, k] for e in chosen_entries for k in (0, 1)]

    # the first round's worlds informed the second, so the first round's result cannot be the claim
    stale = Designer.verify_notebook(path, sample=0, workers=1, submission=submission_for(first, first_report, notebook.head, history))
    assert failed(stale["checks"]) == {"submission"}
    assert any(p.startswith("submission: does not state the recomputed held-out numbers: ") for p in stale["problems"])
    if first.hash() != second.hash():
        assert any(f"submits design {first.hash()}, but the last pre-registration" in p for p in stale["problems"])


# ------------------------------------------------- findings of the design review that concern the designer

def test_flipping_how_memory_designs_are_assigned_leaves_scoring_beliefs_alone():
    from memarena.designer.design import build_seed

    a = Design(init="random").build(build_seed("w"))
    b = Design(init="covering").build(build_seed("w"))
    assert [dict(x.criterion_weights) for x in a] == [dict(x.criterion_weights) for x in b]
    assert [x.memory for x in a] != [x.memory for x in b]
    assert "w" != build_seed("w") and build_seed("w") != build_seed("w2")


def test_a_proposed_design_is_brought_inside_the_search_bounds():
    from memarena.designer.design import clamp_design

    wild = replace(TINY, n_units=0, shade=float("nan"), policy=replace(TINY.policy, dev_streams=0, explore=9.0, trials_per_round=50))
    tame = clamp_design(wild)
    assert tame.n_units == 4 and tame.shade == 1.0
    assert tame.policy.dev_streams == 3 and tame.policy.explore == 0.6 and tame.policy.trials_per_round == 4
    assert clamp_design(TINY) == TINY


def test_the_societys_accounts_do_not_read_private_unit_state():
    """Champions, estimate and value are the same when recomputed from a copy of the board alone."""
    from memarena.designer.design import build_seed
    from memarena.feeds.board import Board
    from memarena.sim.collective import audit_record, collective_estimate, published_since, reputation, select_champions

    scenario = sample_scenario("public-only", rounds=4)
    sim = Simulation(scenario, TINY.build(build_seed("public-only"), scenario.rules))
    sim.run()
    reported = sim.collective()
    # an outside reader: only the posts and blobs, replayed onto a fresh board
    copy = Board()
    for post in sim.board.read(0):
        for h in post.body.get("blobs", []) if isinstance(post.body, dict) else []:
            copy.put_blob(sim.board.get_blob(h))
        copy.post(post.author, post.kind, post.body, to=post.to, round=post.round)
    assert copy.head() == sim.board.head()
    units = sorted(sim.units)
    audit = audit_record(copy, sim.family)
    rep = reputation(audit.signals, units, banned=audit.cheats, seeds=units)
    champions = select_champions(audit.signals, rep, units, published_since(copy, units),
                                 max_items=scenario.rules.exam_streams * scenario.rules.exam_questions)
    assert {c: i["units"] for c, i in champions.items()} == reported["champions"]
    # a unit that has adopted a design but not yet published it does not change the accounts
    for unit in sim.units.values():
        unit.version_round = 99
    assert sim.collective()["champions"] == reported["champions"]


def test_a_notebook_entry_cannot_carry_fields_outside_its_hash_or_duplicate_keys(tmp_path):
    nb = Notebook(str(tmp_path / "a.jsonl"))
    nb.add("header", {"designer": "x"})
    good = (tmp_path / "a.jsonl").read_text()
    entry = json.loads(good)
    for altered in ({**entry, "values": [0.99]}, {**entry, "index": True}, {**entry, "index": 1.0}):
        (tmp_path / "b.jsonl").write_text(json.dumps(altered) + "\n")
        with pytest.raises(ValueError):
            Notebook(str(tmp_path / "b.jsonl"))
    (tmp_path / "c.jsonl").write_text(good.replace('"kind": "header"', '"kind": "header", "kind": "header"'))
    with pytest.raises(ValueError, match="duplicate"):
        Notebook(str(tmp_path / "c.jsonl"))
    empty = tmp_path / "d.jsonl"
    empty.write_text("")
    assert not Designer.verify_notebook(str(empty), sample=0, workers=1)["ok"]
