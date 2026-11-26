import math

import pytest

from memarena.benchmark.schema import Answer, AnswerKey
from memarena.benchmark.synthetic import GenConfig, generate
from memarena.evalcore import metrics
from memarena.evalcore.audit import judge_exam, oracle_answers, run_audit
from memarena.evalcore.calibration import EXTERNAL_LOSS, PEER_LOSS, AsymmetricLoss
from memarena.evalcore.criteria import CriteriaFamily, Criterion, aggregate
from memarena.evalcore.grader import Grader
from memarena.evalcore.judges import HeuristicJudge, longmemeval_prompt
from memarena.evalcore.module import CapabilityTable, EvaluationModule, expected_score, verify_receipt
from memarena.evalcore.normalize import is_abstention, is_echo, is_noncommittal, numbers_in, tokens
from memarena.evalcore.standard import Standard
from memarena.evalcore.stats import bootstrap_ci, cluster_mean_se, mcnemar_exact, mean_se, paired_bootstrap, wilson_interval
from memarena.memory.base import run_system
from memarena.memory.heuristic import FEATURES, HeuristicMemory, MemoryConfig


def test_normalisation():
    assert tokens("The Mazda CX-5, 'Kraków' — three 4.0 1,200!") == ("mazda", "cx", "5", "krakow", "3", "4", "1200")
    assert numbers_in("3 then four, total 7") == [3.0, 4.0, 7.0]
    assert is_abstention("You never mentioned that.") and not is_abstention("") and not is_abstention("Lisbon")
    assert is_echo("Which city do I live in?", "which city do I live in") and not is_echo("Which city do I live in?", "Lisbon")
    assert is_noncommittal("It could be Lisbon, Porto or Valencia") and not is_noncommittal("I think it might be Lisbon")


def test_metrics_basics():
    assert metrics.exact_match("the Lisbon.", "Lisbon") == 1.0
    assert metrics.exact_match("", "") == 0.0
    assert metrics.contains("You live in Lisbon now", "Lisbon") == 1.0
    assert metrics.contains("117 days", "17") == 0.0  # token level, not substring
    assert metrics.token_f1("You live in Lisbon now", "Lisbon") == pytest.approx(1 / 3)
    assert metrics.token_f1("a b", ["x y", "a b"]) == 1.0
    assert metrics.rouge_l("w x y z", "w y z") == pytest.approx(2 * (3 / 4) * 1 / (3 / 4 + 1))
    assert metrics.bleu1("lisbon", "lisbon portugal") == pytest.approx(math.exp(1 - 2))
    assert metrics.list_recall("Peru and Nepal", ["Peru", "Nepal", "Japan"]) == pytest.approx(2 / 3)


def test_judges_follow_the_documented_rules():
    lenient, strict, generous, graded = (HeuristicJudge(m) for m in ("lenient", "strict", "generous", "graded"))
    days = AnswerKey("p", "temporal-reasoning", "18 days", ("18",), meta={"numeric": 18, "tolerance": 1})
    assert lenient.judge("q", days, "19 days").score == 1.0  # off by one is forgiven, as in LongMemEval
    assert strict.judge("q", days, "19 days").score == 0.0
    assert generous.judge("q", days, "20 days").score == 1.0
    assert graded.judge("q", days, "21 days").score == 0.5
    trips = AnswerKey("p", "multi-session", "Peru, Nepal and Japan", meta={"list": ["Peru", "Nepal", "Japan"]})
    assert lenient.judge("q", trips, "Peru and Nepal").score == 0.0  # a subset is not the answer
    assert generous.judge("q", trips, "Peru").score == 1.0
    assert graded.judge("q", trips, "Peru and Nepal").score == pytest.approx(2 / 3)
    unans = AnswerKey("p", "single-session-user", "never mentioned", is_abstention=True)
    for j in (lenient, strict, generous, graded):
        assert j.judge("q", unans, "You never told me that.").score == 1.0
        assert j.judge("q", unans, "Priya").score == 0.0
        assert j.judge("Which city do I live in?", AnswerKey("p", "single-session-user", "Lisbon"), "Which city do I live in?").score == 0.0


def test_official_prompt_is_selected_by_type():
    k = AnswerKey("p", "temporal-reasoning", "18 days")
    assert "off-by-one" in longmemeval_prompt("q", k, "r")
    k = AnswerKey("p", "knowledge-update", "x")
    assert "previous information along with an updated answer" in longmemeval_prompt("q", k, "r")
    k = AnswerKey("p", "single-session-preference", "x", meta={"rubric": "RUBRIC TEXT"})
    assert "Rubric: RUBRIC TEXT" in longmemeval_prompt("q", k, "r")
    k = AnswerKey("p", "multi-session", "x", is_abstention=True)
    assert "unanswerable question" in longmemeval_prompt("q", k, "r")


def test_aggregation_micro_vs_macro():
    rows = [("a", True, 1.0), ("a", True, 1.0), ("a", True, 1.0), ("b", True, 0.0), ("c", False, 1.0)]
    assert aggregate(Criterion("em", "detect", "micro"), rows) == pytest.approx(0.75)
    assert aggregate(Criterion("em", "detect", "macro"), rows) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        Criterion("judge_strict", "literal", "micro")


def test_family_prior_is_a_distribution():
    fam = CriteriaFamily.default()
    assert len(fam) == 52 and sum(fam.prior.values()) == pytest.approx(1.0)
    assert fam.prior["judge_lenient|detect|micro"] == max(fam.prior.values())


@pytest.fixture(scope="module")
def bench():
    return generate("t", GenConfig(n_streams=10, questions_per_stream=12))


def test_grader_oracle_null_and_statuses(bench):
    g = Grader(CriteriaFamily.default())
    oracle = g.grade(oracle_answers(bench.instances), bench.instances)
    assert all(s == pytest.approx(1.0) for cid, s in oracle.scores.items() if "|literal|" not in cid)
    empty = g.grade({i.probe_id: Answer("") for i in bench}, bench.instances)
    assert max(empty.scores.values()) == 0.0
    idk = g.grade({i.probe_id: Answer("I don't know.") for i in bench}, bench.instances)
    share = sum(i.key.is_abstention for i in bench) / len(bench)
    assert idk.score("judge_lenient|detect|micro") == pytest.approx(share)
    assert idk.score("judge_lenient|exclude|micro") == 0.0
    # an error row is never graded on its text, and is counted
    pid = bench.instances[0].probe_id
    answers = oracle_answers(bench.instances)
    answers[pid] = Answer(bench.instances[0].key.answer, status="error")
    rep = g.grade(answers, bench.instances)
    assert rep.n_not_answered == 1 and rep.score("em|detect|micro") < 1.0
    del answers[pid]
    assert g.grade(answers, bench.instances).n_not_answered == 1


def test_evaluation_ignores_who_is_asking(bench):
    """The module has no identity input: same answers, same result, always."""
    std = Standard({"judge_lenient|detect|micro": 1.0})
    module = EvaluationModule(std)
    answers = run_system(HeuristicMemory(MemoryConfig(features=frozenset(FEATURES))), bench.probes())
    a = module.evaluate(answers, bench.instances, "salt")
    b = EvaluationModule(std).evaluate(dict(reversed(list(answers.items()))), bench.instances, "salt")
    assert a.receipt == b.receipt
    import inspect

    assert set(inspect.signature(EvaluationModule.evaluate).parameters) == {"self", "answers", "instances", "salt"}


def test_receipts_detect_every_kind_of_tampering(bench):
    module = EvaluationModule(Standard({"judge_lenient|detect|micro": 0.6, "f1|detect|macro": 0.4}))
    answers = run_system(HeuristicMemory(MemoryConfig(features=frozenset(FEATURES))), bench.probes())
    ev = module.evaluate(answers, bench.instances, "salt")
    assert verify_receipt(ev.receipt, answers, bench.instances, "salt") == (True, "ok")
    lowered = dict(ev.receipt, score=ev.score - 0.05)
    assert not verify_receipt(lowered, answers, bench.instances, "salt")[0]
    assert not verify_receipt(ev.receipt, answers, bench.instances, "other-salt")[0]
    changed = dict(answers)
    pid = bench.instances[3].probe_id
    changed[pid] = Answer(answers[pid].text + " extra")
    assert not verify_receipt(ev.receipt, changed, bench.instances, "salt")[0]
    other_standard = dict(ev.receipt, standard=Standard({"em|detect|micro": 1.0}).to_dict())
    assert not verify_receipt(other_standard, answers, bench.instances, "salt")[0]
    cap = dict(ev.receipt["capability"])
    first = sorted(cap)[0]
    cap[first] = [cap[first][0] - 0.2, cap[first][1]]
    assert not verify_receipt(dict(ev.receipt, capability=cap), answers, bench.instances, "salt")[0]


def test_profile_predicts_exam_scores_without_bias():
    std = Standard({"judge_lenient|detect|micro": 0.7, "f1|detect|macro": 0.3},
                   {"temporal-reasoning": 0.5, "multi-session": 0.3, "abstention": 0.2})
    module = EvaluationModule(std)
    mem = HeuristicMemory(MemoryConfig(features=frozenset(["temporal", "aggregate", "third_party", "expansion", "negation"]), top_k=10))
    hold = generate("hold", std.gen_config(16, 12, balanced=True))
    table = module.evaluate(run_system(mem, hold.probes()), hold.instances, "s").table
    errors = []
    for k in range(20):
        exam = generate(f"exam{k}", std.gen_config(3, 12))
        got = module.evaluate(run_system(mem, exam.probes()), exam.instances, "s").score
        errors.append(expected_score(table, std.criterion_weights, std.capability_weights) - got)
    assert abs(sum(errors) / len(errors)) < 0.05


def test_asymmetric_loss_and_best_report():
    assert PEER_LOSS(0.8, 0.6) == pytest.approx(1.0 * 0.04) and PEER_LOSS(0.4, 0.6) == pytest.approx(0.25 * 0.04)
    assert EXTERNAL_LOSS(0.8, 0.6) < EXTERNAL_LOSS(0.4, 0.6)
    assert PEER_LOSS.best_report(0.6, 0.08) < 0.6 < EXTERNAL_LOSS.best_report(0.6, 0.08)
    assert PEER_LOSS.best_report(0.6, 0.0) == 0.6
    assert PEER_LOSS.best_report(0.6, 0.08, shade=0.0) == 0.6
    sym = AsymmetricLoss(1.0, 1.0)
    assert sym.best_report(0.6, 0.1) == 0.6
    # the report really is the minimiser
    best = PEER_LOSS.best_report(0.6, 0.08)
    assert PEER_LOSS.expected(best, 0.6, 0.08) <= min(PEER_LOSS.expected(best + d, 0.6, 0.08) for d in (-0.02, 0.02)) + 1e-9
    quartic = AsymmetricLoss(0.25, 1.0, power=4)
    assert quartic.best_report(0.6, 0.08) > 0.6


def test_statistics():
    m, se = mean_se([0, 1, 1, 1])
    assert m == 0.75 and se == pytest.approx(0.25)
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2
    xs = [1, 1, 1, 1, 0, 0, 0, 0]
    clusters = ["a"] * 4 + ["b"] * 4
    assert cluster_mean_se(xs, clusters)[1] > mean_se(xs)[1]  # correlated items carry less information
    lo, hi = bootstrap_ci(xs, clusters, n_boot=400, seed=1)
    assert lo == 0.0 and hi == 1.0
    res = paired_bootstrap([1, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0], n_boot=500, seed=1)
    assert res["diff"] == pytest.approx(0.5) and res["p_better"] > 0.9
    assert mcnemar_exact([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]) == pytest.approx(2 / 64)
    assert mcnemar_exact([1, 0], [1, 0]) == 1.0


def test_audit_passes_on_synthetic_benchmark(bench):
    report = run_audit(bench, HeuristicMemory(MemoryConfig(features=frozenset(FEATURES), top_k=10)))
    assert report.passed, report.failures()


def test_judge_exam_on_fresh_cases():
    for mode, target in (("lenient", "official"), ("strict", "strict")):
        res = judge_exam(HeuristicJudge(mode), target, split="fresh")
        assert res["n"] >= 16 and res["accuracy"] >= 0.85 and res["errors"] == 0


def test_capability_table_roundtrip(bench):
    g = Grader(CriteriaFamily.default())
    table = CapabilityTable.from_report(g.grade(oracle_answers(bench.instances), bench.instances))
    again = CapabilityTable.from_dict(table.to_dict())
    assert again.cells == table.cells and again.counts == table.counts
    assert sum(table.counts.values()) == len(bench)
