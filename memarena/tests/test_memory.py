import pytest

from memarena.benchmark.schema import Answer, Probe
from memarena.benchmark.synthetic import GenConfig, generate
from memarena.evalcore.criteria import CriteriaFamily
from memarena.evalcore.grader import Grader
from memarena.memory.base import ABSTAIN_TEXT, run_system
from memarena.memory.heuristic import FEATURES, HeuristicMemory, MemoryConfig, classify_question, observable_class, stem
from memarena.memory.llm_memory import LLMMemory
from memarena.llm.client import LLMResult, MockLLM

CRIT = "judge_lenient|detect|micro"


@pytest.fixture(scope="module")
def bench():
    return generate("mem", GenConfig(n_streams=25, questions_per_stream=12))


def per_type(bench, features, **kw):
    g = Grader(CriteriaFamily.single(CRIT))
    answers = run_system(HeuristicMemory(MemoryConfig(features=frozenset(features), **kw)), bench.probes())
    rep = g.grade(answers, bench.instances)
    return rep.score(CRIT), {t: v[0] for t, v in rep.per_type(CRIT).items()}


def test_each_feature_moves_the_ability_it_is_for(bench):
    _, base = per_type(bench, [])
    assert per_type(bench, ["temporal"])[1]["temporal-reasoning"] > base["temporal-reasoning"] + 0.3
    assert per_type(bench, ["aggregate"], top_k=10)[1]["multi-session"] > base["multi-session"] + 0.15
    assert per_type(bench, ["assistant"])[1]["single-session-assistant"] > base["single-session-assistant"] + 0.3
    assert per_type(bench, ["comparative"])[1]["single-session-preference"] > base["single-session-preference"] + 0.3
    assert per_type(bench, ["recency", "expansion"])[1]["knowledge-update"] > base["knowledge-update"] + 0.2
    full, _ = per_type(bench, FEATURES, top_k=10)
    none, _ = per_type(bench, [])
    assert full > none + 0.4


def test_stemmer_and_question_classes():
    assert stem("lives") == stem("live") == stem("living")
    assert stem("countries") == "country" and stem("studied") == stem("study")
    assert classify_question("How many days ago did I renew my passport?") == "days_ago"
    assert classify_question("How many succulents have I bought in total?") == "sum"
    assert classify_question("Which books have I finished reading?") == "list"
    assert classify_question("Which board game did you recommend to me?") == "assistant"
    assert observable_class("What is my job?") == "fact"


def test_without_history_the_system_declines(bench):
    probe = bench.probes()[0]
    empty = Probe(probe.probe_id, probe.question, probe.question_date, "empty", ())
    assert HeuristicMemory(MemoryConfig(features=frozenset(FEATURES))).answer(empty).text == ABSTAIN_TEXT


def test_harness_turns_exceptions_into_error_rows(bench):
    class Broken:
        name = "broken"

        def answer(self, probe):
            raise RuntimeError("boom")

        def config_hash(self):
            return "x"

    out = run_system(Broken(), bench.probes()[:3])
    assert all(a.status == "error" and a.text == "" for a in out.values())
    g = Grader(CriteriaFamily.single(CRIT))
    rep = g.grade(out, bench.instances[:3])
    assert rep.n_not_answered == 3


def test_config_roundtrip_and_validation():
    cfg = MemoryConfig(features=frozenset(["temporal", "relative"]), top_k=8, style="sentence")
    assert MemoryConfig.from_dict(cfg.to_dict()).hash() == cfg.hash()
    with pytest.raises(ValueError):
        MemoryConfig(features=frozenset(["telepathy"]))


def test_llm_memory_prompts_and_statuses(bench):
    probe = bench.probes()[0]
    seen = {}

    def responder(req):
        seen.update(req)
        return {"answer": "Lisbon", "confidence": 0.9}

    mem = LLMMemory(MockLLM(responder), mode="retrieve", top_k_sessions=4)
    ans = mem.answer(probe)
    assert ans.text == "Lisbon" and ans.confidence == 0.9 and 1 <= ans.meta["sessions_used"] <= 4
    assert probe.question in seen["prompt"] and probe.question_date in seen["prompt"]
    assert seen["json_schema"]["required"] == ["answer", "confidence"]

    full = LLMMemory(MockLLM(responder), mode="full_context")
    full.answer(probe)
    assert "Conversation history" in seen["system"]  # shared history goes in the cacheable system prompt

    refused = LLMMemory(MockLLM(lambda r: LLMResult(text="", status="refused")))
    assert refused.answer(probe).status == "refused"
    broken = LLMMemory(MockLLM(lambda r: LLMResult(text="oops", status="error")))
    assert broken.answer(probe).status == "error"
    blank = LLMMemory(MockLLM(lambda r: {"answer": "", "confidence": "n/a"}))
    assert blank.answer(probe).text == ABSTAIN_TEXT
