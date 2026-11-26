import dataclasses
import json

from memarena.benchmark.adapters import load_locomo, load_longmemeval
from memarena.benchmark.schema import QUESTION_TYPES, Probe, parse_date
from memarena.benchmark.synthetic import GenConfig, generate


def test_generation_is_deterministic_and_seed_dependent():
    cfg = GenConfig(n_streams=3, questions_per_stream=10)
    a, b, c = generate(5, cfg), generate(5, cfg), generate(6, cfg)
    assert a.manifest_hash() == b.manifest_hash()
    assert a.manifest_hash() != c.manifest_hash()


def test_probe_is_structurally_free_of_gold():
    names = {f.name for f in dataclasses.fields(Probe)}
    assert names == {"probe_id", "question", "question_date", "haystack_id", "sessions"}
    bench = generate(1, GenConfig(n_streams=3))
    for inst in bench:
        blob = json.dumps(inst.probe.to_dict())
        assert "is_abstention" not in blob and "question_type" not in blob and "evidence" not in blob
        assert inst.source_id not in blob  # ids are opaque
        for t in QUESTION_TYPES:
            assert t not in inst.probe.probe_id


def test_every_question_type_and_abstention_appear():
    bench = generate(2, GenConfig(n_streams=30, questions_per_stream=12))
    assert set(bench.type_counts()) == set(QUESTION_TYPES)
    assert sum(i.key.is_abstention for i in bench) > 0


def test_split_keeps_histories_together_and_ignores_scores():
    bench = generate(3, GenConfig(n_streams=12, questions_per_stream=8))
    parts = bench.split({"dev": 0.5, "test": 0.5}, seed="s")
    dev = {i.probe.haystack_id for i in parts["dev"]}
    test = {i.probe.haystack_id for i in parts["test"]}
    assert dev and test and not (dev & test)
    assert len(parts["dev"]) + len(parts["test"]) == len(bench)
    again = bench.split({"dev": 0.5, "test": 0.5}, seed="s")
    assert [i.probe_id for i in again["dev"]] == [i.probe_id for i in parts["dev"]]


def test_temporal_gold_matches_session_dates():
    bench = generate(4, GenConfig(n_streams=30, questions_per_stream=12, relative_date_rate=0.0))
    checked = 0
    for inst in bench:
        if inst.probe.question.startswith("How many days ago") and not inst.key.is_abstention:
            ev = [s for s in inst.probe.sessions if s.session_id in inst.key.evidence_session_ids]
            assert len(ev) == 1
            days = (parse_date(inst.probe.question_date) - parse_date(ev[0].date)).days
            assert inst.key.meta["numeric"] == days
            checked += 1
    assert checked > 0


def test_parse_date_formats():
    assert parse_date("2023-05-30").isoformat() == "2023-05-30"
    assert parse_date("2023/05/30 (Tue) 23:40").isoformat() == "2023-05-30"
    assert parse_date("1:56 pm on 8 May, 2023").isoformat() == "2023-05-08"
    assert parse_date("nonsense") is None


def test_longmemeval_adapter_strips_leaks(tmp_path):
    rows = [
        {
            "question_id": "abc123_abs", "question_type": "temporal-reasoning", "question": "When did I go?",
            "answer": "You did not mention this information.", "question_date": "2023/05/30 (Tue) 23:40",
            "haystack_session_ids": ["answer_1", "noise_2"], "haystack_dates": ["2023/05/20 (Sat) 10:00", "2023/05/01 (Mon) 09:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "hi", "has_answer": True}, {"role": "assistant", "content": "hello"}],
                [{"role": "user", "content": "other"}],
            ],
            "answer_session_ids": ["answer_1"],
        },
        {
            "question_id": "q2", "question_type": "single-session-preference", "question": "Suggest a film?",
            "answer": "The user would prefer documentaries.", "question_date": "2023/06/01 (Thu) 08:00",
            "haystack_session_ids": ["s1"], "haystack_dates": ["2023/05/02 (Tue) 09:00"],
            "haystack_sessions": [[{"role": "user", "content": "I like documentaries"}]], "answer_session_ids": ["s1"],
        },
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(rows))
    bench = load_longmemeval(str(path))
    first = bench.instances[0]
    blob = json.dumps(first.probe.to_dict())
    assert "_abs" not in blob and "has_answer" not in blob and "answer_1" not in blob
    assert first.key.is_abstention and first.key.question_type == "temporal-reasoning"
    assert [s.date for s in first.probe.sessions] == ["2023-05-01", "2023-05-20"]  # chronological
    assert len(first.key.evidence_session_ids) == 1
    assert bench.instances[1].key.meta["rubric"].startswith("The user would prefer")


def test_locomo_adapter(tmp_path):
    sample = [{
        "sample_id": "c1",
        "conversation": {
            "speaker_a": "Ann", "speaker_b": "Bob",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "Ann", "dia_id": "D1:1", "text": "I adopted a dog."},
                          {"speaker": "Bob", "dia_id": "D1:2", "text": "Nice!"}],
        },
        "qa": [{"question": "What did Ann adopt?", "answer": "a dog", "evidence": ["D1:1"], "category": 4},
               {"question": "What did Bob adopt?", "adversarial_answer": "a dog", "category": 5}],
    }]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(sample))
    bench = load_locomo(str(path))
    assert len(bench) == 2 and bench.instances[1].key.is_abstention
    assert bench.instances[0].probe.sessions[0].date == "2023-05-08"
    assert len(load_locomo(str(path), include_adversarial=False)) == 1
