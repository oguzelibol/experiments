import dataclasses
import json

import pytest

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
    bench = load_longmemeval(str(path), salt="fixed-for-the-test")
    first = next(i for i in bench.instances if i.source_id == "abc123_abs")
    blob = json.dumps(first.probe.to_dict())
    assert "_abs" not in blob and "has_answer" not in blob and "answer_1" not in blob
    assert first.key.is_abstention and first.key.question_type == "temporal-reasoning"
    assert [s.date for s in first.probe.sessions] == ["2023-05-01 09:00", "2023-05-20 10:00"]  # chronological
    assert first.probe.question_date == "2023-05-30 23:40"
    assert len(first.key.evidence_session_ids) == 1
    second = next(i for i in bench.instances if i.source_id == "q2")
    assert second.key.meta["rubric"].startswith("The user would prefer")


def _lme_rows(n_per_type=12):
    rows = []
    types = ["temporal-reasoning", "multi-session", "single-session-user"]
    for t in types:  # grouped by type, like the public file
        for k in range(n_per_type):
            rows.append({
                "question_id": f"{t}-{k}" + ("_abs" if k % 4 == 0 else ""), "question_type": t, "question": f"q {t} {k}?",
                "answer": "a", "question_date": "2023/06/01 (Thu) 08:00", "haystack_session_ids": ["s1"],
                "haystack_dates": ["2023/05/02 (Tue) 09:00"], "haystack_sessions": [[{"role": "user", "content": "x"}]],
                "answer_session_ids": ["s1"],
            })
    return rows


def test_public_ids_and_order_do_not_leak_labels(tmp_path):
    from memarena.benchmark.schema import opaque_id

    rows = _lme_rows()
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(rows))
    a, b = load_longmemeval(str(path)), load_longmemeval(str(path))
    # ids depend on a secret salt: two loads disagree, and neither matches a guessable salt
    assert {i.probe_id for i in a} != {i.probe_id for i in b}
    guessable = {opaque_id(s, r["question_id"]) for r in rows for s in ("longmemeval", "", "salt")}
    assert not ({i.probe_id for i in a} & guessable)
    # position no longer encodes the type: the first third is not all one type
    first_third = [i.key.question_type for i in a.instances[:12]]
    assert len(set(first_third)) > 1
    # a limited load is a stratified sample, not the first rows
    small = load_longmemeval(str(path), limit=9)
    assert len(small) == 9 and set(small.type_counts().values()) == {3}
    # the same salt reproduces the same benchmark
    assert [i.probe_id for i in load_longmemeval(str(path), salt="s")] == [i.probe_id for i in load_longmemeval(str(path), salt="s")]


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
    bench = load_locomo(str(path), salt="t")
    assert len(bench) == 2 and sum(i.key.is_abstention for i in bench) == 1
    assert bench.instances[0].probe.sessions[0].date == "2023-05-08 13:56"
    assert parse_date(bench.instances[0].probe.sessions[0].date).isoformat() == "2023-05-08"
    assert len(load_locomo(str(path), include_adversarial=False)) == 1


def _row(qid, sessions, qtype="temporal-reasoning", qdate="2023/06/01 (Thu) 08:00"):
    return {
        "question_id": qid, "question_type": qtype, "question": f"q {qid}?", "answer": "a", "question_date": qdate,
        "haystack_session_ids": [f"s{k}" for k in range(len(sessions))], "haystack_dates": [d for d, _ in sessions],
        "haystack_sessions": [[{"role": "user", "content": text}] for _, text in sessions], "answer_session_ids": ["s0"],
    }


def test_longmemeval_same_day_sessions_are_ordered_by_time(tmp_path):
    rows = [_row("q1", [("2023/05/28 (Sun) 21:04", "the evening webinar"), ("2023/05/28 (Sun) 07:17", "the morning workshop")])]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(rows))
    probe = load_longmemeval(str(path), salt="t").instances[0].probe
    assert [s.date for s in probe.sessions] == ["2023-05-28 07:17", "2023-05-28 21:04"]
    assert [s.turns[0].content for s in probe.sessions] == ["the morning workshop", "the evening webinar"]


def test_longmemeval_histories_are_identified_by_content_and_overlaps_are_clustered(tmp_path):
    shared = ("2023/05/02 (Tue) 09:00", "I adopted a dog named Biscuit")
    rows = [
        _row("a1", [shared]), _row("a1_abs", [shared]),  # twins: the same history
        _row("b1", [shared, ("2023/05/03 (Wed) 09:00", "something only b1 has")]),  # overlaps the twins
        _row("c1", [("2023/05/09 (Tue) 09:00", "I adopted a dog named Biscuit")]),  # same text, another date
    ] + [_row(f"z{k}", [("2023/05/04 (Thu) 09:00", f"unrelated {k}")]) for k in range(8)]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(rows))
    bench = load_longmemeval(str(path), salt="t")
    by = {i.source_id: i for i in bench}
    assert by["a1"].probe.haystack_id == by["a1_abs"].probe.haystack_id
    assert by["a1"].probe.sessions is by["a1_abs"].probe.sessions
    assert by["a1"].probe.haystack_id != by["b1"].probe.haystack_id != by["c1"].probe.haystack_id
    assert by["a1"].probe.haystack_id != by["c1"].probe.haystack_id  # the date is part of the history
    assert by["a1"].cluster_id == by["a1_abs"].cluster_id == by["b1"].cluster_id == by["c1"].cluster_id
    assert len(set(bench.clusters().values())) == 9 and bench.info["clusters_in_source"] == 9
    for seed in range(5):
        parts = bench.split({"dev": 0.5, "test": 0.5}, seed=seed)
        text = {n: {t.content for i in part for s_ in i.probe.sessions for t in s_.turns} for n, part in parts.items()}
        assert not (text["dev"] & text["test"])


# ------------------------------------------------- an exam holds what its standard names

def test_a_focused_exam_contains_only_the_named_abilities_and_exactly_the_asked_number_of_questions():
    from memarena.evalcore.standard import Standard, capability_of

    cw = {"judge_lenient|detect|micro": 1.0}
    for focus in ({"single-session-preference": 1.0}, {"knowledge-update": 1.0}, {"multi-session": 0.5, "abstention": 0.5},
                  {"single-session-preference": 1, "single-session-assistant": 1, "knowledge-update": 1}, {"abstention": 1.0}):
        std = Standard(cw, focus)
        for k in range(12):
            exam = generate(f"focus-{k}", std.gen_config(2, 12))
            assert len(exam) == 24
            assert all(capability_of(i.key) in std.capability_weights for i in exam), focus
            assert 2 <= exam.info["histories"] <= 16
            assert max(sum(1 for i in exam if i.probe.haystack_id == h) for h in {i.probe.haystack_id for i in exam}) <= 12


def test_position_in_a_history_says_nothing_about_whether_a_question_is_answerable():
    from memarena.evalcore.standard import Standard

    std = Standard({"judge_lenient|detect|micro": 1.0}, {"abstention": 0.5, "temporal-reasoning": 0.5})
    hits, seen = [0] * 12, [0] * 12
    for k in range(150):
        exam = generate(f"position-{k}", std.gen_config(2, 12))
        order, at = [], {}
        for i in exam:
            if i.probe.haystack_id not in at:
                at[i.probe.haystack_id] = 0
                order.append(i.probe.haystack_id)
            if order.index(i.probe.haystack_id) < 2:  # the two histories that were asked for
                j = at[i.probe.haystack_id]
                seen[j] += 1
                hits[j] += i.key.is_abstention
            at[i.probe.haystack_id] += 1
    rates = [h / n for h, n in zip(hits, seen)]
    assert max(rates) - min(rates) < 0.15, rates


def test_generator_refuses_what_it_cannot_honour_and_never_reuses_an_id_for_other_content():
    for bad in (dict(sessions_per_stream=3), dict(type_mix={}), dict(type_mix={"made-up": 1.0}), dict(update_rate=2.0),
                dict(n_streams=0), dict(exchanges_per_session=(4, 2)), dict(abstention_rate=float("nan")),
                dict(type_mix={"knowledge-update": 1.0}, update_rate=0.0)):  # no updates, so nothing to ask about them
        with pytest.raises(ValueError):
            generate("refused", GenConfig(**bad))
    a = generate("same-seed", GenConfig(n_streams=4))
    b = generate("same-seed", GenConfig(n_streams=4, third_party_rate=1.0))
    assert not {i.probe_id for i in a} & {i.probe_id for i in b}
    assert not {i.probe.haystack_id for i in a} & {i.probe.haystack_id for i in b}
    # a configuration restored from its stored form draws the same benchmark, whatever order its keys come back in
    restored = GenConfig.from_dict(json.loads(json.dumps(a.info["config"], sort_keys=True)))
    assert generate("same-seed", restored).manifest_hash() == a.manifest_hash()
