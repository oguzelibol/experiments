"""Loaders for public long-term-memory benchmarks.

``load_longmemeval`` reads the official LongMemEval JSON files (oracle, S or M).
``load_locomo`` reads LoCoMo's ``locomo10.json``.

Both convert to the canonical schema and, importantly, strip everything a system
must not see.  In LongMemEval that is the question type, the answer, the
evidence session ids, the per-turn ``has_answer`` flags, and the question id
itself, whose ``_abs`` suffix marks unanswerable questions.  Systems receive
opaque ids instead.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .schema import AnswerKey, Benchmark, Instance, Probe, Session, Turn, opaque_id, parse_date

__all__ = ["load_longmemeval", "load_locomo", "LOCOMO_CATEGORIES"]


def _iso(text: str) -> str:
    d = parse_date(text)
    return d.isoformat() if d else str(text)


def load_longmemeval(path: str, limit: int | None = None, salt: str = "longmemeval") -> Benchmark:
    """Load a LongMemEval file (``longmemeval_oracle.json``, ``_s`` or ``_m``)."""
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if limit is not None:
        rows = rows[:limit]
    instances: list[Instance] = []
    for row in rows:
        source_id = str(row["question_id"])
        sessions = []
        ids = row.get("haystack_session_ids") or [f"s{i}" for i in range(len(row["haystack_sessions"]))]
        dates = row.get("haystack_dates") or [""] * len(ids)
        opaque_sessions: dict[str, str] = {}
        for sid, date, turns in zip(ids, dates, row["haystack_sessions"]):
            osid = opaque_id(salt, f"{source_id}:{sid}", prefix="s")
            opaque_sessions[str(sid)] = osid
            sessions.append(
                Session(
                    session_id=osid,
                    date=_iso(date),
                    # only role and content cross over; `has_answer` stays behind
                    turns=tuple(Turn(str(t["role"]), str(t["content"])) for t in turns),
                )
            )
        # sessions are presented in chronological order
        order = sorted(range(len(sessions)), key=lambda i: (sessions[i].date, i))
        sessions = [sessions[i] for i in order]
        pid = opaque_id(salt, source_id, prefix="p")
        probe = Probe(
            probe_id=pid,
            question=str(row["question"]),
            question_date=_iso(row.get("question_date", "")),
            haystack_id=opaque_id(salt, source_id, prefix="h"),
            sessions=tuple(sessions),
        )
        is_abs = source_id.endswith("_abs")
        qtype = str(row["question_type"])
        answer = str(row["answer"])
        meta: dict = {}
        if qtype == "single-session-preference" and not is_abs:
            meta["rubric"] = answer
        key = AnswerKey(
            probe_id=pid,
            question_type=qtype,
            answer=answer,
            is_abstention=is_abs,
            evidence_session_ids=tuple(
                opaque_sessions[str(s)] for s in row.get("answer_session_ids", []) if str(s) in opaque_sessions
            ),
            meta=meta,
        )
        instances.append(Instance(probe, key, source_id))
    return Benchmark("longmemeval", instances, info={"source": path, "n": len(instances)})


# LoCoMo's numeric QA categories.
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
_LOCOMO_TO_TYPE = {
    "multi-hop": "multi-session",
    "temporal": "temporal-reasoning",
    "open-domain": "single-session-user",
    "single-hop": "single-session-user",
    "adversarial": "single-session-user",
}


def load_locomo(path: str, include_adversarial: bool = True, salt: str = "locomo") -> Benchmark:
    """Load LoCoMo (``locomo10.json``): two speakers, many questions per conversation.

    Speaker A is mapped to the ``user`` role and speaker B to ``assistant`` so
    the same systems can read it; each turn keeps the speaker's name in its text.
    Adversarial (category 5) items are marked as abstention questions; most
    published LoCoMo numbers leave them out, so they can be excluded here.
    """
    with open(path, "r", encoding="utf-8") as fh:
        samples = json.load(fh)
    instances: list[Instance] = []
    for sample in samples:
        conv = sample["conversation"]
        sample_id = str(sample.get("sample_id", len(instances)))
        speaker_a = conv.get("speaker_a", "A")
        sessions = []
        n = 1
        while f"session_{n}" in conv:
            turns = []
            for t in conv[f"session_{n}"]:
                role = "user" if t.get("speaker") == speaker_a else "assistant"
                text = f"{t.get('speaker', '')}: {t.get('text', '')}"
                if t.get("blip_caption"):
                    text += f" [shares an image: {t['blip_caption']}]"
                turns.append(Turn(role, text))
            sessions.append(
                Session(opaque_id(salt, f"{sample_id}:session_{n}", prefix="s"), _iso(conv.get(f"session_{n}_date_time", "")), tuple(turns))
            )
            n += 1
        haystack_id = opaque_id(salt, sample_id, prefix="h")
        question_date = sessions[-1].date if sessions else ""
        for q_idx, qa in enumerate(sample.get("qa", [])):
            category = LOCOMO_CATEGORIES.get(int(qa.get("category", 0)), "unknown")
            adversarial = category == "adversarial"
            if adversarial and not include_adversarial:
                continue
            source_id = f"{sample_id}:q{q_idx}"
            pid = opaque_id(salt, source_id, prefix="p")
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            if adversarial:
                answer = "This question cannot be answered from the conversation."
            key = AnswerKey(
                probe_id=pid,
                question_type=_LOCOMO_TO_TYPE.get(category, "single-session-user"),
                answer=str(answer),
                is_abstention=adversarial,
                meta={"locomo_category": category},
            )
            probe = Probe(pid, str(qa["question"]), question_date, haystack_id, tuple(sessions))
            instances.append(Instance(probe, key, source_id))
    return Benchmark("locomo", instances, info={"source": path, "n": len(instances)})
