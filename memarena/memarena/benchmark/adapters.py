"""Loaders for public long-term-memory benchmarks.

``load_longmemeval`` reads the official LongMemEval JSON files (oracle, S or M).
``load_locomo`` reads LoCoMo's ``locomo10.json``.

Both convert to the canonical schema and strip everything a system must not
see.  Three leaks are closed here, because the source files are public:

* **Fields.**  The question type, the answer, the evidence session ids and the
  per-turn ``has_answer`` flags never reach a ``Probe``.
* **Ids.**  LongMemEval marks unanswerable questions with an ``_abs`` suffix on
  the question id.  Probe ids are hashes of the source id under a *secret* salt
  (random per load unless one is supplied), so they cannot be matched against
  the public file.
* **Order.**  The public files group questions by type, and LoCoMo puts its
  adversarial questions last in each conversation, so position alone would
  reveal labels.  Instances are shuffled with a seed derived from the same
  secret salt, and ``limit`` takes a sample stratified by question type instead
  of the first rows.

Two properties of LongMemEval need care on the evaluator's side as well:

* **Histories repeat.**  A question and its unanswerable twin share one
  history, and other questions share individual sessions.  History and session
  ids are derived from content, so the same history has one id, and every
  instance records the cluster of questions its history overlaps with.  Splits
  and cluster-robust intervals use that cluster.
* **Several sessions can fall on one day.**  The time of day is kept and
  sessions are ordered by the full timestamp.
"""

from __future__ import annotations

import json
import re
import secrets
from typing import Any, Mapping

from ..util import make_rng, sha256_hex
from .schema import AnswerKey, Benchmark, Instance, Probe, Session, Turn, opaque_id, parse_date

__all__ = ["load_longmemeval", "load_locomo", "LOCOMO_CATEGORIES"]


_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*(am|pm)?", re.I)


def _iso(text: str) -> str:
    """ISO date, followed by the time of day when the source gives one; the source string if undated."""
    d = parse_date(text)
    if d is None:
        return str(text)
    m = _TIME.search(str(text))
    if not m:
        return d.isoformat()
    hour, minute, half = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
    if half == "pm" and hour < 12:
        hour += 12
    if half == "am" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return d.isoformat()
    return f"{d.isoformat()} {hour:02d}:{minute:02d}"


def _session_digest(date: str, turns: tuple) -> str:
    return sha256_hex(json.dumps([date, [[t.role, t.content] for t in turns]], ensure_ascii=False, separators=(",", ":")))


def _private_order(instances: list[Instance], salt: str, limit: int | None) -> list[Instance]:
    """Shuffle with a secret seed; with ``limit``, sample proportionally from each question type."""
    rng = make_rng("adapter-order", salt)
    instances = list(instances)
    rng.shuffle(instances)
    if limit is None or limit >= len(instances):
        return instances
    strata: dict[tuple, list[Instance]] = {}
    for inst in instances:
        strata.setdefault((inst.key.question_type, inst.key.is_abstention), []).append(inst)
    total = len(instances)
    quota = {k: limit * len(v) / total for k, v in strata.items()}
    take = {k: int(q) for k, q in quota.items()}
    # largest remainders, so the sample has exactly ``limit`` items
    for k in sorted(quota, key=lambda k: (-(quota[k] - take[k]), str(k)))[: limit - sum(take.values())]:
        take[k] += 1
    chosen = [inst for k, members in strata.items() for inst in members[: take[k]]]
    rng.shuffle(chosen)
    return chosen


def load_longmemeval(path: str, limit: int | None = None, salt: str | None = None) -> Benchmark:
    """Load a LongMemEval file (``longmemeval_oracle.json``, ``_s`` or ``_m``).

    ``salt`` keys the probe ids and the order.  Leave it unset to draw a fresh
    secret per load; pass one only to reproduce a previous run, and keep it away
    from the system under test.
    """
    salt = salt if salt is not None else secrets.token_hex(16)
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    drafts: list[dict] = []
    histories: dict[str, tuple] = {}  # history digest -> the one tuple of sessions every twin shares
    parent: dict[str, str] = {}  # union-find over session digests: which questions' histories overlap

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in rows:
        source_id = str(row["question_id"])
        ids = row.get("haystack_session_ids") or [f"s{i}" for i in range(len(row["haystack_sessions"]))]
        dates = row.get("haystack_dates") or [""] * len(ids)
        built = []
        for i, (sid, date, turns) in enumerate(zip(ids, dates, row["haystack_sessions"])):
            stamp = _iso(date)
            # only role and content cross over; `has_answer` stays behind
            tt = tuple(Turn(str(t["role"]), str(t["content"])) for t in turns)
            built.append((stamp, i, str(sid), tt, _session_digest(stamp, tt), _session_digest("", tt)))
        # sessions are presented in chronological order, by the full timestamp
        built.sort(key=lambda b: (b[0], b[1]))
        seen: dict[str, int] = {}
        sessions, by_source = [], {}
        for stamp, _i, sid, tt, digest, _text in built:
            k = seen.get(digest, 0)
            seen[digest] = k + 1
            osid = opaque_id(salt, f"session:{digest}:{k}", prefix="s")
            by_source[sid] = osid
            sessions.append(Session(session_id=osid, date=stamp, turns=tt))
        history_digest = sha256_hex("|".join(b[4] for b in built))
        session_tuple = histories.setdefault(history_digest, tuple(sessions))
        # questions overlap when they share a session's text, whatever date it carries
        for b in built:
            parent[find("s:" + b[5])] = find("q:" + source_id)
        pid = opaque_id(salt, source_id, prefix="p")
        is_abs = source_id.endswith("_abs")
        qtype = str(row["question_type"])
        answer = str(row["answer"])
        meta: dict = {}
        if qtype == "single-session-preference" and not is_abs:
            meta["rubric"] = answer
        drafts.append(dict(
            source_id=source_id,
            probe=Probe(
                probe_id=pid, question=str(row["question"]), question_date=_iso(row.get("question_date", "")),
                haystack_id=opaque_id(salt, "history:" + history_digest, prefix="h"), sessions=session_tuple,
            ),
            key=AnswerKey(
                probe_id=pid, question_type=qtype, answer=answer, is_abstention=is_abs,
                evidence_session_ids=tuple(by_source[str(x)] for x in row.get("answer_session_ids", []) if str(x) in by_source),
                meta=meta,
            ),
        ))
    instances = [
        Instance(d["probe"], d["key"], d["source_id"], cluster=opaque_id(salt, "cluster:" + find("q:" + d["source_id"]), prefix="c"))
        for d in drafts
    ]
    n_clusters = len({i.cluster for i in instances})
    instances = _private_order(instances, salt, limit)
    return Benchmark(
        "longmemeval", instances,
        info={"source": path, "n": len(instances), "salt_id": sha256_hex(salt)[:12], "clusters_in_source": n_clusters},
    )


# LoCoMo's numeric QA categories.
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
_LOCOMO_TO_TYPE = {
    "multi-hop": "multi-session",
    "temporal": "temporal-reasoning",
    "open-domain": "single-session-user",
    "single-hop": "single-session-user",
    "adversarial": "single-session-user",
}


def load_locomo(path: str, include_adversarial: bool = True, salt: str | None = None, limit: int | None = None) -> Benchmark:
    """Load LoCoMo (``locomo10.json``): two speakers, many questions per conversation.

    Speaker A is mapped to the ``user`` role and speaker B to ``assistant`` so
    the same systems can read it; each turn keeps the speaker's name in its text.
    Adversarial (category 5) items are marked as abstention questions; most
    published LoCoMo numbers leave them out, so they can be excluded here.
    """
    salt = salt if salt is not None else secrets.token_hex(16)
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
    instances = _private_order(instances, salt, limit)
    return Benchmark("locomo", instances, info={"source": path, "n": len(instances), "salt_id": sha256_hex(salt)[:12]})
