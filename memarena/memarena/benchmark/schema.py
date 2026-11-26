"""Canonical data model for long-term-memory benchmarks.

The single most important property here is *structural* separation between what
a system under test may see (``Probe``) and what only the evaluator holds
(``AnswerKey``).  A ``Probe`` cannot leak the gold answer, the question type,
the abstention flag or the evidence location, because it simply has no field
that carries them, and its identifier is an opaque salted hash rather than the
source dataset's id (LongMemEval ids, for example, end in ``_abs`` for
abstention questions).
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Sequence

from ..util import canonical_json, derive_seed, hash_obj, make_rng, sha256_hex

# LongMemEval's six question types; abstention is an orthogonal flag.
QUESTION_TYPES: tuple[str, ...] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

ANSWER_STATUSES = ("ok", "error", "truncated", "refused", "timeout")


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str  # ISO "YYYY-MM-DD" where known, else the source string
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Probe:
    """Everything the system under test is given for one question."""

    probe_id: str  # opaque
    question: str
    question_date: str
    haystack_id: str  # opaque; equal for probes that share one history
    sessions: tuple[Session, ...]

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "question": self.question,
            "question_date": self.question_date,
            "haystack_id": self.haystack_id,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "date": s.date,
                    "turns": [{"role": t.role, "content": t.content} for t in s.turns],
                }
                for s in self.sessions
            ],
        }

    @staticmethod
    def from_dict(d: Mapping) -> "Probe":
        return Probe(
            probe_id=d["probe_id"],
            question=d["question"],
            question_date=d["question_date"],
            haystack_id=d["haystack_id"],
            sessions=tuple(
                Session(
                    session_id=s["session_id"],
                    date=s["date"],
                    turns=tuple(Turn(t["role"], t["content"]) for t in s["turns"]),
                )
                for s in d["sessions"]
            ),
        )


@dataclass(frozen=True)
class AnswerKey:
    """Evaluator-only record for one probe."""

    probe_id: str
    question_type: str
    answer: str  # gold answer; for abstention items, the explanation
    aliases: tuple[str, ...] = ()
    is_abstention: bool = False
    evidence_session_ids: tuple[str, ...] = ()
    # Optional structured hints for deterministic scorers:
    #   numeric: bool, unit: str, list: [items], stale: [superseded values], rubric: str
    meta: Mapping = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "question_type": self.question_type,
            "answer": self.answer,
            "aliases": list(self.aliases),
            "is_abstention": self.is_abstention,
            "evidence_session_ids": list(self.evidence_session_ids),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Mapping) -> "AnswerKey":
        return AnswerKey(
            probe_id=d["probe_id"],
            question_type=d["question_type"],
            answer=d["answer"],
            aliases=tuple(d.get("aliases", ())),
            is_abstention=bool(d.get("is_abstention", False)),
            evidence_session_ids=tuple(d.get("evidence_session_ids", ())),
            meta=dict(d.get("meta", {})),
        )


@dataclass(frozen=True)
class Instance:
    """Evaluator-side pairing of a probe and its key."""

    probe: Probe
    key: AnswerKey
    source_id: str = ""  # id in the source dataset; never shown to a system

    @property
    def probe_id(self) -> str:
        return self.probe.probe_id


@dataclass(frozen=True)
class Answer:
    """One response from a system under test.

    ``status`` separates "the system answered" from "the plumbing failed".  A
    non-``ok`` status is never silently folded into a wrong answer: the grader
    reports such rows separately.
    """

    text: str
    confidence: float = 0.5  # system's own estimate of P(correct); optional signal
    status: str = "ok"
    meta: Mapping = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "status": self.status,
        }

    @staticmethod
    def from_dict(d: Mapping) -> "Answer":
        return Answer(
            text=str(d.get("text", "")),
            confidence=float(d.get("confidence", 0.5)),
            status=str(d.get("status", "ok")),
        )


def opaque_id(salt: str, source_id: str, prefix: str = "p") -> str:
    return f"{prefix}_{sha256_hex(salt + '|' + source_id)[:16]}"


class Benchmark:
    """An ordered collection of instances plus provenance."""

    def __init__(self, name: str, instances: Sequence[Instance], info: Mapping | None = None):
        ids = [i.probe_id for i in instances]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate probe ids in benchmark")
        self.name = name
        self.instances: tuple[Instance, ...] = tuple(instances)
        self.info = dict(info or {})

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self) -> Iterator[Instance]:
        return iter(self.instances)

    def probes(self) -> list[Probe]:
        """The only view a system under test receives."""
        return [i.probe for i in self.instances]

    def keys(self) -> dict[str, AnswerKey]:
        return {i.probe_id: i.key for i in self.instances}

    def by_id(self) -> dict[str, Instance]:
        return {i.probe_id: i for i in self.instances}

    def type_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.instances:
            out[i.key.question_type] = out.get(i.key.question_type, 0) + 1
        return dict(sorted(out.items()))

    def manifest_hash(self) -> str:
        """Hash of probes *and* keys: identifies the exact labelled set."""
        return hash_obj(
            [[i.probe.to_dict(), i.key.to_dict()] for i in self.instances]
        )

    def probes_hash(self) -> str:
        return hash_obj([i.probe.to_dict() for i in self.instances])

    def key_commitment(self, salt: str) -> str:
        """Commitment to the answer keys that reveals nothing until ``salt`` is opened."""
        return sha256_hex(
            "keys:" + salt + ":" + canonical_json([i.key.to_dict() for i in self.instances])
        )

    def subset(self, probe_ids: Iterable[str], name: str | None = None) -> "Benchmark":
        wanted = set(probe_ids)
        return Benchmark(
            name or self.name,
            [i for i in self.instances if i.probe_id in wanted],
            self.info,
        )

    def split(self, fractions: Mapping[str, float], seed: int | str) -> dict[str, "Benchmark"]:
        """Random split stratified by (question type, abstention), grouped by haystack.

        Probes that share a haystack stay in the same part, so no history is
        seen in two parts.  The split never looks at any system's scores.
        """
        names = list(fractions)
        total = sum(fractions.values())
        if total <= 0:
            raise ValueError("fractions must sum to a positive number")
        groups: dict[str, list[Instance]] = {}
        for inst in self.instances:
            groups.setdefault(inst.probe.haystack_id, []).append(inst)
        # stratify groups by their dominant stratum
        strata: dict[str, list[str]] = {}
        for hid, members in groups.items():
            counts: dict[str, int] = {}
            for m in members:
                s = f"{m.key.question_type}|{int(m.key.is_abstention)}"
                counts[s] = counts.get(s, 0) + 1
            dominant = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            strata.setdefault(dominant, []).append(hid)
        assignment: dict[str, str] = {}
        for stratum in sorted(strata):
            hids = sorted(strata[stratum])
            make_rng("split", seed, stratum).shuffle(hids)
            n = len(hids)
            start = 0
            acc = 0.0
            for idx, part in enumerate(names):
                acc += fractions[part] / total
                end = n if idx == len(names) - 1 else int(round(acc * n))
                for hid in hids[start:end]:
                    assignment[hid] = part
                start = end
        parts: dict[str, list[Instance]] = {p: [] for p in names}
        for inst in self.instances:
            parts[assignment[inst.probe.haystack_id]].append(inst)
        return {p: Benchmark(f"{self.name}:{p}", parts[p], self.info) for p in names}


# --------------------------------------------------------------------------- dates

_DATE_PATTERNS = (
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"), (1, 2, 3)),
    (re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})"), (1, 2, 3)),
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
        ]
    )
}
_TEXT_DATE = re.compile(
    r"(\d{1,2})(?::\d{2})?\s*(?:am|pm)?\s*on\s*(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", re.I
)


def parse_date(text: str) -> _dt.date | None:
    """Parse the date formats used by supported sources; ``None`` if unknown.

    Handles ISO dates, LongMemEval's ``2023/05/30 (Tue) 23:40`` and LoCoMo's
    ``1:56 pm on 8 May, 2023``.
    """
    if not text:
        return None
    for pat, (yi, mi, di) in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return _dt.date(int(m.group(yi)), int(m.group(mi)), int(m.group(di)))
            except ValueError:
                return None
    m = _TEXT_DATE.search(text)
    if m:
        month = _MONTHS.get(m.group(3).lower())
        if month:
            try:
                return _dt.date(int(m.group(4)), month, int(m.group(2)))
            except ValueError:
                return None
    return None


def stable_salt(*parts) -> str:
    return sha256_hex("salt:" + str(derive_seed(*parts)))[:24]
