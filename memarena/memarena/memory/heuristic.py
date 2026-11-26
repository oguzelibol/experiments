"""An offline, LLM-free memory system: BM25 retrieval plus rule-based reading.

This is a deliberately classical question-answering pipeline.  It exists so the
whole evaluation and peer-review machinery can run for free and at scale, with
real text going in and real answer strings coming out.  It is a weak baseline
on real data; swap in ``LLMMemory`` when a model is available.

Capabilities are switched on by *features*.  Each feature is a distinct piece
of machinery that mostly helps one kind of question, so a system with a limited
feature budget has to specialise:

    expansion     query synonyms: survives paraphrase between statement and question
    recency       prefer the latest matching statement: knowledge updates
    temporal      date arithmetic over session dates: temporal reasoning
    relative      resolve "yesterday" / "two days ago" against the session date
    aggregate     combine evidence across sessions: counts, sums, lists
    assistant     index what the assistant said: assistant-provided facts
    exchange      link an assistant reply to the request it answers
    comparative   parse "X over Y": preferences
    third_party   ignore statements about other people: attribution, abstention
    negation      ignore negated or hypothetical statements: abstention
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..benchmark.schema import Answer, Probe, parse_date
from ..util import hash_obj
from .base import ABSTAIN_TEXT

__all__ = ["FEATURES", "MemoryConfig", "HeuristicMemory", "classify_question", "QUESTION_CLASSES"]

FEATURES = (
    "expansion", "recency", "temporal", "relative", "aggregate", "assistant",
    "exchange", "comparative", "third_party", "negation",
)
STYLES = ("terse", "sentence", "verbose")


@dataclass(frozen=True)
class MemoryConfig:
    features: frozenset = field(default_factory=frozenset)
    top_k: int = 6
    abstain_threshold: float = 0.3
    latest_margin: float = 0.6
    style: str = "terse"
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self):
        unknown = set(self.features) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        if self.style not in STYLES:
            raise ValueError(f"unknown style {self.style!r}")
        object.__setattr__(self, "features", frozenset(self.features))

    def to_dict(self) -> dict:
        return {
            "features": sorted(self.features),
            "top_k": self.top_k,
            "abstain_threshold": self.abstain_threshold,
            "latest_margin": self.latest_margin,
            "style": self.style,
            "k1": self.k1,
            "b": self.b,
        }

    @staticmethod
    def from_dict(d: Mapping) -> "MemoryConfig":
        return MemoryConfig(
            features=frozenset(d.get("features", ())),
            top_k=int(d.get("top_k", 6)),
            abstain_threshold=float(d.get("abstain_threshold", 0.3)),
            latest_margin=float(d.get("latest_margin", 0.6)),
            style=str(d.get("style", "terse")),
            k1=float(d.get("k1", 1.2)),
            b=float(d.get("b", 0.75)),
        )

    def hash(self) -> str:
        return hash_obj(self.to_dict())[:16]


# ------------------------------------------------------------------ linguistics

_WORD = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’/&\-][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_BREAK = re.compile(r"[,;:!?—]|\s-\s|\.(?:\s|$)")

_STOP = frozenset(
    """a an the i me my mine myself we our us you your yours it its this that these those he she they them
    their his her is am are was were be been being do does did doing have has had having will would can could
    should may might must shall of in on at to for from by with about as into over under than then and or but
    if so not no nor too very just also only what which who whom whose when where why how there here all any
    some each few more most other such own same both up down out off again ever once s t d ll re ve m
    got get go going went gone take took taken make made say said know let like want need new old
    sure yes ok okay well happy thanks thank please hey hi oh guess big news finally great nice good
    """.split()
)
# temporal / filler words that never carry the answer
_FILLER = frozenset(
    """today yesterday tomorrow tonight morning evening night week weeks month months year years day days
    last next now recently lately currently already still these those ago time times every single always
    usually daily little one thing lot bit way kind sort am pm""".split()
)
_IRREGULAR = {
    "took": "take", "taken": "take", "went": "go", "gone": "go", "ran": "run", "grew": "grow",
    "grown": "grow", "bought": "buy", "got": "get", "gotten": "get", "gave": "give", "given": "give",
    "saw": "see", "seen": "see", "built": "build", "did": "do", "done": "do", "made": "make",
    "said": "say", "told": "tell", "began": "begin", "begun": "begin", "sold": "sell", "left": "leave",
    "spent": "spend", "ate": "eat", "eaten": "eat", "drove": "drive", "driven": "drive", "was": "be",
    "were": "be", "been": "be", "children": "child", "studied": "study", "lives": "live",
}


def stem(word: str) -> str:
    w = word.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    w = w.replace("'", "")
    w = _IRREGULAR.get(w, w)
    n = len(w)
    if n > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif n > 4 and w.endswith("ied"):
        w = w[:-3] + "y"
    elif n > 5 and w.endswith("ing"):
        w = w[:-3]
    elif n > 4 and w.endswith("ed"):
        w = w[:-2]
    elif n > 4 and (w.endswith("ches") or w.endswith("shes") or w.endswith("sses") or w.endswith("xes")):
        w = w[:-2]
    elif n > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def _stems(text: str) -> list[str]:
    return [stem(m.group(0)) for m in _WORD.finditer(text)]


_STOP_STEMS = frozenset(stem(w) for w in _STOP) | _STOP
_FILLER_STEMS = frozenset(stem(w) for w in _FILLER) | _FILLER

# Generic synonym table over stems (query term -> related terms).
_SYNONYMS_RAW = {
    "live": "move relocate base settle apartment home",
    "work": "job career profession living paycheck team employ",
    "job": "work career profession living professionally",
    "occupation": "job work career profession living professionally",
    "company": "work employer team paycheck join",
    "car": "drive ride vehicle trade",
    "drive": "car ride vehicle around",
    "phone": "carry pocket driver upgrade replace",
    "favorite": "beat love eat",
    "food": "dish eat meal",
    "play": "practice instrument evening",
    "instrument": "play practice",
    "learn": "lesson study flashcard",
    "language": "lesson flashcard",
    "name": "call",
    "pet": "adopt household member",
    "gym": "lift weight workout",
    "coffee": "cafe order",
    "order": "usual always",
    "allergic": "allergy hive away doctor",
    "allergy": "allergic hive away doctor",
    "study": "degree major",
    "college": "degree major university",
    "book": "read page",
    "read": "page book",
    "finish": "done last ending turn",
    "trip": "back travel wander visit spend",
    "country": "trip travel",
    "buy": "pick purchase",
    "recommend": "suggest pick recommendation",
    "prefer": "person give rather preference",
    "preference": "prefer person give rather",
    "run": "ran",
}
_SYNONYMS = {stem(k): tuple(stem(v) for v in vs.split()) for k, vs in _SYNONYMS_RAW.items()}

_THIRD_PARTY = re.compile(
    r"\bmy (friend|cousin|coworker|co-worker|colleague|neighbor|neighbour|uncle|aunt|boss|roommate)\b"
    r"|^[A-Z][\w'’\-]+, my (friend|cousin|coworker|colleague|neighbor|neighbour)\b",
    re.I,
)
_NEGATED = re.compile(
    r"\b(thought about|considered|almost|decided against|changed my mind|fell through|never started|"
    r"didn't|did not|never was|wasn't|was not|used to think|test-drove|but never)\b",
    re.I,
)
_PROPER_CUE = re.compile(r"\b(name|who|city|country|company|phone|car)\b", re.I)

QUESTION_CLASSES = (
    "days_between", "days_ago", "order", "month", "sum", "count", "list",
    "assistant", "preference", "fact",
)
_CLASS_PATTERNS = (
    ("days_between", re.compile(r"how many days (passed|elapsed|went by|were there) between", re.I)),
    ("days_ago", re.compile(r"how many days ago|how long ago", re.I)),
    ("order", re.compile(r"which (did i do|happened|came|was) first", re.I)),
    ("month", re.compile(r"\b(which|what) month\b", re.I)),
    ("sum", re.compile(r"how many .+ in total|total (number|amount)", re.I)),
    ("count", re.compile(r"^\s*how many\b", re.I)),
    ("list", re.compile(r"^\s*which \w+(?: \w+)? have i\b", re.I)),
    ("assistant", re.compile(r"\b(did you|you) (recommend|suggest)|you (recommended|suggested)", re.I)),
    ("preference", re.compile(r"\bprefer|preference\b", re.I)),
)

# Coarse classes that are observable from the question alone, used by
# evaluators to stratify without access to gold labels.
_OBSERVABLE = {
    "days_between": "temporal", "days_ago": "temporal", "order": "temporal", "month": "temporal",
    "sum": "aggregate", "count": "aggregate", "list": "aggregate",
    "assistant": "assistant", "preference": "preference", "fact": "fact",
}


def classify_question(question: str) -> str:
    for name, pat in _CLASS_PATTERNS:
        if pat.search(question):
            return name
    return "fact"


def observable_class(question: str) -> str:
    return _OBSERVABLE[classify_question(question)]


# ----------------------------------------------------------------------- index

@dataclass
class _Unit:
    text: str
    role: str
    session_idx: int
    order: int
    date: _dt.date | None
    stems: list[str]
    weights: dict  # stem -> term frequency (context terms count less)


class _Index:
    def __init__(self, probe: Probe, with_assistant: bool, with_exchange: bool):
        self.units: list[_Unit] = []
        order = 0
        for s_idx, session in enumerate(probe.sessions):
            date = parse_date(session.date)
            last_user_stems: list[str] = []
            for turn in session.turns:
                is_user = turn.role == "user"
                if not is_user and not with_assistant:
                    continue
                sentences = [s for s in _SENT_SPLIT.split(turn.content.strip()) if s.strip()]
                turn_stems: list[str] = []
                for sent in sentences:
                    st = [x for x in _stems(sent) if x not in _STOP_STEMS]
                    turn_stems.extend(st)
                    weights: dict = {}
                    for x in st:
                        weights[x] = weights.get(x, 0.0) + 1.0
                    if not is_user and with_exchange:
                        for x in last_user_stems:
                            weights[x] = weights.get(x, 0.0) + 0.5
                    self.units.append(_Unit(sent.strip(), turn.role, s_idx, order, date, st, weights))
                    order += 1
                if is_user:
                    last_user_stems = turn_stems
        n = len(self.units)
        self.n = n
        self.df: dict = {}
        self.postings: dict = {}
        total_len = 0.0
        self.lengths: list[float] = []
        for i, u in enumerate(self.units):
            length = sum(u.weights.values())
            self.lengths.append(length)
            total_len += length
            for term, tf in u.weights.items():
                self.df[term] = self.df.get(term, 0) + 1
                self.postings.setdefault(term, []).append((i, tf))
        self.avgdl = total_len / n if n else 1.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: Mapping[str, float], k1: float, b: float) -> tuple[dict, float]:
        """BM25 scores per unit index and the best achievable score for the query."""
        scores: dict = {}
        ideal = 0.0
        for term, qw in query.items():
            idf = self.idf(term)
            if qw >= 1.0:
                ideal += idf  # tf saturates towards idf * (k1 + 1) / ... ; use idf as the scale
            for i, tf in self.postings.get(term, ()):
                denom = tf + k1 * (1 - b + b * self.lengths[i] / self.avgdl)
                scores[i] = scores.get(i, 0.0) + qw * idf * tf * (k1 + 1) / denom
        return scores, ideal


class _IndexCache:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._data: OrderedDict = OrderedDict()

    def get(self, probe: Probe, with_assistant: bool, with_exchange: bool) -> _Index:
        key = (probe.haystack_id, len(probe.sessions), with_assistant, with_exchange)
        idx = self._data.get(key)
        if idx is None:
            idx = _Index(probe, with_assistant, with_exchange)
            self._data[key] = idx
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)
        else:
            self._data.move_to_end(key)
        return idx


_CACHE = _IndexCache()


# ---------------------------------------------------------------------- reader

@dataclass
class _Hit:
    unit: _Unit
    score: float
    rel: float  # score relative to the best achievable for the query


def _runs(sentence: str, banned: frozenset) -> list[str]:
    """Maximal spans of informative words in ``sentence`` (original casing)."""
    runs: list[str] = []
    for clause in _CLAUSE_BREAK.split(sentence):
        words = [m.group(0) for m in _WORD.finditer(clause)]
        if not words:
            continue
        keep = []
        for i, w in enumerate(words):
            s = stem(w)
            informative = s not in _STOP_STEMS and s not in _FILLER_STEMS and s not in banned
            if not informative and 0 < i < len(words) - 1:
                # keep a function word inside a title: "A Map of Small Rivers"
                prev_w, next_w = words[i - 1], words[i + 1]
                if (
                    keep and keep[-1] and prev_w[:1].isupper() and next_w[:1].isupper()
                    and stem(next_w) not in banned and s in _STOP_STEMS and len(w) <= 4
                ):
                    informative = True
            keep.append(informative)
        cur: list[str] = []
        for w, k in zip(words, keep):
            if k:
                cur.append(w)
            elif cur:
                runs.append(" ".join(cur))
                cur = []
        if cur:
            runs.append(" ".join(cur))
    return runs


def _pick_span(sentence: str, banned: frozenset, want_proper: bool) -> str | None:
    runs = _runs(sentence, banned)
    if not runs:
        return None
    if want_proper:
        proper = [r for r in runs if any(w[:1].isupper() for w in r.split())]
        if proper:
            return proper[-1]
    return runs[-1]


_COMPARATIVE = (
    re.compile(r"\bprefer (?P<x>.+?) (?:over|to|rather than) (?P<y>.+?)[.!?]?$", re.I),
    re.compile(r"\bmore of an? (?P<x>.+?) person than\b", re.I),
    re.compile(r"\bgive me (?P<x>.+?) over (?P<y>.+?) any day", re.I),
)
_NUM = re.compile(r"(?<![\w.])\d+(?![\w])")
_SMALL_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten".split())}
_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


class HeuristicMemory:
    """BM25 + rules.  See the module docstring for what each feature does."""

    def __init__(self, config: MemoryConfig | None = None, name: str | None = None):
        self.config = config or MemoryConfig()
        self.name = name or f"heuristic:{self.config.hash()}"
        self._f = self.config.features

    def config_hash(self) -> str:
        return self.config.hash()

    # -- retrieval
    def _query(self, text: str) -> dict:
        q: dict = {}
        for s in _stems(text):
            if s in _STOP_STEMS or s in _FILLER_STEMS:
                continue
            q[s] = 1.0
        if "expansion" in self._f:
            for s in list(q):
                for syn in _SYNONYMS.get(s, ()):
                    q.setdefault(syn, 0.5)
        return q

    def _retrieve(self, index: _Index, text: str, roles: tuple[str, ...] | None = None) -> tuple[list[_Hit], frozenset]:
        query = self._query(text)
        scores, ideal = index.search(query, self.config.k1, self.config.b)
        hits: list[_Hit] = []
        for i, sc in scores.items():
            u = index.units[i]
            if roles and u.role not in roles:
                continue
            if "third_party" in self._f and _THIRD_PARTY.search(u.text):
                continue
            if "negation" in self._f and _NEGATED.search(u.text):
                continue
            rel = sc / ideal if ideal > 0 else 0.0
            hits.append(_Hit(u, sc, rel))
        hits.sort(key=lambda h: (-h.score, h.unit.order))
        return hits[: max(1, self.config.top_k)], frozenset(query)

    def _confident(self, hits: Sequence[_Hit]) -> list[_Hit]:
        return [h for h in hits if h.rel >= self.config.abstain_threshold]

    # -- per-class readers; each returns (answer or None, confidence, alternative)
    def _read_fact(self, index, question, roles=None, want_proper=None):
        hits, banned = self._retrieve(index, question, roles)
        good = self._confident(hits)
        if not good:
            return None, 0.55, None
        best = good[0]
        chosen = best
        if "recency" in self._f:
            pool = [h for h in good if h.score >= self.config.latest_margin * best.score]
            chosen = max(pool, key=lambda h: h.unit.order)
        if want_proper is None:
            want_proper = bool(_PROPER_CUE.search(question))
        span = _pick_span(chosen.unit.text, banned, want_proper)
        if not span:
            return None, 0.4, None
        alt = None
        for h in good:
            if h is not chosen:
                cand = _pick_span(h.unit.text, banned, want_proper)
                if cand and cand.lower() != span.lower():
                    alt = cand
                    break
        return span, min(0.95, 0.25 + 0.7 * chosen.rel), alt

    def _read_preference(self, index, question):
        hits, banned = self._retrieve(index, question, ("user",))
        good = self._confident(hits)
        if not good:
            return None, 0.55, None
        best = good[0]
        if "comparative" in self._f:
            for pat in _COMPARATIVE:
                m = pat.search(best.unit.text)
                if m:
                    x = m.group("x").strip(" .,")
                    y = m.groupdict().get("y")
                    return x, min(0.95, 0.35 + 0.6 * best.rel), (y.strip(" .,") if y else None)
        span = _pick_span(best.unit.text, banned, False)
        return span, 0.3, None

    def _event_date(self, index, phrase):
        hits, _ = self._retrieve(index, phrase, ("user",))
        good = self._confident(hits)
        if not good or good[0].unit.date is None:
            return None, 0.0
        date = good[0].unit.date
        if "relative" in self._f:
            text = good[0].unit.text.lower()
            if re.search(r"\byesterday\b", text):
                date = date - _dt.timedelta(days=1)
            else:
                m = re.search(r"\b(\w+) days ago\b", text)
                if m:
                    n = _SMALL_NUMBERS.get(m.group(1))
                    if n is None and m.group(1).isdigit():
                        n = int(m.group(1))
                    if n:
                        date = date - _dt.timedelta(days=n)
        return date, good[0].rel

    def _read_temporal(self, index, question, qclass, qdate):
        text = re.sub(r"[?.!]\s*$", "", question.strip())
        if qclass in ("days_ago", "month"):
            m = re.search(r"\bdid i (.+)$", text, re.I)
            phrase = m.group(1) if m else text
            date, rel = self._event_date(index, phrase)
            if date is None:
                return None, 0.55, None
            if "temporal" not in self._f:
                return date.isoformat(), 0.2, None
            if qclass == "month":
                return _MONTHS[date.month - 1], min(0.95, 0.3 + 0.65 * rel), None
            if qdate is None:
                return None, 0.3, None
            return f"{(qdate - date).days} days", min(0.95, 0.3 + 0.65 * rel), None
        if qclass == "days_between":
            m = re.search(r"between the day i (.+?) and the day i (.+)$", text, re.I)
        else:
            m = re.search(r"first[:,]?\s*(.+?) or (.+)$", text, re.I)
        if not m:
            return None, 0.3, None
        a, b = m.group(1).strip(), m.group(2).strip()
        if "temporal" not in self._f:
            if qclass == "order":
                return a, 0.2, b  # an uninformed guess: the first option
            date, _rel = self._event_date(index, a)
            return (date.isoformat() if date else None), 0.2, None
        da, ra = self._event_date(index, a)
        db, rb = self._event_date(index, b)
        if da is None or db is None:
            return None, 0.5, None
        conf = min(0.95, 0.3 + 0.65 * min(ra, rb))
        if qclass == "days_between":
            return f"{abs((da - db).days)} days", conf, None
        return (a if da <= db else b), conf, None

    def _read_aggregate(self, index, question, qclass):
        hits, banned = self._retrieve(index, question, None)
        good = self._confident(hits)
        if not good:
            return None, 0.55, None
        if "aggregate" not in self._f:
            good = good[:1]
        conf = min(0.9, 0.2 + 0.6 * (sum(h.rel for h in good) / len(good)))
        if "aggregate" not in self._f:
            conf = min(conf, 0.25)
        if qclass == "sum":
            total, found = 0, False
            seen: set = set()
            for h in good:
                if h.unit.order in seen:
                    continue
                seen.add(h.unit.order)
                nums = [int(x) for x in _NUM.findall(h.unit.text)]
                if nums:
                    total += nums[0]
                    found = True
            return (str(total) if found else None), conf, None
        items: list[str] = []
        seen_norm: set = set()
        for h in sorted(good, key=lambda h: h.unit.order):
            span = _pick_span(h.unit.text, banned, True)
            if not span:
                continue
            norm = span.lower()
            if norm not in seen_norm:
                seen_norm.add(norm)
                items.append(span)
        if not items:
            return None, 0.4, None
        if qclass == "count":
            return str(len(items)), conf, None
        if len(items) == 1:
            return items[0], conf, None
        return ", ".join(items[:-1]) + " and " + items[-1], conf, None

    # -- public API
    def answer(self, probe: Probe) -> Answer:
        cfg = self.config
        index = _CACHE.get(probe, "assistant" in self._f, "exchange" in self._f and "assistant" in self._f)
        qclass = classify_question(probe.question)
        qdate = parse_date(probe.question_date)
        if qclass in ("days_ago", "days_between", "order", "month"):
            text, conf, alt = self._read_temporal(index, probe.question, qclass, qdate)
        elif qclass in ("count", "sum", "list"):
            text, conf, alt = self._read_aggregate(index, probe.question, qclass)
        elif qclass == "assistant":
            if "assistant" in self._f:
                text, conf, alt = self._read_fact(index, probe.question, ("assistant",), want_proper=False)
            else:
                text, conf, alt = self._read_fact(index, probe.question, ("user",), want_proper=False)
                conf = min(conf, 0.2)
        elif qclass == "preference":
            text, conf, alt = self._read_preference(index, probe.question)
        else:
            text, conf, alt = self._read_fact(index, probe.question, ("user",))
        if text is None:
            return Answer(ABSTAIN_TEXT, confidence=conf, meta={"class": qclass, "abstained": True})
        if cfg.style == "sentence":
            out = f"Based on what you told me, it's {text}."
        elif cfg.style == "verbose":
            out = f"Based on our earlier conversations, the answer is {text}."
            if alt:
                out += f" You also mentioned {alt} at another point."
        else:
            out = text
        return Answer(out, confidence=conf, meta={"class": qclass, "abstained": False})
