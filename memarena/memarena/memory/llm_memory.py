"""LLM-backed memory system: retrieve the relevant sessions, let a model read them.

This is the system to run on real benchmarks once a model credential is
available.  Retrieval is the same BM25 index the offline system uses, at
session granularity; reading is one model call that receives the retrieved
sessions in chronological order with their dates, the question and the date it
is asked on.  ``mode="full_context"`` skips retrieval and passes the whole
history, which is viable for LongMemEval-S-sized histories on 1M-token models.

When several questions share one history the history is placed in the system
prompt and marked cacheable, so repeated questions reuse the cached prefix.
"""

from __future__ import annotations

import json
from typing import Any

from ..benchmark.schema import Answer, Probe
from ..util import hash_obj
from .base import ABSTAIN_TEXT
from .heuristic import _CACHE, _STOP_STEMS, _stems

__all__ = ["LLMMemory", "READER_SYSTEM"]

READER_SYSTEM = """You answer a user's question about their own past conversations with an assistant.

You are given a conversation history: sessions in chronological order, each with a date and its turns. \
Answer only from what the history says.

Guidelines:
- The user may have updated a fact over time. The most recent statement is the current truth.
- For questions about counts or totals, find every relevant mention across all sessions before answering.
- For questions about time, use the session dates and the date the question is asked on. Resolve relative \
expressions such as "yesterday" against the date of the session in which they were said.
- Things said about other people (a friend, a coworker) are not facts about the user. Plans that were \
considered and dropped did not happen.
- If the history does not contain the information needed, answer exactly: "I don't have that information."
- Give the answer itself, briefly: a name, a number with its unit, a short phrase, or a short list. Do not \
add alternatives or caveats.

Also report how confident you are that your answer is correct, as a probability between 0 and 1."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}


def _render_sessions(probe: Probe, indices: list[int]) -> str:
    sessions = []
    for i in indices:
        s = probe.sessions[i]
        sessions.append(
            {
                "session": i + 1,
                "date": s.date,
                "turns": [{"role": t.role, "content": t.content} for t in s.turns],
            }
        )
    return json.dumps(sessions, ensure_ascii=False, indent=1)


class LLMMemory:
    def __init__(
        self,
        llm: Any,
        mode: str = "retrieve",
        top_k_sessions: int = 10,
        effort: str | None = None,
        name: str | None = None,
        cache_history: bool = False,
    ):
        if mode not in ("retrieve", "full_context"):
            raise ValueError("mode must be 'retrieve' or 'full_context'")
        self.llm = llm
        self.mode = mode
        # In full-context mode the history can be marked for the provider's prompt cache.  That pays
        # only when several questions share one history (LoCoMo, the generated benchmark); where
        # every question has its own (LongMemEval) it only adds the cost of writing the cache.
        self.cache_history = cache_history
        self.top_k_sessions = top_k_sessions
        self.effort = effort
        self.name = name or f"llm-memory:{mode}:{getattr(llm, 'model', 'unknown')}"

    def config_hash(self) -> str:
        return hash_obj(
            {
                "kind": "llm-memory", "mode": self.mode, "top_k_sessions": self.top_k_sessions,
                "model": getattr(self.llm, "model", "unknown"), "system": READER_SYSTEM, "effort": self.effort,
            }
        )[:16]

    def _select(self, probe: Probe) -> list[int]:
        n = len(probe.sessions)
        if self.mode == "full_context" or n <= self.top_k_sessions:
            return list(range(n))
        index = _CACHE.get(probe, True, False)
        query = {s: 1.0 for s in _stems(probe.question) if s not in _STOP_STEMS}
        scores, _ideal = index.search(query, 1.2, 0.75)
        per_session: dict[int, float] = {}
        for i, sc in scores.items():
            s_idx = index.units[i].session_idx
            per_session[s_idx] = max(per_session.get(s_idx, 0.0), sc)
        ranked = sorted(per_session, key=lambda s: (-per_session[s], s))[: self.top_k_sessions]
        return sorted(ranked)

    def answer(self, probe: Probe) -> Answer:
        selected = self._select(probe)
        history = _render_sessions(probe, selected)
        shared_history = self.mode == "full_context"
        if shared_history:
            system = READER_SYSTEM + "\n\nConversation history (JSON):\n" + history
            prompt = f"The question is asked on {probe.question_date}.\n\nQuestion: {probe.question}"
        else:
            system = READER_SYSTEM
            prompt = (
                f"Conversation history (JSON, the sessions most relevant to the question):\n{history}\n\n"
                f"The question is asked on {probe.question_date}.\n\nQuestion: {probe.question}"
            )
        result = self.llm.complete(
            system=system, prompt=prompt, json_schema=_SCHEMA, max_tokens=16000,
            effort=self.effort, purpose="memory-reader",
            # Caching the history only pays when another question reads the same one.
            cache_system=shared_history and self.cache_history,
        )
        meta = {"sessions_used": len(selected), "model": result.model, "usage": dict(result.usage)}
        if result.status == "refused":
            return Answer("", 0.0, status="refused", meta=meta)
        if result.status == "truncated":
            # A reply cut off mid-way is a fragment of JSON, not an answer: it is reported, never graded.
            return Answer("", 0.0, status="error", meta={**meta, "error": "the reply was cut off before the answer was complete"})
        if result.status != "ok" or not isinstance(result.parsed, dict):
            return Answer("", 0.0, status="error", meta={**meta, "error": result.text[:300]})
        text = str(result.parsed.get("answer", "")).strip()
        conf = result.parsed.get("confidence", 0.5)
        try:
            conf = min(1.0, max(0.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.5
        return Answer(text or ABSTAIN_TEXT, conf, meta=meta)
