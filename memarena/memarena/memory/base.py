"""Interface every memory system implements, and the harness that runs one.

A memory system receives a ``Probe`` (question, question date, conversation
history) and returns an ``Answer``.  It never receives an ``AnswerKey``.

``run_system`` is the only way systems are executed for evaluation.  It turns
exceptions into ``status="error"`` rows instead of letting them masquerade as
wrong answers, and records wall-clock time per probe.
"""

from __future__ import annotations

import time
from typing import Iterable, Mapping, Protocol

from ..benchmark.schema import Answer, Probe

__all__ = ["MemorySystem", "run_system", "ABSTAIN_TEXT"]

ABSTAIN_TEXT = "I don't have that information."


class MemorySystem(Protocol):
    name: str

    def answer(self, probe: Probe) -> Answer: ...

    def config_hash(self) -> str: ...


def run_system(system: MemorySystem, probes: Iterable[Probe]) -> dict[str, Answer]:
    """Run ``system`` on every probe and return probe id -> Answer."""
    out: dict[str, Answer] = {}
    for probe in probes:
        start = time.perf_counter()
        try:
            ans = system.answer(probe)
            if not isinstance(ans, Answer):
                raise TypeError(f"system returned {type(ans).__name__}, expected Answer")
        except Exception as exc:  # noqa: BLE001 - any failure is a harness-visible error row
            ans = Answer(text="", confidence=0.0, status="error", meta={"error": f"{type(exc).__name__}: {exc}"})
        elapsed = time.perf_counter() - start
        meta: Mapping = {**dict(ans.meta), "seconds": elapsed}
        out[probe.probe_id] = Answer(ans.text, ans.confidence, ans.status, meta)
    return out
