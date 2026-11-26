"""Interface every memory system implements, and the harnesses that run one.

A memory system receives a ``Probe`` (question, question date, conversation
history) and returns an ``Answer``.  It never receives an ``AnswerKey``.

``run_system`` runs a system in this process.  It turns exceptions into
``status="error"`` rows instead of letting them masquerade as wrong answers,
and records wall-clock time per probe.  The system is handed copies of the
probes, so nothing it does to them can change the question the grader later
reads.  It is still only the right harness for a system you wrote and trust
not to attack its host: Python offers no wall inside one process, so code that
goes looking (walking stack frames, the garbage collector, module globals) can
find the answer keys its caller holds.

``run_isolated`` is the harness for a system you do not trust.  The system is
built and run in a fresh interpreter that is sent the probes and nothing else;
the keys never exist in that process.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from ..benchmark.schema import Answer, Probe, Session, Turn

__all__ = ["MemorySystem", "run_system", "run_isolated", "detached", "ABSTAIN_TEXT"]

ABSTAIN_TEXT = "I don't have that information."


class MemorySystem(Protocol):
    name: str

    def answer(self, probe: Probe) -> Answer: ...

    def config_hash(self) -> str: ...


def detached(probes: Iterable[Probe]) -> list[Probe]:
    """Copies of ``probes`` that share no object with the originals.

    Probes that share one history keep sharing one (copied) history, so a
    system that indexes a history once still does so once.
    """
    histories: dict[int, tuple] = {}
    out: list[Probe] = []
    for p in probes:
        hit = histories.get(id(p.sessions))
        if hit is None or hit[0] is not p.sessions:
            copy = tuple(
                Session(str(s.session_id), str(s.date), tuple(Turn(str(t.role), str(t.content)) for t in s.turns))
                for s in p.sessions
            )
            hit = histories[id(p.sessions)] = (p.sessions, copy)
        out.append(Probe(str(p.probe_id), str(p.question), str(p.question_date), str(p.haystack_id), hit[1]))
    return out


def run_system(system: MemorySystem, probes: Iterable[Probe]) -> dict[str, Answer]:
    """Run ``system`` on a copy of every probe and return probe id -> Answer."""
    out: dict[str, Answer] = {}
    for probe in detached(probes):
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


def _isolated_worker(factory: Callable[[], MemorySystem], probe_dicts: list) -> dict:
    system = factory()
    answers = run_system(system, [Probe.from_dict(d) for d in probe_dicts])
    return {pid: {"text": a.text, "confidence": a.confidence, "status": a.status,
                  "seconds": float(a.meta.get("seconds", 0.0)), "error": a.meta.get("error")}
            for pid, a in answers.items()}


def run_isolated(
    factory: Callable[[], MemorySystem], probes: Sequence[Probe], timeout: float | None = None
) -> dict[str, Answer]:
    """Run an untrusted system in a separate, freshly started interpreter.

    ``factory`` builds the system inside the child, so it must be picklable (a
    module-level function or a ``functools.partial`` of one).  Only the probes
    cross the process boundary.  If the child dies or exceeds ``timeout``
    seconds, every probe comes back as a ``timeout`` or ``error`` row.
    """
    probes = list(probes)
    ctx = multiprocessing.get_context("spawn")  # a clean interpreter: nothing of ours is inherited
    with ctx.Pool(1) as pool:
        pending = pool.apply_async(_isolated_worker, (factory, [p.to_dict() for p in probes]))
        try:
            raw = pending.get(timeout)
        except multiprocessing.TimeoutError:
            return {p.probe_id: Answer("", 0.0, "timeout", {"error": f"no answer within {timeout} seconds"}) for p in probes}
        except Exception as exc:  # noqa: BLE001 - the child failed; report it as rows, not as a crash here
            return {p.probe_id: Answer("", 0.0, "error", {"error": f"{type(exc).__name__}: {exc}"}) for p in probes}
    out: dict[str, Answer] = {}
    for p in probes:
        r = raw.get(p.probe_id) if isinstance(raw, dict) else None
        if not isinstance(r, dict):
            out[p.probe_id] = Answer("", 0.0, "error", {"error": "no row returned by the isolated system"})
            continue
        status = r.get("status") if r.get("status") in ("ok", "error", "truncated", "refused", "timeout") else "error"
        conf = r.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) and conf == conf else 0.5
        out[p.probe_id] = Answer(str(r.get("text", "")), min(1.0, max(0.0, conf)), status,
                                 {"seconds": r.get("seconds", 0.0), "error": r.get("error")})
    return out
