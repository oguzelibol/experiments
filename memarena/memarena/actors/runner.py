"""Run one unit against a board that lives elsewhere (for example over HTTP).

The simulator drives units directly.  A deployed unit instead follows the
board's ``clock`` posts: when a new round is announced it begins the round,
answers whatever arrives in its inbox, and closes the round when the next one
is announced.  Nothing else about the unit changes.
"""

from __future__ import annotations

import time
from typing import Callable

from ..util import ConfigurationError
from .unit import Unit

__all__ = ["UnitRunner"]


class UnitRunner:
    def __init__(self, unit: Unit, clock_author: str = "env", poll_seconds: float = 0.5):
        self.unit = unit
        self.clock_author = clock_author
        self.poll_seconds = poll_seconds
        self._inbox_cursor = 0
        self._clock_cursor = 0
        self._open_round: int | None = None

    def pump(self) -> int:
        """Deliver every new inbox post to the unit; returns how many were handled."""
        posts = self.unit.board.inbox(self.unit.id, since_seq=self._inbox_cursor)
        for post in posts:
            self._inbox_cursor = max(self._inbox_cursor, post.seq)
            self.unit.on_message(post)
        return len(posts)

    def _next_round(self) -> int | None:
        posts = self.unit.board.feed(self.clock_author, since=self._clock_cursor, kinds=["clock"])
        if not posts:
            return None
        self._clock_cursor = posts[-1].author_seq
        return int(posts[-1].body.get("round", posts[-1].round))

    def tick(self) -> None:
        """One polling step: follow the clock, then handle messages."""
        nxt = self._next_round()
        if nxt is not None and nxt != self._open_round:
            if self._open_round is not None:
                self.pump()
                self.unit.end_round()
            self.unit.begin_round(nxt)
            self._open_round = nxt
        if self._open_round is not None:
            self.pump()

    def run(self, should_stop: Callable[[], bool] | None = None, max_seconds: float | None = None,
            max_outage_seconds: float = 120.0) -> str:
        """Follow the clock until told to stop.  Returns why it stopped.

        A board that cannot be reached is retried with a growing pause.  The
        unit acts from what the chain says, so a step that was cut short is
        simply taken again.  After ``max_outage_seconds`` without the board the
        run ends cleanly instead of dying.
        """
        start = time.time()
        down_since: float | None = None
        pause = self.poll_seconds
        why = "stopped"
        while True:
            try:
                self.tick()
                down_since, pause = None, self.poll_seconds
            except ConfigurationError:
                raise
            except Exception as exc:  # noqa: BLE001 - the board is somebody else's machine
                now = time.time()
                down_since = down_since if down_since is not None else now
                self.unit.log.append({"round": self.unit.round, "event": "board_unreachable", "why": f"{type(exc).__name__}: {str(exc)[:160]}"})
                if now - down_since > max_outage_seconds:
                    return "the board could not be reached"
                pause = min(30.0, pause * 2)
            if should_stop is not None and should_stop():
                break
            if max_seconds is not None and time.time() - start > max_seconds:
                why = "time limit"
                break
            time.sleep(pause)
        if self._open_round is not None:
            try:
                self.pump()
                self.unit.end_round()
            except ConfigurationError:
                raise
            except Exception:  # noqa: BLE001
                why = "the board could not be reached"
            self._open_round = None
        return why
