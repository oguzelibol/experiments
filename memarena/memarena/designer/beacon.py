"""Public randomness for held-out worlds.

A designer who derives held-out worlds from its own notebook can re-roll the
notebook until the worlds suit the design.  A randomness beacon removes that
freedom: the pre-registration names a beacon round that has not been produced
yet, the notebook is published, and only then does the round's randomness
exist.  The worlds depend on it, so nobody could have known them when the
design was committed.  The argument needs the publication: a notebook only its
author has seen can still be rewritten once the randomness is out.

The protocol has two steps.  ``commit`` names a *future* round; the result goes
into the pre-registration entry.  ``reveal`` returns that round's randomness,
waiting until the beacon has produced it.  A verifier calls ``reveal`` for the
recorded commitment and must get the recorded randomness.

``DrandBeacon`` talks to drand, the League of Entropy's public beacon.
``FakeBeacon`` is a deterministic stand-in for tests.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.request
from typing import Any, Callable, Mapping, Protocol

from ..util import sha256_hex

__all__ = ["Beacon", "BeaconError", "DrandBeacon", "FakeBeacon", "is_randomness", "public_beacons"]

DRAND_URL = "https://api.drand.sh"
_HEX = frozenset("0123456789abcdef")
_MIN_RANDOMNESS_CHARS = 20  # held-out world seeds use the first 20 hex characters


class BeaconError(RuntimeError):
    """The beacon could not provide the randomness that was asked for."""


class Beacon(Protocol):
    """A source of public randomness that is published in numbered rounds."""

    def commit(self) -> dict:
        """Name a round that does not exist yet: ``{"source": ..., "round": ...}``."""

    def reveal(self, commitment: Mapping) -> str:
        """Hex randomness of the committed round, waiting until the beacon has produced it."""


def is_randomness(value: Any) -> bool:
    """Whether ``value`` can seed held-out worlds: lower-case hex, long enough for the seed prefix."""
    return isinstance(value, str) and len(value) >= _MIN_RANDOMNESS_CHARS and set(value) <= _HEX


def _committed_round(commitment: Mapping, source: str) -> int:
    """The round a commitment names, refusing a commitment made to a different beacon."""
    number = commitment.get("round")
    if commitment.get("source") != source or isinstance(number, bool) or not isinstance(number, int):
        raise BeaconError(f"not a commitment to a round of the {source!r} beacon: {dict(commitment)!r}")
    return number


def _get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "randomness-beacon-client"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class DrandBeacon:
    """drand's default chain, which publishes one round every 30 seconds.

    ``commit`` names the round ``lead_rounds`` after the latest one.  With the
    default of two that round is about 30 to 60 seconds away, which is the
    time the designer has to publish its pre-registration: a pre-registration
    published after the round is no evidence that the design came first.
    ``pending`` says whether the round is still to come; raise ``lead_rounds``
    if publishing takes longer than the lead.

    ``reveal`` polls until the round is out and gives up with a ``BeaconError``
    after ``timeout`` seconds.  ``fetch``, ``clock`` and ``sleep`` exist so the
    polling can be exercised without a network.
    """

    source = "drand"
    # The League of Entropy's default chain: round 1 at GENESIS, one round every PERIOD seconds.
    GENESIS = 1595431050
    PERIOD = 30

    @classmethod
    def round_time(cls, number: int) -> str:
        """When a round of the default chain was (or will be) produced, in UTC."""
        import datetime as _dt
        at = cls.GENESIS + (int(number) - 1) * cls.PERIOD
        return _dt.datetime.fromtimestamp(at, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def __init__(
        self, url: str = DRAND_URL, lead_rounds: int = 2, timeout: float = 180.0, poll_seconds: float = 3.0,
        fetch: Callable[[str], Any] | None = None, clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if lead_rounds < 1:
            raise ValueError("lead_rounds must be at least 1: the committed round has to lie in the future")
        self.url = url.rstrip("/")
        self.lead_rounds = lead_rounds
        self.timeout = timeout
        self.poll_seconds = poll_seconds
        self._fetch = fetch or (lambda address: _get_json(address, timeout=10.0))
        self._clock = clock
        self._sleep = sleep

    def _round(self, which: int | str) -> dict:
        """Fetch ``/public/latest`` or ``/public/<round>`` and insist on a round number and hex randomness."""
        address = f"{self.url}/public/{which}"
        try:
            answer = self._fetch(address)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise BeaconError(f"could not read {address}: {type(exc).__name__}: {exc}") from exc
        number, randomness = (answer.get("round"), answer.get("randomness")) if isinstance(answer, dict) else (None, None)
        if isinstance(number, bool) or not isinstance(number, int) or not isinstance(randomness, str):
            raise BeaconError(f"{address} did not answer with a round and its randomness")
        if not is_randomness(randomness.lower()):
            raise BeaconError(f"{address} did not answer with hex randomness")
        if which != "latest" and number != which:
            raise BeaconError(f"{address} answered for round {number}")
        return {"round": number, "randomness": randomness.lower()}

    def commit(self) -> dict:
        return {"source": self.source, "round": self._round("latest")["round"] + self.lead_rounds}

    def pending(self, commitment: Mapping) -> bool:
        """Whether the committed round is still to come.

        A designer asks this right after publishing its pre-registration: the
        publication shows that the design came first only if the round had not
        been produced by then.
        """
        return self._round("latest")["round"] < _committed_round(commitment, self.source)

    def reveal(self, commitment: Mapping) -> str:
        wanted = _committed_round(commitment, self.source)
        deadline = self._clock() + self.timeout
        while True:
            try:
                latest = self._round("latest")
                if latest["round"] == wanted:
                    return latest["randomness"]
                if latest["round"] > wanted:
                    return self._round(wanted)["randomness"]
                state = f"the latest round is {latest['round']}"
            except BeaconError as exc:
                state = str(exc)
            if self._clock() >= deadline:
                raise BeaconError(f"drand round {wanted} could not be read within {self.timeout:.0f} seconds ({state})")
            self._sleep(self.poll_seconds)


class FakeBeacon:
    """A deterministic beacon for tests: no network, and nothing public about it.

    Each ``commit`` names the next round.  A round's randomness is a hash of the
    seed and the round number, so two FakeBeacons with different seeds disagree
    about every round; a test uses that to show what depends on the randomness
    and what does not.  ``revealed`` lists the rounds asked for so far, which
    shows when in a procedure the randomness became known.
    """

    source = "fake"

    def __init__(self, seed: int | str = 0, first_round: int = 1):
        self._seed = str(seed)
        self._next_round = first_round
        self.revealed: list[int] = []

    def commit(self) -> dict:
        number = self._next_round
        self._next_round += 1
        return {"source": self.source, "round": number}

    def reveal(self, commitment: Mapping) -> str:
        number = _committed_round(commitment, self.source)
        self.revealed.append(number)
        return sha256_hex(f"fake-beacon:{self._seed}:{number}")


def public_beacons() -> dict[str, Beacon]:
    """The beacons a verifier accepts as public, by the source name a commitment carries.

    A verifier looks the source up here and never follows an address found in
    the notebook: the notebook's author must not choose who vouches for the
    randomness.  The short timeout suits verification, where the round is long
    past; a round that still cannot be read is reported, not waited for.
    """
    return {DrandBeacon.source: DrandBeacon(timeout=20.0)}
