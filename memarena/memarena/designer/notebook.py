"""A lab notebook other designers can check.

Every experiment a designer runs is appended here as one JSON line, linked to
the previous line by hash.  Inside one file that makes edits and omissions
visible: an entry cannot be changed or dropped without re-linking everything
after it.

The chain does not stop its author from doing exactly that.  Whoever holds the
file can rewrite it from any point and obtain a different, equally valid chain,
so the chain alone says nothing about *when* an entry was written.  What fixes
a notebook in time is a copy published where its author cannot change it (the
``anchor`` of ``Designer.preregister``); ``extends`` compares a notebook with
such a copy.  ``Designer.verify_notebook`` checks the claims a notebook makes:
it recomputes reports from the logged entries and re-runs experiments.
"""

from __future__ import annotations

import json
import os
from typing import Mapping

from ..util import canonical_json, sha256_hex

__all__ = ["Notebook", "NOTEBOOK_GENESIS"]

NOTEBOOK_GENESIS = sha256_hex("memarena:notebook:genesis")

_ENTRY_KEYS = {"index", "kind", "body", "prev", "hash"}


def _entry_hash(index: int, kind: str, body: Mapping, prev: str) -> str:
    return sha256_hex(canonical_json({"index": index, "kind": kind, "body": body, "prev": prev}))


def _no_duplicates(pairs):
    """A JSON object in which a key occurs twice means different things to different readers: refuse it."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


class Notebook:
    def __init__(self, path: str | None = None):
        self.path = path
        self.entries: list[dict] = []
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for number, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        self.entries.append(json.loads(line, object_pairs_hook=_no_duplicates))
                    except (ValueError, RecursionError) as exc:
                        raise ValueError(f"notebook failed verification: line {number} is not JSON") from exc
            ok, why = self.verify()
            if not ok:
                raise ValueError(f"notebook failed verification: {why}")

    @property
    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else NOTEBOOK_GENESIS

    def add(self, kind: str, body: Mapping) -> dict:
        body = json.loads(canonical_json(body))
        index = len(self.entries) + 1
        entry = {"index": index, "kind": kind, "body": body, "prev": self.head}
        entry["hash"] = _entry_hash(index, kind, body, entry["prev"])
        self.entries.append(entry)
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def verify(self) -> tuple[bool, str]:
        """Check that every entry is well formed and linked to its predecessor.

        An entry is an object with a string ``kind`` and an object ``body``;
        readers of a verified notebook rely on that.  Numbers that are not
        finite cannot be hashed canonically and would compare as neither equal
        nor different, so an entry holding one is rejected here, as is one
        nested too deeply to be hashed.
        """
        prev = NOTEBOOK_GENESIS
        for n, e in enumerate(self.entries, start=1):
            if not isinstance(e, dict) or set(e.keys()) != _ENTRY_KEYS:
                return False, f"entry {n} is not a notebook entry (it must have exactly the fields {sorted(_ENTRY_KEYS)})"
            if not isinstance(e["kind"], str) or not isinstance(e["body"], dict):
                return False, f"entry {n} is not a notebook entry"
            if type(e["index"]) is not int or e["index"] != n:
                return False, f"entry {n} is out of order"
            if e["prev"] != prev:
                return False, f"entry {n} does not link to its predecessor"
            try:
                expected = _entry_hash(n, e["kind"], e["body"], e["prev"])
            except ValueError:
                return False, f"entry {n} holds a number that is not finite"
            except RecursionError:
                return False, f"entry {n} is nested too deeply"
            if e["hash"] != expected:
                return False, f"entry {n} was altered"
            prev = e["hash"]
        return True, "ok"

    def extends(self, earlier: "Notebook") -> bool:
        """True if ``earlier`` is this notebook as it stood at some point.

        That is the question a published copy answers: every entry of the copy
        must still be here, unchanged and in the same place.  A notebook that
        was re-linked after the copy was published fails this, however valid
        its own chain is.
        """
        n = len(earlier.entries)
        return n <= len(self.entries) and earlier.entries == self.entries[:n]

    def of_kind(self, kind: str) -> list[dict]:
        return [e for e in self.entries if e["kind"] == kind]
