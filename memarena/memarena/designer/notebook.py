"""A lab notebook other designers can check.

Every experiment a designer runs is appended here as one JSON line, linked to
the previous line by hash.  Failed ideas stay in: there is no way to drop an
entry without breaking the chain.  ``verify`` re-checks the chain;
``Designer.verify_notebook`` additionally re-runs sampled experiments and
confirms the recorded numbers are reproduced exactly.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from ..util import canonical_json, sha256_hex

__all__ = ["Notebook", "NOTEBOOK_GENESIS"]

NOTEBOOK_GENESIS = sha256_hex("memarena:notebook:genesis")


def _entry_hash(index: int, kind: str, body: Mapping, prev: str) -> str:
    return sha256_hex(canonical_json({"index": index, "kind": kind, "body": body, "prev": prev}))


class Notebook:
    def __init__(self, path: str | None = None):
        self.path = path
        self.entries: list[dict] = []
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self.entries.append(json.loads(line))
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
        prev = NOTEBOOK_GENESIS
        for n, e in enumerate(self.entries, start=1):
            if e.get("index") != n:
                return False, f"entry {n} is out of order"
            if e.get("prev") != prev:
                return False, f"entry {n} does not link to its predecessor"
            if e.get("hash") != _entry_hash(n, e["kind"], e["body"], e["prev"]):
                return False, f"entry {n} was altered"
            prev = e["hash"]
        return True, "ok"

    def of_kind(self, kind: str) -> list[dict]:
        return [e for e in self.entries if e["kind"] == kind]
