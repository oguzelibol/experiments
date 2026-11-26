"""Small shared helpers: canonical serialisation, hashing, commitments, seeded RNGs.

Everything that ends up inside a receipt or a notebook entry goes through
``canonical_json`` so that two parties who hold the same data always compute the
same hash.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Iterable, Sequence

__all__ = [
    "canonical_json",
    "sha256_hex",
    "hash_obj",
    "merkle_root",
    "derive_seed",
    "make_rng",
    "round_floats",
    "commit",
    "verify_commit",
    "clamp",
]

FLOAT_DECIMALS = 6


def round_floats(obj: Any, nd: int = FLOAT_DECIMALS) -> Any:
    """Recursively round floats so hashes are stable across platforms/libm."""
    if isinstance(obj, float):
        r = round(obj, nd)
        # normalise negative zero
        return 0.0 if r == 0 else r
    if isinstance(obj, dict):
        return {str(k): round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, nd) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, ASCII only, rounded floats."""
    try:
        return json.dumps(
            round_floats(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except ValueError as exc:  # NaN or infinity: not JSON, and poison for every comparison downstream
        raise ValueError("non-finite number in data that must be canonical JSON") from exc


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def merkle_root(leaves: Sequence[str]) -> str:
    """Merkle root over hex-string leaves (duplicates the last node on odd levels).

    A root commits to an ordered list of per-item records without publishing
    them; a single item can later be opened with a log-sized proof.
    """
    if not leaves:
        return sha256_hex("merkle:empty")
    level = [sha256_hex("leaf:" + leaf) for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            sha256_hex("node:" + level[i] + level[i + 1])
            for i in range(0, len(level), 2)
        ]
    return level[0]


def derive_seed(*parts: Any) -> int:
    """Derive a 63-bit seed from arbitrary labelled parts (stable across runs)."""
    text = "\x1f".join(str(p) for p in parts)
    return int(sha256_hex(text)[:16], 16) >> 1


def make_rng(*parts: Any) -> random.Random:
    return random.Random(derive_seed(*parts))


def commit(value: Any, nonce: str) -> str:
    """Hash commitment to ``value``. Hiding as long as ``nonce`` is unpredictable."""
    return sha256_hex("commit:" + nonce + ":" + canonical_json(value))


def verify_commit(commitment: str, value: Any, nonce: str) -> bool:
    return commit(value, nonce) == commitment


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
