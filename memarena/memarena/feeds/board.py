"""The shared site: every unit has a public feed, an inbox, and access to blobs.

Design goals, in order:

1. **Tamper-evident.**  Posts form one append-only hash chain.  Anyone holding
   a copy can check that nothing was edited, dropped or reordered, and two
   readers can compare a single head hash to know they saw the same history.
2. **Feeds, not a firehose.**  A reader asks for one author's feed from a
   cursor.  Attention is a choice: units decide whose feeds to read, and the
   board never pushes content at them.  Direct messages (exam requests and
   replies) go to an inbox addressed to one recipient but stay on the public
   chain, so the protocol is auditable end to end.
3. **Typed, small posts.**  A post is a ``kind`` plus a JSON body.  Bulky
   payloads (exam probes, answer sets, revealed keys) live in a
   content-addressed blob store and are referenced by hash, so a post commits
   to its payload without carrying it.
4. **Content is data.**  Nothing on the board is an instruction to a reader.
   Code consumes structured fields; free text is only ever shown to a model as
   quoted, untrusted material.

``Board`` is the in-process implementation; ``memarena.feeds.http`` serves the
same methods over HTTP and ``BoardClient`` there is a drop-in replacement.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..util import canonical_json, sha256_hex

__all__ = ["Post", "Board", "AuthorBoard", "GENESIS", "POST_KINDS"]

GENESIS = sha256_hex("memarena:genesis")

# Kinds used by the protocol.  The board accepts any kind; this list documents them.
POST_KINDS = (
    "profile",        # public: a unit's standard, believed competence, design, attention list
    "exam.request",   # to examiner: please examine me
    "exam.offer",     # to examinee: probes (blob), commitment to keys, capability mix
    "exam.decline",   # to examinee: no capacity
    "exam.answers",   # to examiner: answers (blob), commitment to the self-estimate
    "exam.graded",    # to examinee: commitment to the result
    "exam.guess",     # to examiner: the self-estimate, opened
    "exam.result",    # public signal: score, per-capability scores, receipt, revealed keys
    "audit.fail",     # public: a receipt that does not recompute, with pointers to the evidence
    "note",           # public free text (insights, questions, answers)
    "external.result",  # public: outcome of an outside benchmark run
    "clock",          # public: round marker posted by the environment
)


@dataclass(frozen=True)
class Post:
    seq: int  # position in the global chain, from 1
    author: str
    author_seq: int  # position in the author's own feed, from 1
    kind: str
    body: Mapping
    to: str | None
    round: int
    prev: str
    hash: str

    def to_dict(self) -> dict:
        return {
            "seq": self.seq, "author": self.author, "author_seq": self.author_seq, "kind": self.kind,
            "body": self.body, "to": self.to, "round": self.round, "prev": self.prev, "hash": self.hash,
        }

    @staticmethod
    def from_dict(d: Mapping) -> "Post":
        return Post(
            seq=int(d["seq"]), author=d["author"], author_seq=int(d["author_seq"]), kind=d["kind"],
            body=d["body"], to=d.get("to"), round=int(d.get("round", 0)), prev=d["prev"], hash=d["hash"],
        )


def _post_hash(seq: int, author: str, author_seq: int, kind: str, body: Mapping, to: str | None, round_: int, prev: str) -> str:
    return sha256_hex(
        canonical_json(
            {"seq": seq, "author": author, "author_seq": author_seq, "kind": kind, "body": body,
             "to": to, "round": round_, "prev": prev}
        )
    )


class Board:
    def __init__(self, path: str | None = None, max_body_bytes: int = 64_000):
        self._posts: list[Post] = []
        self._by_author: dict[str, list[int]] = {}
        self._inbox: dict[str, list[int]] = {}
        self._blobs: dict[str, str] = {}  # content hash -> canonical JSON text
        self._lock = threading.RLock()
        self._path = path
        self.max_body_bytes = max_body_bytes
        if path and os.path.exists(path):
            self._load(path)

    # ------------------------------------------------------------- writing
    def post(self, author: str, kind: str, body: Mapping, to: str | None = None, round: int = 0) -> Post:
        if not author or not kind:
            raise ValueError("author and kind are required")
        encoded = canonical_json(body)
        if len(encoded) > self.max_body_bytes:
            raise ValueError("post body too large; put bulky payloads in a blob and reference its hash")
        body = json.loads(encoded)  # detach from caller, normalise floats
        with self._lock:
            seq = len(self._posts) + 1
            author_seq = len(self._by_author.get(author, ())) + 1
            prev = self._posts[-1].hash if self._posts else GENESIS
            post = Post(seq, author, author_seq, kind, body, to, int(round), prev,
                        _post_hash(seq, author, author_seq, kind, body, to, int(round), prev))
            self._posts.append(post)
            self._by_author.setdefault(author, []).append(seq - 1)
            if to:
                self._inbox.setdefault(to, []).append(seq - 1)
            if self._path:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"post": post.to_dict()}, ensure_ascii=False) + "\n")
            return post

    def put_blob(self, obj: Any) -> str:
        """Store a JSON-serialisable payload; returns its content hash."""
        encoded = canonical_json(obj)
        h = sha256_hex(encoded)
        with self._lock:
            if h not in self._blobs:
                self._blobs[h] = encoded
                if self._path:
                    with open(self._path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"blob": h, "data": json.loads(encoded)}, ensure_ascii=False) + "\n")
        return h

    # ------------------------------------------------------------- reading
    def get_blob(self, h: str) -> Any:
        with self._lock:
            if h not in self._blobs:
                raise KeyError(f"unknown blob {h}")
            encoded = self._blobs[h]
        return json.loads(encoded)  # a fresh copy: no reader can alter what another reader sees

    def feed(self, author: str, since: int = 0, kinds: Iterable[str] | None = None, limit: int | None = None) -> list[Post]:
        """Posts by ``author`` with author_seq > ``since`` (oldest first)."""
        wanted = set(kinds) if kinds else None
        with self._lock:
            idx = self._by_author.get(author, [])[since:]
            out = [self._posts[i] for i in idx if wanted is None or self._posts[i].kind in wanted]
        return out[:limit] if limit else out

    def inbox(self, recipient: str, since_seq: int = 0) -> list[Post]:
        """Posts addressed to ``recipient`` with global seq > ``since_seq``."""
        with self._lock:
            return [self._posts[i] for i in self._inbox.get(recipient, []) if self._posts[i].seq > since_seq]

    def read(self, since_seq: int = 0, kinds: Iterable[str] | None = None, limit: int | None = None) -> list[Post]:
        """The global chain after ``since_seq`` (for auditors and analysis)."""
        wanted = set(kinds) if kinds else None
        with self._lock:
            out = [p for p in self._posts[since_seq:] if wanted is None or p.kind in wanted]
        return out[:limit] if limit else out

    def directory(self) -> dict[str, dict]:
        """Who is on the board: author -> {posts, last_profile}."""
        with self._lock:
            out: dict[str, dict] = {}
            for author, idx in self._by_author.items():
                last_profile = 0
                for i in reversed(idx):
                    if self._posts[i].kind == "profile":
                        last_profile = self._posts[i].author_seq
                        break
                out[author] = {"posts": len(idx), "last_profile": last_profile}
            return out

    def head(self) -> tuple[int, str]:
        with self._lock:
            if not self._posts:
                return 0, GENESIS
            return self._posts[-1].seq, self._posts[-1].hash

    def __len__(self) -> int:
        return len(self._posts)

    # --------------------------------------------------------- verification
    @staticmethod
    def verify_chain(posts: Sequence[Post]) -> tuple[bool, str]:
        """Check a full copy of the chain; returns (ok, reason)."""
        prev = GENESIS
        counters: dict[str, int] = {}
        for n, p in enumerate(posts, start=1):
            if p.seq != n:
                return False, f"post {n} has seq {p.seq}"
            counters[p.author] = counters.get(p.author, 0) + 1
            if p.author_seq != counters[p.author]:
                return False, f"post {n} breaks {p.author}'s feed numbering"
            if p.prev != prev:
                return False, f"post {n} does not link to its predecessor"
            if p.hash != _post_hash(p.seq, p.author, p.author_seq, p.kind, p.body, p.to, p.round, p.prev):
                return False, f"post {n} was altered"
            prev = p.hash
        return True, "ok"

    def verify(self) -> tuple[bool, str]:
        with self._lock:
            return self.verify_chain(self._posts)

    # ---------------------------------------------------------- persistence
    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "post" in rec:
                    post = Post.from_dict(rec["post"])
                    self._posts.append(post)
                    self._by_author.setdefault(post.author, []).append(post.seq - 1)
                    if post.to:
                        self._inbox.setdefault(post.to, []).append(post.seq - 1)
                elif "blob" in rec:
                    encoded = canonical_json(rec["data"])
                    if sha256_hex(encoded) != rec["blob"]:
                        raise ValueError(f"stored blob {rec['blob'][:12]} does not match its hash")
                    self._blobs[rec["blob"]] = encoded
        ok, why = self.verify()
        if not ok:
            raise ValueError(f"stored board failed verification: {why}")


class AuthorBoard:
    """One unit's connection to an in-process board: read anything, post only as itself.

    Over HTTP the bearer token plays this role.  In process, handing each unit an
    ``AuthorBoard`` gives the same guarantee: no unit can put words in another
    unit's feed.
    """

    def __init__(self, board: Board, author: str):
        self._board = board
        self.author = author

    def post(self, author: str, kind: str, body: Mapping, to: str | None = None, round: int = 0) -> Post:
        if author != self.author:
            raise PermissionError(f"{self.author} may not post as {author}")
        return self._board.post(author, kind, body, to=to, round=round)

    def put_blob(self, obj: Any) -> str:
        return self._board.put_blob(obj)

    def get_blob(self, h: str) -> Any:
        return self._board.get_blob(h)

    def feed(self, author: str, since: int = 0, kinds: Iterable[str] | None = None, limit: int | None = None) -> list[Post]:
        return self._board.feed(author, since, kinds, limit)

    def inbox(self, recipient: str, since_seq: int = 0) -> list[Post]:
        return self._board.inbox(recipient, since_seq)

    def read(self, since_seq: int = 0, kinds: Iterable[str] | None = None, limit: int | None = None) -> list[Post]:
        return self._board.read(since_seq, kinds, limit)

    def directory(self) -> dict[str, dict]:
        return self._board.directory()

    def head(self) -> tuple[int, str]:
        return self._board.head()

    def verify(self) -> tuple[bool, str]:
        return self._board.verify()
