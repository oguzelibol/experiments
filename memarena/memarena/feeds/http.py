"""HTTP transport for the board (standard library only).

    GET  /v1/head                         -> {"seq": n, "hash": "..."}
    GET  /v1/directory                    -> {author: {"posts": n, "last_profile": k}}
    GET  /v1/feed/<author>?since=&kinds=  -> [post, ...]      one author's feed
    GET  /v1/inbox/<recipient>?since=     -> [post, ...]      posts addressed to a unit
    GET  /v1/posts?since=&kinds=&limit=   -> [post, ...]      the whole chain
    GET  /v1/blobs/<hash>                 -> {"data": ...}
    POST /v1/posts   {kind, body, to, round}   -> post        (author taken from the token)
    POST /v1/blobs   {"data": ...}             -> {"hash": "..."}

Reads are public.  Writes need ``Authorization: Bearer <token>``; the token
identifies the author, so a unit cannot post under another unit's name.

``BoardClient`` exposes the same methods as ``Board``.  Its ``verify`` downloads
the chain and checks it locally, so a client never has to trust the server's
word that history is intact.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Mapping

from .board import Board, Post

__all__ = ["BoardServer", "BoardClient"]


class BoardServer:
    def __init__(self, board: Board, tokens: Mapping[str, str] | None = None, host: str = "127.0.0.1", port: int = 0,
                 reserved: tuple[str, ...] = ()):
        """``tokens`` maps bearer token -> author.  Without it any caller may post (tests only).

        ``reserved`` author names can never be claimed without a token, even on
        an open board; use it for the round clock.
        """
        self.board = board
        self.tokens = dict(tokens or {})
        self.reserved = tuple(reserved)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # silence default stderr logging
                return

            def _send(self, code: int, payload: Any) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _author(self) -> str | None:
                auth = self.headers.get("Authorization", "")
                if not auth.startswith("Bearer "):
                    return None
                return outer.tokens.get(auth[len("Bearer "):])

            def do_GET(self):  # noqa: N802
                url = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(url.query)
                parts = [urllib.parse.unquote(p) for p in url.path.strip("/").split("/")]
                since = int(q.get("since", ["0"])[0])
                kinds = q.get("kinds", [""])[0].split(",") if q.get("kinds", [""])[0] else None
                limit = int(q["limit"][0]) if "limit" in q else None
                try:
                    if parts == ["v1", "head"]:
                        seq, h = outer.board.head()
                        return self._send(200, {"seq": seq, "hash": h})
                    if parts == ["v1", "directory"]:
                        return self._send(200, outer.board.directory())
                    if len(parts) == 3 and parts[:2] == ["v1", "feed"]:
                        return self._send(200, [p.to_dict() for p in outer.board.feed(parts[2], since, kinds, limit)])
                    if len(parts) == 3 and parts[:2] == ["v1", "inbox"]:
                        return self._send(200, [p.to_dict() for p in outer.board.inbox(parts[2], since)])
                    if parts == ["v1", "posts"]:
                        return self._send(200, [p.to_dict() for p in outer.board.read(since, kinds, limit)])
                    if len(parts) == 3 and parts[:2] == ["v1", "blobs"]:
                        return self._send(200, {"data": outer.board.get_blob(parts[2])})
                except KeyError as exc:
                    return self._send(404, {"error": str(exc)})
                except ValueError as exc:
                    return self._send(400, {"error": str(exc)})
                return self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                url = urllib.parse.urlparse(self.path)
                parts = url.path.strip("/").split("/")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, json.JSONDecodeError):
                    return self._send(400, {"error": "invalid JSON"})
                author = self._author()
                if outer.tokens and author is None:
                    return self._send(401, {"error": "missing or unknown bearer token"})
                try:
                    if parts == ["v1", "posts"]:
                        claimed = payload.get("author")
                        if author is None:
                            author = claimed
                            if author in outer.reserved:
                                return self._send(403, {"error": "that author name is reserved"})
                        elif claimed and claimed != author:
                            return self._send(403, {"error": "token does not belong to that author"})
                        post = outer.board.post(
                            author, payload["kind"], payload.get("body", {}),
                            to=payload.get("to"), round=int(payload.get("round", 0)),
                        )
                        return self._send(200, post.to_dict())
                    if parts == ["v1", "blobs"]:
                        return self._send(200, {"hash": outer.board.put_blob(payload["data"])})
                except (KeyError, ValueError, TypeError) as exc:
                    return self._send(400, {"error": str(exc)})
                return self._send(404, {"error": "not found"})

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "BoardServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)


class BoardClient:
    """Drop-in replacement for ``Board`` that talks to a ``BoardServer``."""

    def __init__(self, url: str, token: str | None = None, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _call(self, method: str, path: str, payload: Any = None) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        # The board is a local or directly reachable service; do not route it through a proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if exc.code == 404:
                raise KeyError(detail) from exc
            raise ValueError(f"board returned {exc.code}: {detail}") from exc

    @staticmethod
    def _q(**params) -> str:
        clean = {k: v for k, v in params.items() if v not in (None, "", 0)}
        return ("?" + urllib.parse.urlencode(clean)) if clean else ""

    def post(self, author: str, kind: str, body: Mapping, to: str | None = None, round: int = 0) -> Post:
        return Post.from_dict(
            self._call("POST", "/v1/posts", {"author": author, "kind": kind, "body": body, "to": to, "round": round})
        )

    def put_blob(self, obj: Any) -> str:
        return self._call("POST", "/v1/blobs", {"data": obj})["hash"]

    def get_blob(self, h: str) -> Any:
        return self._call("GET", f"/v1/blobs/{urllib.parse.quote(h)}")["data"]

    def feed(self, author: str, since: int = 0, kinds: Iterable[str] | None = None, limit: int | None = None) -> list[Post]:
        q = self._q(since=since, kinds=",".join(kinds) if kinds else None, limit=limit)
        return [Post.from_dict(d) for d in self._call("GET", f"/v1/feed/{urllib.parse.quote(author)}{q}")]

    def inbox(self, recipient: str, since_seq: int = 0) -> list[Post]:
        return [Post.from_dict(d) for d in self._call("GET", f"/v1/inbox/{urllib.parse.quote(recipient)}{self._q(since=since_seq)}")]

    def read(self, since_seq: int = 0, kinds: Iterable[str] | None = None, limit: int | None = None) -> list[Post]:
        q = self._q(since=since_seq, kinds=",".join(kinds) if kinds else None, limit=limit)
        return [Post.from_dict(d) for d in self._call("GET", f"/v1/posts{q}")]

    def directory(self) -> dict:
        return self._call("GET", "/v1/directory")

    def head(self) -> tuple[int, str]:
        d = self._call("GET", "/v1/head")
        return int(d["seq"]), d["hash"]

    def verify(self) -> tuple[bool, str]:
        posts = self.read(0)
        ok, why = Board.verify_chain(posts)
        if not ok:
            return ok, why
        seq, h = self.head()
        if posts and (posts[-1].seq, posts[-1].hash) != (seq, h):
            return False, "head does not match the downloaded chain"
        return True, "ok"
