import dataclasses

import pytest

from memarena.feeds.board import Board, Post
from memarena.feeds.http import BoardClient, BoardServer


def filled(board):
    board.post("a", "profile", {"x": 1})
    board.post("b", "exam.request", {"xid": "b>a#1"}, to="a", round=1)
    board.post("a", "exam.offer", {"xid": "b>a#1", "score": 0.123456789}, to="b", round=1)
    board.post("a", "note", {"text": "hello"}, round=2)
    return board


def test_feeds_inbox_directory_and_blobs():
    board = filled(Board())
    assert [p.kind for p in board.feed("a")] == ["profile", "exam.offer", "note"]
    assert [p.kind for p in board.feed("a", since=1)] == ["exam.offer", "note"]
    assert [p.kind for p in board.feed("a", kinds=["note"])] == ["note"]
    assert [p.author for p in board.inbox("a")] == ["b"]
    assert board.inbox("b", since_seq=3) == []
    assert board.directory() == {"a": {"posts": 3, "last_profile": 1}, "b": {"posts": 1, "last_profile": 0}}
    h = board.put_blob({"k": [1, 2, 3]})
    assert board.get_blob(h) == {"k": [1, 2, 3]} and board.put_blob({"k": [1, 2, 3]}) == h
    with pytest.raises(KeyError):
        board.get_blob("nope")
    with pytest.raises(ValueError):
        board.post("a", "note", {"text": "x" * 100_000})


def test_chain_detects_edits_drops_and_reordering():
    board = filled(Board())
    posts = board.read(0)
    assert Board.verify_chain(posts) == (True, "ok")
    edited = list(posts)
    edited[1] = dataclasses.replace(edited[1], body={"xid": "forged"})
    assert not Board.verify_chain(edited)[0]
    assert not Board.verify_chain(posts[:1] + posts[2:])[0]
    assert not Board.verify_chain([posts[1], posts[0]] + posts[2:])[0]
    assert board.head() == (4, posts[-1].hash)


def test_persistence_roundtrip_and_tamper_detection(tmp_path):
    path = str(tmp_path / "board.jsonl")
    board = filled(Board(path))
    h = board.put_blob({"big": "payload"})
    again = Board(path)
    assert again.head() == board.head() and again.get_blob(h) == {"big": "payload"}
    text = (tmp_path / "board.jsonl").read_text().replace('"hello"', '"HELLO"')
    (tmp_path / "board.jsonl").write_text(text)
    with pytest.raises(ValueError):
        Board(path)


def test_http_transport_matches_in_process_board():
    board = Board()
    server = BoardServer(board, tokens={"tok-a": "a", "tok-b": "b"}).start()
    try:
        a, b = BoardClient(server.url, "tok-a"), BoardClient(server.url, "tok-b")
        anon = BoardClient(server.url)
        post = a.post("a", "profile", {"x": 1})
        assert isinstance(post, Post) and post.author == "a" and post.seq == 1
        b.post("b", "exam.request", {"xid": "1"}, to="a", round=3)
        assert [p.kind for p in anon.feed("a")] == ["profile"]  # reads are public
        assert [p.body["xid"] for p in a.inbox("a")] == ["1"]
        assert anon.read(0, kinds=["exam.request"])[0].round == 3
        h = a.put_blob({"probes": [1, 2]})
        assert b.get_blob(h) == {"probes": [1, 2]}
        with pytest.raises(KeyError):
            b.get_blob("missing")
        assert anon.directory()["a"]["last_profile"] == 1
        assert a.head() == board.head() and a.verify() == (True, "ok")
        with pytest.raises(ValueError):
            anon.post("a", "note", {"text": "no token"})  # writes need a token
        with pytest.raises(ValueError):
            b.post("a", "note", {"text": "impersonation"})  # and only for one's own name
        assert len(board) == 2
    finally:
        server.stop()


def test_blobs_are_immutable_and_checked_on_load(tmp_path):
    path = str(tmp_path / "board.jsonl")
    board = Board(path)
    h = board.put_blob({"answers": {"p1": "Lisbon"}})
    got = board.get_blob(h)
    got["answers"]["p1"] = "Porto"  # a reader scribbling on what it was handed
    assert board.get_blob(h) == {"answers": {"p1": "Lisbon"}}
    text = (tmp_path / "board.jsonl").read_text().replace("Lisbon", "Porto")
    (tmp_path / "board.jsonl").write_text(text)
    with pytest.raises(ValueError):
        Board(path)  # the stored payload no longer matches its hash


def test_board_refuses_non_finite_numbers():
    board = Board()
    with pytest.raises(ValueError):
        board.post("a", "note", {"x": float("nan")})
    with pytest.raises(ValueError):
        board.put_blob({"x": float("inf")})
    assert len(board) == 0


def test_reserved_author_cannot_be_claimed_on_an_open_board():
    board = Board()
    server = BoardServer(board, reserved=("env",)).start()
    try:
        anon = BoardClient(server.url)
        anon.post("someone", "note", {"text": "open boards accept this"})
        with pytest.raises(ValueError):
            anon.post("env", "clock", {"round": 99})
    finally:
        server.stop()
    assert [p.author for p in board.read(0)] == ["someone"]
