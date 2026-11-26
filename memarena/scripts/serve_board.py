#!/usr/bin/env python3
"""Serve a board over HTTP and (optionally) post the round clock.

    python scripts/serve_board.py --port 8765 --tokens tokens.json --store board.jsonl --round-seconds 20

tokens.json maps bearer token -> author, e.g. {"secret-a": "unit-a", "secret-env": "env"}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memarena.feeds.board import Board  # noqa: E402
from memarena.feeds.http import BoardServer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tokens", help="JSON file mapping bearer token -> author")
    ap.add_argument("--store", help="JSONL file to persist the board to")
    ap.add_argument("--round-seconds", type=float, default=0, help="post a clock tick this often (0: no clock)")
    ap.add_argument("--rounds", type=int, default=0, help="stop after this many rounds (0: run until interrupted)")
    args = ap.parse_args()
    tokens = {}
    if args.tokens:
        with open(args.tokens, "r", encoding="utf-8") as fh:
            tokens = json.load(fh)
    board = Board(args.store)
    server = BoardServer(board, tokens=tokens, host=args.host, port=args.port).start()
    print(f"board at {server.url}  ({len(board)} posts)", flush=True)
    try:
        r = max((p.round for p in board.read(0, kinds=["clock"])), default=0)
        while True:
            if args.round_seconds > 0:
                r += 1
                board.post("env", "clock", {"round": r}, round=r)
                print(f"round {r}: {len(board)} posts", flush=True)
                if args.rounds and r >= args.rounds:
                    time.sleep(args.round_seconds)
                    break
                time.sleep(args.round_seconds)
            else:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
