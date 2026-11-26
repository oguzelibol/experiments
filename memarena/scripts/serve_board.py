#!/usr/bin/env python3
"""Serve a board over HTTP and (optionally) post the round clock.

    python scripts/serve_board.py --port 8765 --tokens tokens.json --store board.jsonl --round-seconds 20

tokens.json maps bearer token -> author, e.g. {"secret-a": "unit-a", "secret-b": "unit-b"}.
Do not issue a token for the author "env": that name is reserved for the round clock,
which this process posts itself.  Without --tokens the server refuses to start unless
--insecure is given.
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
    ap.add_argument("--tokens", help="JSON file mapping bearer token -> author, one per unit (none for 'env': this process posts the clock itself)")
    ap.add_argument("--insecure", action="store_true",
                    help="run WITHOUT tokens: anyone who can reach the port can post as anyone. For local experiments only")
    ap.add_argument("--store", help="JSONL file to persist the board to")
    ap.add_argument("--round-seconds", type=float, default=0, help="post a clock tick this often (0: no clock)")
    ap.add_argument("--rounds", type=int, default=0, help="stop after this many rounds (0: run until interrupted)")
    args = ap.parse_args()
    tokens = {}
    if args.tokens:
        with open(args.tokens, "r", encoding="utf-8") as fh:
            tokens = json.load(fh)
    elif not args.insecure:
        print("refusing to start without --tokens: every writer must be authenticated "
              "(pass --insecure to run an open board for a local experiment)", file=sys.stderr)
        return 2
    if tokens and "env" in set(tokens.values()) and args.round_seconds > 0:
        pass  # the clock is posted in process below, under the reserved author name
    board = Board(args.store)
    clock_tokens = [t for t, a in tokens.items() if a == "env"]
    if clock_tokens:
        print("warning: a token for the reserved author 'env' lets its holder post the round clock", file=sys.stderr)
    server = BoardServer(board, tokens=tokens, host=args.host, port=args.port, reserved=("env",)).start()
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
