#!/usr/bin/env python3
"""Run one unit against a board over HTTP.

    export ANTHROPIC_API_KEY=...        # only needed with --brain llm
    python scripts/run_unit.py --board http://127.0.0.1:8765 --token secret-a --unit unit-a \\
        --features temporal,relative,third_party,expansion,negation --brain llm

The unit's memory system here is the offline rule-based one; the brain (which
design changes to try, whose exams to sit) can be a language model.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memarena.actors.runner import UnitRunner  # noqa: E402
from memarena.actors.unit import Unit, UnitPolicy, UnitSpec, WorldRules  # noqa: E402
from memarena.feeds.http import BoardClient  # noqa: E402
from memarena.memory.heuristic import MemoryConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--owner", default="designer")
    ap.add_argument("--features", default="expansion,recency,third_party,negation,comparative")
    ap.add_argument("--criterion", default="judge_lenient|detect|micro")
    ap.add_argument("--seed", default=None,
                    help="the unit's private seed; exam papers and commitments derive from it. "
                         "Default: 32 random bytes. Never share it and never reuse a guessable value")
    ap.add_argument("--brain", choices=["policy", "llm"], default="policy")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    brain = None
    if args.brain == "llm":
        from memarena.actors.brain import LLMBrain
        from memarena.llm.client import AnthropicClient

        brain = LLMBrain(AnthropicClient(model=args.model))  # credentials come from the environment
    spec = UnitSpec(
        unit_id=args.unit, owner=args.owner, policy=UnitPolicy(),
        memory=MemoryConfig(features=frozenset(f for f in args.features.split(",") if f)),
        criterion_weights={args.criterion: 1.0}, seed=args.seed or secrets.token_hex(32),
    )
    unit = Unit(spec, BoardClient(args.board, args.token), WorldRules(), brain=brain)
    print(f"{args.unit} following the clock on {args.board} (brain: {args.brain})", flush=True)
    UnitRunner(unit).run(max_seconds=args.max_seconds)
    print(f"{args.unit} stopped: {unit.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
