#!/usr/bin/env python3
"""Check a designer's contest package the way a rival designer would.

    python scripts/verify_contest.py contest [--sample 12] [--all] [--check-beacon] [--anchored COPY] [--strict]

Nothing in the package is taken on trust.  The notebook's hash chain is
checked; each pre-registered design must be the one validation chose; the
held-out worlds are re-derived from the pre-registration entry and the beacon
randomness; held-out entries must use exactly those worlds and be the
pre-registered arms; every held-out report is recomputed from its eval entries;
the submission must state the last round's design and the recomputed numbers.
Then experiments are re-run: every held-out world of each chosen design, and
one world from each of a random sample of the other recorded experiments.

The sample is drawn with a seed from the operating system, so the package's
author cannot know what will be re-run; the seed is printed, and ``--seed``
repeats a verification exactly.

What the package alone cannot show is *when* things were written, and that is
printed as warnings:

- Without a randomness beacon the held-out worlds depend only on the notebook
  head, which the designer controls and could have re-rolled.
- With a beacon, the recorded randomness is compared with drand only under
  ``--check-beacon``, which needs the network.
- A notebook can be rewritten as a whole once the randomness is public.
  ``--anchored COPY`` compares it with the copy that was published at
  pre-registration; that the copy was public before the beacon round has to
  be established where it was published.

``--strict`` turns every warning into a failure, for a verifier who accepts
only what could be checked.

The report goes to standard output as JSON; a readable summary and the
warnings go to standard error.  The exit status is 0 only if nothing failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memarena.designer.search import Designer  # noqa: E402


def _read_submission(path: str) -> tuple[dict | None, str | None]:
    """The parsed submission, or the reason it cannot be used.  A package may come without one."""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            submission = json.load(fh)
    except (ValueError, RecursionError) as exc:
        return None, f"submission: submission.json is not JSON ({exc})"
    if not isinstance(submission, dict):
        return None, "submission: submission.json is not a JSON object"
    return submission, None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("--sample", type=int, default=12,
                    help="recorded experiments, besides the chosen designs' held-out runs, of which one random world is re-run")
    ap.add_argument("--all", action="store_true", help="re-run every world of every recorded experiment (slow)")
    ap.add_argument("--seed", default=None,
                    help="seed of the random sample; unpredictable by default, pass a printed sample_seed to repeat a verification")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--check-beacon", action="store_true",
                    help="fetch each committed beacon round from drand and compare the randomness; needs the network, "
                         "and a beacon that cannot be reached then fails verification")
    ap.add_argument("--anchored", action="append", default=[], metavar="COPY",
                    help="a copy of the notebook as it was published at a pre-registration (repeat for several rounds)")
    ap.add_argument("--strict", action="store_true", help="treat every warning as a failure")
    args = ap.parse_args(argv)

    submission, unusable = _read_submission(os.path.join(args.package, "submission.json"))
    check = Designer.verify_notebook(
        os.path.join(args.package, "notebook.jsonl"), sample=None if args.all else args.sample, seed=args.seed,
        workers=args.workers, submission=submission, check_beacon=args.check_beacon, anchored=args.anchored,
    )
    problems = check["problems"] + ([unusable] if unusable else [])
    if args.strict:
        problems += [f"strict: {warning}" for warning in check["warnings"]]

    for name, item in check["checks"].items():
        print(f"{name:<18} {item['status']:<8} {item['rule']}", file=sys.stderr)
    for commitment in check["preregistrations"]:
        beacon, anchor = commitment["beacon"], commitment["anchor"]
        print(f"pre-registration at entry {commitment['entry']}: design {commitment['design_hash']}; "
              + (f"beacon {beacon['source']} round {beacon['round']}; " if beacon else "no beacon; ")
              + (f"reported as published: {json.dumps(anchor, sort_keys=True)}" if anchor is not None else "not published"),
              file=sys.stderr)
    if submission is None and not unusable:
        print("note: the package has no submission.json; only the notebook was checked", file=sys.stderr)
    for problem in problems:
        print(f"FAILED: {problem}", file=sys.stderr)
    for warning in [] if args.strict else check["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"sample seed {check['sample_seed']} (repeat this verification with --seed)", file=sys.stderr)

    print(json.dumps({
        "ok": not problems, "entries": check["entries"], "preregistrations": check["preregistrations"],
        "rerun": check["rerun"], "sample_seed": check["sample_seed"],
        "checks": {name: item["status"] for name, item in check["checks"].items()},
        "failures": [{"check": name, **failure} for name, item in check["checks"].items() for failure in item["failures"]],
        "problems": problems, "warnings": check["warnings"],
    }, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
