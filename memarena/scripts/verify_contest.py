#!/usr/bin/env python3
"""Check a designer's contest package the way a rival designer would.

    python scripts/verify_contest.py contest [--sample 12] [--all]

Checks that the notebook's hash chain is intact, that the held-out worlds were
derived from the notebook head at pre-registration and were not touched before
it, that the submission matches the notebook, and then re-runs recorded
experiments and compares both the values and the simulated histories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memarena.designer.notebook import Notebook  # noqa: E402
from memarena.designer.search import Designer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("--sample", type=int, default=12, help="number of recorded experiments to re-run")
    ap.add_argument("--all", action="store_true", help="re-run every recorded experiment (slow)")
    ap.add_argument("--seed", default="verify")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    nb_path = os.path.join(args.package, "notebook.jsonl")
    nb = Notebook(nb_path)
    sample = len(nb.of_kind("eval")) if args.all else args.sample
    check = Designer.verify_notebook(nb_path, sample=sample, seed=args.seed, workers=args.workers)
    problems = list(check["problems"])

    sub_path = os.path.join(args.package, "submission.json")
    if os.path.exists(sub_path):
        with open(sub_path, "r", encoding="utf-8") as fh:
            sub = json.load(fh)
        reports = nb.of_kind("heldout_report")
        if not reports:
            problems.append("submission present but the notebook has no held-out report")
        else:
            rep = reports[-1]["body"]
            if abs(rep["value_mean"] - sub["claimed_value"]) > 1e-6:
                problems.append("claimed value differs from the notebook's held-out report")
            if rep["design_hash"] != sub["design_hash"]:
                problems.append("submitted design differs from the one evaluated on held-out worlds")
        if sub.get("notebook_head") != nb.head:
            problems.append("notebook was extended or replaced after the submission was written")
    if check["evaluator_code_recorded"] != check["evaluator_code_now"]:
        print("note: the grading code differs from the version the notebook was produced with")
    print(json.dumps({"entries": check["entries"], "rerun": check["rerun"], "ok": not problems, "problems": problems}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
