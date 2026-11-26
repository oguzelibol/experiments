#!/usr/bin/env python3
"""Run the designer end to end and write a contest package.

    python scripts/run_designer.py --out contest --generations 5

Stages: search on training worlds -> validation on fresh worlds ->
pre-registration -> held-out evaluation with baselines and ablations ->
self-verification by re-running recorded experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memarena.designer.design import Design  # noqa: E402
from memarena.designer.notebook import Notebook  # noqa: E402
from memarena.designer.search import Designer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="contest")
    ap.add_argument("--designer-id", default="designer-A")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--rung1", type=int, nargs=2, default=(2, 6), metavar=("WORLDS", "ROUNDS"))
    ap.add_argument("--rung2", type=int, nargs=2, default=(4, 10), metavar=("WORLDS", "ROUNDS"))
    ap.add_argument("--validation-worlds", type=int, default=8)
    ap.add_argument("--heldout-worlds", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--verify-sample", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nb_path = os.path.join(args.out, "notebook.jsonl")
    if os.path.exists(nb_path):
        print(f"refusing to overwrite an existing notebook at {nb_path}", file=sys.stderr)
        return 2
    t0 = time.time()
    designer = Designer(args.designer_id, Notebook(nb_path), workers=args.workers, seed=args.seed)

    finalists = designer.search(args.generations, args.population, tuple(args.rung1), tuple(args.rung2))
    print(f"[{time.time() - t0:6.0f}s] search done: {designer.sims_run} simulations; "
          f"best training mean {finalists[0]['mean']:.4f}", flush=True)

    chosen = designer.validate(finalists, args.validation_worlds, args.rounds)
    print(f"[{time.time() - t0:6.0f}s] validation chose design {chosen.hash()}", flush=True)

    baselines = {"default": Design(), "default_covering": Design(init="covering")}
    prereg = designer.preregister(chosen, args.heldout_worlds, args.rounds, baselines)
    report = designer.heldout(chosen, prereg, baselines)
    print(f"[{time.time() - t0:6.0f}s] held-out value {report['value_mean']:.4f} "
          f"CI95 [{report['value_ci95'][0]:.4f}, {report['value_ci95'][1]:.4f}]", flush=True)

    check = Designer.verify_notebook(nb_path, sample=args.verify_sample, seed=args.seed, workers=args.workers)
    print(f"[{time.time() - t0:6.0f}s] verification: ok={check['ok']} rerun={check['rerun']} problems={check['problems']}", flush=True)

    submission = {
        "designer": args.designer_id,
        "design": chosen.to_dict(),
        "design_hash": chosen.hash(),
        "claimed_value": report["value_mean"],
        "claimed_value_ci95": report["value_ci95"],
        "heldout_worlds": report["worlds"],
        "rounds": report["rounds"],
        "heldout_metrics": report["metrics_mean"],
        "arms": report["arms"],
        "simulations_run": designer.sims_run,
        "notebook_entries": len(designer.notebook.entries),
        "notebook_head": designer.notebook.head,
        "self_verification": check,
        "wall_seconds": round(time.time() - t0, 1),
        "reproduce": f"python scripts/verify_contest.py {args.out}",
    }
    with open(os.path.join(args.out, "submission.json"), "w", encoding="utf-8") as fh:
        json.dump(submission, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: submission[k] for k in ("design_hash", "claimed_value", "claimed_value_ci95", "simulations_run")}, indent=2))
    return 0 if check["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
