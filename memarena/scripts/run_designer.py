#!/usr/bin/env python3
"""Run the designer end to end and write a contest package.

    python scripts/run_designer.py --out contest --generations 5
    python scripts/run_designer.py --out contest --beacon drand --anchor-git --refine-rounds 2

Stages: search on training worlds -> validation on fresh worlds ->
pre-registration -> held-out evaluation with baselines and ablations ->
optional refinement rounds, each pre-registered again and judged on new
held-out worlds -> self-verification.

``--beacon drand`` makes every pre-registration name a future round of the
drand randomness beacon; the held-out worlds depend on that round's randomness.
``--anchor-git`` commits and pushes the notebook, with the pre-registration as
its last entry, before the round is revealed.  Together they let others check
that the design was fixed before anyone could know its test worlds.  Without
them the worlds follow from the notebook alone, and verification says so.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memarena.designer.beacon import BeaconError, DrandBeacon  # noqa: E402
from memarena.designer.design import Design  # noqa: E402
from memarena.designer.notebook import Notebook  # noqa: E402
from memarena.designer.search import Designer  # noqa: E402


class AnchorFailed(RuntimeError):
    """The notebook could not be published, so its pre-registration must not be used."""


def _git(out_dir: str, run: Callable[..., Any], *arguments: str) -> str:
    """Run one git command in ``out_dir`` and return what it printed; a command that fails raises ``AnchorFailed``."""
    command = ["git", *arguments]
    try:
        done = run(command, cwd=out_dir, capture_output=True, text=True)
    except OSError as exc:
        raise AnchorFailed(f"`{' '.join(command)}` could not be run in {out_dir}: {exc}") from exc
    if done.returncode != 0:
        raise AnchorFailed(f"`{' '.join(command)}` failed in {out_dir}: {(done.stderr or done.stdout).strip()}")
    return done.stdout.strip()


def git_anchor(out_dir: str, run: Callable[..., Any] = subprocess.run, trailers: Sequence[str] = ()) -> Callable[[Notebook], dict]:
    """An anchor that publishes the notebook with git, in the repository that contains ``out_dir``.

    It runs when the pre-registration is the notebook's last entry: add the
    notebook, commit it, push.  The commit is limited to the notebook file, so
    that other staged changes are not published along with it.  A step that
    fails raises ``AnchorFailed``, which stops the run: a pre-registration that
    is not public is no evidence of when it was made.  The receipt names the
    commit and the notebook's path in the repository, so that others can fetch
    exactly that copy.  ``trailers`` are lines added at the end of the commit
    message (a repository's attribution lines, for example).
    """
    def anchor(notebook: Notebook) -> dict:
        name = os.path.basename(notebook.path)
        design_hash = notebook.entries[-1]["body"]["design_hash"]
        _git(out_dir, run, "add", name)
        message = f"Pre-register design {design_hash} (notebook head {notebook.head[:12]})"
        if trailers:
            message += "\n\n" + "\n".join(trailers)
        _git(out_dir, run, "commit", "-m", message, "--", name)
        _git(out_dir, run, "push")
        commit, prefix = _git(out_dir, run, "rev-parse", "HEAD"), _git(out_dir, run, "rev-parse", "--show-prefix")
        return {"kind": "git", "commit": commit, "path": prefix + name}

    return anchor


def published_before_round(anchor: Callable[[Notebook], dict], beacon: DrandBeacon) -> Callable[[Notebook], dict]:
    """Wrap an anchor so that it fails unless the notebook was public before the committed beacon round.

    The beacon round is committed when the pre-registration entry is written
    and arrives 30 to 60 seconds later.  If publishing took longer than that,
    the randomness was already known when the notebook became public and the
    publication proves nothing; going on would produce a package that looks
    anchored and is not.
    """
    def checked(notebook: Notebook) -> dict:
        receipt = anchor(notebook)
        commitment = notebook.entries[-1]["body"]["beacon"]
        if not beacon.pending(commitment):
            raise AnchorFailed(
                f"{commitment['source']} round {commitment['round']} was produced before the notebook had been "
                "published, so the publication does not show that the pre-registration came first"
            )
        return receipt

    return checked


def main(argv: Sequence[str] | None = None) -> int:
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
    ap.add_argument("--refine-rounds", type=int, default=1, metavar="N",
                    help="pre-registered rounds in total: 1 is the search alone; each further round refines the "
                         "previous one from its held-out arms and is judged on new held-out worlds")
    ap.add_argument("--beacon", choices=("none", "drand"), default="none",
                    help="derive held-out worlds from a future round of this randomness beacon (drand needs the network)")
    ap.add_argument("--anchor-git", action="store_true",
                    help="after each pre-registration entry, git add, commit and push the notebook before going on")
    ap.add_argument("--anchor-trailer", action="append", default=[], metavar="LINE",
                    help="a line to end each anchor commit message with; may be given more than once")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--verify-sample", type=int, default=6)
    args = ap.parse_args(argv)
    if args.refine_rounds < 1:
        ap.error("--refine-rounds counts all pre-registered rounds and must be at least 1")

    os.makedirs(args.out, exist_ok=True)
    nb_path = os.path.join(args.out, "notebook.jsonl")
    if os.path.exists(nb_path):
        print(f"refusing to overwrite an existing notebook at {nb_path}", file=sys.stderr)
        return 2
    beacon = DrandBeacon() if args.beacon == "drand" else None
    anchor = git_anchor(args.out, trailers=args.anchor_trailer) if args.anchor_git else None
    try:  # learn now, not after the search, that the beacon or the repository is out of reach
        if beacon is not None:
            beacon.commit()
        if anchor is not None:
            _git(args.out, subprocess.run, "rev-parse", "--show-toplevel")
    except (AnchorFailed, BeaconError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 3
    if beacon is not None and anchor is not None:
        anchor = published_before_round(anchor, beacon)
    elif beacon is not None:
        print("note: --beacon without --anchor-git: the pre-registration is not published before its beacon round, "
              "so nothing outside the notebook will show that it came first", file=sys.stderr)
    t0 = time.time()
    designer = Designer(args.designer_id, Notebook(nb_path), workers=args.workers, seed=args.seed)
    baselines = {"default": Design(), "default_covering": Design(init="covering")}

    finalists = designer.search(args.generations, args.population, tuple(args.rung1), tuple(args.rung2))
    print(f"[{time.time() - t0:6.0f}s] search done: {designer.sims_run} simulations; "
          f"best training mean {finalists[0]['mean']:.4f}", flush=True)

    chosen = designer.validate(finalists, args.validation_worlds, args.rounds)
    print(f"[{time.time() - t0:6.0f}s] validation chose design {chosen.hash()}", flush=True)

    history: list[tuple[Design, dict]] = []
    try:
        designer.preregister(chosen, args.heldout_worlds, args.rounds, baselines, beacon, anchor)
        history.append((chosen, designer.heldout(chosen, baselines)))
        while len(history) < args.refine_rounds:
            print(f"[{time.time() - t0:6.0f}s] round {len(history)}: design {history[-1][0].hash()}, "
                  f"held-out value {history[-1][1]['value_mean']:.4f}; refining", flush=True)
            history.append(designer.refine(
                history[-1][1], args.validation_worlds, args.heldout_worlds, args.rounds, baselines, beacon, anchor,
            ))
    except (AnchorFailed, BeaconError) as exc:
        print(f"aborted at the pre-registration of round {len(history) + 1}: {exc}\n"
              f"No held-out world was evaluated for it; {nb_path} records how far the run got.", file=sys.stderr)
        return 3
    chosen, report = history[-1]
    print(f"[{time.time() - t0:6.0f}s] held-out value {report['value_mean']:.4f} "
          f"CI95 [{report['value_ci95'][0]:.4f}, {report['value_ci95'][1]:.4f}] (round {len(history)})", flush=True)

    package = os.path.basename(os.path.normpath(args.out))
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
        # Every pre-registered round; the claim above is the last one's, the only one whose worlds are unspent.
        "refinement_rounds": [
            {"round": number, "design_hash": design.hash(), "design": design.to_dict(),
             "claimed_value": result["value_mean"], "claimed_value_ci95": result["value_ci95"]}
            for number, (design, result) in enumerate(history, start=1)
        ],
        "beacon": args.beacon,
        "anchor": "git" if args.anchor_git else "none",
        "simulations_run": designer.sims_run,
        "notebook_entries": len(designer.notebook.entries),
        "notebook_head": designer.notebook.head,
        "reproduce": f"python scripts/verify_contest.py {package}" + (" --check-beacon" if beacon is not None else ""),
    }
    check = Designer.verify_notebook(
        nb_path, sample=args.verify_sample, workers=args.workers, submission=submission, check_beacon=beacon is not None,
    )
    print(f"[{time.time() - t0:6.0f}s] verification: ok={check['ok']} rerun={check['rerun']} problems={check['problems']}", flush=True)
    for warning in check["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    submission["self_verification"] = check
    submission["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "submission.json"), "w", encoding="utf-8") as fh:
        json.dump(submission, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: submission[k] for k in ("design_hash", "claimed_value", "claimed_value_ci95", "simulations_run")}, indent=2))
    return 0 if check["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
