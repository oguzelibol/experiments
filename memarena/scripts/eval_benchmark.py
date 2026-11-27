#!/usr/bin/env python3
"""Evaluate a memory system on a benchmark with the full evaluation pipeline.

Offline (free, deterministic):
    python scripts/eval_benchmark.py --benchmark synthetic --seed 1
    python scripts/eval_benchmark.py --benchmark longmemeval --path data/longmemeval/longmemeval_oracle.json

With a model (reads credentials from the environment, e.g. ANTHROPIC_API_KEY):
    python scripts/eval_benchmark.py --benchmark longmemeval --path data/longmemeval/longmemeval_s_cleaned.json \\
        --memory llm --judge llm --limit 50 --out runs/lme_s_llm

Writes results.jsonl (one graded row per question), errors.jsonl (rows that never
produced an answer), summary.json (scores with cluster-robust intervals, per-type
breakdown, audit, receipt) into --out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memarena.benchmark.adapters import load_locomo, load_longmemeval  # noqa: E402
from memarena.benchmark.synthetic import GenConfig, generate  # noqa: E402
from memarena.evalcore.audit import dataset_report, judge_exam, run_audit  # noqa: E402
from memarena.evalcore.criteria import CriteriaFamily  # noqa: E402
from memarena.evalcore.grader import default_judges  # noqa: E402
from memarena.evalcore.judges import LLMJudge  # noqa: E402
from memarena.evalcore.module import EvaluationModule  # noqa: E402
from memarena.evalcore.standard import Standard  # noqa: E402
from memarena.evalcore.stats import cluster_mean_se, t_quantile_975  # noqa: E402
from memarena.memory.base import run_isolated, run_system  # noqa: E402
from memarena.memory.heuristic import FEATURES, HeuristicMemory, MemoryConfig  # noqa: E402

HEADLINES = (
    "judge_lenient|detect|micro", "judge_lenient|detect|macro", "judge_strict|detect|micro",
    "judge_generous|detect|micro", "judge_graded|detect|micro", "f1|detect|micro",
    "contains|detect|micro", "em|detect|micro", "judge_lenient|exclude|micro",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=["synthetic", "longmemeval", "locomo"], default="synthetic")
    ap.add_argument("--path", help="data file for longmemeval / locomo")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--streams", type=int, default=30)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--memory", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--features", default="all", help="comma-separated features for the heuristic system, or 'all'")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--mode", choices=["retrieve", "full_context"], default="retrieve")
    ap.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--judge-model", default=None, help="defaults to --model")
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--min-judge-kappa", type=float, default=0.6,
                    help="with --judge llm: stop unless the judge agrees with the hand labels at least this well")
    ap.add_argument("--force", action="store_true", help="report scores even if the judge fails its calibration")
    ap.add_argument("--salt", default=None,
                    help="secret that keys probe ids and question order for public benchmarks; default: random per run")
    ap.add_argument("--isolate", action="store_true",
                    help="run the (offline) system in a separate interpreter that never holds the answer keys")
    args = ap.parse_args()

    if args.benchmark == "synthetic":
        bench = generate(args.seed, GenConfig(n_streams=args.streams))
    elif args.benchmark == "longmemeval":
        bench = load_longmemeval(args.path, limit=args.limit, salt=args.salt)
    else:
        bench = load_locomo(args.path, limit=args.limit, salt=args.salt)
    if args.limit and args.benchmark == "synthetic":
        bench = bench.subset([i.probe_id for i in bench.instances[: args.limit]])

    llm_totals = {}
    transcripts = {}
    if args.memory == "llm" or args.judge == "llm":
        from memarena.llm.client import AnthropicClient, CachedLLM, LLMConfigError
        from memarena.memory.llm_memory import LLMMemory

        if args.out:
            os.makedirs(args.out, exist_ok=True)
        try:  # find out now that the model cannot be reached, not after a table of zeros
            first = AnthropicClient(model=args.model, max_retries=1, timeout=120.0).complete(
                system=None, prompt="Reply with the single word: ready", max_tokens=64, effort="low", purpose="preflight")
        except LLMConfigError as exc:
            print(f"cannot use the model: {exc}", file=sys.stderr)
            return 2
        if first.status != "ok":
            print(f"cannot use the model: the first call came back as {first.status}: {first.text[:200]}", file=sys.stderr)
            return 2

    if args.memory == "llm":
        reader_client = AnthropicClient(model=args.model)
        # every completed reader call is written down as it arrives, so an interrupted run loses nothing
        reader = CachedLLM(reader_client, os.path.join(args.out, "reader_transcript.jsonl") if args.out else None)
        shared = len({i.probe.haystack_id for i in bench}) < len(bench)
        system = LLMMemory(reader, mode=args.mode, top_k_sessions=args.top_k, cache_history=shared)
        llm_totals["reader"] = reader_client.totals
        transcripts["reader"] = reader
    else:
        feats = FEATURES if args.features == "all" else tuple(f for f in args.features.split(",") if f)
        system = HeuristicMemory(MemoryConfig(features=frozenset(feats), top_k=args.top_k))

    judges = default_judges()
    if args.judge == "llm":
        judge_client = AnthropicClient(model=args.judge_model or args.model, effort="low")
        cache_path = os.path.join(args.out, "judge_transcript.jsonl") if args.out else None
        cached = CachedLLM(judge_client, cache_path)
        judges = {f"judge_{m}": LLMJudge(cached, mode=m, votes=args.votes) for m in ("lenient", "strict", "generous", "graded")}
        llm_totals["judge"] = judge_client.totals
        transcripts["judge"] = cached
        print("calibrating the judge on the hand-labelled exam before trusting it ...", flush=True)
        calibration = {}
        for mode, target in (("lenient", "official"), ("strict", "strict")):
            res = judge_exam(judges[f"judge_{mode}"], target)
            calibration[mode] = {k: res[k] for k in ("n", "accuracy", "kappa", "errors")}
            print(f"  {res['judge']}: accuracy {res['accuracy']} kappa {res['kappa']} errors {res['errors']} misses {res['misses']}")
        weak = [m for m, r in calibration.items() if r["kappa"] < args.min_judge_kappa or r["errors"] > 0.05 * r["n"]]
        if weak and not args.force:
            print(f"the judge does not agree with the hand labels well enough ({', '.join(weak)}: kappa below "
                  f"{args.min_judge_kappa} or too many failed calls). No score is reported. Use --force to go on anyway.",
                  file=sys.stderr)
            return 3

    family = CriteriaFamily.default()
    module = EvaluationModule(Standard({"judge_lenient|detect|micro": 1.0}), family, judges)

    t0 = time.time()
    if args.isolate:
        if args.memory != "heuristic":
            print("--isolate currently supports the offline system only", file=sys.stderr)
            return 2
        import functools

        answers = run_isolated(functools.partial(HeuristicMemory, system.config), bench.probes())
    else:
        answers = run_system(system, bench.probes())
    run_seconds = time.time() - t0
    evaluation = module.evaluate(answers, bench.instances, salt="public-benchmark")
    report = evaluation.report

    summary = {
        "benchmark": bench.name, "source": bench.info, "dataset": dataset_report(bench),
        "system": system.name, "n": report.n_items, "not_answered": report.n_not_answered,
        "truncated": report.n_truncated, "judge_errors": report.n_judge_errors,
        "seconds_answering": round(run_seconds, 2), "evaluator": module.fingerprint(),
        "judges": {k: j.spec() for k, j in judges.items()}, "scores": {}, "per_type": {}, "llm_usage": llm_totals,
        # how many model replies were read from a transcript file and how many came from calls made now
        "llm_replies": {name: {"replayed": t.replayed, "live": t.live, "skipped_lines": t.skipped_lines} for name, t in transcripts.items()},
        "receipt": dict(evaluation.receipt),
    }
    print(f"\n{bench.name}: {report.n_items} questions, system {system.name}")
    print(f"not answered: {report.n_not_answered}   truncated: {report.n_truncated}   judge errors: {report.n_judge_errors}")
    if report.n_judge_errors:
        print(f"WARNING: {report.n_judge_errors} item(s) could not be judged under at least one judge; they are left out "
              "of that judge's scores, not counted as wrong. Re-run to fill them in (replies already made are replayed).")
    for name, t in transcripts.items():
        print(f"{name}: {t.live} replies from calls made now, {t.replayed} read from the transcript file")
    if report.n_not_answered:
        reasons = sorted({str(a.meta.get("error"))[:120] for a in answers.values() if a.status != "ok" and a.meta.get("error")})
        if reasons:
            print("why answers are missing: " + " | ".join(reasons[:3]))
    cluster_of = bench.clusters()  # questions whose histories overlap count as one unit
    for cid in HEADLINES:
        rows = [(cluster_of[it.probe_id], s) for it, inc, s in report.item_scores(cid) if inc]
        mean, se = cluster_mean_se([s for _h, s in rows], [h for h, _s in rows])
        # the interval's degrees of freedom are the number of independent histories, not of questions
        half = t_quantile_975(max(1, len({h for h, _s in rows}) - 1)) * se
        summary["scores"][cid] = {"score": report.score(cid), "item_mean": mean, "ci95_half_width": half, "n": len(rows)}
        print(f"  {cid:32} {report.score(cid):.4f}  (item mean {mean:.4f} +/- {half:.4f}, n={len(rows)})")
    headline = HEADLINES[0]
    print(f"per type under {headline}:")
    for qtype, (score, n) in report.per_type(headline).items():
        summary["per_type"][qtype] = {"score": score, "n": n}
        print(f"  {qtype:28} {score:.4f}  (n={n})")
    abst = [s for it, inc, s in report.item_scores(headline) if it.is_abstention]
    if abst:
        summary["abstention"] = {"score": sum(abst) / len(abst), "n": len(abst)}
        print(f"  {'abstention items':28} {sum(abst) / len(abst):.4f}  (n={len(abst)})")

    if not args.skip_audit and args.judge == "heuristic":
        audit = run_audit(bench, system if args.memory == "heuristic" else None, family, judges)
        summary["audit"] = audit.to_dict()
        print("\naudit:")
        print(audit.render())

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as ok, \
                open(os.path.join(args.out, "errors.jsonl"), "w", encoding="utf-8") as bad:
            by_id = bench.by_id()
            for it in report.items:
                inst = by_id[it.probe_id]
                ans = answers.get(it.probe_id)
                row = {
                    "probe_id": it.probe_id, "question_type": it.question_type, "is_abstention": it.is_abstention,
                    "status": it.status, "question": inst.probe.question, "response": ans.text if ans else None,
                    "gold": inst.key.answer, "scores": {c: it.score(family.get(c))[1] for c in HEADLINES},
                    "judge_errors": list(it.judge_errors), "error": (ans.meta.get("error") if ans else None),
                }
                (bad if it.status in ("missing", "error", "timeout", "refused") else ok).write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        print(f"\nwrote {args.out}/results.jsonl, errors.jsonl, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
