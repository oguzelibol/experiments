# memarena

A society of self-improving long-term-memory modules that examine each other, the evaluation pipeline that
makes their judgments trustworthy, and a designer that builds and releases such societies.

- **Task.** A person says things over many sessions; later the system has to recall them. The primary
  benchmark is LongMemEval (cleaned). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- **Unit.** A memory system, one evaluation module, an optional LLM brain, and a feed. It examines other units
  on what it believes it is good at, decides for itself who is high-performing, attends to those units'
  feeds, and strives to be the best at what it does as they measure it.
- **Evaluation.** One grading path for self and others, receipts anyone can recompute, and an audit of the
  evaluator itself.
- **Designer.** Searches over society designs, pre-registers its choice, evaluates on held-out worlds, and
  keeps a hash-chained notebook others can verify.

Design: [docs/DESIGN.md](docs/DESIGN.md). Measured results: [docs/RESULTS.md](docs/RESULTS.md).

## Status

- **Evaluation pipeline, exam protocol, society accounts and contest verifier:** rewritten after three rounds
  of independent review (about 100 findings, each reproduced by a script before it was accepted, all fixed;
  docs/DESIGN.md section 9). The first recorded designer run (commit `651f6c0`) was produced by code those
  reviews broke and has been withdrawn.
- **Striving:** units now improve only where they lead and every exam spreads 30% of its questions over all
  abilities. On 12 fresh worlds this beats switching striving off by +0.023 (95% CI ±0.023); the earlier
  rule lost 0.015. The designer's pre-registered held-out run is the real test (`contest/`, docs/RESULTS.md).
- **No language model has been called**: there is no API key in the build environment. The LLM-backed memory
  system, judge and brain are implemented and tested against a mock, and the Anthropic client against the real
  SDK with a mock transport. The offline memory system is a rule-based stand-in that makes large simulations
  free; on real LongMemEval it scores 0.062 (at the abstention floor), and the audit says so.

To run with a model, set `ANTHROPIC_API_KEY` in the environment (in a cloud session: the environment's
settings, as an environment variable) and start a new session.

## Quick start

```bash
python -m pytest tests -q
```

```bash
python scripts/eval_benchmark.py --benchmark synthetic --seed 1
```

```bash
python scripts/download_data.py longmemeval-oracle
```

```bash
python scripts/eval_benchmark.py --benchmark longmemeval --path data/longmemeval/longmemeval_oracle.json --out runs/lme_oracle
```

With a model (credentials are read from the environment by the Anthropic SDK; `anthropic` must be installed):

```bash
python scripts/eval_benchmark.py --benchmark longmemeval --path data/longmemeval/longmemeval_oracle.json --memory llm --judge llm --limit 50 --out runs/lme_llm
```

That run first calibrates the LLM judge on the hand-labelled exam and prints its agreement before any score.

## Run a society

```python
from memarena.designer.design import Design, sample_scenario
from memarena.sim.engine import Simulation

design = Design()                      # or load contest/submission.json["design"]
world = sample_scenario("demo", rounds=10)
result = Simulation(world, design.build("demo")).run()
print(result.value, result.metrics["outside_score"], result.metrics["estimate_report"])
```

Over HTTP, one process per unit:

```bash
python scripts/serve_board.py --port 8765 --tokens tokens.json --round-seconds 20
```

```bash
python scripts/run_unit.py --board http://127.0.0.1:8765 --token secret-a --unit unit-a --brain llm
```

## Run and check the designer

```bash
python scripts/run_designer.py --out contest --beacon drand --anchor-git --refine-rounds 2
```

```bash
python scripts/verify_contest.py contest --check-beacon
```

Each pre-registration names a future round of the drand public randomness beacon, and with `--anchor-git` the
notebook is committed and pushed before that round is revealed; the held-out worlds depend on its randomness.
`verify_contest.py` checks the notebook's hash chain, recomputes every claimed number, checks that the
held-out worlds follow from the pre-registration and the beacon and were not touched before, re-runs every
world of every arm of the claimed round and a random sample of everything else, and compares values,
metrics and simulated history hashes. Pass `--anchored COPY` with the published copy of each pre-registration
and `--strict` to treat every warning as a failure.

```bash
python scripts/make_results.py
```

regenerates `docs/RESULTS.md`.

## Layout

```
memarena/
  benchmark/   schema (probe vs key), synthetic generator, LongMemEval and LoCoMo adapters
  evalcore/    normalisation, metrics, judges, criteria family, grader, receipts, calibration loss, stats, audit
  memory/      offline BM25 + rules system, LLM-backed system, run harness
  llm/         Anthropic client, transcript cache, mock
  feeds/       hash-chained board with per-unit feeds, inboxes and blobs; HTTP server and client
  actors/      the unit, its brain, a clock-following runner
  sim/         simulation engine, outside evaluator, collective knowledge, dishonest archetypes
  designer/    design genome, search, notebook, live operation
scripts/       evaluate a benchmark, download data, run and verify the designer, serve a board, run a unit
contest/       the recorded designer run: notebook.jsonl and submission.json
tests/         about 450 tests
```
