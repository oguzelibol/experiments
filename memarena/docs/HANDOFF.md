# Handoff

Where the project stands, and what to do next now that an API key is available. Read this first in a new
session; then docs/DESIGN.md for how it works and docs/RESULTS.md for the numbers.

## State (branch `claude/dazzling-carson-v0bcim`)

- About 455 tests pass (`python -m pytest tests -q`, about 6 minutes). Three are expected failures: judge
  phrasings the rule-based judges still get wrong, kept on purpose.
- Three rounds of independent review (about 100 findings, each reproduced by a script) are fixed. The latest
  round's findings are listed in the commit `f567050`.
- `contest/` holds the final designer run: two rounds, each pre-registered against a future round of the drand
  public randomness beacon and pushed to this branch before that round was produced (commits `cf753dc`,
  `1eba243`). Claimed held-out value **0.796** (95% CI 0.764 to 0.828) on 16 fresh worlds. On those worlds,
  world by world:
  - striving off: claimed design better by +0.015 (CI +0.004 to +0.026);
  - the earlier striving rule (`strive_on="specialties"`): better by +0.050 (CI +0.019 to +0.082);
  - the default design: better by +0.105 (CI +0.057 to +0.153);
  - the hand-made covering start without the search's settings: +0.003 (not established).
- `contest/VERIFY.txt` is the external verification: all 16 checks passed, with the randomness fetched from
  drand and both published copies compared. To repeat it:

```bash
git show cf753dc:memarena/contest/notebook.jsonl > /tmp/r1.jsonl
```

```bash
git show 1eba243:memarena/contest/notebook.jsonl > /tmp/r2.jsonl
```

```bash
python scripts/verify_contest.py contest --check-beacon --anchored /tmp/r1.jsonl --anchored /tmp/r2.jsonl --confirmed-timing
```

  `--confirmed-timing` asserts you checked that each copy was public before its beacon round: the copies were
  pushed at 01:49:45 and 02:11:33 UTC on 2026-11-27, and the rounds were produced at 01:50:30 and 02:12:30 UTC
  (the verifier prints round times; the push times are on the remote).

## No language model has been called yet

Everything so far used the offline rule-based memory system and deterministic judges. With
`ANTHROPIC_API_KEY` set in the environment, in this order. First, in a fresh container, install the SDK and
download the data (neither is in the repository):

```bash
pip install anthropic
```

```bash
python scripts/download_data.py longmemeval-oracle
```

1. **Check the key works and calibrate the LLM judge.** The benchmark script calibrates the LLM judge on the
   hand-labelled judge exam (about 170 cases) before reporting any score, and stops if its agreement is below
   kappa 0.6. Start small:

```bash
python scripts/eval_benchmark.py --benchmark longmemeval --path data/longmemeval/longmemeval_oracle.json --memory heuristic --judge llm --limit 50 --out runs/lme_llmjudge_50
```

2. **A model-backed memory system on real LongMemEval**, 100 questions of the oracle split:

```bash
python scripts/eval_benchmark.py --benchmark longmemeval --path data/longmemeval/longmemeval_oracle.json --memory llm --judge llm --limit 100 --out runs/lme_llm_100
```

   Every model call is recorded in a transcript next to `--out`; a re-run replays it instead of paying again.
   The default model is `claude-opus-5-5`; `--model` changes it. Each answer and each judgement is one call,
   so 100 questions cost about 200 calls plus the judge exam.

3. Then the full 500 questions, and the `longmemeval_s` split (`scripts/download_data.py longmemeval-s`), whose
   histories are long enough that retrieval matters.

4. **An LLM brain for units** (`scripts/run_unit.py --brain llm`) has never run live. Try one unit in a small
   society over HTTP before anything larger.

## Known limits (not bugs to chase first)

- The rule-based judges are only exact on generated data. On real keys a decline followed by a recital of
  plausible values still earns 0.16 under the lenient judge, and the audit fails there; use the LLM judge.
- Routing questions to champions uses a regex classifier fitted to the synthetic questions.
- A unit restarted mid-exam abandons exams in flight and is charged walkouts (exam state is not persisted).
- Reading one exam scans both parties' feeds, so a peer that floods an exam id with posts slows its examiner.
- The `peer_scope_flipped` ablation is uninformative under the current striving rule (identical values): with
  `strive_on="lead"`, "criteria" and "full" scope blend peers' scoring the same way.
