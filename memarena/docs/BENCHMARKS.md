# Choosing the benchmark

The setting is: one person tells an assistant things over a long time, and later the system has to recall
them. This note records which public benchmark fits that best, how such benchmarks are scored in practice,
and what that implies for an evaluator whose criteria we cannot see.

It is a survey taken on 2026-11-26. Facts below were checked against primary pages (papers, repositories,
vendor posts) unless marked otherwise. Vendor scores are self-reported.

## Recommendation

1. **LongMemEval, cleaned release (2025-09) — primary.** It is the most widely used benchmark for exactly
   this setting: a single user and an assistant, 500 questions over long multi-session histories, covering
   plain recall, assistant-provided facts, preferences, multi-session aggregation, temporal reasoning,
   knowledge updates and abstention. Its judge script is public, so an outside evaluator is likely to reuse
   it. Weaknesses: only 500 questions (30 preference, 30 abstention), and top systems now report 93–96%.
2. **BEAM (ICLR 2026) — stress test.** The same single-user setting at up to 10M tokens, 2,000 questions,
   scored with partial credit. It shows how a system fares under a graded evaluator and is far from saturated.
3. **ConvoMem (2025-11) — statistical power.** About 75k question–answer pairs across the same categories;
   thinly adopted, but per-category differences become measurable.

LoCoMo should not be the primary benchmark: it is a two-speaker dialogue, about 6.4% of its answer key is
wrong according to one independent audit, its adversarial category has no usable gold answers, and the judge
prompt most vendors use on it is lenient.

## What exists

| Benchmark | Year | Setting | Scale | Scoring |
|---|---|---|---|---|
| LongMemEval (cleaned) | ICLR 2025; cleaned 2025-09 | one user ↔ assistant | 500 questions: 70 user-fact, 56 assistant-fact, 30 preference, 133 multi-session, 133 temporal, 78 knowledge-update; 30 of them abstention. Histories: S ≈115k tokens, M ≈500 sessions, oracle (evidence only) | GPT-4o yes/no judge with per-type prompts; retrieval Recall@k, NDCG@k |
| LongMemEval-V2 | 2026-05 | web/enterprise agent trajectories | 451 questions | string match + LLM judge. A different task, not a revision |
| LoCoMo | ACL 2024 | two people, LLM-generated, human-edited | 10 conversations, 1,540 scored questions + 446 adversarial | token-F1 originally; vendors use an LLM judge ("J") |
| LoCoMo-Plus | ACL 2026 | LoCoMo plus implicit-constraint probes | — | LLM judge; authors note sensitivity to judge model and prompt |
| BEAM | ICLR 2026 | one user ↔ assistant | 100 conversations (128K–10M tokens), 2,000 questions, 10 abilities | rubric "nuggets" scored 0 / 0.5 / 1 |
| ConvoMem | 2025-11 | one user ↔ assistant | 75,336 QA pairs, 100 personas | LLM judge |
| HaluMem | 2025-11 | one user ↔ assistant | 3,467 QA, 20 users | GPT-4o judge on extraction, update and QA |
| PersonaMem v1–v3 | 2025–26 | one user ↔ chatbot, preferences | v2: 5,000 queries | multiple choice; open-ended with three judges |
| MemoryAgentBench | ICLR 2026 | mostly documents fed as chunks | 4 competencies | accuracy, Recall@5, F1 |
| MSC / DMR | 2021 / 2023 | two-human persona chat | 500 conversations | LLM judge + ROUGE-L recall; saturated |

## Known problems

**LoCoMo**
- Answer key: 99 of 1,540 items have score-corrupting errors, so the ceiling is about 93.6% (one audit).
- Category 5 (adversarial): 444 of 446 items have no gold answer; most vendors skip it.
- Judge leniency: the widely shared judge prompt asks the judge to be generous. In one test, deliberately
  wrong answers were accepted 10.6% of the time when specific and 62.8% when vague.
- Setup sensitivity: changing only the answer prompt moved a full-context baseline from 81.95 to 92.62; the
  same system scored 75.14 in its own harness and 85.22 in a competitor's.

**LongMemEval**
- Saturation: full-context GPT-4o scored 60.2 at release; several systems now report 93–96%.
- Headlines are not comparable: some are macro averages, some use a modified judge prompt, one widely quoted
  figure is retrieval Recall@5 rather than answer accuracy.
- Judge noise is small under the official protocol: re-judging identical answers flipped 3 of 500 verdicts,
  and a second judge agreed on 98.6% (single-author preprint).
- No complete audit of the answer key appears to exist (not verified either way).

## How these benchmarks are scored in the wild

This catalogue is the hypothesis space our evaluator computes (`memarena/evalcore/criteria.py`).

| Strategy | Who uses it | Details that move scores |
|---|---|---|
| Exact / multiple-choice match | PersonaMem, PrefEval | LoCoMo's "EM" is token-set equality |
| Substring containment | LoCoMo category 5, LongMemEval-V2 structured answers | lower-cased phrase test |
| Token-F1 | LoCoMo official, A-MEM, Mem0 (secondary) | normalise, stem; multi-hop averages over comma-split gold parts |
| BLEU-1, ROUGE, METEOR, BERTScore | Mem0, A-MEM, MemGPT, RealTalk | |
| Binary judge, LongMemEval official | Zep, Mastra | gpt-4o-2024-08-06, temperature 0, one call; correct iff "yes" appears in the output. Partial answers fail; temporal forgives off-by-one; updates accept old + new; preferences use a rubric; abstention items must be declared unanswerable. Reports per-type, task-averaged (macro), overall (micro) and abstention |
| Binary judge, LoCoMo "J" | Mem0, Zep, EverMemOS | gpt-4o-mini, generous prompt, category 5 skipped, micro average |
| Graded rubric | BEAM, LoCoMo-Plus, HaluMem | partial credit per required piece |
| Multi-judge | EverMemOS, EverMemBench, PersonaMem-v2 | mean or majority of three |
| Retrieval | LongMemEval, MemoryAgentBench | not answer accuracy |

Under the official LongMemEval protocol, judge noise and the micro/macro choice together move a score by
about 1–2 points. A generous prompt or a stronger reader moves it far more. That is why this project scores
every answer set under strict, official-style, generous and graded judges plus the string metrics, and
reports micro and macro, instead of committing to one number.

## Sources

LongMemEval: <https://github.com/xiaowu0162/LongMemEval>,
<https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>, <https://arxiv.org/abs/2410.10813>,
<https://arxiv.org/abs/2605.12493> (V2), <https://arxiv.org/abs/2609.38021> (judge noise),
<https://mastra.ai/research/observational-memory>.

LoCoMo: <https://arxiv.org/abs/2402.17753>, <https://github.com/snap-research/locomo>,
<https://github.com/dial481/locomo-audit>, <https://arxiv.org/abs/2504.19413> (Mem0),
<https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/>,
<https://github.com/getzep/zep-papers/issues/5>, <https://www.letta.com/blog/benchmarking-ai-agent-memory>.

Others: BEAM <https://arxiv.org/abs/2510.27246>, ConvoMem <https://arxiv.org/abs/2511.10523>, HaluMem
<https://arxiv.org/abs/2511.03506>, PersonaMem <https://arxiv.org/abs/2504.14225>, MemoryAgentBench
<https://arxiv.org/abs/2507.05257>, MemGPT/DMR <https://arxiv.org/abs/2310.08560>.
