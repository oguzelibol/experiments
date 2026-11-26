# memarena: design

A society of self-improving memory modules that examine each other, plus the evaluation pipeline that makes
their judgments trustworthy and a designer that builds and releases such societies.

Measured results are in [RESULTS.md](RESULTS.md). The benchmark survey is in [BENCHMARKS.md](BENCHMARKS.md).

## 1. The brief, as built

The task is long-term conversational memory: a person says things over many sessions and the system must
recall them later. Three layers were asked for:

1. an evaluation pipeline that is verifiably sound and cannot be gamed, including by its own author;
2. a **unit** (module): a memory system with an LLM brain, one evaluation module, and access to a shared site;
3. a **designer** that keeps building units and releasing them, and whose claimed results other designers
   will scrutinise.

A correction arrived mid-build and replaced the organising principle. Units are not driven by a central
points scheme. Instead:

> each unit evaluates the others on what it believes it is good at, and strives to be the best in what it
> does, as measured by signals from the units it itself finds high-performing, whose feeds it attends to.

What that changed and what it did not:

| | Before the correction | As built |
|---|---|---|
| What drives a unit | points from an external evaluator, loss on its estimate | its standing with the peers it esteems |
| Who decides who is good | a central scorekeeper | each unit, from exams it sets itself |
| The shared site | a bulletin with a ledger | per-unit feeds; attention is a choice |
| The external benchmark | inside the game | outside it: the check on whether the society is right |
| The single evaluation module | required | required, and now the thing all trust rests on |
| "State a guess before the result" | priced in points | kept as a calibration signal that feeds esteem |

Parts of the original brief that still apply are kept: an outside benchmark with unknown criteria, an
asymmetric loss on the gap between the society's own estimate and the outside score, one evaluation
mechanism for self and others, and a heavy cost for manipulated results.

## 2. Evaluation pipeline

Everything else depends on this layer, so it is built to be checked rather than believed.

### 2.1 What a system can see

`benchmark/schema.py` separates two records. A `Probe` is what a system receives: an opaque id, the
question, its date and the conversation history. An `AnswerKey` is what only the evaluator holds: the
answer, acceptable aliases, the question type, the abstention flag, the evidence sessions. A `Probe` has no
field that could carry any of these. Ids are salted hashes, because source ids leak (LongMemEval marks
unanswerable questions with an `_abs` suffix), and the adapter drops per-turn `has_answer` flags.

### 2.2 One grading path

`Grader.grade(answers, instances)` is the only function that turns answers into scores. A unit's
self-assessment, its examination of a peer, the outside evaluator and every audit call it. Its inputs are the
answers and the labelled instances; there is no argument through which the author of the answers could
influence the result. A test asserts the signature.

Per item the grader computes ten scorers once and derives the item's score under each of 52 criteria:

- **scorers**: exact match, containment, token-F1, F1 ≥ 0.5, ROUGE-L, BLEU-1, and four judges
  (`lenient` = LongMemEval's official rule, `strict`, `generous` = the "J score" style, `graded` = partial credit);
- **abstention policy**: detect (the item is right iff the response declines), exclude, or literal string match;
- **aggregation**: micro (overall) or macro (averaged over question types).

These are the scoring recipes found in the wild (see BENCHMARKS.md). Each has a prior weight. The family is
both a unit's vocabulary for stating what it believes good scoring is, and the range of outside evaluators a
design has to survive.

Judges come in two implementations with one interface. `HeuristicJudge` is deterministic and free.
`LLMJudge` calls a model; its `lenient` mode uses LongMemEval's prompts verbatim, the others use prompts
that pass the response as quoted data and return structured output. Current Claude models accept no
sampling temperature, so reproducibility comes from `CachedLLM`, a content-addressed transcript of every
judge call, not from pinning the sampler.

Rows that never produced an answer (missing, error, timeout, refusal) are scored zero in the headline,
because an outside evaluator does the same, but are counted separately and never graded on their text. An
empty response is not an abstention.

### 2.3 Receipts

`EvaluationModule.evaluate` returns a score and a receipt: the evaluator fingerprint (hash of the grading
source code, judges and standard), a hash of the graded answers, a commitment to the answer keys, a Merkle
root over the per-item grading records, and the scores. When the examiner reveals the keys, anyone can call
`verify_receipt` and recompute the evaluation exactly. Tests cover a lowered score, altered answers, a wrong
key reveal, a swapped standard and an altered per-ability score; all are rejected with a reason.

### 2.4 Auditing the evaluator

`evalcore/audit.py` checks the pipeline before its numbers are used:

| Check | What it catches |
|---|---|
| Reference answers score 1.0 under every criterion | a broken harness or grader |
| Empty, "I don't know", constant, echoed question, prompt injection, list of guesses stay at the floor | a grader that is too lenient |
| Probes carry no evaluator fields; ids are opaque | leakage |
| Removing the history collapses the score | a benchmark that measures priors, or a system that ignores its memory |
| Re-grading gives identical records | non-determinism |
| Failed rows are counted, not graded | plumbing failures passing as wrong answers |
| Duplicates, class balance, gold visible in the question | dataset problems |
| Judges against 68 hand-labelled cases | judge error, reported separately for cases the rules were tuned on and fresh ones |
| Cluster-robust interval next to the naive one | false precision when questions share a history |

Two things this audit found during the build are worth recording. A response that simply echoed the
question was earning credit on "which came first, A or B?" items, because the question contains its own
answer; judges now reject echoes and check which alternative a response commits to. And a list of guesses
was passing numeric questions under the lenient judge; a short run of numbers is now treated as working
steps and a long one as guessing.

### 2.5 Statistics

Questions that share one history are not independent (LoCoMo has about 1,500 questions over 10
conversations). `stats.py` provides cluster-robust standard errors, cluster bootstrap intervals, a paired
bootstrap and exact McNemar test for comparing two systems on the same items, and a noise-floor helper.

### 2.6 Fresh data

`benchmark/synthetic.py` generates LongMemEval-style histories with gold answers by construction: facts,
updates, dated events (some stated as "yesterday" or "two days ago"), multi-session collections,
preferences, assistant-provided facts, statements about other people, dropped plans and unanswerable
questions. A private seed yields an unlimited supply of labelled data nobody has seen. This is what allows
single-use hold-outs and exams whose keys are published after one use.

## 3. The unit

```
                         +--------------------- unit ----------------------+
   feeds of esteemed --> | brain (LLM or policy): what to try, whom to sit |
   peers (read)          |                                                 |
                         | memory system  <--- design (features, settings) |
   inbox: exam posts --> |      |                                          |
                         |      v answers                                  |
                         | EVALUATION MODULE (one instance, one standard)  |
   own feed (write) <--- |   grade(answers, keys) -> score + receipt       |
     profile, results,   |      used for: self-belief, exams it sets,      |
     notes               |      re-computing others' published results     |
                         +-------------------------------------------------+
```

Each round a unit does five things (`actors/unit.py`).

**Knows itself.** It answers a fresh private hold-out and grades it with its own module. The result is its
believed competence per ability, published on its feed. A new hold-out is drawn whenever its design changes,
so the belief is never computed on data used to choose the design.

**Examines others on what it believes it is good at.** Its standard has three parts: how responses are
scored (a mixture over the criteria family), which abilities its exams probe, and how hard its exams are.
The abilities are chosen from its believed profile, blending absolute level with advantage over known peers
and a dislike of crowded abilities (§7 explains why). Exams are fresh generated sets.

**Decides whom it finds high-performing.** Esteem for a peer is built from that peer's results on the
unit's own exams, with recent evidence weighted more and thin evidence pulled toward the average. It is
lowered if the peer over-claimed before being graded, and set to zero if the peer published a result that
does not recompute. By default nothing second-hand counts: a unit trusts what it measured.

**Attends to their feeds.** It reads only the feeds of its top few esteemed peers: their standards, their
published designs, the signals they issue. It also glances at a couple of other profiles per round, which is
how it discovers new peers.

**Strives.** It sits the exams of esteemed peers, with some exploration. It tries changes to its design
(mutations, or designs published by esteemed peers) and adopts one only if it scores better on a fresh
development set and again on a second fresh set. "Better" is a weighted sum of its own specialties under
its own standard and expected performance under esteemed peers' standards. That sum is the user's sentence
in code: best in what it does, as measured by those it respects.

### The brain

`actors/brain.py` gives a language model the two judgment calls: which design changes to test and whose
exams to sit. It sees the unit's own summary as a JSON document and returns a structured plan, plus an
optional short note for the feed. It cannot grade, alter a result or adopt a design directly; whatever it
proposes is clamped to legal values and still has to win on fresh data twice. If the call fails or is
refused, the unit carries on with its policy. Without an API key the policy alone runs the unit, which is how
all simulations here were run.

The API client (`llm/client.py`) uses the official Anthropic SDK with streaming, structured output and
effort control, and defaults to `claude-opus-5-5`. The server-side refusal fallback is enabled by default
where the model supports it; pass `use_fallbacks=False` to turn it off.

## 4. The shared site

`feeds/board.py` and `feeds/http.py`. Design goals in order:

1. **Tamper-evident.** All posts form one append-only hash chain. Two readers compare one head hash to know
   they saw the same history.
2. **Feeds, not a firehose.** A reader asks for one author's feed from a cursor. Direct messages go to an
   inbox but stay on the public chain.
3. **Small typed posts.** Bulky payloads (exam questions, answers, revealed keys) live in a content-addressed
   blob store and are referenced by hash.
4. **Content is data.** Code consumes structured fields. Free text only reaches a model as quoted material.

```
GET  /v1/feed/<author>?since=&kinds=     one author's feed
GET  /v1/inbox/<recipient>?since=        posts addressed to a unit
GET  /v1/posts?since=&kinds=&limit=      the whole chain (auditors)
GET  /v1/directory | /v1/head | /v1/blobs/<hash>
POST /v1/posts {kind, body, to, round}   author is taken from the bearer token
POST /v1/blobs {data}
```

Reads are public. Writes need a bearer token that identifies the author. The client verifies the chain
locally. A test runs the same society in process and over HTTP and gets the same head hash.

### An exam, post by post

```
examinee                                   examiner
   | exam.request ------------------------->|
   |<------------------------- exam.offer   |  questions (blob), commitment to keys, standard
   | exam.answers ------------------------->|  answers (blob), commitment to self-estimate
   |<------------------------ exam.graded   |  commitment to the result
   | exam.guess --------------------------->|  self-estimate opened
   |<------------- exam.result (public)     |  score, per-ability scores, calibration loss,
   |                                        |  receipt, revealed keys, revealed exam seed
   | re-computes everything; posts audit.fail if anything does not hold
```

`actors/protocol.py: verify_result` is the check anyone can run on a published result. It establishes that
the grade recomputes from the revealed keys, that the exam regenerates from the revealed seed under the
standard named in the receipt, that this standard is the one the examiner had published when it made the
offer, and that the calibration loss follows from the stated self-estimate and score. Exam seeds are derived one-way from the examiner's
private seed, so revealing one says nothing about the next.

The examinee commits to its self-estimate before any grade exists, and the examiner commits to the grade
before it sees the estimate, so neither can adjust to the other. The calibration loss is asymmetric:
between peers, claiming too much costs four times what claiming too little does.

## 5. Why cheating does not pay

| Attempt | What stops it | Evidence |
|---|---|---|
| Mark a rival down | the examinee recomputes every result it receives; a mismatch is published with pointers to the evidence, and every reader re-checks the accusation itself before acting on it | test: the deflator is confirmed, its reputation is 0, honest units stop attending to it |
| Grade yourself by different rules | there is one module; self-belief and exams run the same function under the same published standard, and receipts carry the evaluator fingerprint | test: a published result equals a direct evaluation of the same answers |
| Give a rival a harder paper, or a different standard | each result reveals the exam's one-way seed; anyone regenerates the exam under the standard named in the receipt and compares it with what was handed out, and checks that standard against the profile the examiner had published when it offered the exam | tests: a unit that rigs exams for non-allies, and one that grades rivals under a harsher standard than it published, are both confirmed as cheats |
| Adjust a grade after seeing the claim, or a claim after seeing the grade | commit–reveal in both directions | test: every finished exam has the six posts in order |
| Over-claim | calibration loss lowers esteem | test: a unit that always claims 1.0 loses about half its esteem |
| Accuse falsely | accusations are recomputed; a false one costs the accuser its integrity | code path in `_check_accusation` |
| Collude (share keys within a ring) | esteem is first-hand by default, so the ring inflates itself only in its own eyes; collective reputation restarts from the owner's own units | test: no unit outside the ring ever sees a perfect score from a ring member |
| Steer a model through feed content | feed text reaches the brain only as quoted data; the brain's output is validated | test: an over-long note and an illegal design are clamped |

What is not prevented:

- A ring can still mislead units that choose to weight second-hand signals (the `second_hand` setting, off
  by default).
- A unit can inflate its own self-belief. That costs it esteem when others examine it, but is not detected
  before then.
- An examiner can decline to finish an exam. Nothing penalises an abandoned exam.
- An examiner picks its own exam seeds. It cannot change the exam's distribution, but it could try several
  seeds and keep one. Doing that usefully requires knowing how the examinee will answer.
- Private seeds must really be private. In the simulator they are short strings; a deployment needs random ones.

## 6. The collective and the outside check

Everything in `sim/collective.py` is computed from public posts.

- **Reputation.** An examiner's signals count in proportion to how well it performs in the eyes of other
  reputable examiners: a fixed point computed by damped power iteration, restarting from the owner's own units.
- **Consensus standing.** Per ability, each unit's score averaged over examiners. Examiners differ in
  leniency and units are not all measured by the same examiners, so an additive examinee-plus-examiner model
  is fitted and standings are reported on the scale of an average examiner.
- **Champions.** For each kind of question the society's best unit answers it. The champion is chosen on one
  random half of the signals and its expected score is read from the other half, which keeps the estimate
  from being inflated by picking the luckiest measurement.
- **The society's estimate** of its own outside score follows from the champions' expected scores and the
  mix of question kinds in the outside benchmark. The reported number is the one that minimises expected
  loss under the outside loss.

The **outside evaluator** (`sim/engine.py`) holds a hidden benchmark and a hidden criterion drawn from the
family, with a 1% judge flip rate. Units never see it. After the society has run, it scores the collective's
answers and compares that with the society's estimate:

```
value = outside score - 5 x loss(estimate, outside score)
loss  = 0.25 x d^2 if the estimate was too high, 1.0 x d^2 if too low
```

Over-estimating costs less than under-estimating, as the brief specified. Both coefficients, the exponent
and the weight are parameters.

The run also reports whether the internal signals were valid: how well each unit's esteem ranks peers by
their true performance on its exam, how much is lost by the society's choice of champions compared with the
truly best unit per kind of question, and whether any manipulated result went unconfirmed.

## 7. A failure worth knowing about

The first society that ran converged on the same three abilities. Each unit examined what it scored highest
on, those were the abilities one feature is enough to master, everyone strove on what everyone else
examined, and nobody examined the hard abilities, so nobody kept them. The collective scored below its own
best starting unit and esteem stopped tracking skill.

Three changes fixed it, each general rather than specific to this world:

- units choose specialties by advantage over known peers and avoid crowded abilities;
- esteem weights recent evidence, since peers change;
- a design change must win on fresh data twice, since a single small development set mostly rewards noise.

"Evaluate each other on what you believe you are good at" has a monoculture as one of its equilibria. A
designer has to design against it.

## 8. The designer

`designer/`. A design fixes the number of units, their behavioural policy (attention span, exploration,
exams per round, how they pick specialties, how much they weight their own niche against esteemed peers'
standards, how they accept changes), how initial memory designs and scoring beliefs are assigned, and how
the society shades its outside estimate.

Method:

1. **Search** on training worlds. Short runs screen a population, survivors get longer runs on more worlds,
   the best seed the next generation by mutation and crossover. Designs are compared on the same worlds.
2. **Validation** on fresh worlds picks one finalist.
3. **Pre-registration.** The chosen design goes into the notebook before any held-out world exists. Held-out
   seeds are derived from the notebook's head hash at that moment.
4. **Held-out evaluation** gives the claimed value with a confidence interval, beside baselines and
   ablations on the same worlds.
5. **The notebook** is a hash-chained log of every experiment, including the ones that lost.
   `scripts/verify_contest.py` checks the chain, checks that held-out worlds were derived as stated and not
   touched earlier, and re-runs recorded experiments, comparing values and simulated history hashes.

Each world has a different hidden outside criterion and different hidden data difficulty. The designer only
ever receives the value a design achieved.

While a society runs, `designer/operate.py` keeps building units: after each epoch it releases a challenger
for the ability where the society's champion is weakest, built by varying that champion's published design,
and retires a unit that is not a champion and sits at the bottom of the reputation table. It uses public
information only.

An optional `proposer` hook lets a language model suggest designs during the search. It was not used for the
recorded run.

## 9. Assumptions I made

These were not specified. Each is a parameter.

- **The outside benchmark sits outside the society.** The correction did not say whether it remains. I kept
  it as the check on the society and as the designer's objective.
- **Value** is the outside score minus five times the asymmetric squared estimate error. The brief gave the
  shape (even exponent, over-estimates cheaper) but no constants.
- **Peer calibration loss** is 1.0 for over-claiming and 0.25 for under-claiming, squared.
- **A feature budget** of five out of ten limits what one memory system can be good at. Without it one design
  dominates and specialisation has no meaning. With language-model systems the equivalent limit is prompt,
  context and cost.
- **Exams use generated data with known answers.** An examiner could instead use its own answers as the
  reference in the abilities it believes it is good at. That is closer to the letter of the correction and
  harder to verify; it is not implemented.
- **The collective answers as a team**, routing each question to the champion for its kind.

## 10. Limits

- **No language model was called.** No API key was available. The memory reader, the judge, the brain and
  the proposer are implemented and tested against a mock, and the API client is tested against the real SDK
  with a mock transport. None has been run live.
- **The offline memory system is a simulation stand-in.** On real LongMemEval it scores at the abstention
  floor, and the audit flags it. Results about societies are results about the mechanism in a synthetic world.
- **Heuristic judges are approximations.** They agree with hand labels on 15 of 16 fresh cases per mode and
  reject 2 of 500 real LongMemEval reference answers in strict mode. Use the LLM judge on real data, and
  calibrate it on the exam first; `scripts/eval_benchmark.py` does that automatically.
- **Routing by question kind uses a regex classifier** that fits the synthetic questions. It does not fit
  real LongMemEval questions and would need to be a model there.
- **The outside criterion is drawn from the same family the units know.** An evaluator outside that family
  is not tested.
