# memarena: design

A society of self-improving memory modules that examine each other, the evaluation pipeline that makes
their judgments checkable, and a designer that builds and releases such societies.

Measured results are in [RESULTS.md](RESULTS.md). The benchmark survey is in [BENCHMARKS.md](BENCHMARKS.md).
Section 9 records what independent reviews found in earlier versions and what changed because of them.

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
field that could carry any of these.

The public benchmarks leak labels in three other ways, and the adapters close each:

- **Ids.** LongMemEval marks unanswerable questions with an `_abs` suffix. Probe ids are hashes of the source
  id under a secret salt drawn per load, so they cannot be matched against the public file.
- **Order.** The public files group questions by type, and LoCoMo lists its unanswerable questions last in
  each conversation. Instances are shuffled with a seed derived from the same secret, and `limit` takes a
  sample stratified by question type instead of the first rows.
- **Per-turn flags.** LongMemEval's `has_answer` marks never cross over.

Two harnesses run a system. `run_system` runs it in this process on copies of the probes, so nothing it does
to them changes what the grader reads. It is for code you trust: Python has no wall inside one process, and
code that goes looking can find the keys its caller holds. `run_isolated` runs a system in a fresh
interpreter that is sent the probes and nothing else. The audit can run a system both ways and require the
same answers.

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

Rows that never produced an answer (missing, error, timeout, refusal) are scored zero in the headline,
because an outside evaluator does the same, but are counted separately and never graded on their text. An
empty response is not an abstention.

### 2.3 Judges

Judges come in two implementations with one interface.

`LLMJudge` calls a model. Its `lenient` mode uses LongMemEval's prompts verbatim; the others use prompts
that pass the response as quoted data and return structured output. Current Claude models accept no
sampling temperature, so reproducibility comes from `CachedLLM`, a content-addressed transcript of every
judge call, not from pinning the sampler. This is the judge to use on real benchmarks.
`scripts/eval_benchmark.py` calibrates it on the hand-labelled exam before it reports any score.

`HeuristicJudge` is deterministic and free: the offline stand-in, and the only judge used in the
simulations here. Its two reporting modes credit a response only when it **commits** to the answer. Naming
candidates without choosing, denying the answer, restating the question, declining and then listing values,
or burying the answer in a pasted history does not count, however often the right string occurs. On
generated data this is exact, because every key carries the wrong alternatives for its question. On real
benchmark keys, which have no such lists, the same rules run on what the text shows and are weaker; the
audit reports that state rather than hiding it (RESULTS.md, section 1).

The rules came out of two rounds of independent review (section 9). The reviews measured errors in both
directions: responses that scored without answering, and correct answers in natural phrasings that were
rejected. Both kinds are pinned by tests, and the phrasings still judged wrongly are listed as expected
failures in `tests/test_judge_phrasings.py` and counted in RESULTS.md.

### 2.4 Receipts

`EvaluationModule.evaluate` returns a score and a receipt: the evaluator fingerprint (hash of the grading
source code, judges and standard), a hash of the graded answers, a commitment to the answer keys, a Merkle
root over the per-item grading records, and the scores. Given the keys, anyone recomputes the evaluation
and compares every field. A number that is NaN or infinite never compares equal, so it cannot be used to
switch a check off; the board refuses such numbers outright.

A standard hashes the same before and after it is published. Posts store numbers to six decimals, so
weights are normalised once, rounded, and never re-normalised.

### 2.5 Auditing the evaluator

`evalcore/audit.py` checks the pipeline before its numbers are used:

| Check | What it catches |
|---|---|
| Reference answers score 1.0 under every criterion | a broken harness or grader |
| Empty, "I don't know", constant, echoed question, prompt injection, list of guesses, pasted history, decline-then-guess, a recital of candidate values, a run of numbers, a negated candidate: all stay at the floor | a grader that pays for not answering |
| Probes carry no evaluator fields; ids are opaque | leakage |
| Removing the history collapses the score | a benchmark that measures priors |
| Handing the system another conversation's history, under an ordinary id, collapses the score | a system that answers from the question alone and can see an empty history coming |
| The system gives the same answers in a process that never held the keys (optional) | a system that reads the keys from memory |
| Re-grading gives identical records | non-determinism |
| Failed rows are counted, not graded | plumbing failures passing as wrong answers |
| Duplicates, class balance, gold visible in the question | dataset problems |
| Judges against hand-labelled cases, reported separately for cases the rules were shaped on and cases written afterwards | judge error |
| Cluster-robust interval next to the naive one | false precision when questions share a history |

The floor for "I don't know" is the share of unanswerable questions: of all questions under a micro average,
the mean of the share within each type under a macro average.

### 2.6 Statistics

Questions that share one history are not independent (LoCoMo has about 1,500 questions over 10
conversations), and in LongMemEval a question and its unanswerable twin share a history while other pairs
share single sessions. Each instance therefore records a cluster: the group of questions whose histories
overlap. Splits keep clusters together, and intervals are cluster-robust with a t quantile on the number of
clusters. `stats.py` also has a cluster bootstrap, a paired bootstrap and exact McNemar test for comparing
two systems on the same items, and a noise-floor helper.

### 2.7 Fresh data

`benchmark/synthetic.py` generates LongMemEval-style histories with gold answers by construction: facts,
updates, dated events (some stated as "yesterday" or "two days ago"), multi-session collections,
preferences, assistant-provided facts, statements about other people, dropped plans and unanswerable
questions. A private seed yields an unlimited supply of labelled data nobody has seen. This is what allows
single-use hold-outs and exams whose keys are published after one use.

Three properties matter for exams and are tested:

- **A benchmark contains exactly the questions drawn from its mix.** When a history has nothing left to ask
  of a type, the question is carried to another history rather than replaced by a different type, and a mix
  that cannot be met is an error. An exam therefore tests the abilities its standard names and nothing else.
- **Position says nothing.** Question order within a history is shuffled.
- **The same seed and settings always give the same benchmark, and different settings never share an id.**
  Verifiers regenerate exams from seeds, so drift would invalidate every published result; the generator's
  output for three seeds is pinned in the tests.

An independent re-derivation of every label from the history text alone (a separate parser, written by a
reviewer) found no label error in 543 generated histories covering every question kind. Histories shorter
than four sessions cannot hold every statement their answers rely on, and are refused.

## 3. The unit

```
                         +--------------------- unit ----------------------+
   feeds of esteemed --> | brain (LLM or policy): what to try, whom to sit,|
   peers (read)          |                        how to score             |
                         | memory system  <--- design (features, settings) |
   inbox: exam posts --> |      |                                          |
                         |      v answers                                  |
                         | EVALUATION MODULE (one instance, one standard)  |
   own feed (write) <--- |   grade(answers, keys) -> score + receipt       |
     profile, results,   |      used for: self-belief, exams it sets,      |
     notes               |      settling others' exams                     |
                         +-------------------------------------------------+
```

Each round a unit does five things (`actors/unit.py`).

**Knows itself.** It answers a fresh private hold-out and grades it with its own module. The result is its
believed competence per ability, published on its feed. A new hold-out is drawn whenever its design changes,
so the belief is never computed on data used to choose the design.

**Examines others on what it believes it is good at.** Its standard has three parts: how responses are
scored (a mixture over the criteria family), which abilities its exams probe, and how hard its exams are.
The abilities are chosen from its believed profile, blending absolute level with advantage over known peers
and a dislike of crowded abilities (section 7 explains why). Exams are fresh generated sets that contain
those abilities and nothing else.

**Decides whom it finds high-performing.** Esteem for a peer is built from that peer's results on the
unit's own exams, with recent evidence weighted more and thin evidence pulled towards the average. It is
lowered if the peer over-claimed before being graded, halved each time the peer walked away from an exam
it had started to set, and set to zero if the peer published a result that does not verify. By default
nothing second-hand counts: a unit trusts what it measured.

**Attends to their feeds.** It reads only the feeds of its top few esteemed peers: their standards, their
published designs, the signals they issue. It also glances at a couple of other profiles per round, which is
how it discovers new peers.

**Strives.** It sits the exams of esteemed peers, with some exploration. It tries changes to its design
and adopts one only if it scores better on a fresh development set and again on a second fresh set.
"Better" is measured on the abilities where the unit leads, under its own scoring and the scoring of the
peers it esteems, and no change may cost a lead ability under any scoring recipe. It does not chase other
units' specialties (section 7).

### The brain

`actors/brain.py` gives a language model three judgment calls: which design changes to test, whose exams to
sit, and how the unit should score answers (its mixture over the scoring recipes, which is its belief about
what good evaluation is). It sees the unit's own summary as a JSON document and returns a structured plan,
plus an optional short note for the feed. It cannot grade, alter a result or adopt a design directly;
whatever it proposes is clamped to legal values, a design change still has to win on fresh data twice, and a
new scoring belief only takes effect once it has been published with the unit's profile. If the call fails
or is refused, the unit carries on with its policy. Without an API key the policy alone runs the unit,
which is how all simulations here were run.

The API client (`llm/client.py`) uses the official Anthropic SDK with streaming, structured output and
effort control, and defaults to `claude-opus-5-5`. The server-side refusal fallback is enabled by default
where the model supports it; pass `use_fallbacks=False` to turn it off.

## 4. The shared site

`feeds/board.py` and `feeds/http.py`. Design goals in order:

1. **Tamper-evident.** All posts form one append-only hash chain. Two readers compare one head hash to know
   they saw the same history.
2. **Feeds, not a firehose.** A reader asks for one author's feed from a cursor. Direct messages go to an
   inbox but stay on the public chain.
3. **Small typed posts.** Bulky payloads (exam questions, answer sheets) live in a content-addressed blob
   store. A post lists the blobs it relies on and is refused unless they are already stored; blobs are never
   removed. So a post cannot point at questions or answers that nobody can read.
4. **Content is data.** Code consumes structured fields. Free text only reaches a model as quoted material.
   Numbers that are NaN or infinite are refused.

```
GET  /v1/feed/<author>?since=&kinds=     one author's feed
GET  /v1/inbox/<recipient>?since=        posts addressed to a unit
GET  /v1/posts?since=&kinds=&limit=      the whole chain (auditors)
GET  /v1/directory | /v1/head | /v1/blobs/<hash>
POST /v1/posts {kind, body, to, round}   author is taken from the bearer token
POST /v1/blobs {data}
```

Reads are public. Writes need a bearer token that identifies the author; the server will not start without
tokens unless told to, and the environment's own name is reserved. A test runs the same society in process
and over HTTP and gets the same head hash. The outside evaluator posts its verdict on the same board.

### An exam, post by post

```
examinee                                   examiner
   | exam.request ------------------------->|  the standard in force is the examiner's last profile before this post
   |<-------------------------- exam.ready  |  commitment to its seed share; exam size and loss, fixed now
   | exam.nonce --------------------------->|  the examinee's seed share, in the open
   |<-------------------------- exam.offer  |  questions (blob), commitment to the keys, standard
   | exam.sealed -------------------------->|  commitments to its answers and its self-estimate; cites the offer
   |<--------------------------- exam.keys  |  seed share and key salt: the keys are now public
   | exam.open ---------------------------->|  the answers and the self-estimate, opened
   |<------------------------- exam.result  |  score, per-ability scores, calibration loss, receipt
```

The order does the work.

- **Nobody chooses the paper.** The exam is generated from a hash of both seed shares and the exam id. The
  examiner commits to its share, bound to this examinee and this exam id, before it sees the examinee's; the
  examinee sends its share without knowing the examiner's. The size of the exam and the loss are fixed in
  `exam.ready`, before either side can know the paper.
- **The examinee is bound before the keys are public.** It commits to its answers and its self-estimate
  (`exam.sealed`) before the examiner reveals anything about the keys.
- **The examiner gives the keys away before it can read the answers.** After `exam.keys` anyone can
  regenerate the paper and its keys.
- **So the grade is nobody's to decide.** Once the answers are opened, the score, the per-ability table and
  the calibration loss are a function of public posts. `protocol.settle` computes them for any reader, with
  or without the examiner's `exam.result`.

**One reading.** `read_exam` decides, from the chain alone, which posts make up an exam and whose move it is:
of each kind, the first well-formed post that follows the previous step counts, and anything else with the
same exam id is noise. Units act on that reading and nothing else, and `settle` judges by it, so a post a
unit acts on is always the post a verifier counts. (An earlier version had units and verifiers read exams
separately; a review found twenty ways they could disagree.)

**Deadlines are clock posts.** A party called on to act has until the second clock post after the post that
called on it. Every outcome therefore becomes final at a definite position on the chain (`final_seq`), and
nothing posted later can change it:

- an examinee that has seen the questions and seals nothing, or has seen the keys and opens nothing, is
  settled with every answer missing;
- an examiner that knows the paper (it knows it once the examinee's seed share arrives) and then fails to
  offer, to reveal the keys in time, or to offer the paper the joint seed gives, has *abandoned* or *spoiled*
  the exam. Nothing is settled about the examinee, and the exam counts against the examiner;
- a self-estimate that is sealed but not opened is charged the worst loss any claim could have cost.

If a seed cannot be expanded under the examiner's standard, the examiner calls the exam off and reveals its
share, and anyone can check the claim (`called_off`). It may not try another seed, since that would be
choosing the paper.

`check_result` settles the exam a result is about and accepts the post only if every number in it equals the
settled one and it was posted after the verdict became final. `sim/collective.audit_record` settles every exam
on the board whose keys are public, so a result that is false, late or missing changes nothing in the
society's accounts.

Regenerating a paper and grading a sheet are pure functions of public data, and one exam is checked by its
examiner, its examinee, every reader of its result and the audit. The protocol remembers both computations
under a key made of everything they read, which makes a simulation about 40% faster; a test runs societies
with and without the tables and compares the chains.

## 5. Why cheating does not pay

Each row is a unit archetype in `sim/adversaries.py` with a test in `tests/test_society.py`; RESULTS.md
section 5 runs them all against an honest twin.

| Attempt | What happens |
|---|---|
| Mark a rival down | The published numbers differ from what anyone settles, so the examiner is marked as publishing false results and loses all standing. The society's accounts hold the settled grade, not the published one. |
| Hide a false number behind NaN | The board refuses the post. The exam is settled without it. |
| Give a rival a harder paper | The paper does not regenerate from the joint seed. Nothing settles from it, and the exam counts against the examiner. |
| Grade a rival under a harsher standard | The standard is not the one published before the exam was requested. Same outcome. |
| Switch standard just before a rival's exam | The standard in force is the one published before the *request*, a moment the examinee picks. |
| Publish a result about an exam nobody sat, or about yourself | The named examinee took no part, or a unit cannot examine itself: it does not verify. |
| Keep a rival's good result to yourself | The keys are already public by the time answers are opened. Everyone settles the exam without the examiner. |
| Walk away when the paper would suit a rival | Possible: the examiner knows the paper once the examinee's seed share arrives. Each walkout is visible, halves the examinee's trust in the examiner, and discounts the examiner's weight in the society's accounts by the square of its completion rate. |
| See the questions, then go silent | Settled with every answer missing: zero. |
| Change answers after seeing the keys | The answers were sealed before the keys were public. A different sheet does not open. |
| Over-claim before being graded | Calibration loss lowers esteem. |
| Accuse falsely | Accusations are recomputed by whoever reads them; a false one costs the accuser its standing with that reader. |
| Praise each other in a ring | Reputation counts only what the trusted units measured first-hand, so praise between others moves nothing. |
| Share keys within a ring | Not detectable. Ring members ace each other's exams; honest examiners still measure them as they are. |
| Pick an exam id that lands the exam in a convenient half of the evidence | The halves are decided by the exam's joint seed. |
| Flood the accounts with one huge exam | An exam counts for at most the world's exam size, and evidence is weighted by the examiner's reputation. |
| Send malformed or oversized posts | Every reader ends in a verdict or ignores the post; exam size and history length are bounded before anything is regenerated. |
| Steer a model through feed content | Feed text reaches the brain only as quoted data; the brain's output is validated. |

What is not prevented:

- **Collusion out of band.** Two units that share keys or answers privately look like any other pair on the
  record. They gain standing only with each other.
- **Walking away, up to a point.** The discount makes selective walkouts expensive, not impossible. An
  examiner that abandons one exam in ten keeps 81% of its weight.
- **A unit can inflate its own self-belief.** That costs it esteem when others examine it, but is not
  detected before then.
- **The board is trusted** to keep order, to store what it accepted and to bind posts to authors.
- **Private seeds must really be private.** `scripts/run_unit.py` draws a random one by default.
- **Deadlines assume units are up.** A unit that is offline for a round after starting an exam is treated
  as silent or as having walked out. A deployment has to persist exam state across restarts.

## 6. The collective and the outside check

Everything in `sim/collective.py` is computed from public posts; a test replays a board onto a fresh one and
gets the same champions.

- **Audit.** `audit_record` verifies every published result and settles every exam whose keys are public.
  Units with a result that does not verify are marked; exams left unfinished by their examiner are counted.
- **Reputation.** A unit's reputation is the mean score the owner's own units gave it on their exams,
  pulled towards a prior while evidence is thin (the owner's average for its own units, zero for anyone
  else). Nothing is passed on at second hand. It is the weight of a unit's signals as an examiner.
- **Consensus standing.** Per ability, each unit's score averaged over examiners. Examiners differ in
  leniency and units are not all measured by the same examiners, so an additive examinee-plus-examiner model
  is fitted until it settles and standings are reported on the scale of an average examiner. The evidence
  behind a standing is an effective sample size that follows examiner weight.
- **Champions.** For each kind of question, a champion is chosen on one half of the evidence and its
  expected score is read from the other half, in both directions; the two champions (usually the same unit)
  each answer half of that kind's questions. Each reading is independent of the choice it belongs to, so it
  is not inflated by picking the luckiest measurement, and it is taken as measured: pulling it towards the
  average unit would bias it low, on the side the outside loss charges most for. The halves are decided by
  each exam's joint seed.
- **The society's estimate** of its own outside score follows from the champions' expected scores and the
  mix of question kinds in the outside benchmark. The reported number is the one that minimises expected
  loss under the outside loss.

The **outside evaluator** (`sim/engine.py`) holds a hidden benchmark and a hidden criterion drawn from the
family, with a 1% judge flip rate on answers it actually read. Units never see it: units are built from a
one-way hash of the world's name, and the hidden questions are answered by a fresh system of each unit's
design, not by the unit's own object. After the society has run, the evaluator scores the collective's
answers, compares that with the society's estimate, and posts the outcome on the board:

```
value = outside score - 5 x loss(estimate, outside score)
loss  = 0.25 x d^2 if the estimate was too high, 1.0 x d^2 if too low
```

Over-estimating costs less than under-estimating, as the brief specified. Both coefficients, the exponent
and the weight are parameters.

The run also reports whether the internal signals were valid: how well each unit's esteem ranks peers by
their true performance on its exam, how much is lost by the society's choice of champions compared with the
truly best unit per kind of question, and whether any false result went unmarked.

## 7. Three failures worth knowing about

**Everyone examining the same things.** The first society that ran converged on the same three abilities.
Each unit examined what it scored highest on, those were the abilities one feature is enough to master,
everyone strove on what everyone else examined, and nobody examined the hard abilities, so nobody kept
them. The collective scored below its own best starting unit and esteem stopped tracking skill. Units now
choose specialties by advantage over known peers and avoid crowded abilities; esteem weights recent
evidence; a design change must win on fresh data twice.

**Striving towards other units' specialties.** The first recorded designer run showed that turning striving
off raised the society's value. An independent diagnosis (about 180 ten-round simulations) traced it. Units
were scored on their esteemed peers' whole standards, and esteem follows exam scores, which are highest on
abilities that are easy to max. So units copied those designs: 48 of 59 accepted changes came from a peer's
design, units holding the `comparative` feature went from 12 to 42 of 48, and nobody was left who answered
assistant questions well (the best true score on that class fell from 0.77 to 0.40). Value fell by 0.18
against striving off: about half from lost specialists, two fifths from worse champion choice (every design
change throws away the evidence about a unit), the rest from a worse self-estimate. Each unit's own average
went up while the collective went down.

Scoping striving to a unit's own specialties halved the damage but did not remove it (−0.047 on 12 worlds).
The rule now in the code takes "be the best in what you do" literally:

- a unit strives only on the question abilities where it **leads**: where no rival seen in settled results
  (its own exams, and its esteemed peers' exams, which it verifies itself) is more than 0.10 ahead of it. Ties
  count, so twins do not both walk away from what they share;
- it compares designs on fresh data drawn from those abilities;
- it refuses any change that lowers one of them under its own scoring, under each esteemed peer's scoring, or
  under **any single scorer** of the criteria family. The last check is what stops a style that names a second
  candidate, which pleases lenient judges and collapses under string metrics;
- it tunes settings, and peers lend settings rather than features;
- every exam a unit sets carries 30% of its questions spread over all abilities, so a unit that alone holds an
  ability can still be measured on it.

Two collective rules changed with it: a champion answers with the design its last profile describes, which is
the design the evidence is about; and when choosing champions, a unit nobody has measured on an ability is
taken to be no better than the weakest measured unit, rather than average.

On 12 fresh worlds that played no part in designing the rule (same starting design, 10 rounds):

| Striving rule | Value | Minus striving off (95% CI) | Better / worse worlds |
|---|---|---|---|
| off | 0.770 | | |
| previous rule (own specialties) | 0.755 | −0.015 (±0.031) | 4 / 5 |
| lead, no exam floor | 0.781 | +0.011 (±0.019) | 6 / 5 |
| lead with exam floor (default) | 0.793 | +0.023 (±0.023) | 9 / 3 |
| off, with exam floor | 0.776 | +0.006 (±0.011) | 9 / 3 |

The gain is small and its interval only just excludes zero. The designer's held-out worlds are the real test
(RESULTS.md). The old rule is kept as `strive_on="specialties"` and as an ablation.

"Evaluate each other on what you believe you are good at, and strive as those you respect measure you" has a
monoculture among its equilibria. A designer has to design against it.

## 8. The designer

`designer/`. A design fixes the number of units, their policy (attention span, exploration, exams per round,
how they pick specialties, what they strive on, how they accept changes, how much of each exam is spread over
all abilities), how initial memory designs and scoring beliefs are assigned, and how the society shades its
outside estimate.

1. **Search** on training worlds. Short runs screen a population, survivors get longer runs on more worlds,
   the best seed the next generation by mutation and crossover. Designs are compared on the same worlds.
2. **Validation** on fresh worlds picks one finalist.
3. **Pre-registration.** The chosen design goes into the notebook with the number of a *future* round of the
   drand public randomness beacon. The held-out worlds are derived from the notebook's head hash and that
   round's randomness, so they do not exist until the round is published. With `--anchor-git` the notebook
   is committed and pushed before the round is revealed, so a third party can see the choice came first.
4. **Held-out evaluation** gives the claimed value with a 95% t-interval over worlds, beside baselines and
   ablations on the same worlds, compared by paired differences.
5. **Refinement rounds** (optional). Each proposes a successor from the previous round's held-out arms,
   validates it on fresh worlds, pre-registers it against a new beacon round, and is judged on new held-out
   worlds. Earlier rounds' worlds are reported as spent.
6. **The notebook** is a hash-chained log of every experiment, including the ones that lost.
   `scripts/verify_contest.py` checks the chain, recomputes every claimed number from the recorded per-world
   results, checks that held-out worlds were derived as stated and never touched earlier, checks the beacon
   round and the anchor, and re-runs all worlds of the claimed design and a random sample of the rest.

Units are built from a one-way hash of the world's name, so code inside a unit cannot derive the world's
hidden outside criterion, data difficulty or benchmark. The designer only ever receives the value a design
achieved. Without a beacon and an anchor the held-out worlds follow from the notebook alone, so a designer
could rerun the whole pipeline and submit its luckiest notebook; the verifier says so, and `--strict` makes it
a failure.

While a society runs, `designer/operate.py` keeps building units: after each epoch it releases a challenger for
the ability where the society's champion is weakest, built by varying that champion's published design, and
retires a unit that is not a champion and sits at the bottom of the reputation table. It uses public
information only.

## 9. What independent reviews found

Every layer was given to a separate reviewer told to find counterexamples and prove each with a script. Each
finding was re-run before it was accepted. All of the following are fixed in the code.

| Area | Findings | The worst of them |
|---|---|---|
| Data layer | 11 | Unanswerable questions identifiable by id suffix, file order and per-turn flags; clusters split across train and test. |
| Grader and judges | 14 | A list of every candidate answer scored 0.82 under the reporting judges without reading the history. |
| Statistics | 12 | Intervals that ignored shared histories; a champion estimate biased low, on the side the outside loss charges most for. |
| Designer verifier | 12 | The verifier never recomputed the claimed numbers; held-out worlds could be predicted from the notebook. |
| Exam protocol | 20 | An examiner could keep a rival's good result to itself; a unit and a verifier could read one exam differently. |
| Model layer and docs | 14 | A failed model call was scored as a wrong answer; a truncated reply was taken as an answer. |
| Striving diagnosis | 6 | The rule lowered value (section 7); a final-round design change was served but never published; unmeasured units were ranked as average. |

The first recorded designer run (held-out value 0.678) was withdrawn because of these findings. Two review
agents assigned to the protocol were stopped by an automated safety filter before reporting, so the protocol
was restructured to be checkable by construction (section 4) and checked again afterwards.

## 10. Assumptions I made

These were not specified. Each is a parameter.

- **The outside benchmark sits outside the society.** The correction did not say whether it remains. I kept it
  as the check on the society and as the designer's objective.
- **Value** is the outside score minus five times the asymmetric squared estimate error. The brief gave the
  shape (even exponent, over-estimates cheaper) but no constants.
- **Peer calibration loss** is 1.0 for over-claiming and 0.25 for under-claiming, squared: the reverse of the
  outside loss, so a unit is charged more for bragging before an exam.
- **A feature budget** of five out of ten limits what one memory system can be good at. Without it one design
  dominates and specialisation has no meaning. With language-model systems the equivalent limit is prompt,
  context and cost.
- **Exams use generated data with known answers.** An examiner could instead use its own answers as the
  reference in the abilities it believes it is good at. That is closer to the letter of the correction and
  harder to verify; it is not implemented.
- **The collective answers as a team**, routing each question to the champions for its kind.

## 11. Limits

- **No language model has been called.** No API key was available. The memory reader, the judge, the brain and
  the proposer are implemented and tested against a mock, and the API client is tested against the real SDK
  with a mock transport. None has run live.
- **The offline memory system is a simulation stand-in.** On real LongMemEval it scores 0.062 (±0.021), at
  the abstention floor, and the audit flags it. Results about societies are results about the mechanism in a
  synthetic world.
- **The rule-based judges are approximations.** On generated data every key lists the wrong alternatives, and
  the judges keep every non-answer at the floor. On real keys they work from the text alone: a decline
  followed by a recital of plausible values still earns 0.16 under the lenient judge on LongMemEval, and the
  audit fails there. Use the LLM judge on real data; `scripts/eval_benchmark.py` calibrates it on the
  hand-labelled exam before reporting. Phrasings the rules still get wrong are listed as expected failures in
  the tests.
- **Routing by question kind uses a regex classifier** that fits the synthetic questions. Real LongMemEval
  questions would need a model.
- **The outside criterion is drawn from the same family the units know.** An evaluator outside that family is
  not tested.
- **What the protocol does not prevent:** collusion out of band (two units sharing keys look like any other
  pair and gain standing only with each other); selective walkouts, which are discounted, not impossible; a
  unit inflating its own self-belief until others examine it. The board is trusted to keep order, store what
  it accepted and bind posts to authors.
