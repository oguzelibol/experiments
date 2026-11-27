"""The reporting judges must not pay for responses that never commit to an answer.

An independent review showed that the lenient and the strict heuristic judge
both gave high scores to strategies that name candidates without choosing one:
a recital of the generator's vocabulary, every candidate found in the history,
the same after a decline.  This file ports those strategies and the reviewer's
single cases, and holds them to the floor.  It also guards the other side:
honest answers in ordinary phrasings must still be accepted, and the generator
must still produce the same histories, questions and answers as before its
keys learned to list wrong alternatives.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time

import pytest

from memarena.benchmark import synthetic as syn
from memarena.benchmark.schema import Answer, AnswerKey
from memarena.evalcore import metrics
from memarena.evalcore.audit import answer_vocabularies, judge_exam, run_audit
from memarena.evalcore.criteria import CriteriaFamily
from memarena.evalcore.grader import Grader, default_judges
from memarena.evalcore.judge_exam import JUDGE_EXAM
from memarena.evalcore.judges import HeuristicJudge, JudgeVerdict
from memarena.evalcore.normalize import is_abstention, tokens
from memarena.memory.base import run_system
from memarena.memory.heuristic import FEATURES, HeuristicMemory, MemoryConfig

LENIENT, STRICT = "judge_lenient|detect|micro", "judge_strict|detect|micro"
DECLINE = "I don't have that information. "
# How far above its legitimate share a strategy that commits to nothing may score.
SLACK = 0.03


def digest(obj) -> str:
    """A hash that depends on nothing but the data (not on this project's helpers)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()[:16]


# --------------------------------------------------------------------- generator

# What the generator draws today, recorded so that any drift is deliberate.  Verifiers regenerate
# exams from seeds: if the same seed and settings ever gave a different paper, every result published
# before the change would stop verifying.
# seed -> (Benchmark.probes_hash(), digest of the histories, digest of [question, answer, aliases],
#          digest of every other key field).  GenConfig(n_streams=10).
PINNED_BENCHMARKS = {
    1: ("879eee7ac21d067cf5b0f05e5aa7b74f3944a4517ce927e6521757c9aa682137", "beccc508f7bc69ac", "bda014b10c3e0793", "a720380c9db9fec9"),
    "x": ("8d3d7c022e89bf64f172e2b33f43b43bcc408fd6c417281f04eb45545cf9bb9a", "0eaa53c0264e28ea", "6ec94acb3bbd99a3", "587a233cc3d80a60"),
    "audit": ("659255f5b270ea35b3fd9c9e52363498ea387e9de731edc2f6bb16feeb0d386b", "26b4a6f099defa40", "3412a206238569dd", "1ad0be6c77f6768f"),
}


@pytest.mark.parametrize("seed", list(PINNED_BENCHMARKS))
def test_the_generator_draws_the_same_benchmark_from_the_same_seed(seed):
    bench = syn.generate(seed, syn.GenConfig(n_streams=10))
    probes_hash, histories, questions_and_answers, other_key_fields = PINNED_BENCHMARKS[seed]
    assert bench.probes_hash() == probes_hash
    assert digest([
        [i.probe.probe_id, i.probe.question_date, i.probe.haystack_id,
         [[s.session_id, s.date, [[t.role, t.content] for t in s.turns]] for s in i.probe.sessions]]
        for i in bench
    ]) == histories
    assert digest([[i.probe.question, i.key.answer, list(i.key.aliases)] for i in bench]) == questions_and_answers
    assert digest([
        [i.key.probe_id, i.key.question_type, i.key.is_abstention, list(i.key.evidence_session_ids),
         {k: v for k, v in sorted(i.key.meta.items())}]
        for i in bench
    ]) == other_key_fields


def asked_about(question: str) -> str:
    """The kind of question, read off the generator's templates."""
    if question.startswith("How many"):
        return "number"
    if question.startswith("Which did I do first"):
        return "order"
    if "which month" in question.lower():
        return "month"
    if "recommend" in question:
        return "assistant"
    if "prefer" in question:
        return "preference"
    if question in (syn.COLLECTIONS["books"]["list_q"], syn.COLLECTIONS["trips"]["list_q"]):
        return "list"
    return "slot"


def test_keys_say_what_a_response_must_not_name():
    bench = syn.generate("keys", syn.GenConfig(n_streams=40))
    slot_of = {q: name for name, slot in syn.SLOTS.items() for q in slot["ask"]}
    options_of = dict(syn.ASSISTANT_TOPICS)
    seen = set()
    for inst in bench:
        question, key = inst.probe.question, inst.key
        meta, kind = key.meta, asked_about(inst.probe.question)
        wrong = meta["wrong"]
        seen.add((kind, key.is_abstention))
        assert isinstance(wrong, list) and len(set(wrong)) == len(wrong)
        assert bool(meta.get("numeric_question")) == (kind == "number")
        if kind == "slot":
            pool = syn.SLOTS[slot_of[question]]["values"]
            assert wrong == [v for v in pool if key.is_abstention or v != key.answer]
            assert set(meta.get("stale", ())) <= set(wrong)  # superseded values are wrong, and marked as stale
        elif kind == "assistant":
            pool = next(options for topic, options in options_of.items() if topic in question)
            assert wrong == [v for v in pool if key.is_abstention or v != key.answer]
        elif kind == "preference":
            assert wrong == [p for p in syn.PREFERENCE_PHRASES if key.is_abstention or p != key.answer]
            assert key.is_abstention or meta["distractors"][0] in wrong
        elif kind == "month":
            assert wrong[:11] == [m for m in syn.MONTH_NAMES if m != key.answer]
            assert len(wrong) == 22 and key.aliases[1] not in wrong  # the same eleven months written as YYYY-MM
        elif kind == "list":
            pool = syn.BOOKS if "books" in question else syn.COUNTRIES
            assert wrong == [v for v in pool if v not in meta["list"]]
        elif kind == "order":
            offered = re.match(r"Which did I do first: (.*) or (.*)\?", question).groups()
            assert set(offered) <= {key.answer, *meta["distractors"]} and not set(offered) & set(wrong)
            assert len(wrong) == len(syn.EVENT_PHRASES) - 4
        elif key.is_abstention and ("books" in question or "countries" in question):
            assert wrong == (syn.BOOKS if "books" in question else syn.COUNTRIES)
        else:
            assert wrong == []  # a count, a sum or a number of days: any other number is wrong
        if not key.is_abstention:
            # the reference answer never names one of its own wrong alternatives
            for mode in ("lenient", "strict"):
                assert HeuristicJudge(mode).judge(question, key, key.answer).score == 1.0, (question, key.answer)
    assert seen == {(kind, flag) for kind in ("number", "slot", "assistant", "preference") for flag in (False, True)} | {
        ("order", False), ("month", False), ("list", False)}


# -------------------------------------------------------------------- strategies

SLOT_OF = {q: name for name, slot in syn.SLOTS.items() for q in slot["ask"]}


def pool(question: str) -> list[str]:
    """The generator's whole value list for the kind of thing the question asks about."""
    if question in SLOT_OF:
        return list(syn.SLOTS[SLOT_OF[question]]["values"])
    if "books" in question:
        return list(syn.BOOKS)
    if "countries" in question:
        return list(syn.COUNTRIES)
    for topic, options in syn.ASSISTANT_TOPICS:
        if topic in question:
            return list(options)
    return []


def number_run(question: str) -> str | None:
    """A run of numbers for a "how many" question (nothing for day counts), else None."""
    if not question.lower().startswith("how many"):
        return None
    return "8, 9, 10" if "in total" in question else "" if "days" in question else "2, 3, 4"


def blind_vocabulary(probe) -> str:
    """Never opens the history: recites every value the generator could have used."""
    question = probe.question
    if number_run(question) is not None:
        return number_run(question)
    if "which month" in question.lower():
        return " ".join(syn.MONTH_NAMES).lower()
    if question.startswith("Which did I do first"):
        return question + " Good question, let me restate that."
    return " ".join(pool(question)).lower()


def statement_of_preference(probe) -> str:
    """The user's own sentence about the thing a preference question asks about, lower-cased."""
    about = re.search(r"(?:when it comes to|preference for) (.*)\?", probe.question)
    turns = [t.content for s in probe.sessions for t in s.turns]
    return next((t.lower() for t in turns if about and about.group(1) in t), "")


def candidates_in_order(probe) -> str:
    """Opens the history but decides nothing: every candidate value in it, in order of last mention."""
    question = probe.question
    if number_run(question) is not None:
        return number_run(question)
    if "which month" in question.lower():
        return " ".join(syn.MONTH_NAMES).lower()
    turns = [t.content for s in probe.sessions for t in s.turns]
    if question.startswith("Which did I do first"):
        first, second = re.match(r"Which did I do first: (.*) or (.*)\?", question).groups()

        def first_mention(event):
            return next((i for i, t in enumerate(turns) if set(tokens(event)[-2:]) <= set(tokens(t))), len(turns))

        return ", ".join(sorted((first, second), key=first_mention))
    about = re.search(r"(?:when it comes to|preference for) (.*)\?", question)
    if about:
        # Both options of the preference, in the order the user named them, without the words that
        # choose between them.  (This used to paste the user's whole sentence.  "for tea, i prefer
        # oolong to chamomile" states the preference, and a judge that reads "over" / "to" / "than"
        # now accepts it as the answer it is: see test_a_pasted_statement_of_the_preference_is_an_answer.)
        statement = statement_of_preference(probe)
        options = next(((x, y) for category, x, y in syn.PREFERENCES if category == about.group(1)), ())
        return ", ".join(sorted((o for o in options if o in statement), key=statement.find))
    last = {v: max((i for i, t in enumerate(turns) if v.lower() in t.lower()), default=None) for v in pool(question)}
    return ", ".join(v.lower() for v, i in sorted(((v, i) for v, i in last.items() if i is not None), key=lambda x: x[1]))


def scrambled_candidates(probe) -> str:
    """Every word of every candidate, shuffled so that no candidate has to appear in one piece."""
    offered = re.match(r"Which did I do first: (.*) or (.*)\?", probe.question)
    words = [w for value in (offered.groups() if offered else pool(probe.question)) for w in value.split()]
    random.Random(len(probe.question)).shuffle(words)
    return " ".join(words[:95]).lower()


def months_as_digits(probe) -> str:
    """For a month question, every month of this year and the last, written the way the key's alias is."""
    if "which month" not in probe.question.lower():
        return ""
    year = int(probe.question_date[:4])
    return " ".join(f"{y}-{m:02d}" for y in (year - 1, year) for m in range(1, 13))


def one_guess(probe) -> str:
    """Commits to one answer without reading anything: the first value of the pool, the first option."""
    question = probe.question
    if question.lower().startswith("how many"):
        return "9" if "in total" in question else "" if "days" in question else "3"
    if "which month" in question.lower():
        return "June"
    offered = re.match(r"Which did I do first: (.*) or (.*)\?", question)
    if offered:
        return offered.group(1)
    for category, first, _second in syn.PREFERENCES:
        if question.endswith(f" {category}?"):
            return first
    return (pool(question) or [""])[0]


def one_value_of_every_kind(probe) -> str:
    """The same string for every question: one value from each pool, a month and a number."""
    values = [slot["values"][0] for slot in syn.SLOTS.values()] + [options[0] for _topic, options in syn.ASSISTANT_TOPICS]
    return " ".join(values + ["june", "3"])


def declining(strategy):
    return lambda probe: DECLINE + strategy(probe)


class World:
    """The reviewer's benchmark, one grader, and the two reporting scores of a strategy on it."""

    def __init__(self):
        self.bench = syn.generate("x", syn.GenConfig(n_streams=20))
        self.instances = list(self.bench.instances)
        self.grader = Grader(CriteriaFamily.default())
        self.unanswerable = sum(i.key.is_abstention for i in self.instances) / len(self.instances)
        honest = run_system(HeuristicMemory(MemoryConfig(features=frozenset(FEATURES), top_k=10)), self.bench.probes())
        self.honest_answers = honest
        report = self.grader.grade(honest, self.instances)
        self.honest = (report.score(LENIENT), report.score(STRICT))

    def scores(self, strategy) -> tuple[float, float]:
        answers = {i.probe_id: Answer(strategy(i.probe)) for i in self.instances}
        report = self.grader.grade(answers, self.instances)
        return report.score(LENIENT), report.score(STRICT)


@pytest.fixture(scope="module")
def world():
    return World()


@pytest.mark.parametrize("strategy", [
    blind_vocabulary, declining(blind_vocabulary), scrambled_candidates, declining(scrambled_candidates), months_as_digits,
], ids=["blind vocabulary", "decline + blind vocabulary", "scrambled candidates", "decline + scrambled candidates", "months as digits"])
def test_a_recital_that_never_reads_the_history_stays_at_the_floor(world, strategy):
    """Before: the blind vocabulary list scored 0.646 lenient / 0.533 strict."""
    ceiling = world.unanswerable + SLACK
    lenient, strict = world.scores(strategy)
    assert lenient <= ceiling and strict <= ceiling


def test_listing_the_candidates_found_in_the_history_earns_at_most_half_of_an_honest_system(world):
    """Before: 0.758 lenient / 0.713 strict, against 0.775 / 0.762 for the honest system."""
    lenient, strict = world.scores(candidates_in_order)
    assert world.honest[0] > 0.7 and world.honest[1] > 0.7  # the rules did not cost the honest system its score
    assert lenient <= world.honest[0] / 2 and strict <= world.honest[1] / 2


def test_a_pasted_statement_of_the_preference_is_an_answer(world):
    """Before: 0 of 21.  The user's sentence "I prefer X over Y" says which option is preferred.

    The judges used to reject it because the other option is named.  They now
    read what each option is attached to, so the sentence is the correct answer
    wherever it exists, and naming the two options without it is still not one.
    """
    asked = [i for i in world.instances if asked_about(i.probe.question) == "preference" and not i.key.is_abstention]
    assert len(asked) >= 20
    for inst in asked:
        statement = statement_of_preference(inst.probe)
        assert verdicts(inst.probe.question, inst.key, statement) == (1.0, 1.0), statement
        assert verdicts(inst.probe.question, inst.key, candidates_in_order(inst.probe)) == (0.0, 0.0)


def test_a_decline_in_front_of_the_candidates_adds_no_more_than_plain_declines(world):
    """Before: 0.838 lenient, the list's score plus every unanswerable question.

    The prefix can now only help where nothing follows it, which is a plain
    decline; under the strict judge a response that declines never answers.
    """
    listed = world.scores(candidates_in_order)
    lenient, strict = world.scores(declining(candidates_in_order))
    assert lenient <= listed[0] + world.unanswerable
    assert strict <= world.unanswerable + SLACK


def test_naming_many_candidates_never_beats_committing_to_one(world):
    """Chance is the floor for a guess, and a list is not allowed to do better than a guess.

    One committed guess per question is right fairly often here (two-way
    choices, three or four options, counts of two to four); that is a property
    of the benchmark and no judge can remove it.  What a judge can do is make
    sure that a response naming several candidates earns no more.
    """
    guess = world.scores(one_guess)
    for strategy in (blind_vocabulary, scrambled_candidates, months_as_digits, one_value_of_every_kind):
        lenient, strict = world.scores(strategy)
        assert lenient <= guess[0] and strict <= guess[1]


# ------------------------------------------------------------------ single cases

def key(answer, question_type="single-session-user", aliases=(), unanswerable=False, **meta) -> AnswerKey:
    return AnswerKey("p", question_type, answer, tuple(aliases), unanswerable, (), meta)


CITY_Q, CAR_Q = "Which city do I live in?", "What car do I drive now?"
BOOKS_Q, TRIPS_Q = "How many books have I finished reading?", "Which countries have I taken a trip to?"
SISTER_Q, ORDER_Q = "What is my sister's name?", "Which did I do first: renew my passport or get a tattoo?"
# Keys as the generator writes them.
CITY = key("Lisbon", wrong=[c for c in syn.CITIES if c != "Lisbon"])
CAR = key("Mazda CX-5", "knowledge-update", stale=["Honda Civic"], wrong=[c for c in syn.CARS if c != "Mazda CX-5"])
BOOKS = key("3", "multi-session", aliases=("three",), numeric=3, numeric_question=True, wrong=[])
TRIPS = key("Peru, Nepal and Japan", "multi-session", list=["Peru", "Nepal", "Japan"],
            wrong=[c for c in syn.COUNTRIES if c not in ("Peru", "Nepal", "Japan")])
ORDER = key("renew my passport", "temporal-reasoning", aliases=("renewed my passport",),
            distractors=["get a tattoo", "got a tattoo"],
            wrong=[e for e in syn.EVENT_PHRASES if e not in ("renew my passport", "renewed my passport", "get a tattoo", "got a tattoo")])
SISTER = key("The user never mentioned their sister's name.", unanswerable=True, wrong=list(syn.NAMES))
NO_BOOKS = key("The user never mentioned finishing any books.", "multi-session", unanswerable=True,
               numeric_question=True, wrong=list(syn.BOOKS))


def verdicts(question, answer_key, response) -> tuple[float, float]:
    return tuple(HeuristicJudge(mode).judge(question, answer_key, response).score for mode in ("lenient", "strict"))


@pytest.mark.parametrize("question, answer_key, response", [
    (CITY_Q, CITY, " ".join(syn.CITIES).lower()),
    (CITY_Q, CITY, "You definitely do not live in Lisbon."),
    (CITY_Q, CITY, "My best guess is Lisbon, though it might also be Porto or Valencia."),
    (CITY_Q, CITY, "It is false that the answer is Lisbon; it is Porto."),
    (CITY_Q, CITY, "I don't know. Lisbon Porto Valencia"),
    (TRIPS_Q, TRIPS, ", ".join(syn.COUNTRIES).lower()),
    (CAR_Q, CAR, "Eit\u200bher a Honda Civic or a Mazda CX-5."),
    (SISTER_Q, SISTER, "Her name is Priya. You never mentioned otherwise."),
    (BOOKS_Q, BOOKS, "2, 3, 4"),
    (ORDER_Q, ORDER, ORDER_Q + " Good question, let me restate that."),
], ids=["every city", "negated at a distance", "guess with alternatives", "negated in other words", "decline then list",
        "every country", "zero-width space in Either", "answer then decline", "run of numbers", "question restated with filler"])
def test_the_reviewers_cases_are_no_longer_paid(question, answer_key, response):
    """Each of these was scored correct by at least the lenient judge."""
    assert verdicts(question, answer_key, response) == (0.0, 0.0)


def test_an_honest_count_that_lists_a_title_with_a_number_word_is_accepted():
    """Before: the strict judge read "Seventeen" as the number 17 and rejected the answer."""
    assert verdicts(BOOKS_Q, BOOKS, "3 books: The Glass Orchard, Low Orbit and Seventeen Winters.") == (1.0, 1.0)
    assert verdicts(BOOKS_Q, BOOKS, "3 books: Seventeen Winters, The Glass Orchard and Low Orbit.") == (1.0, 1.0)
    assert verdicts(BOOKS_Q, BOOKS, "Total: Three.") == (1.0, 1.0)


def test_a_named_wrong_alternative_is_fatal_and_a_superseded_one_must_come_first():
    assert verdicts(CITY_Q, CITY, "You live in Lisbon.") == (1.0, 1.0)
    assert verdicts(CITY_Q, CITY, "Lisbon or Porto.") == (0.0, 0.0)
    assert verdicts(CITY_Q, CITY, "Lisbon and Porto.") == (0.0, 0.0)
    assert verdicts(CITY_Q, CITY, "Lisbon. Porto.") == (0.0, 0.0)
    # Changed: this was (0.0, 0.0), "an alternative named in order to dismiss it is still named".  A
    # dismissed alternative is not asserted, so the response commits to Lisbon alone; dismissing the
    # right value in favour of a wrong one is still wrong.
    assert verdicts(CITY_Q, CITY, "It is Lisbon, not Porto.") == (1.0, 1.0)
    assert verdicts(CITY_Q, CITY, "It is Porto, not Lisbon.") == (0.0, 0.0)
    assert verdicts(CAR_Q, CAR, "You used to drive a Honda Civic, but you now drive a Mazda CX-5.") == (1.0, 1.0)
    assert verdicts(CAR_Q, CAR, "honda civic, mazda cx-5") == (1.0, 1.0)
    assert verdicts(CAR_Q, CAR, "mazda cx-5, honda civic") == (0.0, 0.0)  # the old value is the last word
    assert verdicts(CAR_Q, CAR, "You sold the Honda Civic and now drive a Mazda CX-5 or a Kia Niro.") == (0.0, 0.0)
    # the hedged form of a right answer: official rule accepts it, the strict judge does not
    assert verdicts(CITY_Q, CITY, "I think it might be Lisbon, but I'm not sure.") == (1.0, 0.0)


def test_negation_reaches_five_tokens_and_stops_at_a_clause_break():
    assert verdicts(CITY_Q, CITY, "Not Lisbon; you live in Porto.") == (0.0, 0.0)
    assert verdicts(CITY_Q, CITY, "You have never, as far as I can tell, lived anywhere but Lisbon.") == (1.0, 1.0)
    assert verdicts(CITY_Q, CITY, "No. You live in Lisbon.") == (1.0, 1.0)
    assert verdicts(CITY_Q, CITY, "No doubt about it: Lisbon.") == (1.0, 1.0)
    assert verdicts(CITY_Q, CITY, "It isn't the city you grew up in, but Lisbon.") == (1.0, 1.0)
    assert verdicts(CITY_Q, CITY, "You don't live in the city of Lisbon.") == (0.0, 0.0)
    pasta = key("orecchiette with broccoli rabe", "single-session-assistant")
    assert verdicts("What pasta recipe did you recommend?", pasta, "It was not orecchiette with broccoli rabe.") == (0.0, 0.0)


def test_the_last_stated_number_is_the_answer():
    total = key("7", "multi-session", aliases=("seven",), numeric=7, numeric_question=True, wrong=[])
    assert verdicts("How many succulents have I bought in total?", total, "You bought 3, then 4, for a total of 7.") == (1.0, 1.0)
    assert verdicts("How many succulents have I bought in total?", total, "You bought 3, then 4, then 7.") == (1.0, 1.0)
    assert verdicts("How many succulents have I bought in total?", total, "7, 8, 9") == (0.0, 0.0)
    assert verdicts(BOOKS_Q, BOOKS, "It is 4, not 3.") == (0.0, 0.0)
    assert verdicts(BOOKS_Q, BOOKS, "Not 4 but 3.") == (1.0, 1.0)
    assert verdicts(BOOKS_Q, BOOKS, "You have not finished 3 books.") == (0.0, 0.0)
    days = key("18 days", "temporal-reasoning", aliases=("18",), numeric=18, unit="days", tolerance=1, numeric_question=True, wrong=[])
    assert verdicts("How many days ago did I renew my passport?", days, "17, 18, 19") == (1.0, 0.0)  # reads as 19; off by one
    assert verdicts("How many days ago did I renew my passport?", days, "16, 17, 20") == (0.0, 0.0)


def test_a_list_answer_may_not_recite_the_pool():
    assert verdicts(TRIPS_Q, TRIPS, "Japan, Peru, and Nepal") == (1.0, 1.0)
    assert verdicts(TRIPS_Q, TRIPS, "Peru, Nepal, Japan and Chile.") == (1.0, 0.0)  # one stray item: the hand label
    assert verdicts(TRIPS_Q, TRIPS, "Peru, Nepal, Japan, Chile and Kenya.") == (0.0, 0.0)
    assert verdicts(TRIPS_Q, TRIPS, "peru, nepal, japan and chile") == (1.0, 0.0)  # lower case hides nothing
    assert verdicts(TRIPS_Q, TRIPS, "You have not been to Peru, Nepal or Japan.") == (0.0, 0.0)


def test_declining_counts_only_if_nothing_is_offered_with_it():
    assert verdicts(SISTER_Q, SISTER, "You haven't mentioned your sister's name.") == (1.0, 1.0)
    assert verdicts(SISTER_Q, SISTER, DECLINE + "priya dana tomasz") == (0.0, 0.0)
    assert verdicts(BOOKS_Q, NO_BOOKS, DECLINE.strip()) == (1.0, 1.0)
    assert verdicts(BOOKS_Q, NO_BOOKS, DECLINE + "2, 3, 4") == (0.0, 0.0)
    assert verdicts(BOOKS_Q, NO_BOOKS, DECLINE + "Three.") == (0.0, 0.0)
    assert verdicts(BOOKS_Q, NO_BOOKS, "You never mentioned finishing any books, only that Dana read Low Orbit.") == (0.0, 0.0)
    # a decline that restates the question is still a decline: there is no answer in the question to give away
    assert verdicts(BOOKS_Q, NO_BOOKS, "You never mentioned how many books you have finished reading.") == (1.0, 1.0)
    # the same template earns nothing where there is an answer
    assert verdicts(BOOKS_Q, BOOKS, "You never mentioned how many books you have finished reading.") == (0.0, 0.0)


def test_which_first_questions_take_the_right_option_first_and_no_third_event():
    assert verdicts(ORDER_Q, ORDER, "You renewed your passport first, and got the tattoo afterwards.") == (1.0, 1.0)
    assert verdicts(ORDER_Q, ORDER, "get a tattoo, renew my passport") == (0.0, 0.0)
    assert verdicts(ORDER_Q, ORDER, "renew my passport go skydiving donate blood") == (0.0, 0.0)
    # every word of both options, no option in one piece: the paraphrase allowance must not accept it
    assert verdicts(ORDER_Q, ORDER, "passport tattoo my get renew") == (0.0, 0.0)
    assert verdicts(ORDER_Q, key("get a tattoo", "temporal-reasoning", aliases=("got a tattoo",),
                                 distractors=["renew my passport", "renewed my passport"]), "passport tattoo my get renew") == (0.0, 0.0)


def test_keys_without_alternatives_are_judged_as_before():
    """Public benchmarks' keys carry no "wrong" list; the rules that need one do not apply and nothing breaks."""
    bare_city = key("Lisbon")
    assert verdicts(CITY_Q, bare_city, "It is Lisbon, not Porto.") == (1.0, 1.0)
    assert verdicts(CITY_Q, bare_city, "You definitely do not live in Lisbon.") == (0.0, 0.0)  # negation needs no list
    assert verdicts(CITY_Q, bare_city, CITY_Q + " Let me think.") == (0.0, 0.0)  # nor does the echo rule
    assert verdicts(TRIPS_Q, key("Peru, Nepal and Japan", "multi-session", list=["Peru", "Nepal", "Japan"]),
                    "Peru, Nepal, Japan and Chile.") == (1.0, 0.0)
    assert verdicts(BOOKS_Q, key("3", "multi-session", numeric=3), "2, 3, 4") == (0.0, 0.0)
    assert verdicts(SISTER_Q, key("never mentioned", unanswerable=True), "You never told me that.") == (1.0, 1.0)
    for mode in ("generous", "graded"):  # the lax judges read no lists at all
        assert HeuristicJudge(mode).judge(CITY_Q, CITY, "Lisbon").score == 1.0
    assert HeuristicJudge("generous").judge(CITY_Q, CITY, " ".join(syn.CITIES).lower()).score == 1.0  # lax by design


# -------------------------------------------------------------- false rejections

# Phrasings a judge has no excuse to reject: template -> the share of honest answers the lenient
# and the strict judge must accept (None where the strict judge rejects the phrasing by design).
PHRASINGS = {
    "{gold}": (0.95, 0.95),
    "It is {gold}.": (0.95, 0.95),
    "You told me {gold}.": (0.95, 0.95),
    "{gold}.": (0.95, 0.95),
    "The answer is {gold}.": (0.95, 0.95),
    "Based on what you told me, it's {gold}.": (0.95, 0.95),
    "{gold}, I think.": (0.95, None),  # hedged
    # These five were the known false rejections of the first round (each was xfail); the rules that
    # read a phrase by what it is attached to accept them.
    "{question} {gold}": (0.95, 0.95),  # the question restated, then the answer
    "{number} days (roughly two weeks)": (0.95, 0.95),  # the number attached to the unit asked for
    "{gold}; before that it was {stale}.": (0.95, 0.95),  # the old value marked as old
    "{gold} over {distractor}.": (0.95, 0.95),  # the other side of a preference, dismissed
    "It is {gold}, not {wrong}.": (0.95, 0.95),  # an alternative named in order to dismiss it
}
# How many items a phrasing must apply to; the ones that need an update, a preference or a day count
# apply to fewer.
APPLICABLE = {"{number} days (roughly two weeks)": 20, "{gold}; before that it was {stale}.": 20, "{gold} over {distractor}.": 20}
# Answers a careful grader would accept and these judges do not: what the rules cost.  Each is
# expected to fail the 95% bar; if a later change makes one pass, move it up.
KNOWN_FALSE_REJECTIONS = {
    "{gold}. {stale} was a while ago.": "a superseded value named after the current one counts as the last word unless "
                                        "one of the phrases the rule knows marks it as old; 'a while ago' is not one",
}


def acceptance(world: World, template: str) -> tuple[float, float, int]:
    """Share of honest answers in this phrasing that each judge accepts, and how many items it applies to."""
    judges = [HeuristicJudge("lenient"), HeuristicJudge("strict")]
    accepted, applicable = [0, 0], 0
    for inst in world.instances:
        meta = inst.key.meta
        if inst.key.is_abstention:
            continue
        fields = {"gold": inst.key.answer, "question": inst.probe.question}
        if meta.get("unit") == "days":
            fields["number"] = meta["numeric"]
        if meta.get("stale"):
            fields["stale"] = meta["stale"][-1]
        if "list" not in meta and meta.get("wrong"):
            fields["wrong"] = next(v for v in meta["wrong"] if v not in meta.get("stale", ()))
            if meta.get("distractors", [None])[0] in meta["wrong"]:
                fields["distractor"] = meta["distractors"][0]
        try:
            response = template.format(**fields)
        except KeyError:
            continue  # the phrasing needs something this item does not have
        applicable += 1
        for k, judge in enumerate(judges):
            accepted[k] += judge.judge(inst.probe.question, inst.key, response).score >= 1.0
    return accepted[0] / applicable, accepted[1] / applicable, applicable


@pytest.mark.parametrize("template", list(PHRASINGS))
def test_honest_answers_are_accepted_in_ordinary_phrasings(world, template):
    lenient_floor, strict_floor = PHRASINGS[template]
    lenient, strict, applicable = acceptance(world, template)
    assert applicable > APPLICABLE.get(template, 100)
    assert lenient >= lenient_floor
    assert strict_floor is None or strict >= strict_floor


@pytest.mark.parametrize("template", [
    pytest.param(template, marks=pytest.mark.xfail(strict=True, reason=reason))
    for template, reason in KNOWN_FALSE_REJECTIONS.items()
])
def test_known_false_rejections(world, template):
    lenient, strict, applicable = acceptance(world, template)
    assert applicable > 0 and lenient >= 0.95 and strict >= 0.95


def test_the_reference_answer_is_accepted_in_each_answer_style_of_the_offline_system(world):
    """Whatever the rules reject, they must not reject the reference itself, terse or wrapped in a sentence."""
    for inst in world.instances:
        if inst.key.is_abstention:
            continue
        for response in (inst.key.answer, f"Based on what you told me, it's {inst.key.answer}.",
                         f"Based on our earlier conversations, the answer is {inst.key.answer}."):
            assert verdicts(inst.probe.question, inst.key, response) == (1.0, 1.0), (inst.probe.question, response)


# -------------------------------------------------------------------- judge exam

# Digest of the 68 cases that existed before the "review" part was added.
EXAM_BEFORE_REVIEW = "b3bde842eecba5d1"


def test_the_exam_gained_a_review_part_and_lost_nothing():
    earlier = [c for c in JUDGE_EXAM if c["split"] in ("dev", "fresh")]  # changed: the exam has since gained "review2" too
    assert [c["split"] for c in earlier] == ["dev"] * 52 + ["fresh"] * 16
    assert digest([
        [c["split"], c["tag"], c["question"], c["response"], c["official"], c["strict"], c["key"].question_type,
         c["key"].answer, list(c["key"].aliases), c["key"].is_abstention, dict(sorted(c["key"].meta.items()))]
        for c in earlier
    ]) == EXAM_BEFORE_REVIEW
    # a judge under calibration is told only the question and the response, so the pair must identify the case
    assert len({(c["question"], c["response"]) for c in JUDGE_EXAM}) == len(JUDGE_EXAM)


@pytest.mark.parametrize("mode, target", [("lenient", "official"), ("strict", "strict")])
def test_both_judges_agree_with_the_hand_labels_on_the_reviewers_cases(mode, target):
    result = judge_exam(HeuristicJudge(mode), target, split="review")
    assert result["n"] == 9 and result["errors"] == 0
    assert result["accuracy"] == 1.0, result["misses"]


# ------------------------------------------------------------------------- audit

NEW_NULLS = ("vocabulary_list", "decline_plus_vocabulary", "number_run", "negated_vocabulary")


def null_checks(report) -> dict:
    return {c["check"].split("'")[1]: c for c in report.checks if c["check"].startswith("null baseline")}


@pytest.fixture(scope="module")
def audit_bench():
    return syn.generate("audit", syn.GenConfig(n_streams=30))


def test_audit_passes_with_the_candidate_naming_nulls(audit_bench):
    report = run_audit(audit_bench, HeuristicMemory(MemoryConfig(features=frozenset(FEATURES), top_k=10)))
    assert report.passed, report.failures()
    checks = null_checks(report)
    assert set(NEW_NULLS) <= set(checks)
    assert checks["negated_vocabulary"]["detail"]["highest_judged_score"] == 0.0


class ContainmentJudge:
    """The judge the review warned about: correct whenever the reference occurs in the response."""

    name = "containment"

    def judge(self, question, answer_key, response):
        if answer_key.is_abstention:
            return JudgeVerdict(1.0 if is_abstention(response) else 0.0)
        return JudgeVerdict(metrics.contains(response, (answer_key.answer, *answer_key.aliases)))

    def spec(self):
        return {"kind": "containment"}


def test_audit_fails_a_judge_that_pays_for_naming_candidates(audit_bench):
    """The new nulls have to be able to fail: seat a containment judge where the reporting judges sit."""
    seats = dict(default_judges(), judge_lenient=ContainmentJudge(), judge_strict=ContainmentJudge())
    report = run_audit(audit_bench, judges=seats)
    failed = {name for name, check in null_checks(report).items() if not check["passed"]}
    assert set(NEW_NULLS) <= failed


def test_a_recited_vocabulary_never_includes_the_questions_own_key(audit_bench):
    instances = list(audit_bench.instances)
    vocabulary = answer_vocabularies(instances)

    def opening(inst):
        return " ".join(inst.probe.question.lower().split()[:3])

    for inst in instances:
        others = {i.key.answer.lower() for i in instances
                  if i is not inst and not i.key.is_abstention and opening(i) == opening(inst)}
        assert set(vocabulary[inst.probe_id]) <= others
        assert sum(len(v.split()) for v in vocabulary[inst.probe_id]) <= 60
    assert any(vocabulary.values())


# ------------------------------------------------------------------- performance

def test_grading_stays_well_under_half_a_millisecond_per_item(world):
    best = min(_timed(lambda: world.grader.grade(world.honest_answers, world.instances)) for _ in range(5))
    assert 1000 * best / len(world.instances) < 0.5


def _timed(call) -> float:
    start = time.perf_counter()
    call()
    return time.perf_counter() - start
