"""What the reporting judges must accept and must reject, phrasing by phrasing.

A second review measured the lenient and the strict heuristic judge in both
directions: responses that earned credit without answering, and natural correct
answers that were rejected.  This file pins the rules written in reply, one
group per rule, with the reviewer's own phrasings as the cases:

  A  unanswerable questions: a decline is correct only if no answer is stated with it
  B  ways of declining
  C  alternatives: the option bound to the asserting phrase decides
  D  updated values: the value marked as current decides
  E  numbers: the number the response commits to for the thing asked
  F  lists under the strict judge
  G  hedges and several candidates under the strict judge
  H  other correct answers (question restated, closing remarks, singular for plural)
  I  normalisation
  J  cost
  K  rows that are not answers
  R  what had to keep holding (exam, honest system, references, floors, determinism)

Where two rules pull in opposite directions the judges prefer not to give credit
to a response that does not answer; the phrasing that loses is marked
``xfail(strict=True)`` with the reason.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from memarena.benchmark import synthetic as syn
from memarena.benchmark.adapters import load_longmemeval
from memarena.benchmark.schema import Answer, AnswerKey, Instance, Probe
from memarena.evalcore.audit import judge_exam, oracle_answers, run_audit
from memarena.evalcore.criteria import CriteriaFamily
from memarena.evalcore.grader import Grader, default_judges
from memarena.evalcore.judge_exam import JUDGE_EXAM, exam_key
from memarena.evalcore.judges import JUDGE_MODES, HeuristicJudge, JudgeVerdict
from memarena.evalcore.normalize import (
    MAX_RESPONSE_CHARS,
    is_abstention,
    is_clean_abstention,
    is_hedged,
    is_noncommittal,
    number_mentions,
    numbers_in,
    read,
    tokens,
)
from memarena.memory.base import ABSTAIN_TEXT, run_system
from memarena.memory.heuristic import FEATURES, HeuristicMemory, MemoryConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LENIENT_ID, STRICT_ID = "judge_lenient|detect|micro", "judge_strict|detect|micro"
JUDGES = (HeuristicJudge("lenient"), HeuristicJudge("strict"))


def verdicts(question, answer_key, response) -> tuple[float, float]:
    return tuple(judge.judge(question, answer_key, response).score for judge in JUDGES)


def key(answer, question_type="single-session-user", aliases=(), unanswerable=False, **meta) -> AnswerKey:
    return AnswerKey("p", question_type, answer, tuple(aliases), unanswerable, (), meta)


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()[:16]


@pytest.fixture(scope="module")
def bench():
    """The reviewer's benchmark: every count below is measured on it."""
    return syn.generate("rev-grader", syn.GenConfig(n_streams=25))


def accepted(items, phrase) -> tuple[int, int]:
    """How many of ``items`` each judge accepts when answered with ``phrase(item)``."""
    scores = [verdicts(i.probe.question, i.key, phrase(i)) for i in items]
    return int(sum(s[0] for s in scores)), int(sum(s[1] for s in scores))


CITY_Q, CAR_Q, SISTER_Q = "Which city do I live in?", "What car do I drive now?", "What is my sister's name?"
BOOKS_Q, TRIPS_Q, SKY_Q = "How many books have I finished reading?", "Which countries have I taken a trip to?", "How many days ago did I go skydiving?"
ORDER_Q, PREF_Q = "Which did I do first: renew my passport or get a tattoo?", "What do I prefer when it comes to coffee beans?"
CITY = key("Lisbon")  # as a public benchmark writes it: no list of wrong alternatives
CITY_POOL = key("Lisbon", wrong=[c for c in syn.CITIES if c != "Lisbon"])
CAR = key("Mazda CX-5", "knowledge-update", stale=["Honda Civic"], wrong=[c for c in syn.CARS if c != "Mazda CX-5"])
SISTER = key("The user never mentioned their sister's name.", unanswerable=True)
SISTER_POOL = key("The user never mentioned their sister's name.", unanswerable=True, wrong=list(syn.NAMES))
NO_SKY = key("The user never mentioned going skydiving.", "temporal-reasoning", unanswerable=True, numeric_question=True, wrong=[])
TRIPS = key("Peru, Nepal and Japan", "multi-session", list=["Peru", "Nepal", "Japan"])
TRIPS_POOL = key("Peru, Nepal and Japan", "multi-session", list=["Peru", "Nepal", "Japan"],
                 wrong=[c for c in syn.COUNTRIES if c not in ("Peru", "Nepal", "Japan")])
ORDER = key("renew my passport", "temporal-reasoning", aliases=("renewed my passport",), distractors=["get a tattoo", "got a tattoo"],
            wrong=[e for e in syn.EVENT_PHRASES if e not in ("renew my passport", "renewed my passport", "get a tattoo", "got a tattoo")])
PREF = key("dark roast", "single-session-preference", distractors=["light roast"],
           wrong=[p for p in syn.PREFERENCE_PHRASES if p != "dark roast"])
BOOKS = key("3", "multi-session", aliases=("three",), numeric=3, numeric_question=True, wrong=[])
DAYS = key("77 days", "temporal-reasoning", aliases=("77",), numeric=77, unit="days", tolerance=1, numeric_question=True, wrong=[])
ECLIPSE_Q = "How many days ago did I see a solar eclipse?"


# ------------------------------------------------- A. unanswerable questions

@pytest.mark.parametrize("answer_key, question, response", [
    (SISTER, SISTER_Q, "I don't know."),
    (SISTER, SISTER_Q, "You haven't mentioned your sister's name."),
    (SISTER, SISTER_Q, "I don't have that information, and I think it's better not to speculate."),
    (SISTER, SISTER_Q, "I don't have any record of you mentioning your sister's name in our conversations. "
                       "If you'd like to share it, I'd be happy to remember it for next time!"),
    (SISTER, SISTER_Q, "Based on our conversation history, I don't have any information about your sister's name."),
    # other information is given, the asked information is not: LongMemEval's own description of a correct decline
    (SISTER, SISTER_Q, "You mentioned your friend Dana has a sister, but you never told me your own sister's name."),
    (SISTER, SISTER_Q, "You mentioned your cat Luna but not your sister's name."),
    (key("x", unanswerable=True), "How often do I see Dr. Johnson?",
     "The information provided is not enough. You mentioned seeing Dr. Smith but not Dr. Johnson."),
    (key("x", unanswerable=True), "When did I book the Airbnb in Sacramento?",
     "The information provided is not enough. You only mentioned booking Airbnb in San Francisco."),
    (NO_SKY, SKY_Q, "You never mentioned going skydiving."),
    (SISTER_POOL, SISTER_Q, "You never told me your sister's name."),
])
def test_a_decline_that_states_no_answer_is_correct(answer_key, question, response):
    assert verdicts(question, answer_key, response) == (1.0, 1.0)


@pytest.mark.parametrize("answer_key, question, response", [
    (SISTER, SISTER_Q, "Priya. I don't know."),
    (SISTER, SISTER_Q, "I don't have that information. Priya."),
    (SISTER, SISTER_Q, "Your sister's name is Priya. You never mentioned her age."),
    (SISTER, SISTER_Q, "Priya (you never mentioned anything else)."),
    (SISTER, SISTER_Q, "Her name is Priya. You never mentioned otherwise."),
    (SISTER, SISTER_Q, "I don't know, but my best guess is Priya."),
    (SISTER, SISTER_Q, "I don't know, but my best guess is pastry chef."),
    (SISTER, SISTER_Q, "Definitely Priya. I'm not aware of any nickname."),
    (SISTER, SISTER_Q, "I don't know. \"Priya\"?"),
    (SISTER, SISTER_Q, "pastry chef. I don't know."),
    (SISTER, SISTER_Q, "I don't have that information. pastry chef."),
    (SISTER, SISTER_Q, "I don't know. 3."),
    (SISTER, SISTER_Q, "I don't know. Maybe in March?"),
    # a report of what was said is other information only next to a decline of the question itself
    (SISTER, SISTER_Q, "You mentioned that your sister's name is Priya, but you never told me her age."),
    (NO_SKY, SKY_Q, "I don't have that information. 12 days."),
    (NO_SKY, SKY_Q, "You never mentioned going skydiving. It was on 2023-03-14."),
    (NO_SKY, SKY_Q, "I don't have that information. Forty-two."),
    (SISTER_POOL, SISTER_Q, "You never told me, but it could be dana."),
    (SISTER_POOL, SISTER_Q, "I don't have that information. priya dana tomasz"),
    # with the pool of possible answers in the key, naming one of them is naming an answer, whatever the frame
    (SISTER_POOL, SISTER_Q, "You mentioned your friend Dana has a sister, but you never told me your own sister's name."),
    (SISTER, SISTER_Q, "Priya."),
    (SISTER, SISTER_Q, ""),
])
def test_a_decline_with_an_answer_next_to_it_is_an_answer(answer_key, question, response):
    assert verdicts(question, answer_key, response) == (0.0, 0.0)


def test_on_generated_data_no_invented_answer_is_credited_as_a_decline(bench):
    """Before: "I don't have that information. {invented}." and "{invented}. I don't know." 12 of 12, both judges."""
    unanswerable = [i for i in bench if i.key.is_abstention]
    assert len(unanswerable) == 12
    templates = ["{a}", "I don't have that information, but my best guess is {a}.", "I don't have that information. {a}.",
                 "{a}. I don't know.", "{a} (you never mentioned anything else).", "You never told me that; {a}?"]
    for inst in unanswerable:
        question, answer_key = inst.probe.question, inst.key
        invented = ["Priya", "42", "pastry chef", "2023-08-19", "March"] + list(answer_key.meta.get("wrong", ()))[:3]
        assert verdicts(question, answer_key, "I don't have that information.") == (1.0, 1.0)
        for answer in invented:
            for template in templates:
                assert verdicts(question, answer_key, template.format(a=answer)) == (0.0, 0.0), template.format(a=answer)
            if answer_key.meta.get("wrong"):
                assert verdicts(question, answer_key, f"I don't have that information. {answer.lower()}.") == (0.0, 0.0)


def test_one_template_cannot_collect_on_both_kinds_of_question(bench):
    """The answer where there is one, an invented one where there is none, a decline attached to both."""
    grader = Grader(CriteriaFamily.default())
    share = sum(i.key.is_abstention for i in bench) / len(bench)
    for template in ("I don't have that information. {a}.", "{a}. I don't know."):
        answers = {i.probe_id: Answer(template.format(a="Priya" if i.key.is_abstention else i.key.answer)) for i in bench}
        report = grader.grade(answers, bench.instances)
        credited = [it.base for it in report.items if it.is_abstention]
        assert sum(b["judge_lenient"] + b["judge_strict"] for b in credited) == 0
        assert report.score(LENIENT_ID) == pytest.approx(1.0 - share)  # the answers, and nothing for the "declines"
        assert report.score(STRICT_ID) == 0.0  # a response that declines the question has not committed


# ------------------------------------------------------ B. ways of declining

DECLINES = """I don't have that information.
You haven't mentioned that.
I don't know.
You never told me that.
There is no record of that in our conversations.
That was never mentioned.
I can't find anything about that.
I couldn't find that in our conversations.
I'm not aware of that.
That information was not provided.
The conversation history does not mention that.
Our conversations don't mention that.
Nothing in our chats says anything about that.
I have no idea.
I'm not able to answer that from what you've shared.
That isn't something you've shared with me.
That information isn't available.
It hasn't come up in our conversations.
I don't believe you've told me that.
I don't think you ever mentioned it.
You have not told me.
I wasn't told that.
Unknown.
No idea, sorry.
There's nothing about that in what you've told me.
I have no memory of you sharing that.
You've not said.
I can not answer that.
This is unclear from our conversations.
I'm not sure; you haven't brought that up.
Sorry, that's not something I know.
I don't have that information. Maybe you mentioned it somewhere I can't see?
You never mentioned that. Perhaps you are thinking of your friend Dana?
You haven't told me that; it is likely something we never discussed.
I don't know. I could guess, but I'd rather not.
I don't have that information, and I think it's better not to speculate.
You never told me, so it could be anything.""".splitlines()
# The one decline of the reviewer's 37 that is not accepted.  It names a person, and what remains
# after the decline may not hold a capitalised name that is not part of the question: "Perhaps it is
# Dana?" would be a guess, and the rule cannot tell a guess from a suggestion.  Not giving credit wins.
NAMES_SOMEONE = "You never mentioned that. Perhaps you are thinking of your friend Dana?"
SISTER_NAME = key("The user never mentioned the name of their sister.", unanswerable=True)


@pytest.mark.parametrize("decline", [
    pytest.param(d, marks=pytest.mark.xfail(strict=True, reason="names a person next to the decline; rule A rejects a "
                                                                "capitalised name that is not part of the question"))
    if d == NAMES_SOMEONE else d for d in DECLINES
])
def test_an_ordinary_decline_is_recognised(decline):
    """Before: 24 of these 37 scored 0 (19 not recognised, 5 disqualified by a hedge word).  Now 36 are accepted."""
    assert len(DECLINES) == 37
    assert verdicts("What is the name of my sister?", SISTER_NAME, decline) == (1.0, 1.0)
    assert is_abstention(decline) and is_clean_abstention(decline, "What is the name of my sister?")


@pytest.mark.parametrize("decline", DECLINES)
def test_a_decline_followed_or_preceded_by_an_answer_stays_rejected(decline):
    question = "What is the name of my sister?"
    for response in (f"{decline} Priya.", f"Priya. {decline}", f"{decline} It's probably Priya.", f"{decline} 42."):
        assert verdicts(question, SISTER_NAME, response) == (0.0, 0.0), response


def test_a_hedge_word_alone_does_not_disqualify_a_decline():
    for response in ("I don't know. I could guess, but I'd rather not.", "You never told me, so it could be anything.",
                     "I don't have that information. Maybe you mentioned it somewhere I can't see?",
                     "You haven't told me that; it is likely something we never discussed."):
        assert verdicts(SISTER_Q, SISTER, response) == (1.0, 1.0)
        assert verdicts(SISTER_Q, SISTER, response + " Maybe Priya.") == (0.0, 0.0)


# ---------------------------------------------------------- C. alternatives

WHICH_FIRST = "Which did I do first: "


def your(phrase: str) -> str:
    return " ".join("your" if w == "my" else w for w in phrase.split())


def words_reversed(inst) -> str:
    """Question text only: both options with the words of each reversed (an option of fewer than three tokens stays)."""
    options = inst.probe.question[len(WHICH_FIRST):-1].split(" or ")
    short = [o for o in options if len(tokens(o)) < 3]
    long_ = [o for o in options if len(tokens(o)) >= 3]
    return "; ".join(short[:1] + [" ".join(reversed(o.split())) for o in short[1:] + long_])


def test_which_came_first_is_read_by_what_each_option_is_attached_to(bench):
    """Before: "Before you OTHER, you GOLD." and "You OTHER after you GOLD." were rejected 17 of 17."""
    asked = [i for i in bench if i.probe.question.startswith(WHICH_FIRST)]
    n = len(asked)
    assert n == 17
    gold, other = (lambda i: your(i.key.aliases[0])), (lambda i: your(i.key.meta["distractors"][1]))
    assert accepted(asked, lambda i: f"You {gold(i)} first.") == (n, n)
    assert accepted(asked, lambda i: f"Before you {other(i)}, you {gold(i)}.") == (n, n)
    assert accepted(asked, lambda i: f"You {other(i)} after you {gold(i)}.") == (n, n)
    assert accepted(asked, lambda i: f"You {gold(i)} first, and {other(i)} afterwards.") == (n, n)
    assert accepted(asked, lambda i: f"First you {gold(i)}, then you {other(i)}.") == (n, n)
    assert accepted(asked, lambda i: f"You did not {your(i.key.answer)} first; you {other(i)} first.") == (0, 0)
    assert accepted(asked, lambda i: f"You {gold(i)} after you {other(i)}.") == (0, 0)
    assert accepted(asked, lambda i: f"Before you {gold(i)}, you {other(i)}.") == (0, 0)
    assert accepted(asked, lambda i: f"You {other(i)} first.") == (0, 0)
    # the question-only trick must not beat naming the first option (it scored 7 of 17, as much as that guess)
    first_option = accepted(asked, lambda i: i.probe.question[len(WHICH_FIRST):-1].split(" or ")[0])
    trick = accepted(asked, words_reversed)
    assert trick == (0, 0) and first_option == (7, 7)


def test_a_preference_is_read_by_what_each_option_is_attached_to(bench):
    """Before: 0 of 14 for each of the three phrasings."""
    asked = [i for i in bench if i.key.question_type == "single-session-preference" and not i.key.is_abstention]
    n = len(asked)
    assert n == 14
    gold, other = (lambda i: i.key.answer), (lambda i: i.key.meta["distractors"][0])
    for phrase in ("You prefer {g} over {o}.", "Rather than {o}, you prefer {g}.", "Between {o} and {g}, you prefer {g}.",
                   "Not {o} - you prefer {g}.", "{g} instead of {o}.", "You prefer {g} to {o}.", "{g}, not {o}."):
        assert accepted(asked, lambda i: phrase.format(g=gold(i), o=other(i))) == (n, n), phrase
    for phrase in ("You prefer {o} over {g}.", "Rather than {g}, you prefer {o}.", "Between {o} and {g}, you prefer {o}.",
                   "{o} instead of {g}.", "{g}, {o}", "{o}, {g}", "{g} and {o}."):
        assert accepted(asked, lambda i: phrase.format(g=gold(i), o=other(i))) == (0, 0), phrase


@pytest.mark.parametrize("answer_key, question, response, expected", [
    (CITY_POOL, CITY_Q, "It is Lisbon, not Porto.", (1.0, 1.0)),
    (CITY_POOL, CITY_Q, "Lisbon rather than Porto or Valencia.", (1.0, 1.0)),
    (CITY_POOL, CITY_Q, "Not Porto, not Valencia: Lisbon.", (1.0, 1.0)),
    (CITY_POOL, CITY_Q, "It is Porto, not Lisbon.", (0.0, 0.0)),
    (CITY_POOL, CITY_Q, "Porto rather than Lisbon.", (0.0, 0.0)),
    (CITY_POOL, CITY_Q, "Lisbon and Porto.", (0.0, 0.0)),
    (CITY_POOL, CITY_Q, "Lisbon, Porto", (0.0, 0.0)),
    (CITY_POOL, CITY_Q, "It's none other than Lisbon.", (1.0, 1.0)),
    (CITY_POOL, CITY_Q, "No doubt it's Lisbon.", (1.0, 1.0)),
    (ORDER, ORDER_Q, "Before you got a tattoo, you renewed your passport.", (1.0, 1.0)),
    (ORDER, ORDER_Q, "You got a tattoo after you renewed your passport.", (1.0, 1.0)),
    (ORDER, ORDER_Q, "You got a tattoo second; you renewed your passport first.", (1.0, 1.0)),
    (ORDER, ORDER_Q, "You renewed your passport after you got a tattoo.", (0.0, 0.0)),
    (ORDER, ORDER_Q, "You got a tattoo, then renewed your passport.", (0.0, 0.0)),
    (ORDER, ORDER_Q, "tattoo a get; passport my renew", (0.0, 0.0)),
    (PREF, PREF_Q, "Dark roast over light roast.", (1.0, 1.0)),
    (PREF, PREF_Q, "Light roast over dark roast.", (0.0, 0.0)),
    (PREF, PREF_Q, "You're more of a dark roast person than a light roast person.", (1.0, 1.0)),
    (PREF, PREF_Q, "Light roast is not for you; dark roast is.", (1.0, 1.0)),
])
def test_single_cases_of_alternatives(answer_key, question, response, expected):
    assert verdicts(question, answer_key, response) == expected


# -------------------------------------------------------- D. updated values

def test_an_update_is_read_by_which_value_is_marked_as_current(bench):
    """Before: "NEW (previously OLD)." was rejected 50 of 50 by both judges."""
    updated = [i for i in bench if not i.key.is_abstention and i.key.meta.get("stale")]
    n = len(updated)
    assert n == 50
    new, old = (lambda i: i.key.answer), (lambda i: i.key.meta["stale"][-1])
    for phrase in ("{new} (previously {old}).", "{new}; before that it was {old}.", "{new}, not {old} any more.",
                   "{new} - you switched from {old}.", "It used to be {old}, but now it is {new}.", "{old} (now {new}).",
                   "{new} now; {old} before.", "{old} at first, then {new}.", "{old}, {new}"):
        assert accepted(updated, lambda i: phrase.format(new=new(i), old=old(i))) == (n, n), phrase
    for phrase in ("{old}", "{old} (previously {new}).", "It used to be {new}, but now it is {old}.", "{new}, {old}",
                   "{old}, not {new} any more.", "{new}, though you may have switched back to {old}."):
        assert accepted(updated, lambda i: phrase.format(new=new(i), old=old(i))) == (0, 0), phrase


def test_single_cases_of_updates():
    assert verdicts(CAR_Q, CAR, "You now drive a Mazda CX-5; before that it was a Honda Civic.") == (1.0, 1.0)
    assert verdicts(CAR_Q, CAR, "A Mazda CX-5, though you may have switched back to the Honda Civic.") == (0.0, 0.0)
    assert verdicts(CAR_Q, CAR, "You sold the Honda Civic and now drive a Mazda CX-5.") == (1.0, 1.0)
    assert verdicts(CAR_Q, CAR, "Mazda CX-5 (previously a Kia Niro).") == (1.0, 1.0)  # wrong history, right answer
    assert verdicts(CAR_Q, CAR, "Mazda CX-5 or Kia Niro.") == (0.0, 0.0)
    # only a question about the current value makes "used to" a reason to doubt the reference
    assert verdicts("Where did I live before?", key("Porto"), "You used to live in Porto.") == (1.0, 1.0)
    assert verdicts("Where did I move from?", key("Porto"), "You moved from Porto.") == (1.0, 1.0)


# ---------------------------------------------------------------- E. numbers

def test_a_number_is_read_with_its_unit_and_without_the_dates_around_it(bench):
    """Before: every line here but the first, fifth and sixth was rejected by both judges on every item."""
    numeric = [i for i in bench if "numeric" in i.key.meta]
    ago = [i for i in numeric if i.probe.question.startswith("How many days ago")]
    between = [i for i in numeric if i.probe.question.startswith("How many days passed")]
    counts = [i for i in numeric if "days" not in i.probe.question and "in total" not in i.probe.question]
    assert (len(ago), len(between), len(counts)) == (14, 19, 33)
    n = lambda i: int(i.key.meta["numeric"])  # noqa: E731
    asked_on = lambda i: dt.date.fromisoformat(i.probe.question_date)  # noqa: E731
    event = lambda i: asked_on(i) - dt.timedelta(days=n(i))  # noqa: E731
    for items, phrase in [
        (ago, lambda i: f"About {n(i)} days ago."),
        (ago, lambda i: f"{n(i)} days (roughly {round(n(i) / 7)} weeks)."),
        (ago, lambda i: f"{n(i)} days ago, on {event(i):%B} {event(i).day}."),
        (ago, lambda i: f"{n(i)} days ago ({event(i)})."),
        (ago, lambda i: f"You did that on {event(i)} and today is {asked_on(i)}, so that was {n(i)} days ago."),
        (ago, lambda i: f"You did that on {event(i):%B} {event(i).day}, {event(i).year} and today is "
                        f"{asked_on(i):%B} {asked_on(i).day}, {asked_on(i).year}, so that was {n(i)} days ago."),
        (ago, lambda i: f"{n(i)} days ({n(i) // 7} weeks and {n(i) % 7} days)."),
        (between, lambda i: f"{n(i)} days passed between the two."),
        (between, lambda i: f"From {event(i)} to {asked_on(i)} is {n(i)} days."),
        (counts, lambda i: f"{n(i)} - one more than last time you asked."),
        (counts, lambda i: f"{n(i)} (as of {i.probe.question_date})."),
    ]:
        assert accepted(items, phrase) == (len(items), len(items)), phrase(items[0])


def test_a_date_or_another_unit_is_not_a_number_of_days(bench):
    """Before: "N weeks" passed for N days 33 of 33; a bare date was read as a count where its digits fitted."""
    days = [i for i in bench if "numeric" in i.key.meta and "days" in i.probe.question]
    assert len(days) == 33
    n = lambda i: int(i.key.meta["numeric"])  # noqa: E731
    event = lambda i: dt.date.fromisoformat(i.probe.question_date) - dt.timedelta(days=n(i))  # noqa: E731
    for phrase in (lambda i: f"{n(i)} weeks", lambda i: f"{n(i)} months ago", lambda i: f"{n(i)} hours.",
                   lambda i: str(event(i)), lambda i: f"It was on {event(i).day} {event(i):%B}.",
                   lambda i: f"On {event(i):%B} {event(i).day}, {event(i).year}.", lambda i: f"At 8:{n(i) % 60:02d} pm."):
        assert accepted(days, phrase) == (0, 0), phrase(days[0])
    # the same dates with a day of the month that equals the answer, or is one off: still not a count
    for inst in days:
        for near in {n(inst) - 1, n(inst), n(inst) + 1} & set(range(1, 29)):
            for response in (f"2023-08-{near:02d}", f"It was on {near} August.", f"August {near}, 2023"):
                assert verdicts(inst.probe.question, inst.key, response) == (0.0, 0.0), response
    whole_weeks = [i for i in days if n(i) % 7 == 0 and n(i)]
    assert whole_weeks and accepted(whole_weeks, lambda i: f"{n(i) // 7} weeks.") == (len(whole_weeks), len(whole_weeks))


def test_a_run_of_numbers_is_worth_no_more_than_its_last_element(bench):
    counts = [i for i in bench if "numeric" in i.key.meta and "days" not in i.probe.question and "in total" not in i.probe.question]
    guess = accepted(counts, lambda i: "4")
    for run in ("2, 3, 4", "2 or 3 or 4", "2 books, 3 books, 4 books", "2\n3\n4"):
        got = accepted(counts, lambda i: run)
        assert got[0] <= guess[0] and got[1] <= guess[1], run
    assert accepted(counts, lambda i: "2, 3, 4") == guess  # exactly its last element


@pytest.mark.parametrize("answer_key, question, response, expected", [
    (DAYS, ECLIPSE_Q, "77 days (roughly 11 weeks).", (1.0, 1.0)),
    (DAYS, ECLIPSE_Q, "11 weeks.", (1.0, 1.0)),
    (DAYS, ECLIPSE_Q, "10 weeks and 7 days.", (1.0, 1.0)),
    (DAYS, ECLIPSE_Q, "77 weeks.", (0.0, 0.0)),
    (DAYS, ECLIPSE_Q, "Seventy-seven days ago.", (1.0, 1.0)),
    (DAYS, ECLIPSE_Q, "Seventy six days.", (1.0, 0.0)),  # off by one: the official rule forgives it
    (DAYS, ECLIPSE_Q, "2023-04-16", (0.0, 0.0)),
    (DAYS, ECLIPSE_Q, "77", (1.0, 1.0)),
    (DAYS, ECLIPSE_Q, "It has been 77 days. That was in 2023.", (1.0, 1.0)),
    (DAYS, ECLIPSE_Q, "77 days? Not sure.", (1.0, 0.0)),
    (BOOKS, BOOKS_Q, "3 - one more than last time you asked.", (1.0, 1.0)),
    (BOOKS, BOOKS_Q, "The three of them: A, B and C. So 3.", (1.0, 1.0)),
    (BOOKS, BOOKS_Q, "You finished 3 books in 2 months.", (1.0, 1.0)),
    (BOOKS, BOOKS_Q, "2 months.", (0.0, 0.0)),
    (BOOKS, BOOKS_Q, "No one told me, but 3.", (1.0, 1.0)),
    (key("7", "multi-session", numeric=7), "How many succulents have I bought in total?",
     "So 7 in total: 3 from the nursery and 4 from the market.", (1.0, 1.0)),
    (key("7", "multi-session", numeric=7), "How many succulents have I bought in total?",
     "You bought 3 succulents, then 4 succulents, for a total of 7.", (1.0, 1.0)),
    (key("7", "multi-session", numeric=7), "How many succulents have I bought in total?", "3 + 4 = 7", (1.0, 1.0)),
    (key("2000 steps", numeric=2000), "How many steps did I walk?", "2000 steps.", (1.0, 1.0)),
])
def test_single_cases_of_numbers(answer_key, question, response, expected):
    assert verdicts(question, answer_key, response) == expected


def test_number_reading():
    assert numbers_in("forty-three, twenty one and ninety-nine") == [43.0, 21.0, 99.0]
    values = lambda text: [m.value for m in number_mentions(text)]  # noqa: E731
    assert values("109 days passed between the two.") == [109.0]
    assert values("3 - one more than last time; the one you liked, no one else, one of them") == [3.0]
    assert values("on April 16, 16 April 2023, the 16th of April, 2023-04-16 and at 8:20 pm") == []
    assert values("2 weeks and 6 days") == [20.0] and values("1 hour 30 minutes") == [90.0]
    assert values("not 3 but 4") == [4.0]
    assert [m.unit for m in number_mentions("77 days, 11 weeks, 3 books, 5")] == ["day", "week", None, None]


# ------------------------------------------------------------------ F. lists

def test_a_correct_list_behind_a_lead_in_or_before_a_closing_is_accepted_by_strict(bench):
    """Before: the strict judge rejected each of these four on all 31 list items."""
    lists = [i for i in bench if "list" in i.key.meta]
    n = len(lists)
    assert n == 31
    for phrase in (lambda i: f"From what I can see, you have {i.key.answer}.",
                   lambda i: f"As far as I can tell, {i.key.answer}.",
                   lambda i: f"You have {len(i.key.meta['list'])} of them, namely {i.key.answer}.",
                   lambda i: f"{i.key.answer}. Hope that helps, and let me know if you want more detail. Happy to help!",
                   lambda i: f"Sure! Here they are: {i.key.answer}.",
                   lambda i: f"{len(i.key.meta['list'])} of them: {i.key.answer}.",
                   lambda i: "\n".join(f"- {item}" for item in i.key.meta["list"]),
                   lambda i: i.key.answer.lower()):
        assert accepted(lists, phrase) == (n, n), phrase(lists[0])
    pool = lambda i: syn.BOOKS if "books" in i.probe.question else syn.COUNTRIES  # noqa: E731
    assert accepted(lists, lambda i: ", ".join(pool(i))) == (0, 0)
    assert accepted(lists, lambda i: ", ".join(pool(i)).lower()) == (0, 0)
    assert accepted(lists, lambda i: " ".join(pool(i)).lower()) == (0, 0)
    extra = lambda i: next(v for v in pool(i) if v not in i.key.meta["list"])  # noqa: E731
    assert accepted(lists, lambda i: f"{i.key.answer} and {extra(i)}.") == (n, 0)  # one stray item: lenient lets it pass
    assert accepted(lists, lambda i: f"{i.key.answer} and {extra(i)}.".lower()) == (n, 0)


@pytest.mark.parametrize("response, expected", [
    ("Peru, Nepal, Japan and Chile.", (1.0, 0.0)),
    ("peru, nepal, japan and chile.", (1.0, 0.0)),  # before: accepted by strict on a key without a "wrong" list
    ("chile, peru, nepal and japan", (1.0, 0.0)),
    ("Peru, Chile, Nepal and Japan.", (1.0, 0.0)),
    ("Peru, Nepal and Japan. Chile.", (1.0, 0.0)),
    ("peru chile nepal japan", (1.0, 0.0)),
    ("Peru, Nepal, Japan and others.", (1.0, 0.0)),
    ("Peru, Nepal and Japan.", (1.0, 1.0)),
    ("Peru in 2019, Nepal in 2021 and Japan last year.", (1.0, 1.0)),
    ("You have taken trips to Nepal, Japan and Peru.", (1.0, 1.0)),
    ("Peru? Nepal? Japan?", (1.0, 0.0)),
    ("Peru and Nepal.", (0.0, 0.0)),
])
def test_an_extra_item_is_extra_whatever_its_case_and_whatever_the_key(response, expected):
    assert verdicts(TRIPS_Q, TRIPS, response) == expected  # a key without a "wrong" list
    assert verdicts(TRIPS_Q, TRIPS_POOL, response) == expected  # and one with it


# --------------------------------------- G. hedges and several candidates

HEDGED = """I think it's Lisbon.
Probably Lisbon.
Lisbon, I believe.
Lisbon, I guess.
I'm guessing Lisbon.
My guess is Lisbon.
It's likely Lisbon.
Lisbon, I suppose.
I'd say Lisbon, but don't quote me.
Lisbon? I honestly can't remember.
Lisbon, if I remember correctly.
Lisbon, though I may be wrong.
Lisbon (unsure).
Presumably Lisbon.
Lisbon, as far as I know.
I want to say Lisbon.
Lisbon seems right, but I'm unsure.""".splitlines()
SEVERAL = """Lisbon or Porto.
Lisbon / Porto / Valencia
Lisbon? Porto? Valencia?
One of Lisbon, Porto and Valencia.
Lisbon, unless it was Porto.""".splitlines()


@pytest.mark.parametrize("response", HEDGED)
def test_strict_rejects_a_hedged_answer_and_lenient_keeps_it(response):
    """Before: the strict judge accepted 18 of the reviewer's 22 lines (these 17 and the 5 below)."""
    assert verdicts(CITY_Q, CITY, response) == (1.0, 0.0)
    assert is_hedged(response) or "?" in response


@pytest.mark.parametrize("response", SEVERAL)
def test_several_candidates_are_not_an_answer_even_without_a_list_of_wrong_ones(response):
    assert verdicts(CITY_Q, CITY, response) == (0.0, 0.0)


@pytest.mark.parametrize("response", ["It is Lisbon.", "Lisbon.", "You told me Lisbon.", "Lisbon? Yes, Lisbon.",
                                      "Lisbon. Anything else?", "Lisbon, for sure.", "About that: Lisbon."])
def test_a_committed_answer_is_accepted_by_both(response):
    assert len(HEDGED) + len(SEVERAL) == 22
    assert verdicts(CITY_Q, CITY, response) == (1.0, 1.0)


# --------------------------------------------------- H. other correct answers

SINGULAR = {"window seats": "a window seat", "aisle seats": "an aisle seat", "mountain cabins": "a mountain cabin",
            "beach resorts": "a beach resort", "documentaries": "a documentary", "action films": "an action film",
            "morning runs": "a morning run", "evening swims": "an evening swim", "paperbacks": "a paperback"}


def test_ordinary_answers_around_a_single_fact_are_accepted(bench):
    """Before, of 146 facts: the closing remark with "could be" 146 rejected, the restated question 146, the side remark 146 (strict)."""
    facts = [i for i in bench if not i.key.is_abstention and not any(k in i.key.meta for k in ("numeric", "list", "distractors"))]
    n = len(facts)
    assert n == 146
    answer = lambda i: i.key.answer  # noqa: E731
    for phrase in ("It is {a}.", "You told me {a}.", "{a}. You didn't mention anything else about it.",
                   "{a}. That could be useful for planning, reminders, or recommendations.", "{q} {a}.", "{q}\n{a}",
                   "You asked: {q} The answer is {a}.", "{a}. I don't have any more details than that.",
                   "{a}. Is there anything else you'd like to know?", "{a} (you told me this a while ago).",
                   "{a}. You never said why, though."):
        assert accepted(facts, lambda i: phrase.format(a=answer(i), q=i.probe.question)) == (n, n), phrase
    assert accepted(facts, lambda i: f"{answer(i)}, I believe.") == (n, 0)  # hedged: rejected by strict by design
    assert accepted(facts, lambda i: f"{answer(i)}. I don't know.") == (n, 0)  # declines the question itself
    assert accepted(facts, lambda i: i.probe.question) == (0, 0)  # the question alone is still an echo
    assert accepted(facts, lambda i: f"{i.probe.question} Good question, let me restate that.") == (0, 0)


def test_the_singular_answers_for_the_plural(bench):
    """Before: "On Tuesday." for Tuesdays and "a beach resort" for beach resorts were rejected 18 of 18."""
    gym = [i for i in bench if "gym" in i.probe.question and not i.key.is_abstention]
    preferences = [i for i in bench if i.key.question_type == "single-session-preference" and i.key.answer in SINGULAR]
    assert (len(gym), len(preferences)) == (11, 7)
    assert accepted(gym, lambda i: f"On {i.key.answer[:-1]}.") == (11, 11)
    assert accepted(preferences, lambda i: f"You prefer {SINGULAR[i.key.answer]}.") == (7, 7)
    # the wrong day in the singular is as wrong as in the plural
    wrong_day = lambda i: next(d for d in syn.WEEKDAYS if d != i.key.answer and d not in i.key.meta.get("stale", ()))  # noqa: E731
    assert accepted(gym, lambda i: f"On {wrong_day(i)[:-1]}.") == (0, 0)
    assert accepted(gym, lambda i: f"On {i.key.answer[:-1]} or {wrong_day(i)[:-1]}.") == (0, 0)


def test_what_is_only_noted_is_not_asserted():
    game = key("Patchwork", "single-session-assistant", wrong=["Jaipur", "Hive", "Lost Cities"])
    question = "Which board game did you recommend to me?"
    assert verdicts(question, game, "The answer is Patchwork. You also mentioned Jaipur at another point.") == (1.0, 1.0)
    assert verdicts(question, game, "The answer is Jaipur. You also mentioned Patchwork at another point.") == (0.0, 0.0)
    assert verdicts(question, game, "You mentioned Patchwork, but you never told me which game I recommended.") == (0.0, 0.0)


@pytest.mark.xfail(strict=True, reason="'no more details' after an answer is read as a decline by the strict judge")
def test_known_failure_no_more_details_after_an_answer():
    assert verdicts(CITY_Q, CITY, "Lisbon. I have no more details.") == (1.0, 1.0)


def test_a_decline_of_something_else_does_not_cost_the_answer():
    for response in ("Lisbon. You didn't mention anything else about it.", "Lisbon, though you never said which neighbourhood.",
                     "Lisbon. You never told me why."):
        assert verdicts(CITY_Q, CITY, response) == (1.0, 1.0), response
    for response in ("Lisbon. I don't know.", "Lisbon, but I don't have that information.", "Lisbon. You never told me which city."):
        assert verdicts(CITY_Q, CITY, response) == (1.0, 0.0), response


# ----------------------------------------------------------- I. normalisation

@pytest.mark.parametrize("gold, response, expected, meta", [
    ("Plan A", "Plan B", 0.0, {}), ("Plan A", "Plan A", 1.0, {}), ("Plan A", "plan a.", 1.0, {}), ("Plan B", "Plan A", 0.0, {}),
    ("Vitamin A", "Vitamin D", 0.0, {}), ("Vitamin A", "You take vitamin A.", 1.0, {}),
    ("C++", "C#", 0.0, {}), ("C++", "C", 0.0, {}), ("C++", "It is C++.", 1.0, {}), ("C#", "C++", 0.0, {}), ("C", "C++", 0.0, {}),
    ("C", "C", 1.0, {}),
    ("A", "A", 1.0, {}), ("A", "The answer is A.", 1.0, {}), ("A", "It is a city.", 0.0, {}), ("A", "B", 0.0, {}),
    ("東京", "東京", 1.0, {}), ("東京", "あなたは東京に住んでいます。", 1.0, {}), ("東京", "大阪", 0.0, {}),
    ("May", "It may have been in April.", 0.0, {}), ("May", "In May.", 1.0, {}), ("May", "may", 1.0, {}), ("May", "in may", 1.0, {}),
    ("Split", "You split your time between Porto and Ghent.", 0.0, {}), ("Split", "You moved to Split.", 1.0, {}),
    ("April", "In April, as you may recall.", 1.0, {"wrong": ["May", "June"]}),  # "may" is not the month here either
    ("18 days", "18 days", 1.0, {"numeric": True}), ("18 days", "1", 0.0, {"numeric": True}),
    ("18 days", "18 days", 1.0, {"numeric": float("nan")}), ("18 days", "1", 0.0, {"numeric": "18"}),
])
def test_normalisation_keeps_what_tells_two_answers_apart(gold, response, expected, meta):
    """Before: "Plan A" accepted "Plan B", "C++" accepted "C#", "A" and "東京" could not match themselves, "May"
    accepted "It may have been in April.", and ``numeric: True`` was read as the number 1."""
    assert verdicts("q?", key(gold, **meta), response) == (expected, expected)


def test_a_numeric_hint_must_be_a_real_number():
    days = key("18 days", "temporal-reasoning", numeric=True)
    for mode in JUDGE_MODES:  # the lax judges ignore the flag the same way
        assert HeuristicJudge(mode).judge("q?", days, "1").score == 0.0
        assert HeuristicJudge(mode).judge("q?", days, "18 days").score == 1.0


# -------------------------------------------------------------------- J. cost

def crafted(n: int) -> list[str]:
    """Responses built to make a careless pattern slow: all of them over the character limit."""
    return ["either'" * n, "either " * n, "could be " * n, "I don't know, " * (n // 2), "not " * (2 * n), "a, " * (3 * n),
            "1 2 3 " * n, "? " * (3 * n), "but " * (2 * n), "Lisbon or " * n, "\n" * (7 * n), "'" * (7 * n)]


def test_an_over_long_response_is_refused_before_anything_reads_it():
    assert MAX_RESPONSE_CHARS == 4000
    long_answer = "You live in Lisbon. " + "x" * MAX_RESPONSE_CHARS
    assert verdicts(CITY_Q, CITY, long_answer) == (0.0, 0.0)
    assert verdicts(CITY_Q, CITY, "You live in Lisbon." + " " * (MAX_RESPONSE_CHARS - 19)) == (1.0, 1.0)  # at the limit
    assert verdicts(SISTER_Q, SISTER, "I don't know. " + " " * MAX_RESPONSE_CHARS) == (0.0, 0.0)
    assert not is_abstention("I don't know. " + "x" * MAX_RESPONSE_CHARS) and not is_hedged("maybe " * 1000)
    assert is_noncommittal("x" * (MAX_RESPONSE_CHARS + 1)) and number_mentions("7 " * MAX_RESPONSE_CHARS) == ()
    assert read("x" * (MAX_RESPONSE_CHARS + 1)).clauses == []


def test_grading_time_is_linear_in_the_length_of_a_crafted_answer():
    """Before: "either'" repeated to 56,000 characters took about 6 seconds, four times more per doubling."""
    instance = Instance(Probe("p", CITY_Q, "2023-06-01", "h", ()), CITY)
    grader = Grader(CriteriaFamily.default())

    def seconds(text: str) -> float:
        best = 9.0
        for k in range(3):  # a new string each time, so no cache answers
            start = time.perf_counter()
            grader.grade({"p": Answer(text + " " * k)}, [instance])
            best = min(best, time.perf_counter() - start)
        return best

    assert seconds("either'" * 8000) < 0.05
    for text in crafted(8000):
        assert len(text) > 10 * MAX_RESPONSE_CHARS and seconds(text) < 0.05, text[:20]
    # under the limit the patterns do run: still nothing that grows faster than the text
    for text in crafted(560):
        assert seconds(text[:MAX_RESPONSE_CHARS]) < 0.05, text[:20]


def test_typical_grading_speed(bench):
    """Measured for the report: 0.14 ms per item on unseen answers before the change, 0.14 after (0.08 / 0.07 on seen ones)."""
    instances = list(bench.instances)
    grader = Grader(CriteriaFamily.default())
    answers = run_system(HeuristicMemory(MemoryConfig(features=frozenset(FEATURES), top_k=10, style="sentence")), bench.probes())
    best = 9.0
    for k in range(5):
        unseen = {pid: Answer(a.text + " " * (k + 1)) for pid, a in answers.items()}
        start = time.perf_counter()
        grader.grade(unseen, instances)
        best = min(best, time.perf_counter() - start)
    assert 1000 * best / len(instances) < 0.5


# ----------------------------------------------- K. rows that are not answers

def raw_row(text, status):
    """A row that never went through ``Answer``'s checks, as a careless harness might hand one over."""
    row = object.__new__(Answer)
    for name, value in (("text", text), ("confidence", 0.5), ("status", status), ("meta", {})):
        object.__setattr__(row, name, value)
    return row


@pytest.mark.parametrize("make", [raw_row, lambda text, status: SimpleNamespace(text=text, status=status),
                                  lambda text, status: Answer.from_dict({"text": text, "status": status})],
                         ids=["unchecked Answer", "any object with text and status", "Answer.from_dict"])
def test_only_ok_and_truncated_rows_are_graded_on_their_text(bench, make):
    """Before: a row with status "failed" or "Error" was graded on its text; a text that was not a string raised."""
    instances = list(bench.instances)
    grader = Grader(CriteriaFamily.default())
    message = "I don't have that information."  # reads as a decline: must not earn the unanswerable questions
    share = sum(i.key.is_abstention for i in instances) / len(instances)
    for status in ("ok", "truncated"):
        report = grader.grade({i.probe_id: make(message, status) for i in instances}, instances)
        assert report.n_not_answered == 0 and report.score(LENIENT_ID) == pytest.approx(share)
    for status in ("error", "timeout", "refused", "missing", "Error", "failed", "refusal", "max_tokens", "", None, 3, ["ok"]):
        report = grader.grade({i.probe_id: make(message, status) for i in instances}, instances)
        assert report.n_not_answered == len(instances), status
        assert max(report.scores.values()) == 0.0, status
        assert len(report.answered_only().items) == 0 and not any(it.abstained for it in report.items)
    for text in (3, 3.0, ["Lisbon"], None, b"Lisbon", {"text": "Lisbon"}):
        report = grader.grade({instances[0].probe_id: make(text, "ok")}, instances[:1])
        assert report.n_not_answered == 1 and report.items[0].status == "error" and max(report.scores.values()) == 0.0
        assert isinstance(report.items_root(), str)


# ------------------------------------------------- R. what had to keep holding

# Digest of the 77 cases the exam held before the second review (dev, fresh, review), ids included.
EXAM_BEFORE_SECOND_REVIEW = "2401e4333e34574d"


def test_the_exam_gained_a_second_review_part_and_lost_nothing():
    earlier = [c for c in JUDGE_EXAM if c["split"] != "review2"]
    assert [c["split"] for c in earlier] == ["dev"] * 52 + ["fresh"] * 16 + ["review"] * 9
    assert digest([
        [c["split"], c["tag"], c["question"], c["response"], c["official"], c["strict"], c["key"].probe_id, c["key"].question_type,
         c["key"].answer, list(c["key"].aliases), c["key"].is_abstention, dict(sorted(c["key"].meta.items()))]
        for c in earlier
    ]) == EXAM_BEFORE_SECOND_REVIEW
    second = [c for c in JUDGE_EXAM if c["split"] == "review2"]
    assert len(second) == 90 and len({c["tag"] for c in second}) == 90
    assert {c["tag"][0] for c in second} == set("abcdefghi")  # every rule family has cases
    assert len({(c["question"], c["response"]) for c in JUDGE_EXAM}) == len(JUDGE_EXAM)
    assert exam_key("eclipse").meta["numeric"] == 77


@pytest.mark.parametrize("mode, target", [("lenient", "official"), ("strict", "strict")])
def test_exam_agreement_by_part(mode, target):
    """Before: dev 52/52, fresh 15/16 (the miss was "update-new-then-old"), review 9/9; on the new part
    the judges as they stood agreed on 43 (lenient) and 39 (strict) of the 90 cases."""
    judge = HeuristicJudge(mode)
    for split, n in (("dev", 52), ("fresh", 16), ("review", 9), ("review2", 90)):
        result = judge_exam(judge, target, split=split)
        assert result["n"] == n and result["errors"] == 0
        assert result["accuracy"] == 1.0, (split, result["misses"])


@pytest.fixture(scope="module")
def honest_world():
    world = syn.generate("x", syn.GenConfig(n_streams=20))
    return world, Grader(CriteriaFamily.default())


@pytest.mark.parametrize("style, lenient_before, strict_before", [("terse", 185, 179), ("sentence", 185, 179), ("verbose", 151, 145)])
def test_the_honest_reference_system_did_not_lose(honest_world, style, lenient_before, strict_before):
    """Items of 240 the full-feature offline system got right before the change, per answer style."""
    world, grader = honest_world
    system = HeuristicMemory(MemoryConfig(features=frozenset(FEATURES), top_k=10, style=style))
    report = grader.grade(run_system(system, world.probes()), world.instances)
    lenient = sum(it.base["judge_lenient"] for it in report.items)
    strict = sum(it.base["judge_strict"] for it in report.items)
    assert lenient >= lenient_before and strict >= strict_before
    if style != "verbose":
        assert (lenient, strict) == (185, 183)  # measured: unchanged under lenient, four more under strict
    else:
        # "You also mentioned <another value> at another point." no longer sinks a right answer or rescues a wrong one
        assert (lenient, strict) == (185, 183)


def test_reference_answers_score_one(honest_world):
    world, grader = honest_world
    report = grader.grade(oracle_answers(world.instances), world.instances)
    assert all(score == pytest.approx(1.0) for cid, score in report.scores.items() if "|literal|" not in cid)


def test_longmemeval_references_are_accepted_at_least_as_often_as_before():
    """Before: lenient 500/500, strict 496/500 (0.992).  Now 500 and 498: a reference's own words are not held against it."""
    path = os.path.join(ROOT, "data", "longmemeval", "longmemeval_oracle.json")
    if not os.path.exists(path):
        pytest.skip("LongMemEval is not on this machine")
    real = load_longmemeval(path, salt="t")
    answers = oracle_answers(real.instances)
    accepted_by = {judge.mode: sum(judge.judge(i.probe.question, i.key, answers[i.probe_id].text).score for i in real)
                   for judge in JUDGES}
    assert len(real) == 500 and accepted_by["lenient"] == 500 and accepted_by["strict"] >= 496
    # its own explanations of why a question cannot be answered: 22 of 30 were read as declines before
    explained = [i for i in real if i.key.is_abstention]
    assert sum(JUDGES[0].judge(i.probe.question, i.key, i.key.answer).score for i in explained) >= 29


class PaysForDeclines:
    """A judge with the fault the review found: a decline phrase anywhere earns the unanswerable questions."""

    name = "pays-for-declines"

    def __init__(self, mode):
        self.inner = HeuristicJudge(mode)

    def judge(self, question, answer_key, response):
        if answer_key.is_abstention:
            return JudgeVerdict(1.0 if is_abstention(response) else 0.0)
        return self.inner.judge(question, answer_key, response)

    def spec(self):
        return {"kind": "pays-for-declines"}


def null_checks(report) -> dict:
    return {c["check"].split("'")[1]: c for c in report.checks if c["check"].startswith("null baseline")}


def test_an_invented_answer_next_to_a_decline_stays_at_the_floor():
    audit_bench = syn.generate("audit", syn.GenConfig(n_streams=30))
    instances = list(audit_bench.instances)
    report = run_audit(audit_bench, HeuristicMemory(MemoryConfig(features=frozenset(FEATURES), top_k=10)))
    assert report.passed, report.failures()
    checks = null_checks(report)
    assert {"answer_then_decline", "decline_then_answer"} <= set(checks)
    # the bound itself, not the audit's reading of it
    grader = Grader(CriteriaFamily.default())
    invented = [i.key.answer for i in instances if not i.key.is_abstention][0]
    alone = grader.grade({i.probe_id: Answer(invented) for i in instances}, instances)
    for template in ("{a}. I don't know.", ABSTAIN_TEXT + " {a}."):
        graded = grader.grade({i.probe_id: Answer(template.format(a=invented)) for i in instances}, instances)
        unanswerable = [it for it in graded.items if it.is_abstention]
        assert unanswerable and sum(it.base["judge_lenient"] + it.base["judge_strict"] for it in unanswerable) == 0
        assert graded.score(STRICT_ID) == 0.0  # what a plain decline earns on the answerable questions: nothing
        assert graded.score(LENIENT_ID) == alone.score(LENIENT_ID)  # the decline adds nothing to the answer
    # and the two new nulls can fail: a judge that pays for the decline phrase does not get past them
    seats = dict(default_judges(), judge_lenient=PaysForDeclines("lenient"), judge_strict=PaysForDeclines("strict"))
    failed = {name for name, check in null_checks(run_audit(audit_bench, judges=seats)).items() if not check["passed"]}
    assert {"answer_then_decline", "decline_then_answer"} <= failed


def test_the_generator_was_not_touched():
    """Rule A is exact on generated data with the keys as they were: an unanswerable question's key already
    lists the values the question could have had.  Nothing was added, so every pinned value still holds."""
    bench = syn.generate("keys", syn.GenConfig(n_streams=40))
    for inst in bench:
        if inst.key.is_abstention:
            assert "wrong" in inst.key.meta
            assert inst.key.meta["wrong"] or inst.key.meta.get("numeric_question")


RESPONSES = ["Lisbon.", "I think it's Lisbon.", "Lisbon or Porto.", "It is Lisbon, not Porto.", "I don't know.", "Priya. I don't know.",
             "77 days (roughly 11 weeks).", "2, 3, 4", "Peru, Nepal, Japan and chile.", "Before you got a tattoo, you renewed your passport.",
             "Mazda CX-5 (previously a Honda Civic).", "", "   ", "?", "東京", "Plan A", "either'" * 30, "​", "a" * 5000]
VERDICT_SCRIPT = """
import sys, json, hashlib
sys.path.insert(0, {root!r})
from memarena.benchmark import synthetic as syn
from memarena.evalcore.judges import HeuristicJudge
from memarena.evalcore.judge_exam import JUDGE_EXAM
bench = syn.generate("hash", syn.GenConfig(n_streams=3))
out = []
for mode in ("lenient", "strict", "generous", "graded"):
    judge = HeuristicJudge(mode)
    for case in JUDGE_EXAM:
        out.append(judge.judge(case["question"], case["key"], case["response"]).score)
    for inst in bench:
        for response in {responses!r} + [inst.key.answer, inst.probe.question, inst.key.answer + ". I don't know."]:
            out.append(judge.judge(inst.probe.question, inst.key, response).score)
print(hashlib.sha256(json.dumps(out).encode()).hexdigest())
"""


def test_verdicts_do_not_depend_on_the_hash_seed():
    script = VERDICT_SCRIPT.format(root=ROOT, responses=RESPONSES)
    seen = set()
    for seed in ("0", "1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        seen.add(subprocess.run([sys.executable, "-B", "-c", script], env=env, capture_output=True, text=True, check=True).stdout.strip())
    assert len(seen) == 1 and len(seen.pop()) == 64


def test_judging_is_repeatable_and_never_raises_on_arbitrary_text(bench):
    rng = random.Random(20240)
    pieces = ["I don't know", "Lisbon", "not", "or", "either", "could be", "maybe", "?", "!", ".", ",", ";", ":", "(", ")", "\n", "-", " - ",
              "'", "’", "\"", "/", "+", "#", "77 days", "2023-04-16", "April 16", "8:20 pm", "one", "the two", "but", "before", "after",
              "first", "previously", "you never mentioned", "unless it was", "​", " ", "é", "東京", "\x00", "😀", "I'm",
              "can't", "1,234", "3.0", "a.m.", "Dr.", "%", "*", "__", "<b>", "{}", "\\"]
    texts = RESPONSES + ["".join(rng.choice(pieces) + rng.choice(["", " ", " ", "  "]) for _ in range(rng.randint(1, 40))) for _ in range(1500)]
    texts += ["".join(chr(rng.choice([rng.randint(32, 126), rng.randint(160, 0x2FF), rng.randint(0x2000, 0x206F)])) for _ in range(rng.randint(1, 80)))
              for _ in range(500)]
    instances = [i for i in bench.instances[:60]] + [Instance(Probe("p", c["question"], "2023-06-01", "h", ()), c["key"]) for c in JUDGE_EXAM[::7]]
    judges = [HeuristicJudge(mode) for mode in JUDGE_MODES]
    grader = Grader(CriteriaFamily.default())
    for k, text in enumerate(texts):
        inst = instances[k % len(instances)]
        for judge in judges:
            first = judge.judge(inst.probe.question, inst.key, text)
            assert first.status == "ok" and 0.0 <= first.score <= 1.0
            assert judge.mode == "graded" or first.score in (0.0, 1.0)
            assert judge.judge(inst.probe.question, inst.key, text) == first
        assert grader.grade({inst.probe_id: Answer(text)}, [inst]).items_root() == grader.grade({inst.probe_id: Answer(text)}, [inst]).items_root()
    # odd keys and odd arguments
    odd = AnswerKey("p", "single-session-user", "Lisbon", meta={"wrong": "Porto", "stale": None, "list": 3, "numeric": "x", "unit": 5})
    for judge in judges:
        for response in ("Lisbon", "", None, 3):
            assert judge.judge(None, odd, response).status == "ok"
        assert judge.judge("q", AnswerKey("p", "single-session-user", ""), "anything").score == 0.0


def test_the_judge_declares_a_new_version():
    assert HeuristicJudge("lenient").spec() == {"kind": "heuristic", "mode": "lenient", "version": 6}
    assert HeuristicJudge("strict").spec()["version"] == 6
