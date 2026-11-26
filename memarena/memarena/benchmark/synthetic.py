"""Procedural LongMemEval-style benchmark: one person talks to an assistant over
many sessions, then is asked recall questions.

Why generate data when LongMemEval exists?

* Public benchmarks get memorised and over-fitted.  A generator with a private
  seed gives an unlimited supply of *fresh, labelled, never-published* test
  sets, which is what makes single-use hold-outs and burn-after-reveal peer
  exchanges possible.
* The generator exposes the difficulty knobs (paraphrase distance, third-party
  and negated distractors, knowledge updates, history length) so an evaluator
  can be stress-tested under controlled distribution shift.

The question taxonomy mirrors LongMemEval: single-session-user,
single-session-assistant, single-session-preference, multi-session,
temporal-reasoning, knowledge-update, plus abstention variants.

Every answer key also says what a response must *not* say.  ``meta["wrong"]``
lists the values of the same kind that are not a correct answer to the question:
the rest of the pool the right value was drawn from, which covers what was said
about a friend, plans that were dropped and values that were later replaced
(those also stay in ``meta["stale"]``).  A value is listed in every form the
key would accept for the right one (a month by name and as "YYYY-MM", an event
in both tenses).  The list is empty where the answer is a number, since every
other number is wrong.  ``meta["numeric_question"]`` marks questions whose
answer is a number, including the ones that cannot be answered.  A judge that
only checks whether the right value appears somewhere pays for a recital of the
whole vocabulary; with these two fields it can tell an answer from a list of
candidates.

Nothing in this module is visible to a system under test except the ``Probe``
objects it produces.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from ..util import hash_obj, make_rng
from .schema import QUESTION_TYPES, AnswerKey, Benchmark, Instance, Probe, Session, Turn, opaque_id

# Proportions of the six types in LongMemEval (500 questions).
LONGMEMEVAL_MIX: Mapping[str, float] = {
    "single-session-user": 0.14,
    "single-session-assistant": 0.112,
    "single-session-preference": 0.06,
    "multi-session": 0.266,
    "temporal-reasoning": 0.266,
    "knowledge-update": 0.156,
}


# The shortest history the generator can label correctly, and how many times more histories than asked
# for a benchmark may use when a narrow mix exhausts what one history can be asked about.
MIN_SESSIONS = 4
MAX_STREAM_FACTOR = 8


@dataclass(frozen=True)
class GenConfig:
    n_streams: int = 8
    questions_per_stream: int = 12
    sessions_per_stream: int = 20
    exchanges_per_session: tuple[int, int] = (2, 4)
    type_mix: Mapping[str, float] = field(default_factory=lambda: dict(LONGMEMEVAL_MIX))
    abstention_rate: float = 0.06
    hard_paraphrase_rate: float = 0.3
    third_party_rate: float = 0.5
    negation_rate: float = 0.3
    update_rate: float = 0.5
    echo_rate: float = 0.3
    relative_date_rate: float = 0.4

    def to_dict(self) -> dict:
        return {
            "n_streams": self.n_streams,
            "questions_per_stream": self.questions_per_stream,
            "sessions_per_stream": self.sessions_per_stream,
            "exchanges_per_session": list(self.exchanges_per_session),
            "type_mix": dict(self.type_mix),
            "abstention_rate": self.abstention_rate,
            "hard_paraphrase_rate": self.hard_paraphrase_rate,
            "third_party_rate": self.third_party_rate,
            "negation_rate": self.negation_rate,
            "update_rate": self.update_rate,
            "echo_rate": self.echo_rate,
            "relative_date_rate": self.relative_date_rate,
        }

    def validate(self) -> None:
        """Raise ``ValueError`` for settings the generator cannot honour."""

        def whole(name: str, lo: int) -> int:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ValueError(f"{name} must be a whole number of at least {lo}")
            return v

        whole("n_streams", 1)
        whole("questions_per_stream", 1)
        # Fewer sessions than this and a history cannot hold every statement its answers rely on.
        whole("sessions_per_stream", MIN_SESSIONS)
        try:
            lo, hi = self.exchanges_per_session
        except (TypeError, ValueError):
            raise ValueError("exchanges_per_session must be a pair") from None
        if isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(lo, int) or not isinstance(hi, int) or not 1 <= lo <= hi:
            raise ValueError("exchanges_per_session must be a pair of whole numbers, low <= high, both at least 1")
        for name in ("abstention_rate", "hard_paraphrase_rate", "third_party_rate", "negation_rate",
                     "update_rate", "echo_rate", "relative_date_rate"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must lie between 0 and 1")
        if not isinstance(self.type_mix, Mapping) or not self.type_mix:
            raise ValueError("type_mix must name at least one question type")
        total = 0.0
        for t, w in self.type_mix.items():
            if t not in QUESTION_TYPES:
                raise ValueError(f"unknown question type in type_mix: {t!r}")
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0.0 <= w < float("inf"):
                raise ValueError("type_mix weights must be finite and not negative")
            total += w
        if total <= 0:
            raise ValueError("type_mix must have positive weight")

    @staticmethod
    def from_dict(d: Mapping) -> "GenConfig":
        base = GenConfig()
        kwargs = {k: d[k] for k in base.to_dict() if k in d}
        if "exchanges_per_session" in kwargs:
            kwargs["exchanges_per_session"] = tuple(kwargs["exchanges_per_session"])
        return replace(base, **kwargs)


# ----------------------------------------------------------------------- vocabulary

CITIES = [
    "Lisbon", "Porto", "Valencia", "Kraków", "Tallinn", "Ghent", "Bergen", "Graz",
    "Split", "Leipzig", "Lyon", "Bologna", "Tampere", "Aarhus", "Gdańsk", "Utrecht",
    "Salzburg", "Bilbao", "Cork", "Ljubljana", "Denver", "Austin", "Portland",
    "Tucson", "Madison", "Halifax", "Calgary", "Osaka", "Busan", "Hobart",
    "Wellington", "Adelaide", "Cusco", "Medellín", "Montevideo", "Nairobi",
    "Kigali", "Hanoi", "Chiang Mai", "Tbilisi",
]
JOBS = [
    "pastry chef", "structural engineer", "veterinary nurse", "tax accountant",
    "landscape architect", "radiology technician", "court interpreter",
    "wind turbine technician", "museum curator", "pediatric dentist",
    "freight dispatcher", "sound designer", "patent attorney", "hydrologist",
    "bike mechanic", "speech therapist", "flight dispatcher", "cheese maker",
    "set designer", "actuary", "marine surveyor", "locksmith", "glassblower",
    "soil scientist", "subtitle translator", "elevator inspector",
    "floral designer", "data librarian", "tugboat captain", "piano tuner",
]
PET_TYPES = ["beagle", "tabby cat", "parrot", "rabbit", "greyhound", "ferret", "corgi", "iguana"]
PET_NAMES = [
    "Biscuit", "Mochi", "Pepper", "Waffles", "Juniper", "Clover", "Nimbus", "Pickle",
    "Marzipan", "Tofu", "Sprocket", "Olive", "Maple", "Zephyr", "Noodle", "Figaro",
    "Truffle", "Pesto", "Dumpling", "Quill", "Basil", "Cricket", "Pippin", "Sable",
]
CARS = [
    "Honda Civic", "Toyota Corolla", "Mazda CX-5", "Subaru Outback", "Kia Niro",
    "Volvo V60", "Ford Maverick", "Hyundai Ioniq", "Skoda Octavia", "Fiat Panda",
    "Nissan Leaf", "Renault Clio", "Peugeot 208", "Suzuki Jimny", "Mini Cooper",
    "Dacia Duster", "Seat Leon", "Opel Corsa", "Chevy Bolt", "Jeep Renegade",
]
PHONES = [
    "Pixel 8", "Galaxy S23", "iPhone 14", "Fairphone 5", "Xperia 10", "OnePlus 11",
    "Moto G84", "Nothing Phone 2", "Pixel 7a", "Galaxy A54", "iPhone SE", "Zenfone 10",
]
FOODS = [
    "pad thai", "mushroom risotto", "lamb tagine", "shakshuka", "pierogi", "ramen",
    "paella", "bibimbap", "moussaka", "gnocchi", "falafel", "ceviche", "pho",
    "goulash", "dosa", "empanadas", "khachapuri", "laksa", "tamales", "okonomiyaki",
]
INSTRUMENTS = [
    "cello", "banjo", "clarinet", "ukulele", "trombone", "accordion", "harp",
    "mandolin", "oboe", "bassoon", "sitar", "marimba", "viola", "flute",
]
LANGUAGES = [
    "Portuguese", "Korean", "Swahili", "Finnish", "Turkish", "Icelandic", "Hindi",
    "Polish", "Vietnamese", "Hungarian", "Greek", "Tagalog", "Czech", "Hebrew",
]
COMPANIES = [
    "Northwind Labs", "Bluefin Logistics", "Harbor & Pine", "Quillstone Media",
    "Redwood Analytics", "Tessellate Health", "Ironbark Robotics", "Lumen Forge",
    "Saltmarsh Energy", "Copperline Studios", "Granite Peak Bank", "Opal Transit",
    "Wrenfield Foods", "Halcyon Aerospace", "Mossgate Software", "Kestrel Freight",
    "Thistledown Books", "Emberly Games", "Driftwood Travel", "Larkspur Biotech",
]
NAMES = [
    "Priya", "Dana", "Tomasz", "Ingrid", "Mateo", "Yuki", "Farah", "Callum",
    "Noor", "Elif", "Rafael", "Sanna", "Kwame", "Liesel", "Arjun", "Maren",
    "Joaquin", "Thandi", "Viktor", "Aoife", "Bashir", "Corinne", "Dmitri", "Esme",
    "Gustav", "Hana", "Idris", "Jolene", "Kaito", "Leona", "Milos", "Nadia",
]
WEEKDAYS = ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays"]
COFFEES = [
    "flat white", "cortado", "oat milk latte", "double espresso", "cold brew",
    "macchiato", "americano", "mocha", "cappuccino", "pour-over", "iced latte",
]
ALLERGENS = [
    "shellfish", "peanuts", "penicillin", "pollen", "latex", "sesame", "eggs",
    "walnuts", "kiwi", "dust mites", "soy", "mustard",
]
MAJORS = [
    "marine biology", "art history", "civil engineering", "linguistics",
    "astrophysics", "nursing", "economics", "philosophy", "geology",
    "graphic design", "sociology", "chemistry", "anthropology", "statistics",
    "architecture", "music theory",
]
BOOKS = [
    "The Glass Orchard", "Winter Circuit", "A Map of Small Rivers", "The Lantern Thief",
    "Salt and Ember", "The Ninth Lighthouse", "Paper Constellations", "Hollow Tide",
    "The Clockmaker's Niece", "Ashes of Juniper", "Low Orbit", "The Quiet Cartographer",
    "Borrowed Thunder", "The Marrow Road", "Seventeen Winters", "The Brass Sparrow",
    "Field Notes on Fog", "The Last Orchardist", "Copper Sky", "The Velvet Algorithm",
    "Tidewater Hymn", "The Wren Protocol", "Slow Lightning", "The Saffron Ledger",
]
COUNTRIES = [
    "Portugal", "Japan", "Peru", "Iceland", "Morocco", "Vietnam", "Georgia",
    "Slovenia", "Kenya", "Uruguay", "Estonia", "Jordan", "Nepal", "Croatia",
    "Chile", "Malta", "Finland", "Rwanda", "Laos", "Albania", "Bolivia", "Oman",
]
PLANTS = ["succulents", "ferns", "tomato plants", "orchids", "basil pots", "cacti", "lavender plants"]

# Single-valued personal facts.  ``easy`` statements share words with the
# questions; ``hard`` ones are paraphrases with little lexical overlap.
SLOTS: dict[str, dict] = {
    "home_city": dict(
        label="which city they live in",
        values=CITIES,
        updatable=True,
        easy=["I live in {v}.", "I'm living in {v} these days.", "The city I live in is {v}."],
        hard=["I'm based in {v}.", "My apartment is in {v}.", "I've settled down in {v}."],
        update=["I just moved to {v}.", "I relocated to {v} last week.", "Big news: I now live in {v}."],
        ask=["Which city do I live in?", "Where am I living now?", "What city do I live in?"],
        third=["My friend {p} lives in {v}.", "{p}, my cousin, just moved to {v}."],
        neg=[
            "I thought about moving to {v}, but decided against it.",
            "I almost relocated to {v}, but it fell through.",
        ],
    ),
    "occupation": dict(
        label="what they do for work",
        values=JOBS,
        updatable=True,
        easy=["I work as a {v}.", "My job is being a {v}.", "For work, I'm a {v}."],
        hard=["I earn my living as a {v}.", "Professionally, I'm a {v}."],
        update=["I started a new job as a {v}.", "I switched careers and I'm now a {v}."],
        ask=["What do I do for work?", "What is my job?", "What is my current occupation?"],
        third=["My friend {p} works as a {v}.", "My neighbor {p} has a job as a {v}."],
        neg=["I considered becoming a {v}, but changed my mind."],
    ),
    "pet_name": dict(
        label="their pet's name",
        values=PET_NAMES,
        updatable=False,
        easy=["My pet's name is {v}.", "I have a pet named {v}."],
        hard=["We adopted a little one and called it {v}.", "{v} is the newest member of our household."],
        update=[],
        ask=["What is my pet's name?", "What is the name of my pet?"],
        third=["My friend {p} has a pet named {v}."],
        neg=["I almost named my pet {v}, but didn't."],
    ),
    "car": dict(
        label="what car they drive",
        values=CARS,
        updatable=True,
        easy=["I drive a {v}.", "My car is a {v}."],
        hard=["My daily ride is a {v}.", "I get around in a {v}."],
        update=["I traded in my old car for a {v}.", "I sold my car and bought a {v}."],
        ask=["What car do I drive?", "Which car do I own now?"],
        third=["My coworker {p} drives a {v}."],
        neg=["I test-drove a {v}, but decided against it."],
    ),
    "phone": dict(
        label="what phone they have",
        values=PHONES,
        updatable=True,
        easy=["My phone is a {v}.", "I have a {v} as my phone."],
        hard=["I carry a {v} in my pocket.", "My daily driver is the {v}."],
        update=["I upgraded my phone to a {v}.", "I replaced my phone with a {v}."],
        ask=["What phone do I have?", "Which phone am I using now?"],
        third=["My friend {p} has a {v} as a phone."],
        neg=["I considered getting a {v} phone, but changed my mind."],
    ),
    "favorite_food": dict(
        label="their favorite food",
        values=FOODS,
        updatable=True,
        easy=["My favorite food is {v}.", "My favorite dish is {v}."],
        hard=["Nothing beats {v} for me.", "I could eat {v} every single day."],
        update=["My new favorite food is {v}."],
        ask=["What is my favorite food?", "Which food is my favorite?"],
        third=["My friend {p} says their favorite food is {v}."],
        neg=["I used to think my favorite food would be {v}, but it never was."],
    ),
    "instrument": dict(
        label="which instrument they play",
        values=INSTRUMENTS,
        updatable=False,
        easy=["I play the {v}.", "The instrument I play is the {v}."],
        hard=["I've been practicing the {v} every evening.", "My evenings are spent with my {v}."],
        update=[],
        ask=["What instrument do I play?", "Which instrument do I play?"],
        third=["My cousin {p} plays the {v}."],
        neg=["I thought about learning to play the {v}, but never started."],
    ),
    "language": dict(
        label="which language they are learning",
        values=LANGUAGES,
        updatable=True,
        easy=["I'm learning {v}.", "The language I'm learning is {v}."],
        hard=["I've been taking {v} lessons.", "My evenings go to {v} flashcards."],
        update=["I switched to learning {v} instead."],
        ask=["Which language am I learning?", "What language am I learning now?"],
        third=["My friend {p} is learning {v}."],
        neg=["I considered learning {v}, but changed my mind."],
    ),
    "employer": dict(
        label="which company they work for",
        values=COMPANIES,
        updatable=True,
        easy=["I work at {v}.", "The company I work for is {v}."],
        hard=["My paychecks come from {v}.", "I'm on the team at {v}."],
        update=["I left my old company and now work at {v}.", "I changed companies and now work for {v}."],
        ask=["Which company do I work for?", "What company do I work at?"],
        third=["My friend {p} works at {v}."],
        neg=["I almost took an offer to work at {v}, but decided against it."],
    ),
    "sister_name": dict(
        label="their sister's name",
        values=NAMES,
        updatable=False,
        easy=["My sister's name is {v}.", "My sister is named {v}."],
        hard=["{v} and I grew up sharing a bedroom, as sisters do."],
        update=[],
        ask=["What is my sister's name?", "What is the name of my sister?"],
        third=["My friend {p} has a sister named {v}."],
        neg=[],
    ),
    "gym_day": dict(
        label="which day they go to the gym",
        values=WEEKDAYS,
        updatable=True,
        easy=["I go to the gym on {v}.", "My gym day is {v}."],
        hard=["You'll find me lifting weights on {v}."],
        update=["I moved my gym day to {v}."],
        ask=["Which day do I go to the gym?", "What is my gym day?"],
        third=["My coworker {p} goes to the gym on {v}."],
        neg=["I thought about going to the gym on {v}, but decided against it."],
    ),
    "coffee_order": dict(
        label="their usual coffee order",
        values=COFFEES,
        updatable=True,
        easy=["My usual coffee order is a {v}.", "My coffee order is always a {v}."],
        hard=["At the café I always get a {v}."],
        update=["I've switched my usual coffee order to a {v}."],
        ask=["What is my usual coffee order?", "What coffee do I usually order?"],
        third=["My friend {p} has a usual coffee order of a {v}."],
        neg=["I almost made a {v} my usual coffee order, but didn't."],
    ),
    "allergy": dict(
        label="what they are allergic to",
        values=ALLERGENS,
        updatable=False,
        easy=["I'm allergic to {v}.", "I have an allergy to {v}."],
        hard=["I break out in hives around {v}.", "My doctor told me to stay away from {v}."],
        update=[],
        ask=["What am I allergic to?", "What allergy do I have?"],
        third=["My friend {p} is allergic to {v}."],
        neg=["I thought I was allergic to {v}, but the test said I wasn't."],
    ),
    "major": dict(
        label="what they studied in college",
        values=MAJORS,
        updatable=False,
        easy=["I studied {v} in college.", "In college I studied {v}."],
        hard=["My degree is in {v}.", "I majored in {v}."],
        update=[],
        ask=["What did I study in college?", "What did I study at college?"],
        third=["My friend {p} studied {v} in college."],
        neg=["I considered studying {v} in college, but changed my mind."],
    ),
}

# (past tense phrase, base-form phrase) for dated one-off events.
EVENTS = [
    ("ran my first half marathon", "run my first half marathon"),
    ("got my scuba certification", "get my scuba certification"),
    ("started a pottery class", "start a pottery class"),
    ("visited the botanical garden", "visit the botanical garden"),
    ("bought a kayak", "buy a kayak"),
    ("finished painting the garage", "finish painting the garage"),
    ("attended a jazz festival", "attend a jazz festival"),
    ("gave a talk at a conference", "give a talk at a conference"),
    ("planted a vegetable garden", "plant a vegetable garden"),
    ("renewed my passport", "renew my passport"),
    ("hosted a dinner party", "host a dinner party"),
    ("donated blood", "donate blood"),
    ("went skydiving", "go skydiving"),
    ("signed up for a triathlon", "sign up for a triathlon"),
    ("repainted my bedroom", "repaint my bedroom"),
    ("took a sourdough baking workshop", "take a sourdough baking workshop"),
    ("saw a solar eclipse", "see a solar eclipse"),
    ("got a tattoo", "get a tattoo"),
    ("adopted a beehive", "adopt a beehive"),
    ("built a bookshelf", "build a bookshelf"),
    ("volunteered at the food bank", "volunteer at the food bank"),
    ("fixed my bicycle", "fix my bicycle"),
    ("joined a choir", "join a choir"),
    ("installed solar panels", "install solar panels"),
]
# Both forms of every event, as a response might name it.
EVENT_PHRASES = [phrase for event in EVENTS for phrase in event]
# (template, days before the session date that the event actually happened)
EVENT_TEMPLATES = [
    ("Today I {e}.", 0), ("I {e} this morning.", 0), ("Guess what, I {e} today!", 0),
]
EVENT_TEMPLATES_RELATIVE = [("Yesterday I {e}.", 1), ("Two days ago I {e}.", 2)]
EVENT_THIRD = ["My friend {p} {e_third} last weekend."]

COLLECTIONS: dict[str, dict] = {
    "books": dict(
        values=BOOKS,
        easy=["I just finished reading {v}.", "I finished reading {v} last night."],
        hard=["Just turned the last page of {v}.", "{v} is done, what an ending."],
        count_q="How many books have I finished reading?",
        list_q="Which books have I finished reading?",
        third="My friend {p} finished reading {v}.",
        label="finishing any books",
    ),
    "trips": dict(
        values=COUNTRIES,
        easy=["I took a trip to {v}.", "I went on a trip to {v}."],
        hard=["Just got back from {v}.", "I spent last week wandering around {v}."],
        count_q="How many countries have I taken a trip to?",
        list_q="Which countries have I taken a trip to?",
        third="My friend {p} took a trip to {v}.",
        label="taking any trips abroad",
    ),
}

PREFERENCES = [
    ("seats on flights", "window seats", "aisle seats"),
    ("coffee beans", "dark roast", "light roast"),
    ("vacations", "mountain cabins", "beach resorts"),
    ("movie nights", "documentaries", "action films"),
    ("workouts", "morning runs", "evening swims"),
    ("background music", "instrumental jazz", "pop radio"),
    ("reading formats", "paperbacks", "e-books"),
    ("pizza", "thin crust", "deep dish"),
    ("tea", "oolong", "chamomile"),
    ("commuting", "cycling", "taking the bus"),
]
# Every option a preference can be about; one of them is right for a given question.
PREFERENCE_PHRASES = [phrase for (_c, x, y) in PREFERENCES for phrase in (x, y)]
PREF_EASY = ["When it comes to {c}, I prefer {x} over {y}.", "For {c}, I prefer {x} to {y}."]
PREF_HARD = ["For {c}, I'm more of a {x} person than a {y} person.", "Give me {x} over {y} any day, as far as {c} go."]
PREF_ASK = ["What do I prefer when it comes to {c}?", "What is my preference for {c}?"]

ASSISTANT_TOPICS = [
    ("sci-fi novel", ["Low Tide on Europa", "The Ansible Garden", "Vector of Saints", "Cold Harbor Station"]),
    ("hiking trail", ["Juniper Ridge Loop", "Falcon Crest Trail", "the Saddleback Traverse", "Mirror Lake Path"]),
    ("pasta recipe", ["cacio e pepe", "orecchiette with broccoli rabe", "pasta alla Norma", "tagliatelle al ragù"]),
    ("history podcast", ["The Long Century", "Footnotes of Empire", "Yesterday's Atlas", "Chronicle Hour"]),
    ("board game", ["Patchwork", "Jaipur", "Hive", "Lost Cities"]),
    ("houseplant", ["a snake plant", "a ZZ plant", "a pothos", "a cast iron plant"]),
    ("stretching routine", ["the sun salutation flow", "the 90/90 hip sequence", "the couch stretch circuit"]),
    ("budgeting app", ["Pennywise", "LedgerLeaf", "CoinNest", "Tally Trail"]),
]
ASSIST_REQ = ["Can you recommend a {t}?", "Could you suggest a {t} for me?"]
ASSIST_EASY = ["For a {t}, I'd recommend {v}.", "My recommendation for a {t} is {v}."]
ASSIST_HARD = ["Sure, {v} would be my pick.", "Happy to help! I'd go with {v}."]
ASSIST_ASK = ["Which {t} did you recommend to me?", "What {t} did you recommend?"]

ACKS = [
    "Got it, thanks for sharing!",
    "Thanks for letting me know.",
    "Noted! Anything else on your mind?",
    "That's great to hear.",
    "Understood. I'll keep that in mind.",
    "Good to know!",
]
ECHO_ACKS = ["Got it: {v}. Thanks for telling me!", "Noted, {v}. I'll remember that."]

FILLERS = [
    ("The weather has been strange lately.", "It does seem unpredictable. Staying comfortable?"),
    ("Can you explain how tides work?", "Tides are caused mainly by the gravitational pull of the moon and the sun."),
    ("I can't decide what to watch tonight.", "Would you like something light or something gripping?"),
    ("Why is the sky blue?", "Shorter blue wavelengths scatter more in the atmosphere."),
    ("I need a synonym for 'happy'.", "You could use cheerful, content, or delighted."),
    ("How long should I boil an egg?", "About seven minutes gives a jammy yolk."),
    ("Tell me a fun fact.", "Honey never spoils if it is stored properly."),
    ("My neighbor's dog barks all night.", "That sounds exhausting. Earplugs might help."),
    ("What's a good way to stay focused?", "Short timed sessions with breaks work well for many people."),
    ("I slept badly last night.", "Sorry to hear that. A consistent bedtime can help."),
    ("How do I remove a coffee stain?", "Blot it, then use cold water with a little dish soap."),
    ("Is it going to rain tomorrow?", "I can't check live forecasts, but a local weather app will know."),
    ("I keep forgetting to drink water.", "Keeping a bottle on your desk is a simple reminder."),
    ("What's the capital of Australia?", "Canberra is the capital of Australia."),
    ("Traffic was terrible today.", "That's frustrating. Did it make you late?"),
    ("I want to tidy my desk.", "Start by clearing everything off, then return only what you use daily."),
    ("How many planets are there?", "There are eight planets in our solar system."),
    ("My internet keeps dropping.", "Restarting the router is a good first step."),
    ("What should I cook for guests?", "A big tray bake is forgiving and feeds a crowd."),
    ("I watched a strange documentary.", "What was it about?"),
    ("Do you know any riddles?", "What has keys but can't open locks? A piano."),
    ("I'm thinking about rearranging my living room.", "Moving the sofa away from the wall can open up the space."),
    ("How do I convert Celsius to Fahrenheit?", "Multiply by nine fifths and add thirty-two."),
    ("The bakery near me closed down.", "That's a shame. Was it a favorite of yours?"),
    ("Can you help me write a toast?", "Of course. Who is it for and what's the occasion?"),
    ("I lost my umbrella again.", "A bright color makes them harder to leave behind."),
    ("What's a healthy snack?", "Fruit with a handful of nuts is a solid choice."),
    ("I have a long meeting tomorrow.", "Preparing two or three key points can make it easier."),
]

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


# ------------------------------------------------------------------------ generation

@dataclass
class _Item:
    u: float  # position in the timeline, 0..1
    user: str
    assistant: str
    tag: str  # for evidence tracking, e.g. "slot:home_city:1"


def _third_person(past_phrase: str) -> str:
    return past_phrase.replace(" my ", " their ")


def _wrong(pool: Sequence[str], correct: Sequence[str] = ()) -> list[str]:
    """Values of ``pool`` that are not a correct answer: what a response must not name."""
    return [v for v in pool if v not in correct]


def _join_list(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _generate_stream(cfg: GenConfig, seed: int | str, stream_idx: int, salt: str, demand: Sequence[tuple[str, bool]]):
    """One history and the questions it can answer from ``demand``.

    Returns the instances placed in this history and the demand still outstanding.  What the history
    says does not depend on the demand, only which questions are asked about it.
    """
    rng = make_rng("stream", seed, stream_idx)
    S = cfg.sessions_per_stream
    items: list[_Item] = []
    candidates: dict[str, list[dict]] = {}  # stratum -> question specs

    def add_q(qtype: str, abstain: bool, **spec):
        key = f"{qtype}|{int(abstain)}"
        candidates.setdefault(key, []).append(dict(qtype=qtype, abstain=abstain, **spec))

    def stmt(templates_easy, templates_hard):
        if templates_hard and rng.random() < cfg.hard_paraphrase_rate:
            return rng.choice(templates_hard)
        return rng.choice(templates_easy)

    def ack(value: str | None = None) -> str:
        if value is not None and rng.random() < cfg.echo_rate:
            return rng.choice(ECHO_ACKS).format(v=value)
        return rng.choice(ACKS)

    friends = rng.sample(NAMES, 6)

    # -- single-valued facts, with optional updates and distractors
    slot_names = list(SLOTS)
    rng.shuffle(slot_names)
    n_filled = max(5, min(len(slot_names) - 3, int(round(len(slot_names) * rng.uniform(0.55, 0.8)))))
    filled = slot_names[:n_filled]
    unfilled = slot_names[n_filled:]
    for name in slot_names:
        slot = SLOTS[name]
        pool = list(slot["values"])
        rng.shuffle(pool)
        used: list[str] = []
        if name in filled:
            v0 = pool.pop()
            used.append(v0)
            items.append(
                _Item(rng.uniform(0.0, 0.45), stmt(slot["easy"], slot["hard"]).format(v=v0), ack(v0), f"slot:{name}:0")
            )
            n_updates = 0
            if slot["updatable"] and slot["update"] and rng.random() < cfg.update_rate:
                n_updates = 1 if rng.random() < 0.7 else 2
            us = sorted(rng.uniform(0.5, 0.98) for _ in range(n_updates))
            for k, u in enumerate(us, start=1):
                v = pool.pop()
                used.append(v)
                items.append(_Item(u, rng.choice(slot["update"]).format(v=v), ack(v), f"slot:{name}:{k}"))
            question = rng.choice(slot["ask"])
            if n_updates:
                add_q(
                    "knowledge-update", False, question=question, answer=used[-1],
                    aliases=(), evidence=[f"slot:{name}:{k}" for k in range(len(used))],
                    meta={"stale": used[:-1], "wrong": _wrong(slot["values"], [used[-1]])},
                )
            else:
                add_q(
                    "single-session-user", False, question=question, answer=v0,
                    aliases=(), evidence=[f"slot:{name}:0"], meta={"wrong": _wrong(slot["values"], [v0])},
                )
        else:
            qtype = "knowledge-update" if slot["updatable"] else "single-session-user"
            add_q(
                qtype, True, question=rng.choice(slot["ask"]),
                answer=f"The user never mentioned {slot['label']}.",
                aliases=(), evidence=[], meta={"wrong": _wrong(slot["values"])},
            )
        if slot["third"] and rng.random() < cfg.third_party_rate:
            v = pool.pop()
            items.append(
                _Item(rng.random(), rng.choice(slot["third"]).format(p=rng.choice(friends), v=v), ack(), f"third:{name}")
            )
        if slot["neg"] and rng.random() < cfg.negation_rate:
            v = pool.pop()
            items.append(_Item(rng.random(), rng.choice(slot["neg"]).format(v=v), ack(), f"neg:{name}"))

    # -- dated events (temporal reasoning)
    n_events = min(S, rng.randint(5, 7))
    events = rng.sample(EVENTS, n_events + 2)
    happened, never = events[:n_events], events[n_events:]
    event_sessions = rng.sample(range(S), n_events)
    event_offset: dict[str, int] = {}
    for (past, _base), s_idx in zip(happened, event_sessions):
        u = (s_idx + rng.uniform(0.05, 0.95)) / S
        pool_ = EVENT_TEMPLATES_RELATIVE if rng.random() < cfg.relative_date_rate else EVENT_TEMPLATES
        template, offset = rng.choice(pool_)
        event_offset[past] = offset
        items.append(_Item(u, template.format(e=past), ack(), f"event:{past}"))
    if rng.random() < cfg.third_party_rate:
        past, _ = never[0]
        items.append(
            _Item(
                rng.random(),
                EVENT_THIRD[0].format(p=rng.choice(friends), e_third=_third_person(past)),
                ack(), "third:event",
            )
        )

    # -- collections (multi-session aggregation)
    for cname, coll in COLLECTIONS.items():
        present = rng.random() < 0.8
        pool = list(coll["values"])
        rng.shuffle(pool)
        if present:
            n = rng.randint(2, 4)
            vals = [pool.pop() for _ in range(n)]
            for k, (v, s_idx) in enumerate(zip(vals, rng.sample(range(S), min(n, S)))):
                u = (s_idx + rng.uniform(0.05, 0.95)) / S
                items.append(_Item(u, stmt(coll["easy"], coll["hard"]).format(v=v), ack(v), f"coll:{cname}:{k}"))
            ev = [f"coll:{cname}:{k}" for k in range(len(vals))]
            add_q(
                "multi-session", False, question=coll["count_q"], answer=str(len(vals)),
                aliases=(NUMBER_WORDS[len(vals)],), evidence=ev,
                meta={"numeric": len(vals), "numeric_question": True, "wrong": []},
            )
            add_q(
                "multi-session", False, question=coll["list_q"], answer=_join_list(vals),
                aliases=(), evidence=ev, meta={"list": vals, "wrong": _wrong(coll["values"], vals)},
            )
        else:
            add_q(
                "multi-session", True, question=coll["count_q"],
                answer=f"The user never mentioned {coll['label']}.",
                aliases=(), evidence=[], meta={"numeric_question": True, "wrong": _wrong(coll["values"])},
            )
        if rng.random() < cfg.third_party_rate * 0.6:
            items.append(
                _Item(rng.random(), coll["third"].format(p=rng.choice(friends), v=pool.pop()), ack(), f"third:{cname}")
            )
    # summed quantities
    plant = rng.choice(PLANTS)
    if rng.random() < 0.8:
        n = rng.randint(2, 4)
        qtys = [rng.randint(1, 5) for _ in range(n)]
        for k, (q, s_idx) in enumerate(zip(qtys, rng.sample(range(S), min(n, S)))):
            u = (s_idx + rng.uniform(0.05, 0.95)) / S
            if rng.random() < cfg.hard_paraphrase_rate:
                text = f"Picked up {q} more {plant} at the nursery."
            else:
                text = f"I bought {q} {plant} for my balcony."
            items.append(_Item(u, text, ack(), f"coll:plants:{k}"))
        total = sum(qtys)
        add_q(
            "multi-session", False, question=f"How many {plant} have I bought in total?",
            answer=str(total), aliases=(NUMBER_WORDS[total],) if total < len(NUMBER_WORDS) else (),
            evidence=[f"coll:plants:{k}" for k in range(len(qtys))],
            meta={"numeric": total, "numeric_question": True, "wrong": []},
        )
    else:
        add_q(
            "multi-session", True, question=f"How many {plant} have I bought in total?",
            answer=f"The user never mentioned buying any {plant}.", aliases=(), evidence=[],
            meta={"numeric_question": True, "wrong": []},
        )

    # -- preferences
    prefs = rng.sample(PREFERENCES, 4)
    for (c, x, y) in prefs[:3]:
        if rng.random() < 0.5:
            x, y = y, x
        items.append(_Item(rng.random(), stmt(PREF_EASY, PREF_HARD).format(c=c, x=x, y=y), ack(), f"pref:{c}"))
        add_q(
            "single-session-preference", False, question=rng.choice(PREF_ASK).format(c=c),
            answer=x, aliases=(), evidence=[f"pref:{c}"],
            meta={
                "rubric": f"The user prefers {x} over {y} for {c}. A correct response reflects that preference.",
                "distractors": [y],
                "wrong": _wrong(PREFERENCE_PHRASES, [x]),
            },
        )
    c, x, y = prefs[3]
    add_q(
        "single-session-preference", True, question=rng.choice(PREF_ASK).format(c=c),
        answer=f"The user never mentioned a preference about {c}.", aliases=(), evidence=[],
        meta={"wrong": _wrong(PREFERENCE_PHRASES)},
    )

    # -- facts provided by the assistant
    topics = rng.sample(ASSISTANT_TOPICS, 4)
    for (t, pool) in topics[:3]:
        v = rng.choice(pool)
        items.append(
            _Item(rng.random(), rng.choice(ASSIST_REQ).format(t=t), stmt(ASSIST_EASY, ASSIST_HARD).format(t=t, v=v), f"assist:{t}")
        )
        add_q(
            "single-session-assistant", False, question=rng.choice(ASSIST_ASK).format(t=t),
            answer=v, aliases=(), evidence=[f"assist:{t}"], meta={"wrong": _wrong(pool, [v])},
        )
    t, _pool = topics[3]
    add_q(
        "single-session-assistant", True, question=rng.choice(ASSIST_ASK).format(t=t),
        answer=f"The assistant never recommended a {t}.", aliases=(), evidence=[], meta={"wrong": _wrong(_pool)},
    )

    # -- timeline
    start = _dt.date(2023, 1, 5) + _dt.timedelta(days=rng.randint(0, 200))
    dates: list[_dt.date] = []
    d = start
    for _ in range(S):
        dates.append(d)
        d = d + _dt.timedelta(days=rng.randint(1, 12))
    question_date = dates[-1] + _dt.timedelta(days=rng.randint(1, 30))

    by_session: dict[int, list[_Item]] = {i: [] for i in range(S)}
    for it in items:
        by_session[min(S - 1, int(it.u * S))].append(it)

    haystack_id = opaque_id(salt, f"stream{stream_idx}", prefix="h")
    sessions: list[Session] = []
    tag_to_session: dict[str, str] = {}
    lo, hi = cfg.exchanges_per_session
    for i in range(S):
        content = sorted(by_session[i], key=lambda it: it.u)
        n_ex = max(len(content), rng.randint(lo, hi))
        slots_ = [None] * n_ex
        positions = sorted(rng.sample(range(n_ex), len(content)))
        for pos, it in zip(positions, content):
            slots_[pos] = it
        sid = opaque_id(salt, f"stream{stream_idx}:session{i}", prefix="s")
        turns: list[Turn] = []
        for slot_item in slots_:
            if slot_item is None:
                fu, fa = rng.choice(FILLERS)
                turns.append(Turn("user", fu))
                turns.append(Turn("assistant", fa))
            else:
                turns.append(Turn("user", slot_item.user))
                turns.append(Turn("assistant", slot_item.assistant))
                tag_to_session[slot_item.tag] = sid
        sessions.append(Session(sid, dates[i].isoformat(), tuple(turns)))

    session_date = {s.session_id: _dt.date.fromisoformat(s.date) for s in sessions}

    # -- temporal questions need the realised dates
    ev_dates = {
        past: session_date[tag_to_session[f"event:{past}"]] - _dt.timedelta(days=event_offset[past])
        for (past, _b) in happened
    }
    for (past, base) in happened:
        n_days = (question_date - ev_dates[past]).days
        add_q(
            "temporal-reasoning", False, question=f"How many days ago did I {base}?",
            answer=f"{n_days} days", aliases=(str(n_days),), evidence=[f"event:{past}"],
            meta={"numeric": n_days, "unit": "days", "tolerance": 1, "numeric_question": True, "wrong": []},
        )
        month = MONTH_NAMES[ev_dates[past].month - 1]
        year_month = f"{ev_dates[past].year}-{ev_dates[past].month:02d}"
        # The key accepts the month by name or as "YYYY-MM", so the other eleven
        # months are wrong in both forms; otherwise "2023-01 2023-02 ..." would pass.
        other_months = _wrong(MONTH_NAMES, [month]) + _wrong(
            [f"{ev_dates[past].year}-{m:02d}" for m in range(1, 13)], [year_month]
        )
        add_q(
            "temporal-reasoning", False, question=f"In which month did I {base}?",
            answer=month,
            aliases=(f"{month} {ev_dates[past].year}", year_month),
            evidence=[f"event:{past}"], meta={"wrong": other_months},
        )
    pairs = [(a, b) for i, a in enumerate(happened) for b in happened[i + 1:]]
    rng.shuffle(pairs)
    for (pa, ba), (pb, bb) in pairs[:6]:
        da, db = ev_dates[pa], ev_dates[pb]
        if da == db:
            continue
        gap = abs((da - db).days)
        add_q(
            "temporal-reasoning", False,
            question=f"How many days passed between the day I {pa} and the day I {pb}?",
            answer=f"{gap} days", aliases=(str(gap),), evidence=[f"event:{pa}", f"event:{pb}"],
            meta={"numeric": gap, "unit": "days", "tolerance": 1, "numeric_question": True, "wrong": []},
        )
        first_past, first_base = (pa, ba) if da < db else (pb, bb)
        later_past, later_base = (pb, bb) if da < db else (pa, ba)
        # The question offers both events, so naming the later one after the earlier
        # is part of a full answer ("distractors" orders them).  Any third event is wrong.
        add_q(
            "temporal-reasoning", False,
            question=f"Which did I do first: {ba} or {bb}?",
            answer=first_base, aliases=(first_past,), evidence=[f"event:{pa}", f"event:{pb}"],
            meta={
                "distractors": [later_base, later_past],
                "wrong": _wrong(EVENT_PHRASES, [pa, ba, pb, bb]),
            },
        )
    for (past, base) in never:
        add_q(
            "temporal-reasoning", True, question=f"How many days ago did I {base}?",
            answer=f"The user never mentioned that they {past.replace(' my ', ' their ')}.",
            aliases=(), evidence=[], meta={"numeric_question": True, "wrong": []},
        )

    # -- take from the outstanding demand what this history can answer
    for specs in candidates.values():
        rng.shuffle(specs)

    def take(qtype: str, abstain: bool) -> dict | None:
        pool = candidates.get(f"{qtype}|{int(abstain)}")
        return pool.pop() if pool else None

    chosen: list[dict] = []
    outstanding: list[tuple[str, bool]] = []
    for qtype, abstain in demand:
        spec = None
        if len(chosen) < cfg.questions_per_stream:
            spec = take(qtype, abstain)
            if spec is None and abstain:
                # What an unanswerable question measures does not depend on its type, so any will do.
                # The type of an answerable question is never substituted.
                order = list(QUESTION_TYPES)
                rng.shuffle(order)
                for alt in order:
                    spec = take(alt, True)
                    if spec is not None:
                        break
        if spec is None:
            outstanding.append((qtype, abstain))
        else:
            chosen.append(spec)
    # Position within a history must say nothing about a question.
    rng.shuffle(chosen)

    instances: list[Instance] = []
    session_tuple = tuple(sessions)
    for q_idx, spec in enumerate(chosen):
        source_id = f"stream{stream_idx}:q{q_idx}"
        pid = opaque_id(salt, source_id, prefix="p")
        probe = Probe(
            probe_id=pid, question=spec["question"], question_date=question_date.isoformat(),
            haystack_id=haystack_id, sessions=session_tuple,
        )
        evidence = tuple(sorted({tag_to_session[t] for t in spec["evidence"] if t in tag_to_session}))
        key = AnswerKey(
            probe_id=pid, question_type=spec["qtype"], answer=spec["answer"],
            aliases=tuple(spec["aliases"]), is_abstention=spec["abstain"],
            evidence_session_ids=evidence, meta=dict(spec["meta"]),
        )
        instances.append(Instance(probe=probe, key=key, source_id=source_id))
    return instances, outstanding


def generate(seed: int | str, cfg: GenConfig | None = None, name: str | None = None) -> Benchmark:
    """Deterministically generate a labelled benchmark from ``seed``.

    The benchmark holds exactly ``n_streams * questions_per_stream`` questions.  Each is drawn from
    ``type_mix`` and ``abstention_rate``, and an answerable question is always of the type drawn: when a
    history has nothing left to ask of that type, the question is carried over, and extra histories are
    added until every question is placed.  A mix that cannot be met within ``MAX_STREAM_FACTOR`` times the
    requested number of histories raises ``ValueError``.
    """
    cfg = cfg or GenConfig()
    cfg.validate()
    # Ids depend on the settings as well as the seed, so two different benchmarks never share one.
    salt = f"synthetic:{seed}:{hash_obj(cfg.to_dict())[:16]}"
    types = [t for t in QUESTION_TYPES if cfg.type_mix.get(t, 0.0) > 0]  # canonical order, whatever the mapping's
    weights = [float(cfg.type_mix[t]) for t in types]
    rng = make_rng("demand", seed)
    demand: list[tuple[str, bool]] = []
    for _ in range(cfg.n_streams * cfg.questions_per_stream):
        abstain = rng.random() < cfg.abstention_rate
        demand.append((rng.choices(types, weights=weights, k=1)[0], abstain))
    instances: list[Instance] = []
    histories = 0
    q = cfg.questions_per_stream
    # Each requested history is offered its own share of the questions, so the make-up of a history
    # follows the mix.  What a history cannot answer is carried over to extra histories at the end.
    carried: list[tuple[str, bool]] = []
    for stream in range(cfg.n_streams):
        # its own share first; whatever room is left goes to questions carried over from earlier histories
        placed, carried = _generate_stream(cfg, seed, stream, salt, demand[stream * q:(stream + 1) * q] + carried)
        instances.extend(placed)
        histories += bool(placed)
    demand = carried
    for stream in range(cfg.n_streams, cfg.n_streams * MAX_STREAM_FACTOR):
        if not demand:
            break
        placed, demand = _generate_stream(cfg, seed, stream, salt, demand)
        instances.extend(placed)
        histories += bool(placed)
    if demand:
        missing: dict[str, int] = {}
        for qtype, abstain in demand:
            label = f"unanswerable {qtype}" if abstain else qtype
            missing[label] = missing.get(label, 0) + 1
        raise ValueError(
            f"cannot place {len(demand)} of {cfg.n_streams * cfg.questions_per_stream} questions within "
            f"{cfg.n_streams * MAX_STREAM_FACTOR} histories under these settings: {missing}"
        )
    return Benchmark(
        name or f"synthetic-{seed}",
        instances,
        info={"source": "synthetic", "config": cfg.to_dict(), "seed": str(seed), "histories": histories},
    )
