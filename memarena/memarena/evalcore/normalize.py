"""Text normalisation shared by every deterministic scorer.

The rules follow the SQuAD convention (lower-case, drop punctuation and
articles, collapse whitespace) with additions that matter for memory QA:
diacritics are folded ("Kraków" == "Krakow"), small number words are mapped to
digits ("three" == "3"), and trailing ".0" / thousands separators are removed.

Everything here reads text through ``clean`` first.  It removes characters a
reader cannot see (zero-width spaces and joiners, soft hyphens, byte-order
marks), so that a response cannot hide a trigger word from a detector, or a
wrong answer from a matcher, by splitting it with something invisible.  The
tokeniser and every detector (abstention, hedging, lists of alternatives,
echoed questions) work on the cleaned text and therefore agree on what the
response says.

Besides the tokeniser the module holds what the judges need to read a response
*by clause*:

``read``            a response split into sentences and clauses, each with its
                    tokens, so that a phrase can be judged by what it is
                    attached to rather than by where it stands;
``Reading.declines`` / ``Reading.offers``
                    which clauses decline to answer, what they decline about,
                    and whether anything outside them states an answer;
``number_mentions`` the numbers a response states, with dates, clock times and
                    pronoun-like "one" / "the two" left out and the unit each
                    number is attached to.

Cost.  Every regular expression here is linear in the length of its input, and
nothing is read beyond ``MAX_RESPONSE_CHARS`` characters by a detector: a
response longer than that is not a decline, not a hedge and not an answer.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from functools import lru_cache
from typing import Iterable, Sequence

# Longest response a detector will read.  The judges stop at 100 normalised tokens; at a generous
# 40 characters per token that is 4,000 characters, and nothing longer can be an answer they accept.
MAX_RESPONSE_CHARS = 4000

_ARTICLES = frozenset({"a", "an", "the"})
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}
_APOSTROPHES = re.compile(r"[’'`]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_NON_ALNUM = re.compile(r"[^0-9a-z\.]+")
_TRAILING_ZERO = re.compile(r"^(\d+)\.0+$")
_DECIMAL = re.compile(r"\d+\.\d+")

# Unicode categories that never show on the page: format characters (zero-width
# space and joiners, soft hyphen, byte-order mark, direction marks) and
# non-spacing or enclosing marks (accents once decomposed, variation selectors).
_UNSEEN = frozenset({"Cf", "Mn", "Me"})
# Several scorers read the same response while one item is graded; this many
# recent texts are remembered so that each is prepared once.
_RECENT = 1024


def clean(text: str) -> str:
    """``text`` with compatibility forms folded and unseen characters removed.

    Rule: a character that a reader cannot see must not change what a scorer
    reads.  "Either" with a zero-width space after the "t" is still "Either",
    and "Lisbon" with a soft hyphen in it is still "Lisbon".  Case and
    punctuation are kept, so detectors that need them can run on the result.
    """
    if text.isascii():
        return text
    text = unicodedata.normalize("NFKD", text)
    return "".join(
        ch for ch in text
        if ch.isascii() or not (unicodedata.combining(ch) or unicodedata.category(ch) in _UNSEEN)
    )


@lru_cache(maxsize=200_000)
def tokens(text: str) -> tuple[str, ...]:
    """Normalised tokens of ``text`` (cached; inputs repeat heavily in grading)."""
    text = clean(str(text)).lower()
    text = _APOSTROPHES.sub("", text)
    text = _THOUSANDS.sub("", text)
    out: list[str] = []
    for tok in _NON_ALNUM.split(text):
        if "." in tok:
            tok = tok.strip(".")
            if "." in tok:
                m = _TRAILING_ZERO.match(tok)
                if m:
                    tok = m.group(1)
                elif not _DECIMAL.fullmatch(tok):
                    # a dot that survives inside a non-number token is punctuation
                    out.extend(t for t in tok.split(".") if t and t not in _ARTICLES)
                    continue
        if not tok:
            continue
        tok = _NUMBER_WORDS.get(tok, tok)
        if tok not in _ARTICLES:
            out.append(tok)
    return tuple(out)


def normalize(text: str) -> str:
    return " ".join(tokens(text))


# ---------------------------------------------------------- clauses and negation

# Where one clause ends and the next begins.  The word "but" also starts a new
# clause; that is handled on tokens.
_CLAUSE_BREAK = re.compile(
    r"[.!?]+[\"'”’)\]]*(?=\s|$)"  # the end of a sentence, with any closing quote or bracket
    r"|\n+"  # a line break
    r"|[;:—–()]"  # semicolon, colon, dash, bracket
    r"|,(?!\d)|(?<!\d),"  # a comma, unless it sits inside a number
    r"|\s-\s"  # a hyphen used as a dash
)

NEGATIONS = frozenset({"not", "never", "no", "isnt", "dont", "doesnt", "didnt", "wasnt", "arent", "werent", "havent",
                       "hasnt", "hadnt", "wont", "wouldnt", "neither", "nor"})
NEGATION_WINDOW = 5


@lru_cache(maxsize=_RECENT)
def clause_tokens(text: str) -> tuple[tuple[str, ...], ...]:
    """The tokens of ``text``, in order, grouped by clause.

    Concatenating the groups gives exactly ``tokens(text)``.  A negation word
    only reaches as far as its own clause, which is what lets "It is not Porto,
    it is Lisbon" assert Lisbon.
    """
    clauses: list[tuple[str, ...]] = []
    for segment in _CLAUSE_BREAK.split(clean(str(text))):
        current: list[str] = []
        for tok in tokens(segment):
            if tok == "but" and current:
                clauses.append(tuple(current))
                current = []
            current.append(tok)
        if current:
            clauses.append(tuple(current))
    return tuple(clauses)


def negated(preceding: Sequence[str]) -> bool:
    """True if the tokens that lead up to a phrase, inside its clause, negate it.

    Rule: a mention does not assert anything when a negation word stands within
    the five tokens before it with no clause break in between.  "You definitely
    do not live in Lisbon" does not say Lisbon; "Not Porto, but Lisbon" does,
    and so does "No doubt it's Lisbon".
    """
    window = list(preceding[-NEGATION_WINDOW:])
    if "but" in window:
        window = window[len(window) - window[::-1].index("but"):]
    for k, tok in enumerate(window):
        # "no doubt it's Lisbon" affirms
        if tok in NEGATIONS and not (tok == "no" and window[k + 1:k + 2] in (["doubt"], ["question"])):
            return True
    return False


# ------------------------------------------------------------------ words

# A response speaks to the user ("your passport"); the reference is written from
# the user's side ("my passport").  Readings compare after folding perspective.
PERSPECTIVE = {
    "your": "my", "yours": "mine", "you": "i", "yourself": "myself",
    "youre": "im", "youve": "ive", "youd": "id", "youll": "ill",
}

_STEMS: dict[str, str] = {}


def stem(token: str) -> str:
    """Singular form of a normalised token, by rule.

    "Tuesdays" -> "tuesday", "resorts" -> "resort", "documentaries" ->
    "documentary", "boxes" -> "box".  Matching on stems lets "On Tuesday"
    answer a reference of "Tuesdays".  Both sides of every comparison are
    stemmed, so a rule that is wrong for some word is wrong the same way twice.
    """
    hit = _STEMS.get(token)
    if hit is None:
        hit = token
        n = len(token)
        if n > 3 and token[-1] == "s" and token[-2].isalpha():
            if n > 4 and token.endswith("ies"):
                hit = token[:-3] + "y"
            elif token.endswith(("ches", "shes", "sses", "xes", "zes")):
                hit = token[:-2]
            elif not token.endswith(("ss", "us", "is")):
                hit = token[:-1]
        if len(_STEMS) < 500_000:
            _STEMS[token] = hit
    return hit


# Words that carry no answer.  ``GRAMMAR_WORDS`` are function words; ``FUNCTION_WORDS`` adds the
# vocabulary of talking about a conversation ("mentioned", "record", "sorry", "guess").  Written as
# the tokeniser gives them: no apostrophes, "you" folded to "i", "one" as "1".
GRAMMAR_WORDS = frozenset("""
i me my mine myself we us our ours it its itself he she him her his hers they them their theirs this that these those
there here what which who whom whose when where why how whatever whether if then than so as of in on at to for from by
with without about into onto over under between through during before after since until up down out off and or but nor
yet not no never none nothing anything something everything anyone someone everyone anybody somebody nobody any some
all both each every either neither such same own very really quite just only also too even still already ever again
once now today currently far much many most less least few lot bit kind sort thing things way 1 be am is are was were
been being im ive id ill isnt arent wasnt werent do does did doing done dont doesnt didnt have has had having havent
hasnt hadnt can cant cannot could couldnt will wont would wouldnt shall should shouldnt may might must lets let thats
theres whats heres weve were hes shes theyre theyve else otherwise other others more further another next back down
away around together apart along across ahead always often sometimes usually however though although because while
unless rather instead maybe perhaps
""".split())
FUNCTION_WORDS = GRAMMAR_WORDS | frozenset("""
mention mentioned mentioning mentions tell told telling tells say said saying says share shared sharing shares discuss
discussed discussing talk talked talking bring brought come came coming comes give gave given provide provided specify
specified state stated note noted notes record records recorded information info detail details data context
conversation conversations chat chats history memory memories message messages session sessions transcript transcripts
log logs know known knowing knows knowledge recall remember remembered see seen saw find found look looking looked
check checked checking answer answers answered question questions sure certain aware able unable sorry afraid
unfortunately apologies apologize honestly frankly please thanks thank guess guessing speculate speculating assume
assuming think thinking believe suppose want wanted like liked need needed help helps helping happy glad feel free
hope ask asked asking remind clarify confirm earlier previously past previous prior point time moment based
according regarding concerning anywhere somewhere elsewhere nowhere possibly probably likely presumably better best
wise safe safer fair true false right wrong mistaken confused confusing referring mean meant ok okay fyi idk tbd yes
yeah nope hmm well oh ah actually simply truly exactly clearly specifically particular particularly available unknown
unclear unspecified missing enough sufficient insufficient incomplete anymore longer up additional beyond besides rest
topic subject matter fact case regard mind hand side part place people person try consider keep meanwhile anyway
regardless typically generally normally whenever wherever
""".split())
# Words that make a clause a statement of its own ("I could guess", "it is ...") rather than a bare phrase.
SUBJECTS = frozenset("i im ive id ill we weve were it its that thats this there theres they theyre he she hes shes me us "
                      "lets let".split())
# Words a lead-in or a remark opens with ("Based on our chats, ...", "Sorry, ...", "Hope that helps").
OPENERS = frozenset("""
based according from as in on at for to so after before looking checking going given with without if unless until when
while whether because since though although however but and or yet then also plus not no never nothing none nope sorry
unfortunately sadly honestly frankly hmm well oh ah yes yeah ok okay sure right good great interesting tough hard
apologies thanks thank alas regrettably currently presently still again maybe perhaps possibly probably presumably
likely hope hoping happy glad feel please let lets here there that this these those what which who where why how do
does did is are was were can could would should will may might must have has had any some all every each either
neither both such namely specifically including like overall total altogether finally lastly firstly first second
judging reviewing searching unknown unclear of certainly absolutely definitely yep alright got noted
""".split())
_WH = frozenset({"why", "when", "where", "how", "who", "which", "what"})
# A decline about one of these is about something other than the question ("anything else", "her other pets").
_ELSE = frozenset("else otherwise other others more further additional beyond besides rest".split())


# ---------------------------------------------------------------------- numbers

_TENS = {"twenty": 20.0, "thirty": 30.0, "forty": 40.0, "fifty": 50.0, "sixty": 60.0, "seventy": 70.0,
         "eighty": 80.0, "ninety": 90.0}
_WORD_VALUES = {**{w: float(d) for w, d in _NUMBER_WORDS.items()}, **_TENS}
_NUM_OR_WORD = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w]|\.\d)|[A-Za-z]+")
# What may stand between the end of one sentence and the first word of the next.
_SENTENCE_LEADERS = " \t\r\n\"'“”‘’([{*_-•>#"

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?"
          r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
# Dates and clock times: their digits are not counts.  "April 16", "16 April 2023", "the 16th of April",
# "2023-04-16", "16/04/2023", "April 2023", "8:20 pm", "8 pm".
_DATE_OR_TIME = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"
    r"|\b\d{1,4}/\d{1,2}/\d{1,4}\b"
    rf"|\b{_MONTH}\.? \d{{1,2}}(?:st|nd|rd|th)?(?!\d)(?:,? \d{{4}}\b(?![-/]\d))?"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?(?: of)? {_MONTH}\b\.?(?:,? \d{{4}}\b(?![-/]\d))?"
    rf"|\b{_MONTH}\.?,? \d{{4}}\b(?![-/]\d)"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?(?: ?[ap]\.?m\b\.?)?"
    r"|\b\d{1,2} ?[ap]\.?m\b\.?",
    re.I,
)
# Words before a number word that make it a pronoun ("the two", "this one"), and words after "one"
# that do the same ("one of", "one more").
_POINTING = frozenset({"the", "this", "that", "these", "those", "which", "no", "any", "every", "each", "another",
                       "other", "either", "neither"})
_POINTING_ONE = _POINTING | frozenset({"last", "first", "next", "same", "only", "right", "wrong", "good", "big",
                                       "new", "old", "single"})
_ONE_IS_NOT_A_COUNT_BEFORE = frozenset({"of", "more", "another", "thing", "way", "guess", "possibility", "option"})
_WORD_BEFORE = re.compile(r"([A-Za-z]+)[ ]$")
_WORD_AFTER = re.compile(r"[ -]([A-Za-z]+)(?: ([A-Za-z]+))?")

# Units of time and what each is worth in seconds, where that is exact.
TIME_UNITS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "night": None, "week": 604800,
              "fortnight": 1209600, "month": None, "year": None, "decade": None, "weekend": None}


def _starts_sentence(text: str, start: int, end: int) -> bool:
    """True if the word at ``text[start:end]`` owes its capital letter to its position.

    That is the first word of the response, of a sentence or of a line.  After a
    colon it is the answer ("Total: Seven.") unless another capitalised word
    follows, in which case the colon introduced a name ("3 books: Seventeen
    Winters, ...").
    """
    lead = text[:start].rstrip(_SENTENCE_LEADERS)
    if not lead or lead[-1] in ".!?" or "\n" in text[len(lead):start]:
        return True
    if lead[-1] == ":":
        return not text[end:].lstrip()[:1].isupper()
    return False


def _scan_numbers(text: str) -> list[tuple[float, int, int, bool]]:
    """Every number in ``text`` as (value, start, end, written as a word), in order.

    Rules.  A number word with a capital letter that is not explained by its
    position is part of a name, not a number: "Seventeen Winters" is a book,
    and an answer that lists it has not said "17".  Number words are read up
    to ninety-nine, joined by a hyphen or a space ("forty-three", "forty
    three").
    """
    found: list[tuple[float, int, int, bool]] = []
    tens_end = -1  # where a word such as "forty" ended, if the last number was one
    for m in _NUM_OR_WORD.finditer(text):
        tok = m.group(0)
        start = m.start()
        if not tok[0].isalpha():
            found.append((float(tok), start, m.end(), False))
            tens_end = -1
            continue
        value = _WORD_VALUES.get(tok.lower())
        if value is None:
            tens_end = -1
            continue
        if tens_end >= 0 and 1 <= value <= 9 and text[tens_end:start] in ("-", " "):
            tens, first, _end, _word = found[-1]
            found[-1] = (tens + value, first, m.end(), True)
            tens_end = -1
            continue
        if tok[0].isupper() and tok[1:].islower() and not _starts_sentence(text, start, m.end()):
            tens_end = -1
            continue
        found.append((value, start, m.end(), True))
        tens_end = m.end() if value >= 20 and value % 10 == 0 else -1
    return found


def _numbers(text: str) -> tuple[str, list[tuple[float, int]]]:
    """The cleaned text and every number in it as (value, offset), in order."""
    text = _THOUSANDS.sub("", clean(str(text)))
    return text, [(value, start) for value, start, _end, _word in _scan_numbers(text)]


def numbers_in(text: str) -> list[float]:
    """Numbers mentioned in ``text`` in order of appearance, including number words up to ninety-nine."""
    return [value for value, _at in _numbers(text)[1]]


def asserted_numbers(text: str) -> list[float]:
    """Numbers ``text`` states, in order, leaving out the ones it negates ("not 3")."""
    cleaned, found = _numbers(text)
    breaks = [m.end() for m in _CLAUSE_BREAK.finditer(cleaned)]
    stated: list[float] = []
    for value, at in found:
        k = bisect_right(breaks, at)
        clause_start = breaks[k - 1] if k else 0
        if not negated(tokens(cleaned[clause_start:at])):
            stated.append(value)
    return stated


class NumberMention:
    """One number a response states: its value, what it counts and how it is introduced."""

    __slots__ = ("value", "words", "unit", "year_like", "concludes", "follows_verb")

    def __init__(self, value: float, words: tuple[str, ...], year_like: bool, concludes: bool, follows_verb: bool):
        self.value = value
        self.words = words  # the one or two words after the number, in the singular: ("day", "ago"), ("full", "day")
        self.unit = next((w for w in words if w in TIME_UNITS), None)  # the unit of time among them, if any
        self.year_like = year_like  # a bare 1900..2099: a year unless the question is about such a number
        self.concludes = concludes  # introduced as the result: "a total of 7", "so that was 77", "= 7"
        self.follows_verb = follows_verb  # "it is 4", "that was 77"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"NumberMention({self.value}, {self.words!r})"


_CONCLUDING_BEFORE = frozenset({"total", "totals", "totalling", "totaling", "so", "equals", "equal", "makes", "making",
                                "altogether", "sum", "overall", "therefore", "hence", "thus", "giving", "gives"})
_CONCLUDING_AFTER = frozenset({"total", "altogether", "overall", "combined"})
_VERB_BEFORE = frozenset({"is", "was", "are", "were", "its", "thats", "be", "been"})
_EQUALS = re.compile(r"=\s*$")
_IN_TOTAL = re.compile(r"\s*(?:\w+ ){0,2}?in (?:total|all)\b", re.I)


@lru_cache(maxsize=_RECENT)
def number_mentions(text: str) -> tuple[NumberMention, ...]:
    """The numbers ``text`` states, in order, as a careful reader takes them.

    Rules.
    * Dates and clock times are masked first: "April 16", "2023-04-16" and
      "8:20 pm" hold no count.  A bare year stays, marked ``year_like``.
    * A number word that points rather than counts is skipped: "the two",
      "this one", "one of", "one more", "no one".
    * A negated number ("not 3") is not stated.
    * Each mention carries the noun that follows it, in the singular ("77
      days" -> "day"), so a judge can tell the number attached to the unit it
      asked for from every other number.  A duration given in two units is
      one mention: "2 weeks and 6 days" is 20 days.
    """
    if len(text) > MAX_RESPONSE_CHARS:
        return ()
    cleaned = _THOUSANDS.sub("", clean(str(text)))
    masked = _DATE_OR_TIME.sub(lambda m: " " * len(m.group(0)), cleaned)
    breaks = [m.end() for m in _CLAUSE_BREAK.finditer(masked)]
    out: list[NumberMention] = []
    spans: list[tuple[int, int]] = []
    for value, start, end, word in _scan_numbers(masked):
        k = bisect_right(breaks, start)
        clause_start = breaks[k - 1] if k else 0
        lead = masked[clause_start:start]
        after = _WORD_AFTER.match(masked, end)
        next_word = after.group(1).lower() if after else ""
        if word:
            before = _WORD_BEFORE.search(lead)
            prev_word = before.group(1).lower() if before else ""
            if value == 1 and (prev_word in _POINTING_ONE or next_word in _ONE_IS_NOT_A_COUNT_BEFORE):
                continue
            if value <= 3 and prev_word in _POINTING and next_word in ("", "of", "that", "who", "which"):
                continue
        lead_tokens = tokens(lead)
        if negated(lead_tokens):
            continue
        words = tuple(stem(w.lower()) for w in after.groups() if w) if after else ()
        unit = next((w for w in words if w in TIME_UNITS), None)
        window = lead_tokens[-3:]
        concludes = (
            any(t in _CONCLUDING_BEFORE for t in window)
            or bool(_EQUALS.search(lead))
            or next_word in _CONCLUDING_AFTER
            or bool(_IN_TOTAL.match(masked, end))
        )
        follows_verb = any(t in _VERB_BEFORE for t in lead_tokens[-2:])
        year_like = not word and 1900 <= value <= 2099 and value == int(value) and unit is None
        # "2 weeks and 6 days", "1 hour 30 minutes": the smaller unit continues the larger one
        if out and unit in TIME_UNITS and TIME_UNITS[unit] and out[-1].unit in TIME_UNITS and TIME_UNITS[out[-1].unit]:
            gap = masked[spans[-1][1]:start].strip(" ,").lower()
            bigger = out[-1].unit
            if gap in (bigger, bigger + "s", bigger + " and", bigger + "s and") and TIME_UNITS[bigger] > TIME_UNITS[unit]:
                total = out[-1].value * TIME_UNITS[bigger] + value * TIME_UNITS[unit]
                out[-1] = NumberMention(total / TIME_UNITS[unit], words, False, out[-1].concludes or concludes,
                                        out[-1].follows_verb)
                spans[-1] = (spans[-1][0], end)
                continue
        out.append(NumberMention(value, words, year_like, concludes, follows_verb))
        spans.append((start, end))
    return tuple(out)


# ------------------------------------------------------------- reading by clause

# Where one clause ends.  A full stop after a single letter or a title ("U.S.", "Dr. Smith") ends nothing.
_BREAK = re.compile(
    r"(?P<s>(?:(?<!\b[A-Za-z])(?<!\b(?:Dr|Mr|Ms|St|vs|Jr|Sr|Mt))(?<!\bMrs)\.+|[!?][.!?]*)[\"'”’)\]]*(?=\s|$))"
    r"|(?P<n>\n+)"
    r"|(?P<m>[;—–()\[\]]|:(?!\d)|(?<!\d):|,(?!\d)|(?<!\d),|\s-\s)"
)
_BUT = re.compile(r"\bbut\b", re.I)


@lru_cache(maxsize=_RECENT)
def _spaced(text: str) -> str:
    """What the phrase detectors read: cleaned text, one apostrophe form, single spaces, line breaks kept.

    The detectors are regular expressions over words separated by one space, so
    an invisible character inside a word, a no-break space or a doubled space
    between two words would otherwise switch a detector off.
    """
    text = _APOSTROPHES.sub("'", clean(str(text)))
    if "\n" not in text:
        return " ".join(text.split())
    return "\n".join(" ".join(line.split()) for line in text.split("\n")).strip()


class Clause:
    """One clause: where it sits in the text, which tokens are its own, and what separates it from its neighbours.

    ``before`` is "^" at the start of a sentence, "b" after the word "but", else
    the punctuation mark that opened the clause.  ``after`` is the mark that
    closed it ("?" whenever a question mark is among them), "b" before "but" and
    "$" at the end of the text.
    """

    __slots__ = ("start", "end", "t0", "t1", "sentence", "before", "after")

    def __init__(self, start: int, end: int, t0: int, t1: int, sentence: int, before: str, after: str):
        self.start, self.end, self.t0, self.t1 = start, end, t0, t1
        self.sentence, self.before, self.after = sentence, before, after


_NT = r"(?:isn'?t|wasn'?t|aren'?t|weren'?t|hasn'?t|haven'?t|hadn'?t|doesn'?t|don'?t|didn'?t|won'?t|wouldn'?t|couldn'?t|can'?t)"
_ADV = r"(?:(?:really|honestly|truly|simply|just|still|actually|currently|even|ever|yet|quite|exactly|sadly) )*"
_SAID = (r"(?:mention|mentioned|tell|told|say|said|share|shared|specify|specified|bring|brought|discuss|discussed|talk"
         r"|talked|state|stated|give|gave|given|provide|provided|indicate|indicated|reveal|revealed|note|noted"
         r"|let me know|disclose|disclosed|recommend|recommended|suggest|suggested)")
# Phrases that decline to answer for lack of information.  Each is a plain alternation of words: no
# pattern can take more than one pass over the text.
_DECLINE = re.compile("|".join(f"(?:{p})" for p in [
    # "I don't know", "I can't find anything about that", "I honestly can't remember"
    rf"\bi {_ADV}(?:do not|don'?t|did not|didn'?t|can ?not|can'?t|could not|couldn'?t) {_ADV}(?:seem to |appear to )?"
    r"(?:know|have|recall|remember|see|find|locate|answer|tell|say|determine)\b",
    # "I don't think you ever mentioned it", "I don't believe you've told me that"
    r"\bi (?:do not|don'?t) (?:think|believe) (?:that )?(?:you|we|i|it|this|that|anyone)"
    r"(?:'?ve|'?d| have| had| has| did| was| were)? (?:ever |actually )?"
    r"(?:mentioned|told|said|shared|discussed|brought|came|come|talked|covered|gave|given|specified|stated"
    r"|recommended|suggested)\b",
    # "no record", "no idea", "not enough information"
    rf"\b(?:no|not enough|insufficient|lack of|without(?: any)?|not any|{_NT} any|nothing|zero) "
    r"(?:relevant |such |more |further |additional |specific |clear |prior |previous |stored |saved )?"
    r"(?:information|info|record|records|mention|mentions|data|details|context|indication|evidence|memory"
    r"|recollection|idea|clue|knowledge|way (?:to|of) (?:know|tell|knowing|telling))\b",
    # "was never mentioned", "isn't available", "that's not something I know", "I wasn't told", "is not enough"
    rf"\b(?:not|never|{_NT})(?: yet| ever)?(?: been)? (?:mentioned|specified|stated|provided|discussed|shared|given"
    r"|recorded|said|told|covered|brought up|addressed|disclosed|revealed|available|known|clear|something"
    r"|recommended|suggested|talked|indicated|enough|sufficient|come up)\b",
    # "cannot be determined", "I'm not able to answer that", "hard to say"
    r"\b(?:can ?not|can'?t|unable to|not able to|could not|couldn'?t|impossible to|no way to|hard to|difficult to) "
    r"(?:be |really |actually )?(?:determine|determined|answer|answered|find|tell|say|know|known|confirm|verify"
    r"|recall|remember|provide|give)\b",
    r"\b(?:unanswerable|unknown|unclear|unspecified|undetermined|undisclosed)\b",
    # "you haven't mentioned that", "you never told me", "you've not said"
    rf"\byou(?:'?ve| have| had| did| do)? (?:not|never) {_ADV}{_SAID}\b",
    rf"\byou (?:haven'?t|hadn'?t|didn'?t|don'?t) {_ADV}{_SAID}\b",
    # "there's nothing about that", "I have nothing on that"
    r"\b(?:there is|there'?s|there was|i have|i'?ve|i see|i find|i found|i can find|i can see|we have|i'?ve got) nothing\b",
    r"\bnothing (?:in|about|on|from|that|to (?:suggest|indicate|say)|i|we|you|was|has|is|here|there)\b",
    r"\bi(?:'?m| am) not aware\b",
    # "the history does not mention that", "it hasn't come up", "I didn't recommend one"
    rf"\b(?:does not|doesn'?t|do not|don'?t|did not|didn'?t|has not|hasn'?t|have not|haven'?t) {_ADV}"
    r"(?:mention|recommend|suggest|say|tell|come up|contain|include|cover|record|show|indicate|specify|state|address"
    r"|reveal|appear)\b",
    r"\b(?:is|was|are|were) (?:insufficient|incomplete|missing|lacking|too limited)\b",
    r"\b(?:beats me|your guess is as good as mine)\b",
]), re.I)
# Every pattern above needs one of these tokens; a response without any of them is not searched.
_DECLINE_WORDS = frozenset("""
not never no nothing dont didnt doesnt cant cannot couldnt isnt wasnt arent werent hasnt havent hadnt wont wouldnt
unable impossible hard difficult insufficient lack without zero unanswerable unknown unclear unspecified undetermined
undisclosed incomplete missing lacking limited beats guess
""".split())
# "... but not your hamster": after a report of what was said, the rest of a decline.
_BUT_NOT = re.compile(r"\W*but (?:you |i |it |that |there )?(?:did not|didn'?t|not|never|have not|haven'?t|nothing|no)\b", re.I)
# A clause that reports what the user said rather than answering: "You mentioned ...", "I only know ...".
_REPORT = re.compile(
    r"\byou(?:'?ve| have| had| did)? (?:(?:only|also|just|once|previously|earlier) )*"
    r"(?:mention|mentioned|said|told me|talked about|spoke about|brought up|noted|shared|described|discussed"
    r"|referred to|wrote|stated)\b"
    r"|\bi (?:only|just) (?:know|see|have|recall|remember|found)\b",
    re.I,
)
_TURNS = frozenset({"however", "though", "although", "yet", "so", "while", "whereas", "instead", "actually"})
# Phrases that introduce a guess.  What follows one is an offered answer unless it is empty of content.
_OFFER = re.compile(
    r"\b(?:best guess|my guess|i(?:'?d| would)? guess|if i had to (?:guess|pick|choose|say|bet)|i(?:'?d| would) say"
    r"|i(?:'?d| would) go with|my bet|probably|maybe|perhaps|possibly|presumably|likely|(?:might|could|may) be"
    r"|(?:might|could|may) have been|i think|i believe|i suppose|i assume|i suspect|i reckon)\b",
    re.I,
)
_QUOTED = re.compile(r"[\"“”«»]([^\"“”«»]{1,120})[\"“”«»]|(?<![\w])'([^']{1,120})'(?![\w])")
_CAPITALISED = re.compile(r"(?<![\w'])[^\W\d_][\w'’-]*")
# Month and weekday names are dates.  "may" and "march" are also verbs, so those two need their capital.
_CALENDAR = re.compile(
    r"\b(?:january|february|april|june|july|august|september|october|november|december|monday|tuesday|wednesday"
    r"|thursday|friday|saturday|sunday)s?\b",
    re.I,
)
_MAY_OR_MARCH = re.compile(r"\b(?:May|March)\b(?! (?:I|we|you|be|have|not)\b)")


@lru_cache(maxsize=_RECENT)
def question_words(question: str) -> tuple[frozenset, frozenset, frozenset]:
    """What a question is about: its content stems, all of its stems, and the numbers it mentions."""
    folded = [PERSPECTIVE.get(t, t) for t in tokens(question)]
    return (
        frozenset(stem(t) for t in folded if t not in FUNCTION_WORDS),
        frozenset(stem(t) for t in folded),
        frozenset(numbers_in(question)),
    )


def is_bare_phrase(own: Sequence[str], asked: frozenset = frozenset()) -> bool:
    """True if the tokens ``own`` are a phrase standing alone, as an answer or a list item does.

    Rule.  "pastry chef", "Chile" and "the brass sparrow" are bare phrases.  A
    clause with a subject of its own ("I could guess", "it helps"), one that
    opens the way a remark or a lead-in does ("Sorry", "As far as I can tell",
    "Hope that helps"), and one made only of function words or of the
    question's own words are not.
    """
    if not own or own[0] in OPENERS or any(t in SUBJECTS for t in own):
        return False
    return any(t not in FUNCTION_WORDS and not t.isdigit() and stem(t) not in asked for t in own)


class Reading:
    """A response as a careful grader reads it: sentence by sentence, clause by clause.

    ``hay`` holds the perspective-folded tokens ("your" read as "my"), ``stems``
    their singular forms, and every token knows its clause.  A negation, a
    "previously" or a "rather than" reaches no further than the clause it
    stands in, and a decline is told apart from what is said around it.
    """

    __slots__ = ("plain", "clauses", "hay", "stems", "clause_of", "_starts", "_declines", "_about", "_memo")

    def __init__(self, text: str):
        plain = _spaced(text) if len(text) <= MAX_RESPONSE_CHARS else ""
        self.plain = plain
        clauses: list[Clause] = []
        hay: list[str] = []
        clause_of: list[int] = []
        fold = PERSPECTIVE.get
        buts = [m.start() for m in _BUT.finditer(plain)] if "but" in plain.lower() else ()
        sentence, pos, before = 0, 0, "^"
        breaks = [(m.start(), m.end(), m.lastgroup, m.group()) for m in _BREAK.finditer(plain)]
        breaks.append((len(plain), len(plain), "s", "$"))
        for stop, resume, kind, mark in breaks:
            after = "?" if "?" in mark else (mark.strip()[:1] or "\n")
            # the word "but" opens a new clause, unless nothing stands before it
            cuts = [at for at in buts if pos < at < stop] if buts else ()
            start = pos
            for end in (*cuts, stop):
                found = tokens(plain[start:end])
                if not found:
                    continue
                n = len(hay)
                clause_of.extend([len(clauses)] * len(found))
                clauses.append(Clause(start, end, n, n + len(found), sentence,
                                      "b" if found[0] == "but" else before, after if end == stop else "b"))
                hay.extend([fold(t, t) for t in found])
                start, before = end, "b"
            if kind == "m":
                before = mark.strip()[:1]
            else:
                before = "^"
                sentence += 1
            pos = resume
        self.clauses = clauses
        self.hay = tuple(hay)
        self.stems = tuple(map(stem, hay))
        self.clause_of = clause_of
        self._starts = [c.start for c in clauses]
        self._declines = None
        self._about = None
        self._memo: dict = {}

    # -- positions

    def clause_at(self, offset: int) -> int:
        """Index of the clause that holds the character at ``offset``."""
        return max(0, bisect_right(self._starts, offset) - 1)

    def token_at(self, offset: int) -> int:
        """Index of the token at (or first after) the character at ``offset``."""
        if not self.clauses:
            return 0
        clause = self.clauses[self.clause_at(offset)]
        return min(clause.t1, clause.t0 + len(tokens(self.plain[clause.start:max(clause.start, offset)])))

    def text_of(self, index: int) -> str:
        clause = self.clauses[index]
        return self.plain[clause.start:clause.end]

    def sentence_spans(self) -> list[tuple[int, int]]:
        """(start, end) characters of every sentence, each with the punctuation that closes it."""
        starts: list[int] = []
        last = None
        for clause in self.clauses:
            if clause.sentence != last:
                starts.append(clause.start if starts else 0)
                last = clause.sentence
        return list(zip(starts, starts[1:] + [len(self.plain)]))

    # -- declining

    def declines(self) -> tuple[tuple[int, int, int], ...]:
        """Every phrase that declines to answer, as (clause index, first character, last character + 1)."""
        if self._declines is None:
            if _DECLINE_WORDS.isdisjoint(self.hay):
                self._declines = ()
                return self._declines
            found = [(self.clause_at(m.start()), m.start(), m.end()) for m in _DECLINE.finditer(self.plain)]
            if any(c.before == "b" for c in self.clauses):
                reports = self.reports()
                taken = {i for i, _a, _b in found}
                for i, clause in enumerate(self.clauses):
                    if clause.before != "b" or i in taken:
                        continue
                    m = _BUT_NOT.match(self.plain, clause.start, clause.end)
                    if m and any(self.clauses[r].sentence == clause.sentence and r < i for r in reports):
                        found.append((i, m.start(), m.end()))
                found.sort()
            self._declines = tuple(found)
        return self._declines

    def reports(self) -> tuple[int, ...]:
        """Clauses that open a report of what the user said ("You mentioned ...", "I only know ...")."""
        hit = self._memo.get("reports")
        if hit is None:
            found = []
            for i, clause in enumerate(self.clauses):
                m = _REPORT.search(self.plain, clause.start, clause.end)
                if m and all(t in FUNCTION_WORDS or t in OPENERS for t in tokens(self.plain[clause.start:m.start()])):
                    found.append(i)
            hit = self._memo["reports"] = tuple(found)
        return hit

    def decline_about(self, question: str = "") -> tuple[bool, ...]:
        """For each decline: True if it is about the question, False if about something else.

        Rule.  "You didn't mention anything else about it" and "You never said
        which neighbourhood" decline something other than what was asked; they
        are remarks, and an answer next to them is still an answer.  A decline
        is about something else when it says so ("else", "other", "more",
        "beyond"), when it names a subject of its own that shares no word with
        the question, or when all it names is a question word the question
        does not ask ("you never said why").  A decline that names nothing ("I
        don't know", "you never told me that") is about the question.
        """
        if self._about is None or self._about[0] != question:
            about_stems = question_words(question)[0] if question else frozenset()
            flags = []
            for index, start, end in self.declines():
                clause = self.clauses[index]
                own = [PERSPECTIVE.get(t, t) for t in tokens(self.plain[clause.start:start]) + tokens(self.plain[end:clause.end])]
                if any(t in _ELSE for t in own):
                    flags.append(False)
                    continue
                content = [stem(t) for t in own if t not in FUNCTION_WORDS and not t.isdigit()]
                if not content and about_stems:
                    # "you never said why": a question word the question does not ask
                    asked = question_words(question)[1]
                    if any(t in _WH and t not in asked for t in own):
                        flags.append(False)
                        continue
                flags.append(not content or not about_stems or any(s in about_stems for s in content))
            self._about = (question, tuple(flags))
        return self._about[1]

    def context(self, question: str = "") -> frozenset:
        """Clauses that only report other information next to a decline about the question.

        "You mentioned your friend Dana has a sister, but you never told me
        your own sister's name": the first clause gives other information, the
        second declines.  What such a report names is neither an answer offered
        on an unanswerable question nor an answer given on an answerable one.
        A report counts as context only when the response declines the
        question itself, and it runs to the end of its sentence or to the next
        "but".
        """
        key = ("context", question)
        hit = self._memo.get(key)
        if hit is None:
            found: set[int] = set()
            if any(self.decline_about(question)):
                declining = {i for i, _a, _b in self.declines()}
                for r in self.reports():
                    if r in declining:
                        continue
                    found.add(r)
                    for k in range(r + 1, len(self.clauses)):
                        clause = self.clauses[k]
                        if (clause.sentence != self.clauses[r].sentence or clause.before == "b" or k in declining
                                or self.hay[clause.t0] in _TURNS):
                            break
                        found.add(k)
            hit = self._memo[key] = frozenset(found)
        return hit

    def offers(self, question: str = "") -> bool:
        """True if, outside its declines, the response states something that could be an answer.

        Rule.  A response to an unanswerable question is a correct decline only
        if it declines and asserts no answer.  After the declining clauses and
        the reports that go with them are set aside, what remains offers an
        answer when it holds

        * a number or a date that the question does not contain,
        * a quoted span that is not a quotation from the question,
        * a capitalised name that is not part of the question,
        * a guess ("maybe", "my best guess is", "I think it's") followed by
          anything of substance, or
        * a bare phrase standing where an answer would stand ("pastry chef. I
          don't know.").

        A hedge word alone disqualifies nothing: "I could guess, but I'd rather
        not" offers no answer.
        """
        key = ("offers", question)
        hit = self._memo.get(key)
        if hit is None:
            skip = {i for i, _a, _b in self.declines()} | self.context(question)
            hit = self._memo[key] = any(
                i not in skip and self.states_a_candidate(i, question) for i in range(len(self.clauses))
            )
        return hit

    def hedges(self) -> Sequence[int]:
        """Offsets of the hedging phrases in the response."""
        hit = self._memo.get("hedges")
        if hit is None:
            hit = self._memo["hedges"] = () if _HEDGE_WORDS.isdisjoint(self.hay) else hedges_in(self.plain)
        return hit

    def noncommittal(self) -> int:
        """Offset at which the response starts offering alternatives ("either ... or", "it could be A, B or C"), or -1."""
        hit = self._memo.get("noncommittal")
        if hit is None:
            hit = self._memo["noncommittal"] = noncommittal_at(self.plain) if ("either" in self.hay or "be" in self.hay) else -1
        return hit

    def states_a_candidate(self, index: int, question: str = "") -> bool:
        """True if clause ``index`` holds something that could be an answer (the rule is given under ``offers``)."""
        clause = self.clauses[index]
        text = self.plain[clause.start:clause.end]
        own = self.hay[clause.t0:clause.t1]
        about, asked, asked_numbers = question_words(question) if question else (frozenset(), frozenset(), frozenset())
        # a number or a date
        if _DATE_OR_TIME.search(text):
            return True
        for m in list(_CALENDAR.finditer(text)) + list(_MAY_OR_MARCH.finditer(text)):
            if stem(m.group(0).lower()) not in asked:
                return True
        if own in (("may",), ("march",)) and own[0] not in asked:
            return True  # nothing but the name of a month
        if any(m.value not in asked_numbers for m in number_mentions(text)):
            return True
        # a quoted span
        for m in _QUOTED.finditer(text):
            if any(stem(PERSPECTIVE.get(t, t)) not in asked for t in tokens(m.group(1) or m.group(2))):
                return True
        # a capitalised name
        for m in _CAPITALISED.finditer(text):
            word = m.group(0)
            if not word[0].isupper():
                continue
            folded = [PERSPECTIVE.get(t, t) for t in tokens(word)]
            if not folded:
                continue  # an article
            if all(t in FUNCTION_WORDS or t in SUBJECTS or t in OPENERS or stem(t) in asked for t in folded):
                continue
            return True
        content = [t for t in own if t not in FUNCTION_WORDS and not t.isdigit() and stem(t) not in asked]
        if not content:
            return False
        # a guess with something in it
        m = _OFFER.search(text)
        if m and any(t not in FUNCTION_WORDS and not t.isdigit() and stem(t) not in asked
                     for t in (PERSPECTIVE.get(t, t) for t in tokens(text[m.end():]))):
            return True
        # a bare phrase
        return is_bare_phrase(own, asked)


@lru_cache(maxsize=_RECENT)
def read(text: str) -> Reading:
    """The reading of ``text``, shared by every scorer that grades the same response."""
    return Reading(text)


# ------------------------------------------------------------------ abstention

def is_abstention(text: str) -> bool:
    """True if the response declines to answer for lack of information.

    An empty response is *not* an abstention: "no output" and "I don't know"
    are different outcomes and must not be conflated.  Neither is a response
    too long to be read (``MAX_RESPONSE_CHARS``).
    """
    if not text or len(text) > MAX_RESPONSE_CHARS or not text.strip():
        return False
    return bool(read(text).declines())


def is_clean_abstention(text: str, question: str = "") -> bool:
    """True if the response declines *and does not offer an answer alongside*.

    "I don't know, but my guess is X", "X. I don't know." and "X (you never
    mentioned anything else)" all give an answer.  Counting them as declining
    would let one template collect credit on answerable and unanswerable
    questions alike.  What disqualifies is the answer, not a hedge word: "I
    don't know, and I think it's better not to speculate" is a clean decline.
    ``question`` lets names and numbers that the question itself contains be
    told from answers; see ``Reading.offers`` for the rule.
    """
    return is_abstention(text) and not read(text).offers(question)


# --------------------------------------------------------------------- hedging

# Words by which a response withholds commitment.  The list is a vocabulary, not a grammar: it is
# matched anywhere outside the answer itself, except that "could be" / "might be" before a word of
# praise ("that could be useful for ...") is a remark about the answer, not a doubt about it.
_HEDGE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|presumably|supposedly|apparently|not (?:\w+ )?(?:sure|certain)|unsure"
    r"|uncertain|i think|i believe|i guess|i suppose|i assume|i suspect|i reckon|i(?:'?m| am) guessing"
    r"|(?:my|a|best|wild|rough) guess|i(?:'?d| would) (?:say|guess)|i want to say"
    r"|(?:might|could|may) (?:well |also |possibly )?(?:be|have)|either(?! way\b)|or maybe"
    r"|if i (?:recall|remember)|if memory serves|as far as i (?:know|recall|remember)"
    r"|to the best of my (?:knowledge|recollection|memory)|don'?t quote me|likely|seems?|unless"
    r"|can'?t be (?:sure|certain)|not positive)\b",
    re.I,
)
# Every phrase above holds one of these tokens.
_HEDGE_WORDS = frozenset("""
maybe perhaps possibly probably presumably supposedly apparently sure certain unsure uncertain think believe guess
suppose assume suspect reckon guessing say might could may either recall remember serves know knowledge recollection
memory quote likely seem seems unless positive
""".split())
_EVALUATIVE = re.compile(
    r" (?:(?:very|quite|really|also|particularly|especially|more|most|a|an|the|of) )*"
    r"(?:useful|helpful|handy|relevant|important|worth|worthwhile|good|great|nice|valuable|interesting|fun"
    r"|convenient|use|interest|help|something|why|because|due|used|able|wise|best|better|easier|easy|hard|harder"
    r"|difficult|time)\b",
    re.I,
)
_EITHER = re.compile(r"\beither\b(?! way\b)", re.I)
_OR = re.compile(r"\bor\b", re.I)
_MODAL_BE = re.compile(r"\b(?:could|might|may) be\b", re.I)


def hedges_in(plain: str) -> list[int]:
    """Offsets of the hedging phrases in ``plain`` (text as ``_spaced`` returns it)."""
    found = []
    for m in _HEDGE.finditer(plain):
        if m.group(0)[-3:].lower() == " be" and _EVALUATIVE.match(plain, m.end()):
            continue
        found.append(m.start())
    return found


def is_hedged(text: str) -> bool:
    """True if the response hedges ("I think", "probably", "my guess is", "unless", ...)."""
    if not text or len(text) > MAX_RESPONSE_CHARS:
        return False
    return bool(hedges_in(_spaced(text)))


def noncommittal_at(plain: str) -> int:
    """Offset at which ``plain`` starts offering alternatives instead of an answer, or -1.

    Two linear steps each: the first "either" and one search for a later "or";
    the first "could / might / may be" that is not a remark ("that could be
    useful for planning, reminders, or recommendations") and one look at what
    follows it.
    """
    m = _EITHER.search(plain)
    if m and _OR.search(plain, m.end() + 1):
        return m.start()
    for m in _MODAL_BE.finditer(plain):
        if _EVALUATIVE.match(plain, m.end()):
            continue
        if plain.count(",", m.end()) >= 2 or _OR.search(plain, m.end()):
            return m.start()
        break
    return -1


def is_noncommittal(text: str) -> bool:
    """True if the response offers several alternatives instead of an answer.

    "Either X or Y" and "it could be A, B or C" name the right answer only by
    accident; no careful grader counts them as answering the question.  A
    response too long to be read commits to nothing either.
    """
    if not text:
        return False
    if len(text) > MAX_RESPONSE_CHARS:
        return True
    return noncommittal_at(_spaced(text)) >= 0


# ------------------------------------------------------------------------ echo

def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of ``a`` and ``b``.

    Bit-parallel (Hyyrö 2004): one big-integer step per token of ``b``, so a
    very long response costs little when ``a`` is a question.
    """
    masks: dict[str, int] = {}
    for i, tok in enumerate(a):
        masks[tok] = masks.get(tok, 0) | (1 << i)
    full = (1 << len(a)) - 1
    state = full
    for tok in b:
        mask = masks.get(tok)
        if mask:
            shared = state & mask
            state = ((state + shared) | (state - shared)) & full
    return len(a) - state.bit_count()


@lru_cache(maxsize=_RECENT)
def is_echo(question: str, response: str) -> bool:
    """True if the response restates the question, whatever else it adds.

    Rule: a response is an echo when the longest common subsequence of the two
    token sequences covers at least 80% of the question's tokens.  Some
    questions contain their own answer as one of several options ("which came
    first, A or B?"), so the question restated must never earn credit, and
    padding the restatement with other words must not rescue it.
    """
    q, r = tokens(question), tokens(response)
    if not q or not r:
        return False
    return 5 * _lcs_length(q, r) >= 4 * len(q)


def without_restated_question(question: str, response: str) -> str:
    """``response`` without the sentences that restate the question; "" if nothing else is left.

    Rule.  "Which budgeting app did you recommend to me? LedgerLeaf." restates
    the question and then answers it.  The restatement earns nothing, and what
    follows it is judged as if it stood alone.  A response in which no single
    sentence is the restatement (the question spread over several sentences, or
    padded) is left an echo.
    """
    reading = read(response)
    spans = reading.sentence_spans()
    kept = [reading.plain[a:b].strip() for a, b in spans if not is_echo(question, reading.plain[a:b])]
    if len(kept) == len(spans):
        return ""
    rest = " ".join(part for part in kept if part)
    return "" if is_echo(question, rest) else rest
