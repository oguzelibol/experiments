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
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from functools import lru_cache
from typing import Sequence

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
    for raw in _NON_ALNUM.split(text):
        tok = raw.strip(".")
        if not tok:
            continue
        m = _TRAILING_ZERO.match(tok)
        if m:
            tok = m.group(1)
        tok = _NUMBER_WORDS.get(tok, tok)
        if tok in _ARTICLES:
            continue
        # a dot that survives inside a non-number token is punctuation
        if "." in tok and not re.fullmatch(r"\d+\.\d+", tok):
            out.extend(t for t in tok.split(".") if t and t not in _ARTICLES)
        else:
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

NEGATIONS = frozenset({"not", "never", "no", "isnt", "dont", "doesnt", "didnt", "wasnt", "arent", "neither", "nor"})
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
    do not live in Lisbon" does not say Lisbon; "Not Porto, but Lisbon" does.
    """
    window = list(preceding[-NEGATION_WINDOW:])
    if "but" in window:
        window = window[len(window) - window[::-1].index("but"):]
    return any(tok in NEGATIONS for tok in window)


# ---------------------------------------------------------------------- numbers

_NUM_OR_WORD = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w]|\.\d)|[A-Za-z]+")
# What may stand between the end of one sentence and the first word of the next.
_SENTENCE_LEADERS = " \t\r\n\"'“”‘’([{*_-•>#"


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


def _numbers(text: str) -> tuple[str, list[tuple[float, int]]]:
    """The cleaned text and every number in it as (value, offset), in order.

    Rule: a number word with a capital letter that is not explained by its
    position is part of a name, not a number.  "Seventeen Winters" is a book,
    and an answer that lists it has not said "17".
    """
    text = _THOUSANDS.sub("", clean(str(text)))
    found: list[tuple[float, int]] = []
    for m in _NUM_OR_WORD.finditer(text):
        tok = m.group(0)
        if tok[0].isalpha():
            digits = _NUMBER_WORDS.get(tok.lower())
            if digits is None:
                continue
            if tok[0].isupper() and tok[1:].islower() and not _starts_sentence(text, m.start(), m.end()):
                continue
            found.append((float(digits), m.start()))
        else:
            found.append((float(tok), m.start()))
    return text, found


def numbers_in(text: str) -> list[float]:
    """Numbers mentioned in ``text`` in order of appearance, including small number words."""
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


# ------------------------------------------------------------------ abstention

_ABSTAIN_PATTERNS = [
    r"\bi (do not|dont|don't|did not|didn't|cannot|can't|cant) (know|have|recall|remember|see|find)\b",
    r"\b(no|not enough|insufficient|lack of|without) (relevant )?(information|info|record|records|mention|data|details|context)\b",
    r"\b(not|never|wasn't|was not|isn't|is not|hasn't been|has not been|weren't) (been )?(mentioned|specified|stated|provided|discussed|shared|given|recorded|said)\b",
    r"\b(cannot|can't|cant|unable to|could not|couldn't|impossible to) (be )?(determine|determined|answer|answered|find|tell|say|know)\b",
    r"\bunanswerable\b",
    r"\bnot (available|known|clear from)\b",
    r"\byou (did not|didn't|never|haven't|have not) (mention|mentioned|tell|told|say|said|share|shared|specify|specified)\b",
    r"\b(there is|there's|there was|i have|i see) no (mention|record|information|indication|evidence)\b",
    r"\bno such (information|record|mention)\b",
    r"\bi'?m not aware\b",
    r"\bnever (mentioned|told|said|shared|recommended|suggested|discussed)\b",
    r"\b(did not|didn't) (mention|recommend|suggest|say|tell)\b",
]
_ABSTAIN_RE = re.compile("|".join(f"(?:{p})" for p in _ABSTAIN_PATTERNS), re.I)

_HEDGE_RE = re.compile(
    r"\b(maybe|perhaps|possibly|not sure|not certain|i think|i believe|i guess|might be|could be|"
    r"either|or maybe|probably|if i recall|not entirely sure)\b",
    re.I,
)


@lru_cache(maxsize=_RECENT)
def _plain(text: str) -> str:
    """What the phrase detectors read: cleaned text, one apostrophe form, single spaces.

    The detectors are regular expressions over words separated by one space, so
    an invisible character inside a word, a no-break space or a doubled space
    between two words would otherwise switch a detector off.
    """
    return " ".join(_APOSTROPHES.sub("'", clean(str(text))).split())


def is_abstention(text: str) -> bool:
    """True if the response declines to answer for lack of information.

    An empty response is *not* an abstention: "no output" and "I don't know"
    are different outcomes and must not be conflated.
    """
    if not text or not text.strip():
        return False
    return bool(_ABSTAIN_RE.search(_plain(text)))


_GUESS_AFTER_DECLINING = re.compile(
    r"\b(best guess|my guess|i(?:'d| would)? guess|probably|maybe|perhaps|might be|could be|i think|"
    r"i believe|likely|possibly|if i had to)\b",
    re.I,
)


def is_clean_abstention(text: str) -> bool:
    """True if the response declines *and does not then offer an answer anyway*.

    "I don't know, but my guess is X" is a guess.  Counting it as declining
    would let one template collect credit on answerable and unanswerable
    questions alike.
    """
    return is_abstention(text) and not _GUESS_AFTER_DECLINING.search(_plain(text))


def is_hedged(text: str) -> bool:
    return bool(_HEDGE_RE.search(_plain(text or "")))


_EITHER_OR = re.compile(r"\beither\b.+\bor\b", re.I | re.S)
_GUESS_LIST = re.compile(r"\b(could|might|may) be\b(?P<rest>.+)", re.I | re.S)


def is_noncommittal(text: str) -> bool:
    """True if the response offers several alternatives instead of an answer.

    "Either X or Y" and "it could be A, B or C" name the right answer only by
    accident; no careful grader counts them as answering the question.
    """
    if not text:
        return False
    text = _plain(text)
    if _EITHER_OR.search(text):
        return True
    m = _GUESS_LIST.search(text)
    if m:
        rest = m.group("rest")
        if rest.count(",") >= 2 or re.search(r"\bor\b", rest, re.I):
            return True
    return False


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
