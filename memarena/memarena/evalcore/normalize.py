"""Text normalisation shared by every deterministic scorer.

The rules follow the SQuAD convention (lower-case, drop punctuation and
articles, collapse whitespace) with three additions that matter for memory QA:
diacritics are folded ("Kraków" == "Krakow"), small number words are mapped to
digits ("three" == "3"), and trailing ".0" / thousands separators are removed.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

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
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w]|\.\d)")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


@lru_cache(maxsize=200_000)
def tokens(text: str) -> tuple[str, ...]:
    """Normalised tokens of ``text`` (cached; inputs repeat heavily in grading)."""
    text = _fold(str(text)).lower()
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


_NUM_OR_WORD = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w]|\.\d)|[a-z]+")


def numbers_in(text: str) -> list[float]:
    """Numbers mentioned in ``text`` in order of appearance, including small number words."""
    found: list[float] = []
    folded = _THOUSANDS.sub("", _fold(str(text)).lower())
    for m in _NUM_OR_WORD.finditer(folded):
        tok = m.group(0)
        if tok in _NUMBER_WORDS:
            found.append(float(_NUMBER_WORDS[tok]))
        elif tok[0].isdigit() or tok[0] == "-":
            try:
                found.append(float(tok))
            except ValueError:
                pass
    return found


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


def is_abstention(text: str) -> bool:
    """True if the response declines to answer for lack of information.

    An empty response is *not* an abstention: "no output" and "I don't know"
    are different outcomes and must not be conflated.
    """
    if not text or not text.strip():
        return False
    return bool(_ABSTAIN_RE.search(_APOSTROPHES.sub("'", text)))


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
    return is_abstention(text) and not _GUESS_AFTER_DECLINING.search(_APOSTROPHES.sub("'", text or ""))


def is_hedged(text: str) -> bool:
    return bool(_HEDGE_RE.search(text or ""))


_EITHER_OR = re.compile(r"\beither\b.+\bor\b", re.I | re.S)
_GUESS_LIST = re.compile(r"\b(could|might|may) be\b(?P<rest>.+)", re.I | re.S)


def is_noncommittal(text: str) -> bool:
    """True if the response offers several alternatives instead of an answer.

    "Either X or Y" and "it could be A, B or C" name the right answer only by
    accident; no careful grader counts them as answering the question.
    """
    if not text:
        return False
    if _EITHER_OR.search(text):
        return True
    m = _GUESS_LIST.search(text)
    if m:
        rest = m.group("rest")
        if rest.count(",") >= 2 or re.search(r"\bor\b", rest, re.I):
            return True
    return False


def is_echo(question: str, response: str) -> bool:
    """True if the response merely restates the question.

    Some questions contain their own answer as one of several options ("which
    came first, A or B?"); echoing the question must never earn credit.
    """
    q, r = set(tokens(question)), set(tokens(response))
    if not q or not r:
        return False
    return len(q & r) / len(q | r) >= 0.8
