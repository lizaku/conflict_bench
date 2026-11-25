"""Answer matching - one definition, used by every step that reads an answer.

A name matches a text when it occurs in it on word boundaries, after
lower-casing and stripping accents and punctuation.  Names shorter than
MIN_LEN characters are never used: the old substring matcher let the alias
"IN" match the word "in", and a two-letter alias is not worth that.
"""
import re
import unicodedata
from functools import lru_cache

MIN_LEN = 3


@lru_cache(maxsize=500_000)
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def norm(s) -> str:
    return _norm(str(s))


def names(answer, aliases=()) -> list:
    """Normalised, de-duplicated, usable names of one answer."""
    out = []
    for n in [answer, *aliases]:
        n = norm(n)
        if len(n.replace(" ", "")) >= MIN_LEN and n not in out:
            out.append(n)
    return out


def find(text, name) -> int:
    """Position of normalised `name` in `text` on word boundaries, or -1."""
    return f" {norm(text)} ".find(f" {name} ")


def has(text, names_) -> bool:
    t = f" {norm(text)} "
    return any(f" {n} " in t for n in names_)


def which(text, candidates: dict) -> list:
    """Keys of `candidates` ({key: names}) that occur in `text`."""
    t = f" {norm(text)} "
    return [k for k, ns in candidates.items() if any(f" {n} " in t for n in ns)]


def hit(text, target, candidates: dict) -> bool:
    """`text` gives `target` as the answer: it names the target and none of
    the other candidate answers.  An answer naming two candidates is not a
    hit for either."""
    return which(text, candidates) == [target]
