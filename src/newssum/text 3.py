"""Text utilities: sentence splitting, normalisation and BBC summary repair."""

from __future__ import annotations

import re

# Abbreviations that end with a full stop but do not end a sentence.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "rev", "gen", "col", "lt", "sgt",
    "capt", "gov", "sen", "rep", "no", "vs", "etc", "inc", "ltd", "co", "corp", "plc",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "u.n", "e.g", "i.e", "approx", "dept", "est", "fig",
}

_SPLIT = re.compile(r"(?<=[.!?])[\"'”’]?\s+(?=[\"'“‘(\[]?[A-Z0-9£$])")
_WORD = re.compile(r"[A-Za-z0-9£$%']+")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def split_sentences(text: str) -> list[str]:
    """Split English news text into sentences.

    Paragraph breaks always end a sentence. Inside a paragraph the text is split
    after ``.``, ``!`` or ``?`` followed by whitespace and a capital letter or
    digit, unless the full stop belongs to a known abbreviation or an initial.
    """
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        start = 0
        for m in _SPLIT.finditer(para):
            prefix = para[start:m.start()].rstrip()
            last = prefix.rsplit(None, 1)[-1] if prefix.split() else ""
            token = last.rstrip(".!?\"'”’").lower()
            if last.endswith(".") and (token in _ABBREV or re.fullmatch(r"[a-z]", token)):
                continue
            piece = para[start:m.end()].strip()
            if piece:
                sentences.append(piece)
            start = m.end()
        tail = para[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def repair_bbc_summary(summary: str) -> str:
    """Insert the missing space between sentences in BBC News Summary references.

    The reference files concatenate extracted sentences without a space, for
    example ``"...to $11.1bn.For the full year..."``. A space is inserted after
    sentence-final punctuation when it is directly followed by a capital letter
    and preceded by a lower-case letter, digit, ``%`` or ``)``. Decimals such as
    ``11.1`` and initialisms such as ``U.S.`` are left alone.
    """
    s = summary.strip()
    s = re.sub(r"(?<=[a-z0-9%)£])([.!?])(?=[A-Z£])", r"\1 ", s)
    s = re.sub(r"(?<=[.!?])([\"'”’])(?=[A-Z])", r"\1 ", s)
    s = re.sub(r"(?<=[a-z0-9%)])([.!?])([\"“])(?=[A-Z])", r"\1 \2", s)
    return re.sub(r"[ \t]+", " ", s)


def normalise(text: str) -> str:
    """Lower-case, keep only letters and digits, collapse spaces (for matching)."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def n_words(text: str) -> int:
    return len(words(text))


def tokens_lower(text: str) -> list[str]:
    return [w.lower() for w in words(text)]


def numbers(text: str) -> set[str]:
    """Numbers in a text, with thousands separators removed (``1,200`` -> ``1200``)."""
    return {m.replace(",", "") for m in _NUM.findall(text)}


def lsum_format(text: str) -> str:
    """One sentence per line, as required by ROUGE-Lsum."""
    return "\n".join(split_sentences(text))
