"""Sentence splitting tuned for SEC filings.

A naive split on ". " cuts "Mr. Hancock has pledged ..." into "Mr." and "Hancock has pledged ...",
which loses the holder's name. We only split after a full stop or semicolon when the word
before it is not an honorific, initial or corporate abbreviation."""
import re

KEY = re.compile(r"pledg|margin loan|margin account", re.I)
NO_SPLIT_AFTER = {
    "mr", "ms", "mrs", "dr", "messrs", "jr", "sr", "st", "no", "nos", "inc", "co", "corp", "ltd",
    "llc", "lp", "l.p", "n.a", "u.s", "s.a", "n.v", "plc", "vs", "e.g", "i.e", "approx", "sec",
}
CANDIDATE = re.compile(r"[.;]\s+(?=[A-Z(])")
PREV_WORD = re.compile(r"([A-Za-z.]+)$")


def split_sentences(text: str) -> list[str]:
    out, start = [], 0
    for m in CANDIDATE.finditer(text):
        before = text[start:m.start()]
        w = PREV_WORD.search(before)
        word = (w.group(1) if w else "").lower().rstrip(".")
        if text[m.start()] == "." and (word in NO_SPLIT_AFTER or len(word) == 1):
            continue  # "Mr.", "J.", "Inc." ... keep going
        out.append(text[start:m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def pledge_sentences(text: str, max_len: int = 1500, max_n: int = 60) -> list[str]:
    return [s[:max_len] for s in split_sentences(text) if KEY.search(s)][:max_n]
