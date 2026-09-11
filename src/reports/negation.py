"""A small, from-scratch, NegEx-style negation/uncertainty/history scoper.

No new NLP dependency (stdlib `re` only), by design -- a full negation model
would be overkill for short radiology-report sentences, and the spec calls
for exactly this trigger-list + fixed-window simplification.

Callers are expected to have already split the report into sentences (on
`.`/`;`/newlines) before calling `analyze_scope`, so that a negation trigger
for one finding never bleeds into a different finding described later in the
same report. `analyze_scope` additionally stops its own scan at an in-sentence
clause boundary (e.g. "but", "however") for the same reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Triggers that negate the finding when they appear BEFORE the mention.
PRE_NEGATION_TRIGGERS = [
    r"\bno\b",
    r"\bwithout\b",
    r"absence of",
    r"\bdenies\b",
    r"negative for",
    r"rule(d)? out",
    r"r/o\b",
    r"\bsin\b",
    r"ausencia de",
    r"\bniet\b",
    r"\bgeen\b",
]

# Triggers that negate the finding when they appear AFTER the mention.
POST_NEGATION_TRIGGERS = [
    r"(is|was|were|are) (ruled out|excluded|negative)",
    r"\bresolved\b",
    r"no (longer )?(present|seen|identified)",
]

UNCERTAINTY_TRIGGERS = [
    r"\bpossible\b",
    r"\bprobable\b",
    r"cannot exclude",
    r"\bquestionable\b",
    r"suspect(ed)?",
    r"\bmay represent\b",
    r"\bposible\b",
    r"\bprobable\b",
    r"\bmogelijk\b",
]

HISTORICAL_TRIGGERS = [
    r"\bprior\b",
    r"\bprevious(ly)?\b",
    r"\bold\b",
    r"history of",
    r"\bremote\b",
    r"status post",
    r"\bantiguo\b",
    r"\bprevio\b",
    r"eerder(e)?\b",
]

# Word-level scope boundaries: a backward/forward scan stops here even if it
# hasn't yet covered SCOPE_WINDOW_TOKENS, so a trigger before "but"/"however"
# doesn't get attributed to a finding described after it.
SCOPE_TERMINATORS = [r"\bbut\b", r"\bexcept\b", r"\bhowever\b", r"\bpero\b"]

SCOPE_WINDOW_TOKENS = 6

_PRE_NEG_RE = [re.compile(p, re.IGNORECASE) for p in PRE_NEGATION_TRIGGERS]
_POST_NEG_RE = [re.compile(p, re.IGNORECASE) for p in POST_NEGATION_TRIGGERS]
_UNCERTAIN_RE = [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY_TRIGGERS]
_HISTORICAL_RE = [re.compile(p, re.IGNORECASE) for p in HISTORICAL_TRIGGERS]
_TERMINATOR_RE = [re.compile(p, re.IGNORECASE) for p in SCOPE_TERMINATORS]

_TOKEN_RE = re.compile(r"\S+")

STATUS_CONFIDENCE = {
    "affirmed": 1.0,
    "negated": 1.0,   # confident that the label is ABSENT, not confident it's present
    "uncertain": 0.5,
    "historical": 0.3,
}


@dataclass
class NegationResult:
    status: str        # "affirmed" | "negated" | "uncertain" | "historical"
    confidence: float  # how confident we are in `status`, in [0, 1]


def _tokens_with_offsets(sentence: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(sentence)]


def _window_before(sentence: str, tokens: list[tuple[str, int, int]], match_start: int) -> str:
    """Text of up to SCOPE_WINDOW_TOKENS tokens immediately before
    `match_start`, truncated at the nearest preceding scope terminator."""
    before = [t for t in tokens if t[2] <= match_start]
    window = before[-SCOPE_WINDOW_TOKENS:]
    for i in range(len(window) - 1, -1, -1):
        word = window[i][0]
        if any(term.search(word) for term in _TERMINATOR_RE):
            window = window[i + 1 :]
            break
    if not window:
        return ""
    return sentence[window[0][1] : window[-1][2]]


def _window_after(sentence: str, tokens: list[tuple[str, int, int]], match_end: int) -> str:
    """Text of up to SCOPE_WINDOW_TOKENS tokens immediately after
    `match_end`, truncated at the nearest following scope terminator."""
    after = [t for t in tokens if t[1] >= match_end]
    window = after[:SCOPE_WINDOW_TOKENS]
    for i, (word, _s, _e) in enumerate(window):
        if any(term.search(word) for term in _TERMINATOR_RE):
            window = window[:i]
            break
    if not window:
        return ""
    return sentence[window[0][1] : window[-1][2]]


def analyze_scope(sentence: str, match_start: int, match_end: int) -> NegationResult:
    """Classify one target mention (`sentence[match_start:match_end]`) as
    affirmed / negated / uncertain / historical, using the surrounding
    within-sentence context only.

    Priority when multiple trigger types are present: negation beats
    uncertainty beats historical -- e.g. "no history of fracture" is treated
    as `negated` (the finding is absent), not `historical`.
    """
    tokens = _tokens_with_offsets(sentence)
    pre = _window_before(sentence, tokens, match_start)
    post = _window_after(sentence, tokens, match_end)

    if any(p.search(pre) for p in _PRE_NEG_RE) or any(p.search(post) for p in _POST_NEG_RE):
        return NegationResult("negated", STATUS_CONFIDENCE["negated"])
    if any(p.search(pre) for p in _UNCERTAIN_RE):
        return NegationResult("uncertain", STATUS_CONFIDENCE["uncertain"])
    if any(p.search(pre) for p in _HISTORICAL_RE):
        return NegationResult("historical", STATUS_CONFIDENCE["historical"])
    return NegationResult("affirmed", STATUS_CONFIDENCE["affirmed"])
