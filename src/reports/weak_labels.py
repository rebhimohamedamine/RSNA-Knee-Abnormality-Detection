"""Orchestrates synonym mapping (`label_mapping.py`) and negation/uncertainty
scoping (`negation.py`) into a 12-dimensional weak-label vector plus a
matching confidence vector, per report.

Deliberately torch-free (numpy only) -- this module is pure text processing
and should be usable/testable without the heavier ML dependencies.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from src.reports.label_mapping import TARGETS, find_target_mentions
from src.reports.negation import analyze_scope

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"[.;\n]+")


def normalize_report_text(text: str) -> str:
    """Lowercase, strip diacritics (NFKD decompose + drop combining marks),
    and collapse whitespace. Applied uniformly so a Spanish "contusión" and
    an unaccented "contusion" both match the same synonym pattern, and so
    that positions returned by regex matches stay valid for `analyze_scope`."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split already-normalized text on '.', ';', and newlines. Sentence
    scoping is what keeps a negation trigger for one finding from bleeding
    into an unrelated finding mentioned later in the same report."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def extract_weak_labels(report_text: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Extract a weak label and confidence for each of the 12 targets from
    one report string.

    Returns `(labels, confidence)`, each `(12,)` float32, in `TARGETS` order:
      - no mention anywhere -> (0.0, 0.0): unknown, NOT a confident negative;
        callers should exclude it from any loss via the confidence value.
      - any affirmed mention -> (1.0, 1.0)
      - every mention negated -> (0.0, 1.0)
      - else any uncertain mention -> (1.0, 0.5)
      - else every mention historical -> (1.0, 0.3)
      - else (a mix of negated and historical mentions, no affirmed/uncertain)
        -> (1.0, 0.2): genuinely ambiguous, kept as the lowest-confidence
        positive rather than silently discarded or treated as a confident
        negative.
    """
    n = len(TARGETS)
    labels = np.zeros(n, dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)

    if not report_text or not isinstance(report_text, str):
        return labels, confidence

    normalized = normalize_report_text(report_text)
    sentences = split_sentences(normalized)

    statuses_per_target: dict[str, list[str]] = {t: [] for t in TARGETS}
    for sentence in sentences:
        mentions = find_target_mentions(sentence)
        for target, matches in mentions.items():
            for match in matches:
                result = analyze_scope(sentence, match.start(), match.end())
                statuses_per_target[target].append(result.status)

    for i, target in enumerate(TARGETS):
        statuses = statuses_per_target[target]
        if not statuses:
            continue  # stays (0.0, 0.0): unknown
        if "affirmed" in statuses:
            labels[i], confidence[i] = 1.0, 1.0
        elif all(s == "negated" for s in statuses):
            labels[i], confidence[i] = 0.0, 1.0
        elif "uncertain" in statuses:
            labels[i], confidence[i] = 1.0, 0.5
        elif all(s == "historical" for s in statuses):
            labels[i], confidence[i] = 1.0, 0.3
        else:
            labels[i], confidence[i] = 1.0, 0.2  # mixed negated/historical: ambiguous

    return labels, confidence


def batch_extract_weak_labels(report_texts: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized-in-name-only convenience wrapper: stacks per-report results
    into `(N, 12)` label and confidence arrays."""
    results = [extract_weak_labels(text) for text in report_texts]
    labels = np.stack([r[0] for r in results], axis=0) if results else np.zeros((0, len(TARGETS)), dtype=np.float32)
    confidence = np.stack([r[1] for r in results], axis=0) if results else np.zeros((0, len(TARGETS)), dtype=np.float32)
    return labels, confidence
