"""Unit tests for src/reports/negation.py."""

from __future__ import annotations

from src.reports.label_mapping import find_target_mentions
from src.reports.negation import analyze_scope
from src.reports.weak_labels import normalize_report_text, split_sentences


def _status_for(sentence: str, target: str) -> list[str]:
    normalized = normalize_report_text(sentence)
    mentions = find_target_mentions(normalized)
    return [analyze_scope(normalized, m.start(), m.end()).status for m in mentions[target]]


def test_affirmed_mention():
    assert _status_for("There is a tear of the ACL.", "ACL") == ["affirmed"]


def test_pre_negation():
    assert _status_for("No effusion is seen.", "Effusion") == ["negated"]


def test_post_negation():
    assert _status_for("Fracture is ruled out.", "Fracture") == ["negated"]


def test_uncertainty():
    statuses = _status_for("Possible ACL tear is noted.", "ACL")
    assert statuses == ["uncertain"]


def test_historical_finding():
    statuses = _status_for("History of fracture of the tibia.", "Fracture")
    assert statuses == ["historical"]


def test_negation_beats_historical_when_both_present():
    # "no history of X" means X is absent, not merely an old finding.
    statuses = _status_for("No history of fracture.", "Fracture")
    assert statuses == ["negated"]


def test_negation_does_not_bleed_across_sentences():
    text = "No ACL tear. Effusion is present."
    for sentence in split_sentences(normalize_report_text(text)):
        mentions = find_target_mentions(sentence)
        for m in mentions["Effusion"]:
            assert analyze_scope(sentence, m.start(), m.end()).status == "affirmed"
        for m in mentions["ACL"]:
            assert analyze_scope(sentence, m.start(), m.end()).status == "negated"


def test_scope_terminator_limits_backward_window():
    # "but" should stop the negation from reaching across the clause boundary.
    statuses = _status_for("No joint effusion, but there is a medial meniscus tear.", "Medial Meniscus")
    assert statuses == ["affirmed"]
