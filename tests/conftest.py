"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

RESEARCH_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def research_root() -> Path:
    return RESEARCH_ROOT


@pytest.fixture(scope="session")
def train_csv_path() -> Path:
    return RESEARCH_ROOT / "data" / "train.csv"


@pytest.fixture(scope="session")
def sample_submission_path() -> Path:
    return RESEARCH_ROOT / "data" / "sample_submission.csv"
