"""Writes the competition submission CSV, validated against
`sample_submission.csv`'s exact column names and order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def write_submission(pred_df: pd.DataFrame, sample_submission_path: str | Path, out_path: str | Path) -> None:
    """`pred_df` must have a `StudyInstanceUID` column plus the 12 label
    columns (any order). Output columns/order exactly match
    `sample_submission_path`. Any `StudyInstanceUID` present in the sample
    submission but missing from `pred_df` is filled with 0.5 across every
    label and logged as a warning -- never a crash."""
    sample = pd.read_csv(sample_submission_path)
    columns = list(sample.columns)
    label_columns = columns[1:]

    missing_from_pred = set(sample["StudyInstanceUID"]) - set(pred_df["StudyInstanceUID"])
    if missing_from_pred:
        logger.warning(
            "%d/%d studies in %s have no prediction -- filling with 0.5 for every label: %s",
            len(missing_from_pred), len(sample), sample_submission_path,
            sorted(missing_from_pred)[:5],
        )

    pred_indexed = pred_df.set_index("StudyInstanceUID")
    rows = []
    for study_uid in sample["StudyInstanceUID"]:
        if study_uid in pred_indexed.index:
            row = pred_indexed.loc[study_uid]
            rows.append([study_uid] + [float(row[col]) for col in label_columns])
        else:
            rows.append([study_uid] + [0.5] * len(label_columns))

    out_df = pd.DataFrame(rows, columns=columns)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote submission for %d studies to %s", len(out_df), out_path)
