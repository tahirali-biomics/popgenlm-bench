from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_SCORE_COLUMNS = {"chrom", "pos", "ref", "alt", "score"}


def validate_score_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a genomic language-model variant score table."""

    missing = REQUIRED_SCORE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()

    positions = pd.to_numeric(result["pos"], errors="raise")
    scores = pd.to_numeric(result["score"], errors="raise")

    position_values = positions.to_numpy(dtype=float)
    if (
        not np.isfinite(position_values).all()
        or not (positions > 0).all()
        or not (positions % 1 == 0).all()
    ):
        raise ValueError("Positions must be positive integers")

    result["pos"] = positions.astype("int64")
    result["score"] = scores.astype(float)

    if not result["ref"].astype(str).str.fullmatch("[ACGT]").all():
        raise ValueError("REF must contain single canonical bases")

    if not result["alt"].astype(str).str.fullmatch("[ACGT]").all():
        raise ValueError("ALT must contain single canonical bases")

    if (result["ref"] == result["alt"]).any():
        raise ValueError("REF and ALT must differ")

    if not np.isfinite(result["score"].to_numpy(dtype=float)).all():
        raise ValueError("Scores must be finite")

    key = ["chrom", "pos", "ref", "alt"]
    if result.duplicated(key).any():
        raise ValueError("Duplicate variants detected")

    return result
