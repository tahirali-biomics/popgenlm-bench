import numpy as np
import pandas as pd
import pytest

from popgenlm.validation import validate_score_table


def valid_table():
    return pd.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [100, 200],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "score": [-1.2, 0.4],
        }
    )


def test_valid_table_passes():
    result = validate_score_table(valid_table())

    assert len(result) == 2
    assert result["pos"].dtype.kind in {"i", "u"}
    assert result["score"].dtype.kind == "f"


def test_missing_column_fails():
    frame = valid_table().drop(columns=["score"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_score_table(frame)


def test_ref_equals_alt_fails():
    frame = valid_table()
    frame.loc[0, "alt"] = "A"

    with pytest.raises(ValueError, match="REF and ALT must differ"):
        validate_score_table(frame)


def test_noncanonical_ref_fails():
    frame = valid_table()
    frame.loc[0, "ref"] = "N"

    with pytest.raises(ValueError, match="REF must contain single canonical bases"):
        validate_score_table(frame)


def test_noncanonical_alt_fails():
    frame = valid_table()
    frame.loc[0, "alt"] = "N"

    with pytest.raises(ValueError, match="ALT must contain single canonical bases"):
        validate_score_table(frame)


def test_zero_position_fails():
    frame = valid_table()
    frame.loc[0, "pos"] = 0

    with pytest.raises(ValueError, match="Positions must be positive integers"):
        validate_score_table(frame)


def test_fractional_position_fails():
    frame = valid_table()
    frame["pos"] = frame["pos"].astype(float)
    frame.loc[0, "pos"] = 100.5

    with pytest.raises(ValueError, match="Positions must be positive integers"):
        validate_score_table(frame)


def test_duplicate_variant_fails():
    frame = pd.concat(
        [valid_table(), valid_table().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate variants detected"):
        validate_score_table(frame)


def test_nonfinite_score_fails():
    frame = valid_table()
    frame.loc[0, "score"] = np.nan

    with pytest.raises(ValueError, match="Scores must be finite"):
        validate_score_table(frame)
