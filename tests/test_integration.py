from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from popgenlm.integration import integrate_population_gpn_scores

FASTA_TO_VCF = {"NC_1": "1"}


def population_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [100, 1000],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "af_alt": [0.10, 0.80],
            "maf": [0.10, 0.20],
        }
    )


def gpn_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chrom": ["NC_1", "NC_1"],
            "pos": [100, 1000],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "gpn_score_ref_alt": [-2.0, 3.0],
        }
    )


def test_integrates_and_orients_scores():
    result = integrate_population_gpn_scores(
        population_table(),
        gpn_table(),
        FASTA_TO_VCF,
        window_size=512,
    )

    assert len(result) == 2

    first = result.iloc[0]
    assert first["fasta_chrom"] == "NC_1"
    assert first["orientation"] == "alt_minor"
    assert bool(first["flipped"]) is False
    assert first["gpn_score_minor_vs_major"] == pytest.approx(-2.0)
    assert bool(first["edge_padded"]) is True

    second = result.iloc[1]
    assert second["orientation"] == "ref_minor"
    assert bool(second["flipped"]) is True
    assert second["gpn_score_minor_vs_major"] == pytest.approx(-3.0)
    assert bool(second["edge_padded"]) is False


def test_missing_population_column_fails():
    population = population_table().drop(columns=["af_alt"])

    with pytest.raises(
        ValueError,
        match="Population table is missing required columns",
    ):
        integrate_population_gpn_scores(
            population,
            gpn_table(),
            FASTA_TO_VCF,
        )


def test_duplicate_population_variant_fails():
    population = pd.concat(
        [population_table(), population_table().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate variant keys",
    ):
        integrate_population_gpn_scores(
            population,
            gpn_table(),
            FASTA_TO_VCF,
        )


def test_duplicate_gpn_variant_fails():
    scores = pd.concat(
        [gpn_table(), gpn_table().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate variant keys",
    ):
        integrate_population_gpn_scores(
            population_table(),
            scores,
            FASTA_TO_VCF,
        )


def test_unknown_fasta_chromosome_fails():
    scores = gpn_table()
    scores.loc[0, "chrom"] = "UNKNOWN"

    with pytest.raises(
        ValueError,
        match="Unrecognized GPN FASTA chromosome",
    ):
        integrate_population_gpn_scores(
            population_table(),
            scores,
            FASTA_TO_VCF,
        )


def test_missing_gpn_variant_fails():
    scores = gpn_table().iloc[[0]].copy()

    with pytest.raises(
        ValueError,
        match="lack a matching GPN score",
    ):
        integrate_population_gpn_scores(
            population_table(),
            scores,
            FASTA_TO_VCF,
        )


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf, "bad"])
def test_nonfinite_or_nonnumeric_gpn_score_fails(bad_score):
    scores = gpn_table()
    scores["gpn_score_ref_alt"] = scores["gpn_score_ref_alt"].astype(object)
    scores.loc[0, "gpn_score_ref_alt"] = bad_score

    with pytest.raises(
        ValueError,
        match="GPN scores must be finite numeric values",
    ):
        integrate_population_gpn_scores(
            population_table(),
            scores,
            FASTA_TO_VCF,
        )


def test_inconsistent_maf_fails():
    population = population_table()
    population.loc[0, "maf"] = 0.25

    with pytest.raises(
        ValueError,
        match="MAF is inconsistent",
    ):
        integrate_population_gpn_scores(
            population,
            gpn_table(),
            FASTA_TO_VCF,
        )


def test_nonpositive_window_size_fails():
    with pytest.raises(
        ValueError,
        match="window_size must be positive",
    ):
        integrate_population_gpn_scores(
            population_table(),
            gpn_table(),
            FASTA_TO_VCF,
            window_size=0,
        )


def test_existing_population_fasta_chrom_is_preserved_and_validated():
    population = population_table()
    population["fasta_chrom"] = ["NC_1", "NC_1"]

    result = integrate_population_gpn_scores(
        population,
        gpn_table(),
        FASTA_TO_VCF,
        window_size=512,
    )

    assert list(result.columns).count("fasta_chrom") == 1
    assert result["fasta_chrom"].tolist() == ["NC_1", "NC_1"]


def test_inconsistent_population_fasta_chrom_fails():
    population = population_table()
    population["fasta_chrom"] = ["WRONG", "NC_1"]

    with pytest.raises(
        ValueError,
        match="FASTA chromosome mapping is inconsistent",
    ):
        integrate_population_gpn_scores(
            population,
            gpn_table(),
            FASTA_TO_VCF,
            window_size=512,
        )
