"""Integrate population-genetic summaries with genomic model scores."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from popgenlm.orientation import orient_gpn_score

POPULATION_REQUIRED_COLUMNS = {"chrom", "pos", "ref", "alt", "af_alt"}
GPN_REQUIRED_COLUMNS = {"chrom", "pos", "ref", "alt", "gpn_score_ref_alt"}

POPULATION_KEY = ["chrom", "pos", "ref", "alt"]
GPN_MAPPED_KEY = ["vcf_chrom", "pos", "ref", "alt"]


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    missing = sorted(required - set(frame.columns))

    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")


def integrate_population_gpn_scores(
    population: pd.DataFrame,
    gpn_scores: pd.DataFrame,
    fasta_to_vcf: Mapping[str, str],
    *,
    window_size: int = 512,
) -> pd.DataFrame:
    """Join population summaries to raw REF→ALT GPN scores.

    Population variants use VCF chromosome labels, while GPN scores may use
    reference-FASTA record IDs. ``fasta_to_vcf`` explicitly defines that
    chromosome mapping.

    GPN scores are oriented using ALT allele frequency. When ALT is the major
    allele (AF_ALT > 0.5), the raw REF→ALT score is sign-flipped so that the
    resulting minor→major score follows the published Arabidopsis convention.

    ``edge_padded`` identifies variants whose centered sequence window extends
    beyond the left chromosome boundary. Detecting right-edge padding requires
    chromosome lengths and is therefore outside this function.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    _require_columns(
        population,
        POPULATION_REQUIRED_COLUMNS,
        "Population table",
    )
    _require_columns(
        gpn_scores,
        GPN_REQUIRED_COLUMNS,
        "GPN score table",
    )

    population = population.copy()
    gpn_scores = gpn_scores.copy()

    population["chrom"] = population["chrom"].astype(str)
    gpn_scores["chrom"] = gpn_scores["chrom"].astype(str)

    if population.duplicated(POPULATION_KEY).any():
        raise ValueError("Population table contains duplicate variant keys")

    mapping = {str(fasta_chrom): str(vcf_chrom) for fasta_chrom, vcf_chrom in fasta_to_vcf.items()}

    gpn_scores["vcf_chrom"] = gpn_scores["chrom"].map(mapping)

    if gpn_scores["vcf_chrom"].isna().any():
        unknown = sorted(
            gpn_scores.loc[
                gpn_scores["vcf_chrom"].isna(),
                "chrom",
            ].unique()
        )
        raise ValueError("Unrecognized GPN FASTA chromosome ID(s): " + ", ".join(unknown))

    if gpn_scores.duplicated(GPN_MAPPED_KEY).any():
        raise ValueError("GPN score table contains duplicate variant keys")

    numeric_scores = pd.to_numeric(
        gpn_scores["gpn_score_ref_alt"],
        errors="coerce",
    )

    if not np.isfinite(numeric_scores.to_numpy()).all():
        raise ValueError("GPN scores must be finite numeric values")

    gpn_scores["gpn_score_ref_alt"] = numeric_scores

    merged = population.merge(
        gpn_scores[
            [
                "vcf_chrom",
                "pos",
                "ref",
                "alt",
                "chrom",
                "gpn_score_ref_alt",
            ]
        ],
        left_on=POPULATION_KEY,
        right_on=GPN_MAPPED_KEY,
        how="left",
        validate="one_to_one",
        suffixes=("", "_gpn"),
        indicator=True,
    )

    unmatched = merged["_merge"] != "both"

    if unmatched.any():
        raise ValueError(f"{int(unmatched.sum())} population variant(s) lack a matching GPN score")

    orientation_rows = [
        orient_gpn_score(
            row.ref,
            row.alt,
            row.af_alt,
            row.gpn_score_ref_alt,
        )
        for row in merged.itertuples()
    ]

    orientation = pd.DataFrame(
        orientation_rows,
        index=merged.index,
    )

    if "maf" in merged.columns:
        observed_maf = pd.to_numeric(
            merged["maf"],
            errors="coerce",
        ).to_numpy()

        expected_maf = orientation["maf"].to_numpy(dtype=float)

        if not np.isfinite(observed_maf).all() or not np.allclose(
            observed_maf,
            expected_maf,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Population MAF is inconsistent with AF_ALT")
    else:
        merged["maf"] = orientation["maf"].to_numpy()

    for column in [
        "minor_allele",
        "major_allele",
        "orientation",
        "flipped",
        "gpn_score_paper_oriented",
        "gpn_score_minor_vs_major",
    ]:
        merged[column] = orientation[column].to_numpy()

    # GPN converts the one-based position to zero-based coordinates and centers
    # a window of window_size bases around it. For a 512-bp window, positions
    # 1–256 therefore require left-edge padding.
    merged["edge_padded"] = pd.to_numeric(merged["pos"]).to_numpy() <= window_size // 2

    merged = merged.rename(columns={"chrom_gpn": "fasta_chrom"})

    return merged.drop(columns=["vcf_chrom", "_merge"])
