#!/usr/bin/env python3
"""Analyze population frequency versus GPN scores for PopGenLM Bench v0.2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "benchmarks" / "v0.2" / "1001g_population_10000_gpn.tsv"

REPORT_DIR = ROOT / "reports" / "v0.2"

SUMMARY_PATH = REPORT_DIR / "population_gpn_analysis.json"
CLASS_PATH = REPORT_DIR / "frequency_class_summary.tsv"
PAIRWISE_PATH = REPORT_DIR / "frequency_class_pairwise.tsv"
SENSITIVITY_PATH = REPORT_DIR / "sensitivity_summary.tsv"
CHROMOSOME_PATH = REPORT_DIR / "chromosome_robustness.tsv"

EXPECTED_INPUT_SHA256 = "ef188e83d0e598d016c158b4daeec08be63e19723840fc42b494c76443fca469"

SCORE_COLUMN = "gpn_score_minor_vs_major"

SEED = 20260912
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 10000

FREQUENCY_ORDER = [
    "rare",
    "low_frequency",
    "common",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including correct handling of ties."""
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Calculate Spearman's rho from average ranks."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")

    if len(x) < 2:
        raise ValueError("At least two observations are required")

    rank_x = average_ranks(x)
    rank_y = average_ranks(y)

    rho = np.corrcoef(
        rank_x,
        rank_y,
    )[0, 1]

    return float(rho)


def chromosome_stratified_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    chromosome: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[float, float, int]:
    """Test Spearman association by permuting y within chromosomes.

    This preserves chromosome-specific score distributions but does not model
    local linkage disequilibrium. It is therefore an association diagnostic,
    not a test for natural selection.
    """
    rank_x = average_ranks(x)
    rank_y = average_ranks(y)

    x_centered = rank_x - rank_x.mean()
    y_centered = rank_y - rank_y.mean()

    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))

    observed = float(
        np.dot(
            x_centered,
            y_centered,
        )
        / denominator
    )

    strata = [np.flatnonzero(chromosome == chrom) for chrom in sorted(set(chromosome))]

    rng = np.random.default_rng(seed)

    extreme = 0

    for _ in range(n_permutations):
        permuted = y_centered.copy()

        for indices in strata:
            permuted[indices] = rng.permutation(y_centered[indices])

        null_rho = float(
            np.dot(
                x_centered,
                permuted,
            )
            / denominator
        )

        if abs(null_rho) >= abs(observed):
            extreme += 1

    p_value = (extreme + 1) / (n_permutations + 1)

    return observed, float(p_value), extreme


def rank_biserial(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """Return tie-aware rank-biserial effect size.

    Positive values mean group A tends to have larger scores than group B.
    The quantity is equivalent to Cliff's delta for independent samples.
    """
    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError("Both groups must contain observations")

    sorted_b = np.sort(group_b)

    b_less_than_a = np.searchsorted(
        sorted_b,
        group_a,
        side="left",
    )

    b_less_or_equal_a = np.searchsorted(
        sorted_b,
        group_a,
        side="right",
    )

    b_greater_than_a = len(group_b) - b_less_or_equal_a

    numerator = b_less_than_a.sum() - b_greater_than_a.sum()

    denominator = len(group_a) * len(group_b)

    return float(numerator / denominator)


def bootstrap_pairwise(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap median difference and rank-biserial effect size."""
    rng = np.random.default_rng(seed)

    median_differences = np.empty(
        n_bootstrap,
        dtype=float,
    )

    effects = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for index in range(n_bootstrap):
        boot_a = group_a[
            rng.integers(
                0,
                len(group_a),
                size=len(group_a),
            )
        ]

        boot_b = group_b[
            rng.integers(
                0,
                len(group_b),
                size=len(group_b),
            )
        ]

        median_differences[index] = np.median(boot_a) - np.median(boot_b)

        effects[index] = rank_biserial(
            boot_a,
            boot_b,
        )

    median_ci = np.quantile(
        median_differences,
        [0.025, 0.975],
    )

    effect_ci = np.quantile(
        effects,
        [0.025, 0.975],
    )

    return {
        "median_difference_ci_low": float(median_ci[0]),
        "median_difference_ci_high": float(median_ci[1]),
        "rank_biserial_ci_low": float(effect_ci[0]),
        "rank_biserial_ci_high": float(effect_ci[1]),
    }


def group_summary(
    values: np.ndarray,
) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sd": float(
            np.std(
                values,
                ddof=1,
            )
        ),
        "median": float(np.median(values)),
        "q25": float(
            np.quantile(
                values,
                0.25,
            )
        ),
        "q75": float(
            np.quantile(
                values,
                0.75,
            )
        ),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    observed_hash = sha256_file(INPUT_PATH)

    if observed_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(f"Joined benchmark checksum mismatch: {observed_hash}")

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype={
            "chrom": str,
            "fasta_chrom": str,
        },
    )

    if len(df) != 10_000:
        raise ValueError(f"Expected 10,000 rows, observed {len(df):,}")

    required = {
        "chrom",
        "maf",
        "mac",
        "allele_call_rate",
        "frequency_class",
        "gpn_score_ref_alt",
        SCORE_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError("Missing analysis columns: " + ", ".join(sorted(missing)))

    maf = pd.to_numeric(df["maf"]).to_numpy(dtype=float)

    score = pd.to_numeric(df[SCORE_COLUMN]).to_numpy(dtype=float)

    raw_score = pd.to_numeric(df["gpn_score_ref_alt"]).to_numpy(dtype=float)

    chromosome = df["chrom"].astype(str).to_numpy()

    if not np.isfinite(maf).all():
        raise ValueError("MAF contains non-finite values")

    if not np.isfinite(score).all():
        raise ValueError("Oriented score contains non-finite values")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # Primary continuous association.
    #
    (
        rho,
        permutation_p,
        permutation_extreme,
    ) = chromosome_stratified_permutation_test(
        maf,
        score,
        chromosome,
        n_permutations=N_PERMUTATIONS,
        seed=SEED,
    )

    raw_rho = spearman_rho(
        maf,
        raw_score,
    )

    #
    # Frequency-class summaries.
    #
    class_rows = []
    class_summary_json = {}

    for frequency_class in FREQUENCY_ORDER:
        values = df.loc[
            df["frequency_class"] == frequency_class,
            SCORE_COLUMN,
        ].to_numpy(dtype=float)

        summary = group_summary(values)

        class_summary_json[frequency_class] = summary

        class_rows.append(
            {
                "frequency_class": frequency_class,
                **summary,
            }
        )

    class_table = pd.DataFrame(class_rows)

    class_table.to_csv(
        CLASS_PATH,
        sep="\t",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    #
    # Pairwise effect sizes.
    #
    comparisons = [
        (
            "rare",
            "low_frequency",
        ),
        (
            "rare",
            "common",
        ),
        (
            "low_frequency",
            "common",
        ),
    ]

    pairwise_rows = []

    for index, (
        group_a_name,
        group_b_name,
    ) in enumerate(comparisons):
        group_a = df.loc[
            df["frequency_class"] == group_a_name,
            SCORE_COLUMN,
        ].to_numpy(dtype=float)

        group_b = df.loc[
            df["frequency_class"] == group_b_name,
            SCORE_COLUMN,
        ].to_numpy(dtype=float)

        median_difference = float(np.median(group_a) - np.median(group_b))

        effect = rank_biserial(
            group_a,
            group_b,
        )

        bootstrap = bootstrap_pairwise(
            group_a,
            group_b,
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED + 100 + index,
        )

        pairwise_rows.append(
            {
                "contrast": (f"{group_a_name}_vs_{group_b_name}"),
                "group_a": group_a_name,
                "group_b": group_b_name,
                "n_a": len(group_a),
                "n_b": len(group_b),
                "median_a": float(np.median(group_a)),
                "median_b": float(np.median(group_b)),
                "median_difference_a_minus_b": (median_difference),
                "rank_biserial_a_vs_b": effect,
                **bootstrap,
            }
        )

    pairwise_table = pd.DataFrame(pairwise_rows)

    pairwise_table.to_csv(
        PAIRWISE_PATH,
        sep="\t",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    #
    # Sensitivity analyses.
    #
    sensitivity_masks = {
        "all": np.ones(
            len(df),
            dtype=bool,
        ),
        "call_rate_ge_0.95": (df["allele_call_rate"].to_numpy(dtype=float) >= 0.95),
        "mac_ge_5": (df["mac"].to_numpy(dtype=float) >= 5),
        "call_rate_ge_0.95_and_mac_ge_5": (
            (df["allele_call_rate"].to_numpy(dtype=float) >= 0.95)
            & (df["mac"].to_numpy(dtype=float) >= 5)
        ),
    }

    sensitivity_rows = []

    for name, mask in sensitivity_masks.items():
        subset_maf = maf[mask]
        subset_score = score[mask]

        sensitivity_rows.append(
            {
                "analysis": name,
                "n": int(mask.sum()),
                "spearman_rho": spearman_rho(
                    subset_maf,
                    subset_score,
                ),
                "median_maf": float(np.median(subset_maf)),
                "median_score": float(np.median(subset_score)),
            }
        )

    sensitivity_table = pd.DataFrame(sensitivity_rows)

    sensitivity_table.to_csv(
        SENSITIVITY_PATH,
        sep="\t",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    #
    # Chromosome robustness.
    #
    chromosome_rows = []

    for chrom in ["1", "2", "3", "4", "5"]:
        mask = chromosome == chrom

        chromosome_rows.append(
            {
                "analysis": "per_chromosome",
                "chromosome": chrom,
                "n": int(mask.sum()),
                "spearman_rho": spearman_rho(
                    maf[mask],
                    score[mask],
                ),
            }
        )

    for chrom in ["1", "2", "3", "4", "5"]:
        mask = chromosome != chrom

        chromosome_rows.append(
            {
                "analysis": "leave_one_chromosome_out",
                "chromosome": chrom,
                "n": int(mask.sum()),
                "spearman_rho": spearman_rho(
                    maf[mask],
                    score[mask],
                ),
            }
        )

    chromosome_table = pd.DataFrame(chromosome_rows)

    chromosome_table.to_csv(
        CHROMOSOME_PATH,
        sep="\t",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    #
    # Machine-readable analysis summary.
    #
    summary = {
        "analysis_id": ("popgenlm_v0.2_population_gpn_analysis"),
        "input": {
            "filename": INPUT_PATH.name,
            "sha256": observed_hash,
            "n_variants": len(df),
        },
        "score": {
            "primary": SCORE_COLUMN,
            "semantics": ("log P(minor) - log P(major)"),
        },
        "settings": {
            "random_seed": SEED,
            "bootstrap_replicates": N_BOOTSTRAP,
            "permutations": N_PERMUTATIONS,
            "frequency_classes": {
                "rare": "MAF < 0.01",
                "low_frequency": ("0.01 <= MAF < 0.05"),
                "common": "MAF >= 0.05",
            },
            "sensitivity_filters": {
                "call_rate": ("allele_call_rate >= 0.95"),
                "minor_allele_count": ("MAC >= 5"),
            },
        },
        "continuous_association": {
            "spearman_maf_minor_vs_major": rho,
            "chromosome_stratified_permutation_p": (permutation_p),
            "permutation_extreme_count": (permutation_extreme),
            "spearman_maf_raw_ref_alt": raw_rho,
        },
        "frequency_class_summary": (class_summary_json),
        "pairwise_effect_size_definition": (
            "Tie-aware rank-biserial correlation, "
            "equivalent to Cliff's delta. "
            "Positive values mean group A tends "
            "to have larger GPN scores than group B."
        ),
        "interpretation_boundary": (
            "These analyses quantify concordance "
            "between population frequency and GPN "
            "scores. They are not tests of natural "
            "selection. Demography, population "
            "structure, linkage, ascertainment, "
            "and allele age may contribute. The "
            "chromosome-stratified permutation "
            "preserves chromosome membership but "
            "does not explicitly model local LD."
        ),
        "outputs": {
            "chromosome_robustness": CHROMOSOME_PATH.name,
            "frequency_class_summary": (CLASS_PATH.name),
            "frequency_class_pairwise": (PAIRWISE_PATH.name),
            "sensitivity_summary": (SENSITIVITY_PATH.name),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("PopGenLM Bench v0.2 statistical analysis")
    print("=" * 45)

    print(
        "Input SHA256:",
        observed_hash,
    )

    print()
    print("Primary continuous association")
    print(
        "Spearman rho:",
        f"{rho:.6f}",
    )
    print(
        "Chromosome-stratified permutation p:",
        f"{permutation_p:.6g}",
    )
    print(
        "Extreme permutations:",
        f"{permutation_extreme:,}/{N_PERMUTATIONS:,}",
    )
    print(
        "Raw REF->ALT rho:",
        f"{raw_rho:.6f}",
    )

    print()
    print("Frequency-class score summaries")

    for row in class_rows:
        print(
            row["frequency_class"],
            f"n={row['n']:,}",
            f"median={row['median']:.6f}",
            f"mean={row['mean']:.6f}",
        )

    print()
    print("Pairwise effects")

    for row in pairwise_rows:
        print(
            row["contrast"],
            "median_diff=",
            f"{row['median_difference_a_minus_b']:.6f}",
            "rank_biserial=",
            f"{row['rank_biserial_a_vs_b']:.6f}",
            "95% CI=",
            (f"[{row['rank_biserial_ci_low']:.6f}, {row['rank_biserial_ci_high']:.6f}]"),
        )

    print()
    print("Sensitivity")

    for row in sensitivity_rows:
        print(
            row["analysis"],
            f"n={row['n']:,}",
            f"rho={row['spearman_rho']:.6f}",
        )

    print()
    print("Chromosome robustness")

    for row in chromosome_rows:
        if row["analysis"] == "per_chromosome":
            label = f"chr{row['chromosome']}"
        else:
            label = f"exclude_chr{row['chromosome']}"

        print(
            row["analysis"],
            label,
            f"n={row['n']:,}",
            f"rho={row['spearman_rho']:.6f}",
        )

    print()
    print("Output hashes")

    for path in [
        SUMMARY_PATH,
        CLASS_PATH,
        PAIRWISE_PATH,
        SENSITIVITY_PATH,
        CHROMOSOME_PATH,
    ]:
        print(
            path.name,
            sha256_file(path),
        )

    print()
    print("V0.2 STATISTICAL ANALYSIS PASSED")


if __name__ == "__main__":
    main()
