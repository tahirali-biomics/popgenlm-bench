"""Release-invariant tests for the PopGenLM Bench v0.2 analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "v0.2"

SUMMARY = REPORT_DIR / "population_gpn_analysis.json"
CLASS = REPORT_DIR / "frequency_class_summary.tsv"
PAIRWISE = REPORT_DIR / "frequency_class_pairwise.tsv"
SENSITIVITY = REPORT_DIR / "sensitivity_summary.tsv"
CHROMOSOME = REPORT_DIR / "chromosome_robustness.tsv"

EXPECTED_HASHES = {
    "population_gpn_analysis.json": (
        "c41e2a4e59c5a9fc25f12803ae30149c502096167328874b0890e1bf4b5f11e9"
    ),
    "frequency_class_summary.tsv": (
        "a7dfd82ee5f0850a68dbfcdfc21b47a0df0a694d22e4ecfc462d75afc3785c3e"
    ),
    "frequency_class_pairwise.tsv": (
        "3779fe5578ca4b33b5d66c4a34d4eefe61d85417982a4e5f45e31b3571f80739"
    ),
    "sensitivity_summary.tsv": ("d00b1a016f87102d45cbf276b08b0fa0fea3379ab08194c14a3f4e8091e2a3fa"),
    "chromosome_robustness.tsv": (
        "5f0791cf974a35524393d99cdc248dede8fac0bc25978aca7dbefe1dd2e74dc3"
    ),
}

EXPECTED_INPUT_SHA256 = "ef188e83d0e598d016c158b4daeec08be63e19723840fc42b494c76443fca469"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def test_v02_analysis_artifact_hashes() -> None:
    paths = {
        SUMMARY.name: SUMMARY,
        CLASS.name: CLASS,
        PAIRWISE.name: PAIRWISE,
        SENSITIVITY.name: SENSITIVITY,
        CHROMOSOME.name: CHROMOSOME,
    }

    for filename, path in paths.items():
        assert path.exists()
        assert sha256_file(path) == EXPECTED_HASHES[filename]


def test_v02_primary_analysis_results() -> None:
    summary = json.loads(SUMMARY.read_text())

    assert summary["input"]["sha256"] == EXPECTED_INPUT_SHA256
    assert summary["input"]["n_variants"] == 10_000

    assert summary["score"]["primary"] == ("gpn_score_minor_vs_major")

    association = summary["continuous_association"]

    assert association["spearman_maf_minor_vs_major"] == pytest.approx(
        0.0611223588105,
        abs=1e-12,
    )

    assert association["chromosome_stratified_permutation_p"] == pytest.approx(
        1 / 10_001,
        abs=1e-12,
    )

    assert association["permutation_extreme_count"] == 0

    assert association["spearman_maf_raw_ref_alt"] == pytest.approx(
        0.065715,
        abs=1e-6,
    )

    assert summary["settings"]["random_seed"] == 20260912
    assert summary["settings"]["bootstrap_replicates"] == 5000
    assert summary["settings"]["permutations"] == 10000


def test_v02_frequency_class_results() -> None:
    df = pd.read_csv(
        CLASS,
        sep="\t",
    )

    assert df["frequency_class"].tolist() == [
        "rare",
        "low_frequency",
        "common",
    ]

    assert df["n"].tolist() == [
        7209,
        1561,
        1230,
    ]

    medians = dict(
        zip(
            df["frequency_class"],
            df["median"],
            strict=True,
        )
    )

    assert medians["rare"] == pytest.approx(
        -0.34853882,
        abs=1e-10,
    )

    assert medians["low_frequency"] == pytest.approx(
        -0.27562457,
        abs=1e-10,
    )

    assert medians["common"] == pytest.approx(
        -0.15618226,
        abs=1e-10,
    )

    assert medians["rare"] < medians["low_frequency"] < medians["common"]


def test_v02_pairwise_effects() -> None:
    df = pd.read_csv(
        PAIRWISE,
        sep="\t",
    ).set_index("contrast")

    expected = {
        "rare_vs_low_frequency": -0.0373423710788,
        "rare_vs_common": -0.13216631875,
        "low_frequency_vs_common": -0.095847981542,
    }

    for contrast, effect in expected.items():
        assert df.loc[
            contrast,
            "rank_biserial_a_vs_b",
        ] == pytest.approx(
            effect,
            abs=1e-12,
        )

        assert (
            df.loc[
                contrast,
                "rank_biserial_ci_high",
            ]
            < 0
        )

    rare_common = df.loc["rare_vs_common"]

    assert rare_common["median_difference_a_minus_b"] == pytest.approx(
        -0.19235656,
        abs=1e-10,
    )

    assert rare_common["rank_biserial_ci_low"] == pytest.approx(
        -0.166546249212,
        abs=1e-12,
    )

    assert rare_common["rank_biserial_ci_high"] == pytest.approx(
        -0.097289003019,
        abs=1e-12,
    )


def test_v02_sensitivity_results() -> None:
    df = pd.read_csv(
        SENSITIVITY,
        sep="\t",
    )

    expected_n = {
        "all": 10_000,
        "call_rate_ge_0.95": 6761,
        "mac_ge_5": 5057,
        "call_rate_ge_0.95_and_mac_ge_5": 3279,
    }

    observed_n = dict(
        zip(
            df["analysis"],
            df["n"],
            strict=True,
        )
    )

    assert observed_n == expected_n

    assert np.isfinite(df["spearman_rho"]).all()

    # Every sensitivity analysis retains
    # the same positive direction.
    assert (df["spearman_rho"] > 0).all()

    values = df.set_index("analysis")["spearman_rho"]

    assert values["all"] == pytest.approx(
        0.0611223588105,
        abs=1e-12,
    )

    assert values["call_rate_ge_0.95_and_mac_ge_5"] == pytest.approx(
        0.100183115456,
        abs=1e-12,
    )


def test_v02_chromosome_robustness() -> None:
    df = pd.read_csv(
        CHROMOSOME,
        sep="\t",
        dtype={"chromosome": str},
    )

    assert len(df) == 10

    per_chromosome = df[df["analysis"] == "per_chromosome"]

    leave_one_out = df[df["analysis"] == "leave_one_chromosome_out"]

    assert len(per_chromosome) == 5
    assert len(leave_one_out) == 5

    assert set(per_chromosome["chromosome"]) == {
        "1",
        "2",
        "3",
        "4",
        "5",
    }

    assert (per_chromosome["spearman_rho"] > 0).all()

    assert (leave_one_out["spearman_rho"] > 0).all()

    assert (
        per_chromosome["spearman_rho"]
        .between(
            0.04,
            0.08,
        )
        .all()
    )

    assert (
        leave_one_out["spearman_rho"]
        .between(
            0.05,
            0.07,
        )
        .all()
    )
