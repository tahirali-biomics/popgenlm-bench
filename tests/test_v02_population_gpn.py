"""Release-invariant tests for the PopGenLM Bench v0.2 population + GPN table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TABLE = ROOT / "data" / "benchmarks" / "v0.2" / "1001g_population_10000_gpn.tsv"

METADATA = ROOT / "data" / "benchmarks" / "v0.2" / "1001g_population_10000_gpn.meta.json"

POPULATION_TABLE = ROOT / "data" / "benchmarks" / "v0.2" / "1001g_population_10000.tsv"

GPN_TABLE = ROOT / "data" / "benchmarks" / "v0.2" / "gpn" / "1001g_population_10000_gpn_scores.tsv"

EXPECTED_TABLE_SHA256 = "ef188e83d0e598d016c158b4daeec08be63e19723840fc42b494c76443fca469"

EXPECTED_METADATA_SHA256 = "923a0035a0fcd787b756dd64e7d58c01df48d74616cc3eac564f1b46f762fe8c"

EXPECTED_POPULATION_SHA256 = "cbf535008593ed45fc7ee8619ed7a87c4c3bb2e741e721a4cb97832f3ba5b374"

EXPECTED_GPN_SHA256 = "735155832ff51bcb8c5259fdd678e5fb88bf6e3097368dfcc00c948f899afbc3"

EXPECTED_COLUMNS = [
    "selection_rank",
    "chrom",
    "pos",
    "ref",
    "alt",
    "fasta_chrom",
    "filter",
    "n_genotypes",
    "n_missing",
    "n_heterozygous",
    "ac_ref",
    "ac_alt",
    "an",
    "af_alt",
    "mac",
    "maf",
    "allele_call_rate",
    "frequency_class",
    "alt_major",
    "edge_padded",
    "selection_hash",
    "gpn_score_ref_alt",
    "minor_allele",
    "major_allele",
    "orientation",
    "flipped",
    "gpn_score_paper_oriented",
    "gpn_score_minor_vs_major",
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


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    result = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    assert result.notna().all()

    return result.astype(bool)


def test_v02_population_gpn_release_invariants() -> None:
    assert TABLE.exists()
    assert METADATA.exists()

    assert sha256_file(TABLE) == EXPECTED_TABLE_SHA256
    assert sha256_file(METADATA) == EXPECTED_METADATA_SHA256
    assert sha256_file(POPULATION_TABLE) == EXPECTED_POPULATION_SHA256
    assert sha256_file(GPN_TABLE) == EXPECTED_GPN_SHA256

    df = pd.read_csv(
        TABLE,
        sep="\t",
        dtype={
            "chrom": str,
            "fasta_chrom": str,
        },
    )

    metadata = json.loads(METADATA.read_text())

    assert df.shape == (10_000, 28)
    assert list(df.columns) == EXPECTED_COLUMNS

    key = ["chrom", "pos", "ref", "alt"]

    assert not df.duplicated(key).any()

    assert df["frequency_class"].value_counts().to_dict() == {
        "rare": 7209,
        "low_frequency": 1561,
        "common": 1230,
    }

    alt_major = parse_bool(df["alt_major"])
    flipped = parse_bool(df["flipped"])
    edge_padded = parse_bool(df["edge_padded"])

    assert int(alt_major.sum()) == 246
    assert int(flipped.sum()) == 246
    assert not edge_padded.any()

    assert (alt_major == (df["af_alt"] > 0.5)).all()
    assert (flipped == alt_major).all()

    assert int((df["af_alt"] == 0.5).sum()) == 0

    assert df["orientation"].value_counts().to_dict() == {
        "alt_minor": 9754,
        "ref_minor": 246,
    }

    alt_minor = df["af_alt"] < 0.5
    ref_minor = df["af_alt"] > 0.5

    assert (df.loc[alt_minor, "minor_allele"] == df.loc[alt_minor, "alt"]).all()

    assert (df.loc[alt_minor, "major_allele"] == df.loc[alt_minor, "ref"]).all()

    assert (df.loc[ref_minor, "minor_allele"] == df.loc[ref_minor, "ref"]).all()

    assert (df.loc[ref_minor, "major_allele"] == df.loc[ref_minor, "alt"]).all()

    raw = df["gpn_score_ref_alt"].to_numpy(dtype=float)

    oriented = df["gpn_score_minor_vs_major"].to_numpy(dtype=float)

    paper_oriented = df["gpn_score_paper_oriented"].to_numpy(dtype=float)

    assert np.isfinite(raw).all()
    assert np.isfinite(oriented).all()

    expected_oriented = np.where(
        alt_major.to_numpy(),
        -raw,
        raw,
    )

    np.testing.assert_allclose(
        oriented,
        expected_oriented,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        paper_oriented,
        oriented,
        rtol=0.0,
        atol=1e-12,
    )

    assert metadata["counts"]["variants"] == 10_000
    assert metadata["counts"]["alt_major"] == 246
    assert metadata["counts"]["flipped"] == 246
    assert metadata["counts"]["frequency_ties"] == 0
    assert metadata["counts"]["edge_padded"] == 0

    assert metadata["counts"]["frequency_classes"] == {
        "rare": 7209,
        "low_frequency": 1561,
        "common": 1230,
    }

    assert metadata["inputs"]["population"]["sha256"] == EXPECTED_POPULATION_SHA256

    assert metadata["inputs"]["gpn_scores"]["sha256"] == EXPECTED_GPN_SHA256

    assert metadata["output"]["sha256"] == EXPECTED_TABLE_SHA256

    assert metadata["output"]["n_variants"] == 10_000
