"""Release-invariant tests for the PopGenLM Bench v0.2 benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TABLE = ROOT / "data" / "benchmarks" / "v0.2" / "1001g_population_10000.tsv"

METADATA = ROOT / "data" / "benchmarks" / "v0.2" / "1001g_population_10000.meta.json"

EXPECTED_SOURCE_SHA256 = "7f825afb784b2424e35501fd8b88c32a4798013eb9fe762724addfb249007e7a"

EXPECTED_TABLE_SHA256 = "cbf535008593ed45fc7ee8619ed7a87c4c3bb2e741e721a4cb97832f3ba5b374"

FASTA_MAPPING = {
    "1": "NC_003070.9",
    "2": "NC_003071.7",
    "3": "NC_003074.8",
    "4": "NC_003075.7",
    "5": "NC_003076.8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def parse_bool(series: pd.Series) -> pd.Series:
    result = (
        series.astype(str)
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


def test_v02_population_benchmark_release_invariants() -> None:
    assert TABLE.exists()
    assert METADATA.exists()

    df = pd.read_csv(
        TABLE,
        sep="\t",
        dtype={"chrom": str},
    )

    metadata = json.loads(METADATA.read_text())

    assert len(df) == 10_000

    assert not df.duplicated(["chrom", "pos", "ref", "alt"]).any()

    assert set(df["chrom"]) == set(FASTA_MAPPING)

    assert df["ref"].isin(list("ACGT")).all()
    assert df["alt"].isin(list("ACGT")).all()
    assert (df["ref"] != df["alt"]).all()

    assert (df["maf"] > 0).all()
    assert (df["maf"] <= 0.5).all()

    assert (df["af_alt"] > 0).all()
    assert (df["af_alt"] < 1).all()

    assert (df["allele_call_rate"] >= 0.90).all()

    edge_padded = parse_bool(df["edge_padded"])
    assert not edge_padded.any()

    alt_major = parse_bool(df["alt_major"])

    assert (alt_major == (df["af_alt"] > 0.5)).all()

    expected_class = df["maf"].map(
        lambda maf: "rare" if maf < 0.01 else ("low_frequency" if maf < 0.05 else "common")
    )

    assert (df["frequency_class"] == expected_class).all()

    expected_fasta = df["chrom"].map(FASTA_MAPPING)

    assert (df["fasta_chrom"] == expected_fasta).all()

    assert set(df["selection_rank"]) == set(range(1, 10_001))

    assert df["selection_hash"].str.fullmatch(r"[0-9a-f]{64}").all()

    assert sha256_file(TABLE) == EXPECTED_TABLE_SHA256

    assert metadata["source_sha256"] == EXPECTED_SOURCE_SHA256

    assert metadata["output"]["sha256"] == EXPECTED_TABLE_SHA256

    assert metadata["counts"]["selected_variants"] == 10_000

    assert metadata["selection"]["candidate_population_eligible"] == 28_360

    assert metadata["selection"]["exact_bottom_k_condition_satisfied"]
