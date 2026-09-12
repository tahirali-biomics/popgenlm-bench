#!/usr/bin/env python3
"""Build the canonical PopGenLM Bench v0.2 population + GPN dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from popgenlm.integration import integrate_population_gpn_scores

ROOT = Path(__file__).resolve().parents[1]

POPULATION_PATH = ROOT / "data/benchmarks/v0.2" / "1001g_population_10000.tsv"

GPN_PATH = ROOT / "data/benchmarks/v0.2/gpn" / "1001g_population_10000_gpn_scores.tsv"

GPN_META_PATH = ROOT / "data/benchmarks/v0.2/gpn" / "1001g_population_10000_gpn_scores.meta.json"

CHROM_CONFIG_PATH = ROOT / "configs/references" / "tair10.1_nuclear.tsv"

OUTPUT_PATH = ROOT / "data/benchmarks/v0.2" / "1001g_population_10000_gpn.tsv"

META_PATH = ROOT / "data/benchmarks/v0.2" / "1001g_population_10000_gpn.meta.json"

WINDOW_SIZE = 512

EXPECTED_HASHES = {
    "population": ("cbf535008593ed45fc7ee8619ed7a87c4c3bb2e741e721a4cb97832f3ba5b374"),
    "gpn_scores": ("735155832ff51bcb8c5259fdd678e5fb88bf6e3097368dfcc00c948f899afbc3"),
    "gpn_metadata": ("cbafbd25837c27b23fd42f7f794968513f3a1dbad11faa76938ceb686f9169b3"),
}

EXPECTED_FREQUENCY_COUNTS = {
    "rare": 7209,
    "low_frequency": 1561,
    "common": 1230,
}

EXPECTED_ALT_MAJOR = 246


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def boolean_values(series: pd.Series) -> np.ndarray:
    """Convert a boolean-like column to a strict bool array."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    values = (
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

    if values.isna().any():
        raise ValueError(f"Column {series.name!r} contains non-boolean values")

    return values.to_numpy(dtype=bool)


def main() -> None:
    #
    # Freeze input identity.
    #
    observed_hashes = {
        "population": sha256(POPULATION_PATH),
        "gpn_scores": sha256(GPN_PATH),
        "gpn_metadata": sha256(GPN_META_PATH),
    }

    if observed_hashes != EXPECTED_HASHES:
        raise ValueError(
            "Input checksum mismatch:\n"
            + json.dumps(
                observed_hashes,
                indent=2,
                sort_keys=True,
            )
        )

    #
    # Reference chromosome mapping.
    #
    chrom_config = pd.read_csv(
        CHROM_CONFIG_PATH,
        sep="\t",
        dtype={
            "chrom": str,
            "fasta_chrom": str,
        },
    )

    if set(chrom_config["chrom"]) != {
        "1",
        "2",
        "3",
        "4",
        "5",
    }:
        raise ValueError("Chromosome config must contain exactly chromosomes 1-5")

    fasta_to_vcf = dict(
        zip(
            chrom_config["fasta_chrom"],
            chrom_config["chrom"],
            strict=True,
        )
    )

    chrom_lengths = dict(
        zip(
            chrom_config["chrom"],
            chrom_config["length"],
            strict=True,
        )
    )

    #
    # Load frozen inputs.
    #
    population = pd.read_csv(
        POPULATION_PATH,
        sep="\t",
        dtype={
            "chrom": str,
            "fasta_chrom": str,
        },
    )

    gpn = pd.read_csv(
        GPN_PATH,
        sep="\t",
        dtype={"chrom": str},
    )

    if len(population) != 10_000:
        raise ValueError(f"Expected 10,000 population variants, observed {len(population):,}")

    if len(gpn) != 10_000:
        raise ValueError(f"Expected 10,000 GPN scores, observed {len(gpn):,}")

    key = ["chrom", "pos", "ref", "alt"]

    if population.duplicated(key).any():
        raise ValueError("Population benchmark contains duplicate variant keys")

    if gpn.duplicated(["chrom", "pos", "ref", "alt"]).any():
        raise ValueError("GPN table contains duplicate variant keys")

    #
    # Use the tested package integration API.
    #
    merged = integrate_population_gpn_scores(
        population,
        gpn,
        fasta_to_vcf,
        window_size=WINDOW_SIZE,
    )

    if len(merged) != 10_000:
        raise ValueError(f"Integration produced {len(merged):,} rows")

    #
    # The joined table must preserve the exact
    # benchmark row order and keys.
    #
    pd.testing.assert_frame_equal(
        population[key].reset_index(drop=True),
        merged[key].reset_index(drop=True),
        check_dtype=False,
    )

    #
    # Population-frequency invariants.
    #
    frequency_counts = merged["frequency_class"].value_counts().to_dict()

    if frequency_counts != EXPECTED_FREQUENCY_COUNTS:
        raise ValueError(f"Unexpected frequency classes: {frequency_counts}")

    af_alt = pd.to_numeric(merged["af_alt"]).to_numpy(dtype=float)

    maf = pd.to_numeric(merged["maf"]).to_numpy(dtype=float)

    expected_maf = np.minimum(
        af_alt,
        1.0 - af_alt,
    )

    np.testing.assert_allclose(
        maf,
        expected_maf,
        rtol=0.0,
        atol=1e-12,
    )

    alt_major = af_alt > 0.5

    if int(alt_major.sum()) != EXPECTED_ALT_MAJOR:
        raise ValueError(f"Unexpected ALT-major count: {int(alt_major.sum())}")

    observed_alt_major = boolean_values(merged["alt_major"])

    if not np.array_equal(
        alt_major,
        observed_alt_major,
    ):
        raise ValueError("alt_major disagrees with AF_ALT")

    tie_count = int(np.count_nonzero(af_alt == 0.5))

    if tie_count != 0:
        raise ValueError(f"Expected zero frequency ties, observed {tie_count}")

    #
    # Orientation invariants.
    #
    flipped = boolean_values(merged["flipped"])

    if not np.array_equal(
        flipped,
        alt_major,
    ):
        raise ValueError("Score flips do not exactly match ALT-major variants")

    raw_score = pd.to_numeric(merged["gpn_score_ref_alt"]).to_numpy(dtype=float)

    oriented_score = pd.to_numeric(merged["gpn_score_minor_vs_major"]).to_numpy(dtype=float)

    paper_score = pd.to_numeric(merged["gpn_score_paper_oriented"]).to_numpy(dtype=float)

    if not np.isfinite(raw_score).all():
        raise ValueError("Non-finite raw GPN score")

    if not np.isfinite(oriented_score).all():
        raise ValueError("Non-finite oriented GPN score")

    expected_oriented = np.where(
        alt_major,
        -raw_score,
        raw_score,
    )

    np.testing.assert_allclose(
        oriented_score,
        expected_oriented,
        rtol=0.0,
        atol=0.0,
    )

    # There are no 50:50 ties, so the published
    # >0.5 convention and the explicitly named
    # minor-vs-major quantity are identical here.
    np.testing.assert_allclose(
        paper_score,
        oriented_score,
        rtol=0.0,
        atol=0.0,
    )

    #
    # Verify the whole 512-bp GPN window is
    # inside the reference for every site.
    #
    left_margin = WINDOW_SIZE // 2
    right_margin = WINDOW_SIZE - left_margin - 1

    for row in merged.itertuples():
        chrom_length = int(chrom_lengths[str(row.chrom)])

        if row.pos <= left_margin:
            raise ValueError(f"Left-edge variant detected: {row.chrom}:{row.pos}")

        if row.pos > chrom_length - right_margin:
            raise ValueError(f"Right-edge variant detected: {row.chrom}:{row.pos}")

    edge_padded = boolean_values(merged["edge_padded"])

    if edge_padded.any():
        raise ValueError("Release benchmark unexpectedly contains edge-padded variants")

    #
    # Canonical release schema.
    #
    output_columns = [
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

    result = merged[output_columns].copy()

    if result.columns.duplicated().any():
        raise ValueError("Duplicate output column names")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    output_hash = sha256(OUTPUT_PATH)

    #
    # Provenance metadata.
    #
    score_summary = result["gpn_score_minor_vs_major"].describe()

    metadata = {
        "benchmark_id": ("popgenlm_v0.2_1001g_population_10000_gpn"),
        "release_target": "v0.2.0",
        "species": "Arabidopsis thaliana",
        "dataset_id": "1001g_1135_v3.1",
        "reference": {
            "assembly": "TAIR10.1",
            "ncbi_assembly": "GCF_000001735.4",
            "chromosome_config_sha256": (sha256(CHROM_CONFIG_PATH)),
            "window_size": WINDOW_SIZE,
            "edge_padding_excluded": True,
        },
        "integration": {
            "method": ("one-to-one variant-key join through PopGenLM integration API"),
            "variant_key": ("chrom:pos:ref:alt"),
            "fasta_mapping_validated": True,
            "population_row_order_preserved": True,
        },
        "score_semantics": {
            "gpn_score_ref_alt": ("log P(ALT) - log P(REF), strand-averaged GPN score"),
            "gpn_score_minor_vs_major": (
                "log P(minor) - log P(major); raw score sign-flipped exactly when AF_ALT > 0.5"
            ),
            "gpn_score_paper_oriented": (
                "published Arabidopsis >0.5 frequency-orientation convention"
            ),
        },
        "inputs": {
            "population": {
                "filename": POPULATION_PATH.name,
                "sha256": observed_hashes["population"],
            },
            "gpn_scores": {
                "filename": GPN_PATH.name,
                "sha256": observed_hashes["gpn_scores"],
            },
            "gpn_metadata": {
                "filename": GPN_META_PATH.name,
                "sha256": observed_hashes["gpn_metadata"],
            },
        },
        "counts": {
            "variants": len(result),
            "frequency_classes": (frequency_counts),
            "alt_major": int(alt_major.sum()),
            "flipped": int(flipped.sum()),
            "frequency_ties": tie_count,
            "edge_padded": int(edge_padded.sum()),
        },
        "score_summary_minor_vs_major": {
            "minimum": float(score_summary["min"]),
            "q1": float(score_summary["25%"]),
            "median": float(score_summary["50%"]),
            "mean": float(score_summary["mean"]),
            "q3": float(score_summary["75%"]),
            "maximum": float(score_summary["max"]),
            "standard_deviation": float(score_summary["std"]),
        },
        "interpretation_boundary": (
            "Population-frequency concordance "
            "with a genomic language-model score "
            "is not by itself evidence of natural "
            "selection. Demography, population "
            "structure, linkage, ascertainment, "
            "and allele age may contribute."
        ),
        "output": {
            "filename": OUTPUT_PATH.name,
            "sha256": output_hash,
            "n_variants": len(result),
        },
    }

    META_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("PopGenLM Bench v0.2 population + GPN integration")
    print("=" * 50)

    print("Rows:", f"{len(result):,}")
    print(
        "Unique variant keys:",
        f"{result[key].drop_duplicates().shape[0]:,}",
    )
    print(
        "Frequency classes:",
        frequency_counts,
    )
    print(
        "ALT-major:",
        f"{int(alt_major.sum()):,}",
    )
    print(
        "Flipped:",
        f"{int(flipped.sum()):,}",
    )
    print(
        "Frequency ties:",
        f"{tie_count:,}",
    )
    print(
        "Edge padded:",
        f"{int(edge_padded.sum()):,}",
    )

    print()
    print("Minor-vs-major score summary:")
    print(score_summary.to_string())

    print()
    print(
        "Output SHA256:",
        output_hash,
    )
    print(
        "Metadata SHA256:",
        sha256(META_PATH),
    )

    print()
    print("10K POPULATION + GPN INTEGRATION PASSED")


if __name__ == "__main__":
    main()
