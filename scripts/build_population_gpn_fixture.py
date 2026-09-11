#!/usr/bin/env python3
"""Join the 1001 Genomes population fixture to canonical GPN scores."""

from pathlib import Path

import pandas as pd

from popgenlm.orientation import orient_gpn_score

ROOT = Path(__file__).resolve().parents[1]

POPULATION_PATH = ROOT / "data/fixtures/1001g_v3.1_population_fixture.tsv"
GPN_PATH = ROOT / "data/fixtures/1001g_v3.1_gpn_scores.tsv"
OUTPUT_PATH = ROOT / "data/fixtures/1001g_v3.1_population_gpn_fixture.tsv"

WINDOW_SIZE = 512


def main() -> None:
    population = pd.read_csv(
        POPULATION_PATH,
        sep="\t",
        dtype={"chrom": str},
    )

    gpn = pd.read_csv(
        GPN_PATH,
        sep="\t",
        dtype={"chrom": str},
    )

    # GPN uses FASTA record IDs, whereas the population fixture uses VCF labels.
    fasta_to_vcf = {
        "NC_003070.9": "1",
        "NC_003071.7": "2",
        "NC_003074.8": "3",
        "NC_003075.7": "4",
        "NC_003076.8": "5",
    }

    gpn["vcf_chrom"] = gpn["chrom"].map(fasta_to_vcf)

    if gpn["vcf_chrom"].isna().any():
        raise ValueError("Unrecognized GPN FASTA chromosome ID")

    merged = population.merge(
        gpn[
            [
                "vcf_chrom",
                "pos",
                "ref",
                "alt",
                "chrom",
                "gpn_score_ref_alt",
            ]
        ],
        left_on=["chrom", "pos", "ref", "alt"],
        right_on=["vcf_chrom", "pos", "ref", "alt"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_gpn"),
    )

    if merged["gpn_score_ref_alt"].isna().any():
        raise ValueError("One or more population variants lack a GPN score")

    orientation_rows = [
        orient_gpn_score(
            row.ref,
            row.alt,
            row.af_alt,
            row.gpn_score_ref_alt,
        )
        for row in merged.itertuples()
    ]

    orientation = pd.DataFrame(orientation_rows)

    for column in [
        "minor_allele",
        "major_allele",
        "orientation",
        "flipped",
        "gpn_score_paper_oriented",
        "gpn_score_minor_major",
    ]:
        merged[column] = orientation[column].to_numpy()

    # For a centered 512-bp window, variants with POS <= 256 require
    # left-edge N padding. This flag prevents those sites from being silently
    # treated as ordinary interior sequence contexts.
    merged["edge_padded"] = merged["pos"] <= WINDOW_SIZE // 2

    merged = merged.rename(columns={"chrom_gpn": "fasta_chrom"})

    output_columns = [
        "selection_class",
        "chrom",
        "fasta_chrom",
        "pos",
        "ref",
        "alt",
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
        "gpn_score_ref_alt",
        "minor_allele",
        "major_allele",
        "orientation",
        "flipped",
        "gpn_score_paper_oriented",
        "gpn_score_minor_major",
        "edge_padded",
    ]

    result = merged[output_columns]

    if len(result) != 20:
        raise ValueError(f"Expected 20 variants, observed {len(result)}")

    result.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    print(f"Wrote {len(result)} variants to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
