#!/usr/bin/env python3
"""Join the 1001 Genomes population fixture to canonical GPN scores."""

from pathlib import Path

import pandas as pd

from popgenlm.integration import integrate_population_gpn_scores

ROOT = Path(__file__).resolve().parents[1]

POPULATION_PATH = ROOT / "data/fixtures/1001g_v3.1_population_fixture.tsv"
GPN_PATH = ROOT / "data/fixtures/1001g_v3.1_gpn_scores.tsv"
OUTPUT_PATH = ROOT / "data/fixtures/1001g_v3.1_population_gpn_fixture.tsv"

WINDOW_SIZE = 512

FASTA_TO_VCF = {
    "NC_003070.9": "1",
    "NC_003071.7": "2",
    "NC_003074.8": "3",
    "NC_003075.7": "4",
    "NC_003076.8": "5",
}


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

    merged = integrate_population_gpn_scores(
        population,
        gpn,
        FASTA_TO_VCF,
        window_size=WINDOW_SIZE,
    )

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
