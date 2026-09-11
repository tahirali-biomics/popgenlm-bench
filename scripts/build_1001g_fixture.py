#!/usr/bin/env python3
"""Build a small deterministic population-genetic fixture from 1001 Genomes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from popgenlm.population import summarize_biallelic_genotypes

SELECTION_CLASSES = ("rare", "alt_major", "missing", "common")


def choose_class(
    summary: dict[str, float | int],
    counts: dict[str, int],
    target_per_class: int,
) -> str | None:
    """Assign a variant to a controlled fixture class."""

    maf = float(summary["maf"])
    af_alt = float(summary["af_alt"])
    allele_call_rate = float(summary["allele_call_rate"])

    if maf <= 0:
        return None

    # Frequency classes should not be confounded by heavy missingness.
    if maf < 0.01 and allele_call_rate >= 0.90 and counts["rare"] < target_per_class:
        return "rare"

    if af_alt > 0.5 and allele_call_rate >= 0.90 and counts["alt_major"] < target_per_class:
        return "alt_major"

    # Missingness is deliberately represented as its own stress-test class.
    if allele_call_rate < 0.80 and counts["missing"] < target_per_class:
        return "missing"

    if (
        maf >= 0.05
        and af_alt <= 0.5
        and allele_call_rate >= 0.90
        and counts["common"] < target_per_class
    ):
        return "common"

    return None


def complete(counts: dict[str, int], target_per_class: int) -> bool:
    return all(counts[name] >= target_per_class for name in SELECTION_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--expected-accessions", type=int, default=1135)
    parser.add_argument("--target-per-class", type=int, default=5)
    parser.add_argument("--max-sites", type=int, default=500000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in SELECTION_CLASSES}
    selected: list[dict[str, str | float | int]] = []
    scanned = 0

    view_cmd = [
        "bcftools",
        "view",
        "-f",
        "PASS",
        "-m2",
        "-M2",
        "-v",
        "snps",
        "-Ov",
        str(args.vcf),
    ]

    query_cmd = [
        "bcftools",
        "query",
        "-f",
        r"%CHROM\t%POS\t%REF\t%ALT\t%FILTER[\t%GT]\n",
        "-",
    ]

    view = subprocess.Popen(view_cmd, stdout=subprocess.PIPE)
    assert view.stdout is not None

    query = subprocess.Popen(
        query_cmd,
        stdin=view.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )
    view.stdout.close()

    assert query.stdout is not None

    try:
        for line in query.stdout:
            scanned += 1

            if scanned > args.max_sites:
                break

            fields = line.rstrip("\n").split("\t")

            if len(fields) < 6:
                continue

            chrom, pos, ref, alt, filt = fields[:5]
            genotypes = fields[5:]

            if len(genotypes) != args.expected_accessions:
                raise ValueError(
                    f"{chrom}:{pos} has {len(genotypes)} genotypes; "
                    f"expected {args.expected_accessions}"
                )

            # Additional explicit canonical-SNV guard.
            if (
                len(ref) != 1
                or len(alt) != 1
                or ref not in "ACGT"
                or alt not in "ACGT"
                or ref == alt
            ):
                continue

            summary = summarize_biallelic_genotypes(genotypes)

            selection_class = choose_class(
                summary,
                counts,
                args.target_per_class,
            )

            if selection_class is None:
                continue

            counts[selection_class] += 1

            selected.append(
                {
                    "selection_class": selection_class,
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "filter": filt,
                    **summary,
                }
            )

            if complete(counts, args.target_per_class):
                break
    finally:
        if query.poll() is None:
            query.terminate()
        if view.poll() is None:
            view.terminate()

        query.wait()
        view.wait()

    if not complete(counts, args.target_per_class):
        raise RuntimeError(
            "Could not fill all fixture classes. "
            f"Counts after scanning {scanned} filtered sites: {counts}"
        )

    fieldnames = [
        "selection_class",
        "chrom",
        "pos",
        "ref",
        "alt",
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
    ]

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(selected)

    metadata = {
        "dataset_id": "1001g_1135_v3.1",
        "source_filename": args.vcf.name,
        "source_sha256": args.source_sha256,
        "expected_accessions": args.expected_accessions,
        "fixture_variants": len(selected),
        "target_per_class": args.target_per_class,
        "selection_classes": list(SELECTION_CLASSES),
        "sites_scanned": scanned,
        "filters": {
            "vcf_filter": "PASS",
            "variant_type": "biallelic canonical SNV",
            "require_polymorphic": True,
            "frequency_class_min_allele_call_rate": 0.90,
            "missing_class_max_allele_call_rate_exclusive": 0.80,
            "rare_maf_max_exclusive": 0.01,
            "common_maf_min": 0.05,
        },
    }

    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {len(selected)} variants to {args.output}")
    print(f"Class counts: {counts}")
    print(f"Filtered sites scanned: {scanned}")


if __name__ == "__main__":
    main()
