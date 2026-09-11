#!/usr/bin/env python3
"""Build the deterministic PopGenLM Bench v0.2 population benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from popgenlm.population import summarize_biallelic_genotypes

SELECTION_NAMESPACE = "popgenlm-v0.2-population-benchmark-v1"


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def frequency_class(maf: float) -> str:
    """Assign a population-frequency class."""
    if maf < 0.01:
        return "rare"
    if maf < 0.05:
        return "low_frequency"
    return "common"


def read_chrom_config(path: Path) -> dict[str, dict[str, str | int]]:
    """Read chromosome lengths and FASTA mappings."""
    result: dict[str, dict[str, str | int]] = {}

    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        required = {"chrom", "length", "fasta_chrom"}

        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: " + ", ".join(sorted(required)))

        for row in reader:
            chrom = str(row["chrom"])

            if chrom in result:
                raise ValueError(f"Duplicate chromosome in config: {chrom}")

            result[chrom] = {
                "length": int(row["length"]),
                "fasta_chrom": str(row["fasta_chrom"]),
            }

    if set(result) != {"1", "2", "3", "4", "5"}:
        raise ValueError("Chromosome config must contain exactly chromosomes 1-5")

    return result


def hash_rank(key: str) -> tuple[int, str]:
    """Return deterministic SHA256 rank and digest."""
    digest = hashlib.sha256(f"{SELECTION_NAMESPACE}|{key}".encode()).hexdigest()

    return int(digest, 16), digest


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--chrom-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)

    parser.add_argument("--source-sha256", required=True)

    parser.add_argument(
        "--expected-accessions",
        type=int,
        default=1135,
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--target",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--min-call-rate",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    if args.target <= 0:
        raise ValueError("--target must be positive")

    if args.candidate_count < args.target:
        raise ValueError("--candidate-count must be >= --target")

    if not 0 < args.min_call_rate <= 1:
        raise ValueError("--min-call-rate must lie in (0, 1]")

    chrom_config = read_chrom_config(args.chrom_config)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    left_margin = args.window_size // 2
    right_margin = args.window_size - left_margin - 1

    #
    # Pass 1
    #
    # Scan only variant metadata. Keep the globally lowest
    # SHA256-ranked candidate variants.
    #

    query_sites_cmd = [
        "bcftools",
        "query",
        "-i",
        'FILTER="PASS" && N_ALT=1 && TYPE="snp"',
        "-f",
        r"%CHROM\t%POS\t%REF\t%ALT\t%FILTER\n",
        str(args.vcf),
    ]

    sites = subprocess.Popen(
        query_sites_cmd,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert sites.stdout is not None

    candidate_heap: list[tuple[int, str, dict[str, object]]] = []

    queried_sites = 0
    metadata_eligible_sites = 0

    metadata_excluded: Counter[str] = Counter()

    for line in sites.stdout:
        queried_sites += 1

        fields = line.rstrip("\n").split("\t")

        if len(fields) != 5:
            metadata_excluded["malformed"] += 1
            continue

        chrom, pos_text, ref, alt, filt = fields

        if chrom not in chrom_config:
            metadata_excluded["non_nuclear"] += 1
            continue

        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT" or ref == alt:
            metadata_excluded["noncanonical_snv"] += 1
            continue

        try:
            pos = int(pos_text)
        except ValueError:
            metadata_excluded["invalid_position"] += 1
            continue

        chrom_length = int(chrom_config[chrom]["length"])

        if pos <= left_margin:
            metadata_excluded["left_edge"] += 1
            continue

        if pos > chrom_length - right_margin:
            metadata_excluded["right_edge"] += 1
            continue

        key = f"{chrom}:{pos}:{ref}:{alt}"

        rank, digest = hash_rank(key)

        record: dict[str, object] = {
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "filter": filt,
            "fasta_chrom": chrom_config[chrom]["fasta_chrom"],
            "selection_hash": digest,
            "selection_rank_integer": rank,
        }

        metadata_eligible_sites += 1

        # Python heapq is a min-heap. Using -rank makes
        # heap[0] the worst (largest) selected rank.
        item = (-rank, key, record)

        if len(candidate_heap) < args.candidate_count:
            heapq.heappush(
                candidate_heap,
                item,
            )
            continue

        worst_selected_rank = -candidate_heap[0][0]

        if rank < worst_selected_rank:
            heapq.heapreplace(
                candidate_heap,
                item,
            )

    sites_returncode = sites.wait()

    if sites_returncode != 0:
        raise RuntimeError(f"bcftools metadata query failed with exit code {sites_returncode}")

    if len(candidate_heap) != args.candidate_count:
        raise RuntimeError(
            "Could not collect requested number of candidates: "
            f"{len(candidate_heap):,} / "
            f"{args.candidate_count:,}"
        )

    candidates = [item[2] for item in candidate_heap]

    candidate_keys = {
        (
            str(row["chrom"]),
            int(row["pos"]),
            str(row["ref"]),
            str(row["alt"]),
        ): row
        for row in candidates
    }

    if len(candidate_keys) != len(candidates):
        raise RuntimeError("Duplicate variant keys detected among candidates")

    #
    # Pass 2
    #
    # Retrieve GT only for the deterministic candidate prefix.
    #

    with tempfile.TemporaryDirectory(prefix="popgenlm-v02-") as temp_dir:
        regions_path = Path(temp_dir) / "candidate_regions.tsv"

        region_coordinates = sorted(
            {
                (
                    str(row["chrom"]),
                    int(row["pos"]),
                )
                for row in candidates
            },
            key=lambda value: (
                int(value[0]),
                value[1],
            ),
        )

        with regions_path.open("w") as handle:
            for chrom, pos in region_coordinates:
                print(
                    chrom,
                    pos,
                    pos,
                    sep="\t",
                    file=handle,
                )

        query_gt_cmd = [
            "bcftools",
            "query",
            "-R",
            str(regions_path),
            "-i",
            'FILTER="PASS" && N_ALT=1 && TYPE="snp"',
            "-f",
            r"%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n",
            str(args.vcf),
        ]

        gt_query = subprocess.Popen(
            query_gt_cmd,
            stdout=subprocess.PIPE,
            text=True,
        )

        assert gt_query.stdout is not None

        seen_candidate_keys: set[tuple[str, int, str, str]] = set()

        population_eligible: list[dict[str, object]] = []

        population_excluded: Counter[str] = Counter()

        retrieved_records = 0

        for line in gt_query.stdout:
            fields = line.rstrip("\n").split("\t")

            if len(fields) < 5:
                population_excluded["malformed_gt_row"] += 1
                continue

            chrom, pos_text, ref, alt = fields[:4]
            genotypes = fields[4:]

            try:
                pos = int(pos_text)
            except ValueError:
                population_excluded["invalid_position"] += 1
                continue

            key = (
                chrom,
                pos,
                ref,
                alt,
            )

            # -R selects coordinates, so another VCF record at
            # the same position could also be returned.
            if key not in candidate_keys:
                continue

            if key in seen_candidate_keys:
                raise RuntimeError(
                    f"Duplicate candidate record returned: {chrom}:{pos}:{ref}:{alt}"
                )

            seen_candidate_keys.add(key)

            retrieved_records += 1

            if len(genotypes) != args.expected_accessions:
                raise RuntimeError(
                    f"{chrom}:{pos}:{ref}:{alt} has "
                    f"{len(genotypes)} genotypes; expected "
                    f"{args.expected_accessions}"
                )

            summary = summarize_biallelic_genotypes(genotypes)

            maf = float(summary["maf"])
            call_rate = float(summary["allele_call_rate"])

            if maf <= 0:
                population_excluded["nonpolymorphic"] += 1
                continue

            if call_rate < args.min_call_rate:
                population_excluded["low_call_rate"] += 1
                continue

            candidate = candidate_keys[key]

            population_eligible.append(
                {
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "filter": candidate["filter"],
                    "fasta_chrom": candidate["fasta_chrom"],
                    **summary,
                    "frequency_class": frequency_class(maf),
                    "alt_major": (float(summary["af_alt"]) > 0.5),
                    "edge_padded": False,
                    "selection_hash": candidate["selection_hash"],
                    "selection_rank_integer": candidate["selection_rank_integer"],
                }
            )

        gt_returncode = gt_query.wait()

        if gt_returncode != 0:
            raise RuntimeError(f"bcftools GT query failed with exit code {gt_returncode}")

    if retrieved_records != len(candidates):
        missing = len(candidates) - retrieved_records

        raise RuntimeError(
            "Not every candidate variant was retrieved. "
            f"Expected {len(candidates):,}, "
            f"retrieved {retrieved_records:,}; "
            f"difference {missing:,}."
        )

    if len(population_eligible) < args.target:
        raise RuntimeError(
            "Candidate prefix did not contain enough eligible "
            "variants. "
            f"Eligible: {len(population_eligible):,}; "
            f"target: {args.target:,}. "
            "Rerun with a larger --candidate-count."
        )

    #
    # Because the globally lowest candidate prefix already
    # contains >= target eligible variants, the target lowest
    # eligible SHA256 ranks cannot occur outside that prefix.
    #

    population_eligible.sort(key=lambda row: int(row["selection_rank_integer"]))

    selected = population_eligible[: args.target]

    for rank, row in enumerate(
        selected,
        start=1,
    ):
        row["selection_rank"] = rank

    # Release table is coordinate sorted for human inspection.
    selected.sort(
        key=lambda row: (
            int(str(row["chrom"])),
            int(row["pos"]),
            str(row["ref"]),
            str(row["alt"]),
        )
    )

    fieldnames = [
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
    ]

    with args.output.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(selected)

    chrom_counts = Counter(str(row["chrom"]) for row in selected)

    frequency_counts = Counter(str(row["frequency_class"]) for row in selected)

    alt_major_count = sum(bool(row["alt_major"]) for row in selected)

    tie_count = sum(float(row["af_alt"]) == 0.5 for row in selected)

    bcftools_version = subprocess.check_output(
        ["bcftools", "--version"],
        text=True,
    ).splitlines()[0]

    metadata = {
        "benchmark_id": ("popgenlm_v0.2_1001g_population_10000"),
        "release_target": "v0.2.0",
        "dataset_id": "1001g_1135_v3.1",
        "source_filename": args.vcf.name,
        "source_sha256": args.source_sha256,
        "expected_accessions": args.expected_accessions,
        "bcftools": bcftools_version,
        "selection": {
            "algorithm": ("global deterministic SHA256 bottom-k selection"),
            "namespace": SELECTION_NAMESPACE,
            "variant_key": "chrom:pos:ref:alt",
            "candidate_count": args.candidate_count,
            "target_count": args.target,
            "metadata_eligible_sites": (metadata_eligible_sites),
            "candidate_population_eligible": (len(population_eligible)),
            "exact_bottom_k_condition_satisfied": (len(population_eligible) >= args.target),
        },
        "eligibility": {
            "vcf_filter": "PASS",
            "variant_type": ("biallelic canonical nuclear SNV"),
            "chromosomes": [
                "1",
                "2",
                "3",
                "4",
                "5",
            ],
            "polymorphic_after_gt_parsing": True,
            "min_allele_call_rate": (args.min_call_rate),
            "gpn_window_size": args.window_size,
            "edge_padding_excluded": True,
        },
        "counts": {
            "bcftools_pass_biallelic_snp_rows": (queried_sites),
            "metadata_eligible_sites": (metadata_eligible_sites),
            "candidate_variants": len(candidates),
            "candidate_records_retrieved": (retrieved_records),
            "candidate_population_eligible": (len(population_eligible)),
            "selected_variants": len(selected),
        },
        "metadata_excluded": dict(sorted(metadata_excluded.items())),
        "population_excluded": dict(sorted(population_excluded.items())),
        "selected_chromosome_counts": dict(sorted(chrom_counts.items())),
        "selected_frequency_class_counts": dict(sorted(frequency_counts.items())),
        "selected_alt_major_count": (alt_major_count),
        "selected_frequency_tie_count": (tie_count),
        "chromosome_config": chrom_config,
        "output": {
            "filename": args.output.name,
            "sha256": sha256_file(args.output),
        },
    }

    args.metadata.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("PopGenLM Bench v0.2 population benchmark")
    print("=" * 47)

    print(
        "PASS biallelic SNP rows queried:",
        f"{queried_sites:,}",
    )

    print(
        "Metadata-eligible sites:",
        f"{metadata_eligible_sites:,}",
    )

    print(
        "Candidate prefix:",
        f"{len(candidates):,}",
    )

    print(
        "Population-eligible candidates:",
        f"{len(population_eligible):,}",
    )

    print(
        "Final selected variants:",
        f"{len(selected):,}",
    )

    print(
        "Chromosome counts:",
        dict(sorted(chrom_counts.items())),
    )

    print(
        "Frequency classes:",
        dict(sorted(frequency_counts.items())),
    )

    print(
        "ALT-major variants:",
        f"{alt_major_count:,}",
    )

    print(
        "AF_ALT == 0.5 ties:",
        f"{tie_count:,}",
    )

    print(
        "Output SHA256:",
        sha256_file(args.output),
    )

    print()
    print(
        "EXACT DETERMINISTIC BOTTOM-K:",
        "YES",
    )

    print()
    print("Wrote:", args.output)
    print("Wrote:", args.metadata)


if __name__ == "__main__":
    main()
