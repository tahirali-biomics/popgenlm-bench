"""Validate benchmark REF alleles against an exact reference FASTA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chrom_config(path: Path) -> dict[str, dict[str, str | int]]:
    result: dict[str, dict[str, str | int]] = {}

    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        required = {"chrom", "length", "fasta_chrom"}

        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Chromosome config requires columns: chrom, length, fasta_chrom")

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


def load_target_sequences(
    fasta: Path,
    accessions: set[str],
) -> dict[str, bytearray]:
    sequences = {accession: bytearray() for accession in accessions}

    current: str | None = None

    with fasta.open() as handle:
        for line in handle:
            if line.startswith(">"):
                accession = line[1:].strip().split()[0]
                current = accession if accession in sequences else None
                continue

            if current is not None:
                sequences[current].extend(line.strip().upper().encode("ascii"))

    missing = [accession for accession, sequence in sequences.items() if not sequence]

    if missing:
        raise ValueError(f"Reference FASTA is missing required accessions: {missing}")

    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--chrom-config", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--expected-fasta-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=10_000)
    parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()

    config = read_chrom_config(args.chrom_config)

    required_accessions = {str(entry["fasta_chrom"]) for entry in config.values()}

    observed_fasta_sha256 = sha256_file(args.fasta)

    fasta_hash_ok = observed_fasta_sha256 == args.expected_fasta_sha256

    sequences = load_target_sequences(
        args.fasta,
        required_accessions,
    )

    length_checks: dict[str, dict[str, object]] = {}
    length_mismatches = 0

    for chrom, entry in config.items():
        accession = str(entry["fasta_chrom"])
        expected_length = int(entry["length"])
        observed_length = len(sequences[accession])

        matches = expected_length == observed_length

        if not matches:
            length_mismatches += 1

        length_checks[chrom] = {
            "fasta_chrom": accession,
            "expected_length": expected_length,
            "observed_length": observed_length,
            "matches": matches,
        }

    total_rows = 0
    reference_matches = 0
    reference_mismatches = 0
    mapping_mismatches = 0
    out_of_bounds = 0

    mismatch_examples: list[dict[str, object]] = []

    with args.benchmark.open() as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        required_columns = {
            "chrom",
            "pos",
            "ref",
            "alt",
            "fasta_chrom",
        }

        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Benchmark is missing required reference-validation columns")

        for row in reader:
            total_rows += 1

            chrom = str(row["chrom"])
            pos = int(row["pos"])
            ref = str(row["ref"]).upper()

            if chrom not in config:
                mapping_mismatches += 1
                continue

            expected_accession = str(config[chrom]["fasta_chrom"])

            observed_accession = str(row["fasta_chrom"])

            if observed_accession != expected_accession:
                mapping_mismatches += 1

                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "reason": "fasta_chrom_mapping",
                            "chrom": chrom,
                            "pos": pos,
                            "benchmark_fasta_chrom": observed_accession,
                            "expected_fasta_chrom": expected_accession,
                        }
                    )

                continue

            sequence = sequences[expected_accession]

            if pos < 1 or pos > len(sequence):
                out_of_bounds += 1

                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "reason": "out_of_bounds",
                            "chrom": chrom,
                            "pos": pos,
                            "reference_length": len(sequence),
                        }
                    )

                continue

            observed_ref = chr(sequence[pos - 1])

            if observed_ref == ref:
                reference_matches += 1
            else:
                reference_mismatches += 1

                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "reason": "ref_mismatch",
                            "chrom": chrom,
                            "pos": pos,
                            "benchmark_ref": ref,
                            "reference_ref": observed_ref,
                            "fasta_chrom": expected_accession,
                        }
                    )

    expected_rows_ok = total_rows == args.expected_rows

    passed = all(
        [
            fasta_hash_ok,
            expected_rows_ok,
            length_mismatches == 0,
            mapping_mismatches == 0,
            out_of_bounds == 0,
            reference_mismatches == 0,
            reference_matches == args.expected_rows,
        ]
    )

    report = {
        "status": "PASS" if passed else "FAIL",
        "benchmark": {
            "filename": args.benchmark.name,
            "sha256": sha256_file(args.benchmark),
            "expected_rows": args.expected_rows,
            "observed_rows": total_rows,
        },
        "reference": {
            "filename": args.fasta.name,
            "expected_sha256": args.expected_fasta_sha256,
            "observed_sha256": observed_fasta_sha256,
            "sha256_matches": fasta_hash_ok,
        },
        "chromosome_lengths": length_checks,
        "validation": {
            "reference_matches": reference_matches,
            "reference_mismatches": reference_mismatches,
            "mapping_mismatches": mapping_mismatches,
            "out_of_bounds": out_of_bounds,
        },
        "mismatch_examples": mismatch_examples,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "FASTA SHA256 matches:",
        "YES" if fasta_hash_ok else "NO",
    )

    print(
        "Chromosome lengths matching:",
        f"{len(config) - length_mismatches}/{len(config)}",
    )

    print(
        "Benchmark rows:",
        f"{total_rows:,}",
    )

    print(
        "REF matches:",
        f"{reference_matches:,}/{total_rows:,}",
    )

    print("REF mismatches:", reference_mismatches)
    print("Mapping mismatches:", mapping_mismatches)
    print("Out of bounds:", out_of_bounds)
    print()

    if passed:
        print("TAIR10.1 REFERENCE VALIDATION PASSED")
    else:
        print("TAIR10.1 REFERENCE VALIDATION FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
