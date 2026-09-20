#!/usr/bin/env python3
"""Extract and validate dosages for the frozen PopGenLM 10K variants."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

TABLE_SHA256 = "428f6948180f6aa95366d931a49b1a86ccfb7ea98663875271af8bbb2f9e8b47"
VCF_SHA256 = "7f825afb784b2424e35501fd8b88c32a4798013eb9fe762724addfb249007e7a"
VCF_BYTES = 19230910695
INDEX_SHA256 = "2862ff882d9594875f28b4eef95c907cb517129b6edddd2b7554e25ab38cb370"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--allow-unpinned-inputs", action="store_true")
    args = parser.parse_args()
    try:
        import pysam
    except ImportError as exc:
        raise SystemExit("extract_genotypes.py requires optional dependency pysam") from exc

    table = args.table.resolve()
    vcf_path = args.vcf.resolve()
    index_path = Path(str(vcf_path) + ".tbi")
    output_dir = args.output_dir.resolve()

    if not args.allow_unpinned_inputs and digest(table) != TABLE_SHA256:
        raise ValueError("annotated-table SHA-256 mismatch")
    if not args.allow_unpinned_inputs and vcf_path.stat().st_size != VCF_BYTES:
        raise ValueError("VCF byte-size mismatch")
    if not args.allow_unpinned_inputs and digest(index_path) != INDEX_SHA256:
        raise ValueError("VCF-index SHA-256 mismatch")

    with table.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    if len(rows) != 10000:
        raise ValueError("annotated table must contain 10,000 rows")

    keys = [(row["chrom"], int(row["pos"]), row["ref"], row["alt"]) for row in rows]
    if len(set(keys)) != 10000:
        raise ValueError("annotated table contains duplicate VCF keys")

    vcf = pysam.VariantFile(str(vcf_path))
    samples = tuple(vcf.header.samples)

    if len(samples) != 1135 or len(set(samples)) != 1135:
        raise ValueError("unexpected VCF sample set")

    dosages = np.full((10000, 1135), -1, dtype=np.int8)
    chromosome_counts = Counter()
    total_missing = 0
    format_counts = Counter()

    for row_index, row in enumerate(rows):
        chrom = row["chrom"]
        pos = int(row["pos"])
        ref = row["ref"]
        alt = row["alt"]
        key = (chrom, pos, ref, alt)

        matches = [
            record
            for record in vcf.fetch(chrom, pos - 1, pos)
            if record.pos == pos and record.ref == ref and record.alts == (alt,)
        ]

        if len(matches) != 1:
            raise ValueError(f"{key}: expected one exact biallelic record, found {len(matches)}")

        record = matches[0]
        format_counts[tuple(record.format.keys())] += 1

        ac_ref = 0
        ac_alt = 0
        n_missing = 0
        n_heterozygous = 0

        for sample_index, sample in enumerate(samples):
            gt = record.samples[sample].get("GT")

            if gt is None or len(gt) != 2 or any(allele is None for allele in gt):
                n_missing += 1
                continue

            if any(allele not in {0, 1} for allele in gt):
                raise ValueError(f"{key}: unexpected genotype {gt} for {sample}")

            dosage = int(gt[0]) + int(gt[1])
            dosages[row_index, sample_index] = dosage
            ac_ref += 2 - dosage
            ac_alt += dosage
            n_heterozygous += dosage == 1

        an = ac_ref + ac_alt
        af_alt = ac_alt / an
        call_rate = (len(samples) - n_missing) / len(samples)

        exact_checks = {
            "n_genotypes": (len(samples), int(row["n_genotypes"])),
            "n_missing": (n_missing, int(row["n_missing"])),
            "n_heterozygous": (n_heterozygous, int(row["n_heterozygous"])),
            "ac_ref": (ac_ref, int(row["ac_ref"])),
            "ac_alt": (ac_alt, int(row["ac_alt"])),
            "an": (an, int(row["an"])),
        }

        for field, (observed, expected) in exact_checks.items():
            if observed != expected:
                raise ValueError(f"{key}: {field} mismatch {observed} != {expected}")

        if not math.isclose(
            af_alt,
            float(row["af_alt"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{key}: AF mismatch")

        if not math.isclose(
            call_rate,
            float(row["allele_call_rate"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{key}: call-rate mismatch")

        chromosome_counts[chrom] += 1
        total_missing += n_missing

        if (row_index + 1) % 500 == 0:
            print(
                f"EXTRACTION_PROGRESS {row_index + 1}/10000",
                flush=True,
            )

    dosage_path = output_dir / "dosages_10000x1135.npz"
    sample_path = output_dir / "sample_ids.txt"
    index_output = output_dir / "variant_index.tsv"
    summary_path = output_dir / "summary.json"
    provenance_path = output_dir / "provenance.json"

    for target in (
        dosage_path,
        sample_path,
        index_output,
        summary_path,
        provenance_path,
    ):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

    dosage_partial = dosage_path.with_suffix(".npz.partial")
    with dosage_partial.open("wb") as handle:
        np.savez_compressed(
            handle,
            dosage=dosages,
            sample_ids=np.asarray(samples, dtype="U64"),
            selection_rank=np.asarray(
                [int(row["selection_rank"]) for row in rows],
                dtype=np.int32,
            ),
            chrom=np.asarray([row["chrom"] for row in rows], dtype="U2"),
            pos=np.asarray(
                [int(row["pos"]) for row in rows],
                dtype=np.int32,
            ),
            ref=np.asarray([row["ref"] for row in rows], dtype="U1"),
            alt=np.asarray([row["alt"] for row in rows], dtype="U1"),
        )
    dosage_partial.replace(dosage_path)

    sample_partial = sample_path.with_suffix(".txt.partial")
    sample_partial.write_text("\n".join(samples) + "\n")
    sample_partial.replace(sample_path)

    index_partial = index_output.with_suffix(".tsv.partial")
    with index_partial.open("w", newline="") as handle:
        fields = [
            "row_index",
            "selection_rank",
            "chrom",
            "fasta_chrom",
            "pos",
            "ref",
            "alt",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row_index, row in enumerate(rows):
            writer.writerow(
                {
                    "row_index": row_index,
                    "selection_rank": row["selection_rank"],
                    "chrom": row["chrom"],
                    "fasta_chrom": row["fasta_chrom"],
                    "pos": row["pos"],
                    "ref": row["ref"],
                    "alt": row["alt"],
                }
            )
    index_partial.replace(index_output)

    with np.load(dosage_path, allow_pickle=False) as saved:
        saved_dosage = saved["dosage"]
        if saved_dosage.shape != (10000, 1135):
            raise ValueError("saved dosage shape changed")
        if saved_dosage.dtype != np.int8:
            raise ValueError("saved dosage dtype changed")
        if not np.array_equal(saved_dosage, dosages):
            raise ValueError("saved dosage matrix differs")

    summary = {
        "phase": "H5A",
        "status": "PASS",
        "variants": 10000,
        "samples": 1135,
        "matrix_shape": [10000, 1135],
        "matrix_dtype": "int8",
        "dosage_encoding": {
            "-1": "missing genotype",
            "0": "0 ALT alleles",
            "1": "1 ALT allele",
            "2": "2 ALT alleles",
        },
        "chromosome_counts": dict(sorted(chromosome_counts.items())),
        "total_missing_genotypes": total_missing,
        "format_counts": {
            "|".join(fields): count for fields, count in sorted(format_counts.items())
        },
        "dosage_sha256": digest(dosage_path),
        "variant_index_sha256": digest(index_output),
        "sample_ids_sha256": digest(sample_path),
    }
    write_json(summary_path, summary)

    provenance = {
        "phase": "H5A",
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_commit": args.repo_commit,
        "extractor": str(Path(__file__).resolve()),
        "extractor_sha256": digest(Path(__file__).resolve()),
        "inputs": {
            "annotated_table": {
                "path": str(table),
                "bytes": table.stat().st_size,
                "sha256": digest(table),
            },
            "vcf": {
                "path": str(vcf_path),
                "bytes": vcf_path.stat().st_size,
                "sha256": VCF_SHA256,
                "sha256_status": "reused from completed transfer verification",
            },
            "vcf_index": {
                "path": str(index_path),
                "bytes": index_path.stat().st_size,
                "sha256": digest(index_path),
            },
        },
        "outputs": {
            "dosages": {
                "path": str(dosage_path),
                "bytes": dosage_path.stat().st_size,
                "sha256": digest(dosage_path),
            },
            "variant_index": {
                "path": str(index_output),
                "bytes": index_output.stat().st_size,
                "sha256": digest(index_output),
            },
            "sample_ids": {
                "path": str(sample_path),
                "bytes": sample_path.stat().st_size,
                "sha256": digest(sample_path),
            },
            "summary": {
                "path": str(summary_path),
                "bytes": summary_path.stat().st_size,
                "sha256": digest(summary_path),
            },
        },
        "validations": [
            "10,000 exact indexed biallelic VCF records found",
            "1,135 unique samples preserved in header order",
            "all GT values were diploid, missing, or biallelic 0/1",
            "all AC, AN, AF, MAC-related inputs and call rates matched",
            "saved dosage matrix was reopened and compared exactly",
        ],
    }
    write_json(provenance_path, provenance)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("H5A_GENOTYPE_EXTRACTION_PASS")


if __name__ == "__main__":
    main()
