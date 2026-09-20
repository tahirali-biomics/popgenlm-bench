#!/usr/bin/env python3
"""Annotate the v0.3 master table with pinned TAIR10.1 genomic contexts."""

import argparse
import bisect
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote

MASTER_SHA256 = "f46dcfac597bfc47b9967aa6eb3aa39d9368de94f7b25517eccb0c5902f81b63"
GFF_SHA256 = "d6a3aa3c00df97cf0a458608572fbe721424a4740e64e9ddef9398c4eef7db5f"

LENGTHS = {
    "NC_003070.9": 30427671,
    "NC_003071.7": 19698289,
    "NC_003074.8": 23459830,
    "NC_003075.7": 18585056,
    "NC_003076.8": 26975502,
    "NC_037304.1": 367808,
    "NC_000932.1": 154478,
}
NUCLEAR = {
    "NC_003070.9",
    "NC_003071.7",
    "NC_003074.8",
    "NC_003075.7",
    "NC_003076.8",
}
ORGANELLAR = {"NC_037304.1", "NC_000932.1"}

ADDED_FIELDS = [
    "primary_context",
    "overlaps_cds",
    "overlaps_exon",
    "overlaps_gene_feature",
    "overlaps_pseudogene_feature",
    "overlaps_explicit_utr",
    "gene_overlap_count",
    "gene_ids",
    "locus_tags",
    "gene_symbols",
    "gene_biotypes",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_attributes(text):
    result = {}
    for item in text.split(";"):
        if not item:
            continue
        key, separator, value = item.partition("=")
        if separator:
            result[key] = unquote(value)
    return result


def variant_key(row):
    return (
        row["fasta_chrom"],
        int(row["pos"]),
        row["ref"],
        row["alt"],
    )


def write_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--gff", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--allow-unpinned-inputs", action="store_true")
    args = parser.parse_args()

    master = args.master.resolve()
    gff = args.gff.resolve()
    output_dir = args.output_dir.resolve()

    if not args.allow_unpinned_inputs and sha256(master) != MASTER_SHA256:
        raise ValueError("H3 master SHA-256 mismatch")
    if not args.allow_unpinned_inputs and sha256(gff) != GFF_SHA256:
        raise ValueError("GFF3 SHA-256 mismatch")

    with master.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        input_fields = reader.fieldnames or []
        rows = list(reader)

    required = {"fasta_chrom", "pos", "ref", "alt"}
    if not required <= set(input_fields):
        raise ValueError("master table lacks required variant columns")
    if len(rows) != 10000:
        raise ValueError("master table must contain exactly 10,000 rows")

    keys = [variant_key(row) for row in rows]
    if len(set(keys)) != 10000:
        raise ValueError("master table contains duplicate normalized keys")

    annotations = []
    chromosome_positions = {}
    chromosome_indices = {}

    for index, row in enumerate(rows):
        chrom = row["fasta_chrom"]
        pos = int(row["pos"])
        if chrom not in NUCLEAR:
            raise ValueError(f"non-nuclear master-table chromosome {chrom}")
        if not 1 <= pos <= LENGTHS[chrom]:
            raise ValueError(f"variant outside chromosome bounds: {keys[index]}")

        annotations.append(
            {
                "cds": False,
                "exon": False,
                "gene": False,
                "pseudogene": False,
                "utr": False,
                "gene_ids": set(),
                "locus_tags": set(),
                "gene_symbols": set(),
                "gene_biotypes": set(),
            }
        )
        chromosome_positions.setdefault(chrom, []).append((pos, index))

    for chrom, pairs in chromosome_positions.items():
        pairs.sort()
        chromosome_positions[chrom] = [pair[0] for pair in pairs]
        chromosome_indices[chrom] = [pair[1] for pair in pairs]

    feature_counts = Counter()
    organellar_feature_counts = Counter()
    organellar_nonstandard_coordinate_counts = Counter()
    declared_regions = {}
    build = None
    build_accession = None

    relevant = {
        "CDS",
        "exon",
        "gene",
        "pseudogene",
        "five_prime_UTR",
        "three_prime_UTR",
    }

    with gzip.open(gff, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#!genome-build "):
                build = line.strip().split(maxsplit=1)[1]
                continue
            if line.startswith("#!genome-build-accession "):
                build_accession = line.strip().split(maxsplit=1)[1]
                continue
            if line.startswith("##sequence-region "):
                parts = line.strip().split()
                if len(parts) != 4:
                    raise ValueError(f"line {line_number}: invalid sequence-region")
                declared_regions[parts[1]] = (int(parts[2]), int(parts[3]))
                continue
            if not line.strip() or line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"line {line_number}: expected nine GFF3 fields")

            seqid, _, feature, start_text, end_text, _, _, _, attr_text = fields

            if seqid not in LENGTHS:
                raise ValueError(f"line {line_number}: unexpected sequence ID {seqid}")

            start = int(start_text)
            end = int(end_text)

            if seqid in ORGANELLAR:
                organellar_feature_counts[feature] += 1
                if (
                    start < 1
                    or end < 1
                    or start > end
                    or start > LENGTHS[seqid]
                    or end > LENGTHS[seqid]
                ):
                    organellar_nonstandard_coordinate_counts[feature] += 1
                continue

            if start < 1 or end < start or end > LENGTHS[seqid]:
                raise ValueError(f"line {line_number}: invalid nuclear feature bounds")

            feature_counts[feature] += 1

            if feature not in relevant:
                continue

            positions = chromosome_positions[seqid]
            indices = chromosome_indices[seqid]
            left = bisect.bisect_left(positions, start)
            right = bisect.bisect_right(positions, end)

            if left == right:
                continue

            attrs = parse_attributes(attr_text)

            for offset in range(left, right):
                annotation = annotations[indices[offset]]

                if feature == "CDS":
                    annotation["cds"] = True
                elif feature == "exon":
                    annotation["exon"] = True
                elif feature in {"five_prime_UTR", "three_prime_UTR"}:
                    annotation["utr"] = True
                elif feature in {"gene", "pseudogene"}:
                    if feature == "gene":
                        annotation["gene"] = True
                    else:
                        annotation["pseudogene"] = True

                    gene_id = attrs.get("ID")
                    if not gene_id:
                        raise ValueError(f"line {line_number}: gene feature lacks ID")

                    annotation["gene_ids"].add(gene_id)

                    for source, target in (
                        ("locus_tag", "locus_tags"),
                        ("gene", "gene_symbols"),
                        ("gene_biotype", "gene_biotypes"),
                    ):
                        value = attrs.get(source)
                        if value:
                            annotation[target].add(value)

    if build != "TAIR10.1":
        raise ValueError(f"unexpected genome build {build!r}")
    if build_accession != "NCBI_Assembly:GCF_000001735.4":
        raise ValueError(f"unexpected genome-build accession {build_accession!r}")

    expected_regions = {seqid: (1, length) for seqid, length in LENGTHS.items()}
    if declared_regions != expected_regions:
        raise ValueError("GFF3 sequence-region declarations differ")

    expected_feature_counts = {
        "gene": 33056,
        "pseudogene": 4843,
        "exon": 324327,
        "CDS": 286104,
    }
    for feature, expected in expected_feature_counts.items():
        if feature_counts[feature] != expected:
            raise ValueError(
                f"unexpected nuclear {feature} count: {feature_counts[feature]} != {expected}"
            )

    output_rows = []
    context_counts = Counter()
    multi_gene_rows = 0
    biotype_variant_counts = Counter()

    for row, annotation in zip(rows, annotations):
        genic = annotation["gene"] or annotation["pseudogene"]

        if annotation["cds"]:
            context = "coding_cds"
        elif annotation["exon"]:
            context = "exonic_non_cds"
        elif genic:
            context = "other_genic"
        else:
            context = "intergenic"

        if context == "coding_cds" and not annotation["cds"]:
            raise AssertionError("coding context without CDS")
        if context == "exonic_non_cds" and (annotation["cds"] or not annotation["exon"]):
            raise AssertionError("invalid exonic-non-CDS context")
        if context == "other_genic" and (annotation["cds"] or annotation["exon"] or not genic):
            raise AssertionError("invalid other-genic context")
        if context == "intergenic" and (annotation["cds"] or annotation["exon"] or genic):
            raise AssertionError("invalid intergenic context")

        gene_count = len(annotation["gene_ids"])
        if gene_count > 1:
            multi_gene_rows += 1

        for biotype in annotation["gene_biotypes"]:
            biotype_variant_counts[biotype] += 1

        output = dict(row)
        output.update(
            {
                "primary_context": context,
                "overlaps_cds": str(annotation["cds"]),
                "overlaps_exon": str(annotation["exon"]),
                "overlaps_gene_feature": str(annotation["gene"]),
                "overlaps_pseudogene_feature": str(annotation["pseudogene"]),
                "overlaps_explicit_utr": str(annotation["utr"]),
                "gene_overlap_count": str(gene_count),
                "gene_ids": "|".join(sorted(annotation["gene_ids"])),
                "locus_tags": "|".join(sorted(annotation["locus_tags"])),
                "gene_symbols": "|".join(sorted(annotation["gene_symbols"])),
                "gene_biotypes": "|".join(sorted(annotation["gene_biotypes"])),
            }
        )

        output_rows.append(output)
        context_counts[context] += 1

    if sum(context_counts.values()) != 10000:
        raise AssertionError("context counts do not sum to 10,000")

    output_fields = input_fields + ADDED_FIELDS
    output_path = output_dir / "popgenlm_v03_master_10000_annotated.tsv"
    summary_path = output_dir / "summary.json"
    provenance_path = output_dir / "provenance.json"

    for target in (output_path, summary_path, provenance_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

    partial = output_path.with_suffix(".tsv.partial")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=output_fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    partial.replace(output_path)

    with output_path.open(newline="") as handle:
        check_reader = csv.DictReader(handle, delimiter="\t")
        check_rows = list(check_reader)
        check_fields = check_reader.fieldnames or []

    if check_fields != output_fields:
        raise ValueError("written annotation header changed")
    if len(check_rows) != 10000:
        raise ValueError("written annotation row count changed")
    if [variant_key(row) for row in check_rows] != keys:
        raise ValueError("written annotation row order or keys changed")

    output_hash = sha256(output_path)

    summary = {
        "phase": "H4",
        "status": "PASS",
        "rows": 10000,
        "columns": len(output_fields),
        "context_counts": dict(sorted(context_counts.items())),
        "multi_gene_rows": multi_gene_rows,
        "variants_by_gene_biotype": dict(sorted(biotype_variant_counts.items())),
        "explicit_utr_overlap_rows": sum(item["utr"] for item in annotations),
        "annotation_contract": {
            "coding_cds": "overlaps at least one CDS",
            "exonic_non_cds": "overlaps exon but no CDS",
            "other_genic": "overlaps gene/pseudogene but no CDS or exon",
            "intergenic": "overlaps no gene or pseudogene",
        },
        "assembly": "TAIR10.1 / GCF_000001735.4",
        "coordinates": "GFF3 and variants are one-based; intervals inclusive",
        "organellar_policy": "organellar records parsed and excluded; nonstandard "
        "trans-spliced coordinates counted; nuclear bounds strict",
        "organellar_nonstandard_coordinate_counts": dict(
            sorted(organellar_nonstandard_coordinate_counts.items())
        ),
        "annotated_table": str(output_path),
        "annotated_table_sha256": output_hash,
    }
    write_json(summary_path, summary)

    provenance = {
        "phase": "H4",
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_commit": args.repo_commit,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            "h3_master": {
                "path": str(master),
                "bytes": master.stat().st_size,
                "sha256": sha256(master),
            },
            "gff3": {
                "path": str(gff),
                "bytes": gff.stat().st_size,
                "sha256": sha256(gff),
            },
        },
        "outputs": {
            "annotated_table": {
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": output_hash,
            },
            "summary": {
                "path": str(summary_path),
                "bytes": summary_path.stat().st_size,
                "sha256": sha256(summary_path),
            },
        },
        "nuclear_feature_counts": dict(sorted(feature_counts.items())),
        "organellar_feature_counts": dict(sorted(organellar_feature_counts.items())),
        "organellar_nonstandard_coordinate_counts": dict(
            sorted(organellar_nonstandard_coordinate_counts.items())
        ),
        "validations": [
            "pinned input hashes matched",
            "assembly and build accession matched",
            "all seven sequence-region declarations matched",
            "only the five nuclear chromosomes were annotated",
            "all 10,000 input rows and their order were preserved",
            "exclusive context rules passed",
            "written output was reopened and key-checked",
        ],
    }
    write_json(provenance_path, provenance)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("H4_ANNOTATION_PASS")


if __name__ == "__main__":
    main()
