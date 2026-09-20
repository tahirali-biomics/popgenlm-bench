#!/usr/bin/env python3
"""Build and validate the PopGenLM v0.3 GPN–PlantCAD–PhyloP master table."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_HASHES = {
    "population": "cbf535008593ed45fc7ee8619ed7a87c4c3bb2e741e721a4cb97832f3ba5b374",
    "gpn_plantcad": "9733cbac5fac5eb9fa9f5d9df3095731e85a81b54ab50d16f93c388e6e6eece7",
    "phylop": "5e3a1c868f8b5f59945c595337ac51f4d2d7975d3f59b3ecc8012be3292b12c5",
}

CHROM_MAP = {
    "1": "NC_003070.9",
    "2": "NC_003071.7",
    "3": "NC_003074.8",
    "4": "NC_003075.7",
    "5": "NC_003076.8",
}
VALID_REFSEQ = set(CHROM_MAP.values())

POPULATION_REQUIRED = {
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
}
MODEL_REQUIRED = {
    "chrom",
    "pos",
    "ref",
    "alt",
    "af_alt",
    "alt_major",
    "orientation",
    "gpn_score_ref_alt",
    "gpn_score_minor_vs_major",
    "plantcad_score_ref_alt",
    "plantcad_score_minor_vs_major",
}
PHYLOP_REQUIRED = {
    "fasta_chrom",
    "pos",
    "ref",
    "alt",
    "phylop",
    "status",
}

ADDED_FIELDS = [
    "orientation",
    "gpn_score_ref_alt",
    "gpn_score_minor_vs_major",
    "plantcad_score_ref_alt",
    "plantcad_score_minor_vs_major",
    "phylop",
    "phylop_status",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path, required):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    return fields, rows


def normalized_key(row):
    raw = row.get("fasta_chrom", "").strip() or row.get("chrom", "").strip()
    if raw in CHROM_MAP:
        chrom = CHROM_MAP[raw]
    elif raw in VALID_REFSEQ:
        chrom = raw
    else:
        raise ValueError(f"unknown chromosome {raw!r}")

    return (
        chrom,
        int(row["pos"]),
        row["ref"].upper(),
        row["alt"].upper(),
    )


def keyed(rows, label):
    result = {}
    for row in rows:
        key = normalized_key(row)
        if key in result:
            raise ValueError(f"{label}: duplicate key {key}")
        result[key] = row
    return result


def parse_bool(value):
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid Boolean value {value!r}")


def finite(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: non-finite value {value!r}")
    return number


def close(left, right, tolerance=1e-12):
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def write_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True, type=Path)
    parser.add_argument("--gpn-plantcad", required=True, type=Path)
    parser.add_argument("--phylop", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--allow-unpinned-inputs", action="store_true")
    args = parser.parse_args()

    inputs = {
        "population": args.population.resolve(),
        "gpn_plantcad": args.gpn_plantcad.resolve(),
        "phylop": args.phylop.resolve(),
    }

    for label, path in inputs.items():
        observed = sha256(path)
        expected = EXPECTED_HASHES[label]
        if not args.allow_unpinned_inputs and observed != expected:
            raise ValueError(f"{label}: SHA-256 mismatch: expected {expected}, observed {observed}")

    population_fields, population_rows = read_tsv(inputs["population"], POPULATION_REQUIRED)
    _, model_rows = read_tsv(inputs["gpn_plantcad"], MODEL_REQUIRED)
    _, phylop_rows = read_tsv(inputs["phylop"], PHYLOP_REQUIRED)

    if not (len(population_rows) == len(model_rows) == len(phylop_rows) == 10000):
        raise ValueError("all three inputs must contain exactly 10,000 rows")

    population_by_key = keyed(population_rows, "population")
    model_by_key = keyed(model_rows, "GPN–PlantCAD")
    phylop_by_key = keyed(phylop_rows, "PhyloP")

    key_set = set(population_by_key)
    if key_set != set(model_by_key) or key_set != set(phylop_by_key):
        raise ValueError("input normalized-key sets do not match exactly")

    output_rows = []
    phylop_status = Counter()
    chromosome_counts = Counter()
    frequency_counts = Counter()
    seen_ranks = set()

    for population in population_rows:
        key = normalized_key(population)
        model = model_by_key[key]
        phylop = phylop_by_key[key]

        rank = int(population["selection_rank"])
        if rank in seen_ranks:
            raise ValueError(f"duplicate selection rank {rank}")
        seen_ranks.add(rank)

        chrom, _pos, ref, alt = key
        if ref not in "ACGT" or alt not in "ACGT" or ref == alt:
            raise ValueError(f"invalid canonical SNV {key}")
        if population["fasta_chrom"] != chrom:
            raise ValueError(f"population FASTA chromosome mismatch at {key}")
        if CHROM_MAP[population["chrom"]] != chrom:
            raise ValueError(f"population chromosome mapping mismatch at {key}")

        n_genotypes = int(population["n_genotypes"])
        n_missing = int(population["n_missing"])
        n_heterozygous = int(population["n_heterozygous"])
        ac_ref = int(population["ac_ref"])
        ac_alt = int(population["ac_alt"])
        an = int(population["an"])
        mac = int(population["mac"])

        if not 0 <= n_missing <= n_genotypes:
            raise ValueError(f"invalid missing count at {key}")
        if not 0 <= n_heterozygous <= n_genotypes - n_missing:
            raise ValueError(f"invalid heterozygous count at {key}")
        if an != ac_ref + ac_alt:
            raise ValueError(f"AC/AN inconsistency at {key}")
        if an != 2 * (n_genotypes - n_missing):
            raise ValueError(f"diploid genotype/AN inconsistency at {key}")
        if mac != min(ac_ref, ac_alt):
            raise ValueError(f"MAC inconsistency at {key}")

        af_alt = finite(population["af_alt"], f"population AF at {key}")
        maf = finite(population["maf"], f"population MAF at {key}")
        call_rate = finite(population["allele_call_rate"], f"call rate at {key}")

        if not 0.0 <= af_alt <= 1.0:
            raise ValueError(f"AF outside [0,1] at {key}")
        if af_alt == 0.5:
            raise ValueError(f"unexpected frequency tie at {key}")
        if not close(af_alt, ac_alt / an):
            raise ValueError(f"AF/allele-count inconsistency at {key}")
        if not close(maf, min(af_alt, 1.0 - af_alt)):
            raise ValueError(f"MAF inconsistency at {key}")
        if not close(call_rate, (n_genotypes - n_missing) / n_genotypes):
            raise ValueError(f"call-rate inconsistency at {key}")

        alt_major = parse_bool(population["alt_major"])
        if alt_major != (af_alt > 0.5):
            raise ValueError(f"population orientation mismatch at {key}")

        model_af = finite(model["af_alt"], f"model AF at {key}")
        model_alt_major = parse_bool(model["alt_major"])
        if not close(model_af, af_alt) or model_alt_major != alt_major:
            raise ValueError(f"cross-table AF/orientation mismatch at {key}")

        expected_orientation = "ref_minor" if alt_major else "alt_minor"
        if model["orientation"] != expected_orientation:
            raise ValueError(f"model orientation label mismatch at {key}")

        gpn_raw = finite(model["gpn_score_ref_alt"], f"GPN raw at {key}")
        gpn_oriented = finite(model["gpn_score_minor_vs_major"], f"GPN oriented at {key}")
        plantcad_raw = finite(model["plantcad_score_ref_alt"], f"PlantCAD raw at {key}")
        plantcad_oriented = finite(
            model["plantcad_score_minor_vs_major"],
            f"PlantCAD oriented at {key}",
        )

        sign = -1.0 if alt_major else 1.0
        if not close(gpn_oriented, sign * gpn_raw):
            raise ValueError(f"GPN score orientation mismatch at {key}")
        if not close(plantcad_oriented, sign * plantcad_raw):
            raise ValueError(f"PlantCAD score orientation mismatch at {key}")

        status = phylop["status"]
        score_text = phylop["phylop"].strip()
        if status == "covered":
            finite(score_text, f"PhyloP at {key}")
        elif status == "missing_coverage":
            if score_text:
                raise ValueError(f"missing PhyloP row has a score at {key}")
        else:
            raise ValueError(f"unknown PhyloP status {status!r} at {key}")

        output = {field: population[field] for field in population_fields}
        output.update(
            {
                "orientation": model["orientation"],
                "gpn_score_ref_alt": model["gpn_score_ref_alt"],
                "gpn_score_minor_vs_major": model["gpn_score_minor_vs_major"],
                "plantcad_score_ref_alt": model["plantcad_score_ref_alt"],
                "plantcad_score_minor_vs_major": model["plantcad_score_minor_vs_major"],
                "phylop": score_text,
                "phylop_status": status,
            }
        )
        output_rows.append(output)

        chromosome_counts[population["chrom"]] += 1
        frequency_counts[population["frequency_class"]] += 1
        phylop_status[status] += 1

    if phylop_status != Counter(
        {
            "covered": 9453,
            "missing_coverage": 547,
        }
    ):
        raise ValueError(f"unexpected PhyloP coverage counts: {phylop_status}")

    output_fields = population_fields + ADDED_FIELDS
    output_path = args.output_dir / "popgenlm_v03_master_10000.tsv"
    summary_path = args.output_dir / "summary.json"
    provenance_path = args.output_dir / "provenance.json"

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

    written_fields, written_rows = read_tsv(output_path, set(output_fields))
    if written_fields != output_fields:
        raise ValueError("written master-table header changed unexpectedly")
    if len(written_rows) != 10000:
        raise ValueError("written master table does not contain 10,000 rows")
    if set(keyed(written_rows, "written master")) != key_set:
        raise ValueError("written master-table keys differ from source keys")

    output_hash = sha256(output_path)

    summary = {
        "phase": "H3",
        "status": "PASS",
        "rows": 10000,
        "unique_normalized_keys": 10000,
        "columns": len(output_fields),
        "chromosome_counts": dict(sorted(chromosome_counts.items())),
        "frequency_class_counts": dict(sorted(frequency_counts.items())),
        "phylop_status_counts": dict(sorted(phylop_status.items())),
        "raw_score_orientation": "log P(ALT) - log P(REF)",
        "population_score_orientation": "log P(minor allele) - log P(major allele)",
        "phylop_semantics": "signed source PhyloP value; not allele-oriented",
        "master_table": str(output_path.resolve()),
        "master_table_sha256": output_hash,
    }
    write_json(summary_path, summary)

    provenance = {
        "phase": "H3",
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_commit": args.repo_commit,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            label: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for label, path in inputs.items()
        },
        "outputs": {
            "master_table": {
                "path": str(output_path.resolve()),
                "bytes": output_path.stat().st_size,
                "sha256": output_hash,
            },
            "summary": {
                "path": str(summary_path.resolve()),
                "bytes": summary_path.stat().st_size,
                "sha256": sha256(summary_path),
            },
        },
        "validations": [
            "all pinned input SHA-256 hashes matched",
            "all inputs contained 10,000 unique normalized keys",
            "the three normalized-key sets matched exactly",
            "population allele counts, AF, MAF and call rate were consistent",
            "GPN and PlantCAD raw and population-oriented scores were finite",
            "both population-oriented score equations were verified",
            "covered PhyloP scores were finite and signed",
            "missing PhyloP rows had empty score fields",
            "written output was reopened and independently structure-checked",
        ],
    }
    write_json(provenance_path, provenance)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("H3_MASTER_TABLE_PASS")


if __name__ == "__main__":
    main()
