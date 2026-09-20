#!/usr/bin/env python3
"""Calculate pairwise LD and deterministic thinning structures."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

EXPECTED = {
    "dosage_sha256": "e84eb449d8fc883c156ecb2d37706d3bbe8c757dc783641f3393d426e397e1c1",
    "index_sha256": "a59f243da9ee16e3a5ea52a6616205115a2d0e3a97a14824b237a918a6803650",
    "repo_commit": "da430ff6ce3890e6a1a09f08fdcfd5fdfccc139b",
    "pair_counts": {
        50_000: 46_972,
        100_000: 90_953,
        250_000: 221_866,
    },
}

FASTA_CHROM = {
    "1": "NC_003070.9",
    "2": "NC_003071.7",
    "3": "NC_003074.8",
    "4": "NC_003075.7",
    "5": "NC_003076.8",
}

PROFILES = (
    ("w050kb_r2ge0.20", 50_000, 0.20),
    ("w100kb_r2ge0.20", 100_000, 0.20),
    ("w250kb_r2ge0.20", 250_000, 0.20),
    ("w100kb_r2ge0.10", 100_000, 0.10),
    ("w100kb_r2ge0.50", 100_000, 0.50),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def open_tsv_atomic(path):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    return path, partial, partial.open("x", newline="", encoding="utf-8")


def read_inputs(dosage_path, index_path, allow_unpinned_inputs=False):
    if not allow_unpinned_inputs and sha256(dosage_path) != EXPECTED["dosage_sha256"]:
        raise ValueError("dosage SHA-256 mismatch")
    if not allow_unpinned_inputs and sha256(index_path) != EXPECTED["index_sha256"]:
        raise ValueError("variant-index SHA-256 mismatch")

    with np.load(dosage_path, allow_pickle=False) as archive:
        dosage = archive["dosage"].copy()
        chrom = archive["chrom"].astype(str)
        pos = archive["pos"].astype(np.int64)
        ref = archive["ref"].astype(str)
        alt = archive["alt"].astype(str)
        ranks = archive["selection_rank"].astype(np.int64)
        sample_ids = archive["sample_ids"].astype(str)

    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_header = [
            "row_index",
            "selection_rank",
            "chrom",
            "fasta_chrom",
            "pos",
            "ref",
            "alt",
        ]
        if reader.fieldnames != expected_header:
            raise ValueError(f"unexpected variant-index header: {reader.fieldnames}")
        rows = list(reader)

    if dosage.shape != (10_000, 1_135):
        raise ValueError(f"unexpected dosage shape: {dosage.shape}")
    if dosage.dtype != np.int8:
        raise ValueError(f"unexpected dosage dtype: {dosage.dtype}")
    if len(rows) != 10_000 or len(set(sample_ids)) != 1_135:
        raise ValueError("row or sample count mismatch")

    observed_values, observed_counts = np.unique(dosage, return_counts=True)
    dosage_counts = {
        int(value): int(count) for value, count in zip(observed_values, observed_counts)
    }
    expected_counts = {-1: 463_505, 0: 10_444_053, 2: 442_442}
    if dosage_counts != expected_counts:
        raise ValueError(f"dosage-count mismatch: {dosage_counts}")

    keys = set()
    previous = None
    for i, row in enumerate(rows):
        observed = (
            int(row["row_index"]),
            int(row["selection_rank"]),
            row["chrom"],
            int(row["pos"]),
            row["ref"],
            row["alt"],
        )
        expected = (i, int(ranks[i]), chrom[i], int(pos[i]), ref[i], alt[i])
        if observed != expected:
            raise ValueError(f"index/NPZ mismatch at row {i}")
        if row["fasta_chrom"] != FASTA_CHROM[chrom[i]]:
            raise ValueError(f"RefSeq mapping mismatch at row {i}")

        key = (row["fasta_chrom"], int(pos[i]), ref[i], alt[i])
        if key in keys:
            raise ValueError(f"duplicate key: {key}")
        keys.add(key)

        order_key = (int(chrom[i]), int(pos[i]), ref[i], alt[i])
        if previous is not None and order_key < previous:
            raise ValueError("variants are not in genomic order")
        previous = order_key

    called_counts = (dosage >= 0).sum(axis=1)
    variable = np.zeros(len(rows), dtype=bool)
    for i in range(len(rows)):
        called = dosage[i, dosage[i] >= 0]
        variable[i] = called.size >= 2 and called.min() != called.max()

    if not variable.all():
        raise ValueError("one or more variants are not variable")

    return {
        "dosage": dosage,
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "ranks": ranks,
        "rows": rows,
        "called_counts": called_counts,
        "dosage_counts": dosage_counts,
    }


def candidate_pairs(chrom, pos):
    pairs = []
    counts_by_chrom = Counter()

    for chromosome in ("1", "2", "3", "4", "5"):
        indices = np.flatnonzero(chrom == chromosome)
        right = 0

        for local_left, left in enumerate(indices):
            right = max(right, local_left + 1)

            while right < len(indices) and int(pos[indices[right]]) - int(pos[left]) <= 250_000:
                right += 1

            for local_right in range(local_left + 1, right):
                other = int(indices[local_right])
                distance = int(pos[other]) - int(pos[left])
                pairs.append((int(left), other, distance))
                counts_by_chrom[chromosome] += 1

    counts = {
        window: sum(distance <= window for _, _, distance in pairs)
        for window in EXPECTED["pair_counts"]
    }
    if counts != EXPECTED["pair_counts"]:
        raise ValueError(f"candidate-pair count mismatch: {counts}")

    return pairs, counts, dict(counts_by_chrom)


def pairwise_r2(x, y, minimum_complete=200):
    mask = (x >= 0) & (y >= 0)
    n_complete = int(mask.sum())
    if n_complete < minimum_complete:
        return n_complete, None, "insufficient_complete_samples"

    xv = x[mask].astype(np.float64, copy=False)
    yv = y[mask].astype(np.float64, copy=False)

    sx = float(xv.sum())
    sy = float(yv.sum())
    sxx = float(np.dot(xv, xv))
    syy = float(np.dot(yv, yv))
    sxy = float(np.dot(xv, yv))
    n = float(n_complete)

    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    covariance = sxy - sx * sy / n

    if variance_x <= 0.0 or variance_y <= 0.0:
        return n_complete, None, "pairwise_nonvariable"

    r2 = covariance * covariance / (variance_x * variance_y)
    if r2 < -1e-12 or r2 > 1.0 + 1e-10:
        raise ValueError(f"invalid r2: {r2}")
    r2 = min(1.0, max(0.0, r2))

    if not math.isfinite(r2):
        raise ValueError("non-finite r2")

    return n_complete, r2, "ok"


def variant_id(data, index):
    row = data["rows"][index]
    return f"{row['fasta_chrom']}:{row['pos']}:{row['ref']}:{row['alt']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dosage", required=True, type=Path)
    parser.add_argument("--variant-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--allow-unpinned-inputs", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if not args.allow_unpinned_inputs and args.repo_commit != EXPECTED["repo_commit"]:
        raise ValueError("repository commit mismatch")

    data = read_inputs(args.dosage, args.variant_index, args.allow_unpinned_inputs)
    candidates, candidate_counts, counts_by_chrom = candidate_pairs(data["chrom"], data["pos"])

    preflight = {
        "status": "PREFLIGHT_PASS",
        "variants": 10_000,
        "samples": 1_135,
        "variable_variants": 10_000,
        "minimum_called_samples": int(data["called_counts"].min()),
        "maximum_called_samples": int(data["called_counts"].max()),
        "candidate_pair_counts": {str(key): value for key, value in candidate_counts.items()},
        "pairs_250kb_by_chromosome": counts_by_chrom,
        "maximum_distance_bp": 250_000,
        "minimum_pairwise_complete_samples": 200,
        "hypothesis_testing": False,
    }

    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_results = []
    status_counts = Counter()

    for number, (left, right, distance) in enumerate(candidates, 1):
        n_complete, r2, status = pairwise_r2(
            data["dosage"][left],
            data["dosage"][right],
        )
        pair_results.append((left, right, distance, n_complete, r2, status))
        status_counts[status] += 1

        if number % 25_000 == 0:
            print(
                f"LD_PROGRESS {number}/{len(candidates)}",
                flush=True,
            )

    pair_path = args.output_dir / "nearby_ld_pairs_250kb.tsv"
    final, partial, handle = open_tsv_atomic(pair_path)
    with handle:
        fields = [
            "left_row_index",
            "right_row_index",
            "chrom",
            "left_pos",
            "right_pos",
            "distance_bp",
            "left_variant_id",
            "right_variant_id",
            "n_complete",
            "r2",
            "status",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for left, right, distance, n_complete, r2, status in pair_results:
            writer.writerow(
                {
                    "left_row_index": left,
                    "right_row_index": right,
                    "chrom": data["chrom"][left],
                    "left_pos": int(data["pos"][left]),
                    "right_pos": int(data["pos"][right]),
                    "distance_bp": distance,
                    "left_variant_id": variant_id(data, left),
                    "right_variant_id": variant_id(data, right),
                    "n_complete": n_complete,
                    "r2": "" if r2 is None else format(r2, ".17g"),
                    "status": status,
                }
            )
    os.replace(partial, final)

    outgoing = [[] for _ in range(10_000)]
    for pair_number, result in enumerate(pair_results):
        if result[5] == "ok":
            outgoing[result[0]].append(pair_number)

    thinning_rows = []
    thinning_summary = {}

    for profile, window, threshold in PROFILES:
        retained = np.ones(10_000, dtype=bool)
        removed_by = np.full(10_000, -1, dtype=np.int32)
        removal_distance = np.full(10_000, -1, dtype=np.int32)
        removal_r2 = np.full(10_000, np.nan, dtype=np.float64)

        for left in range(10_000):
            if not retained[left]:
                continue

            for pair_number in outgoing[left]:
                _, right, distance, _, r2, _ = pair_results[pair_number]
                if distance <= window and r2 is not None and r2 >= threshold and retained[right]:
                    retained[right] = False
                    removed_by[right] = left
                    removal_distance[right] = distance
                    removal_r2[right] = r2

        retained_count = int(retained.sum())
        thinning_summary[profile] = {
            "window_bp": window,
            "r2_threshold": threshold,
            "retained": retained_count,
            "removed": 10_000 - retained_count,
        }

        for index in range(10_000):
            cause = int(removed_by[index])
            thinning_rows.append(
                {
                    "profile": profile,
                    "window_bp": window,
                    "r2_threshold": format(threshold, ".2f"),
                    "row_index": index,
                    "selection_rank": int(data["ranks"][index]),
                    "chrom": data["chrom"][index],
                    "fasta_chrom": data["rows"][index]["fasta_chrom"],
                    "pos": int(data["pos"][index]),
                    "ref": data["ref"][index],
                    "alt": data["alt"][index],
                    "variant_id": variant_id(data, index),
                    "retained": str(bool(retained[index])),
                    "removed_by_row_index": "" if cause < 0 else cause,
                    "removed_by_variant_id": "" if cause < 0 else variant_id(data, cause),
                    "removal_distance_bp": "" if cause < 0 else int(removal_distance[index]),
                    "removal_r2": "" if cause < 0 else format(float(removal_r2[index]), ".17g"),
                }
            )

    thinning_path = args.output_dir / "ld_thinning_membership.tsv"
    final, partial, handle = open_tsv_atomic(thinning_path)
    with handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(thinning_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(thinning_rows)
    os.replace(partial, final)

    bins_path = args.output_dir / "physical_bins.tsv"
    final, partial, handle = open_tsv_atomic(bins_path)
    bin_sets = {50_000: set(), 100_000: set(), 250_000: set()}

    with handle:
        fields = [
            "row_index",
            "selection_rank",
            "chrom",
            "fasta_chrom",
            "pos",
            "ref",
            "alt",
            "variant_id",
            "physical_bin_50kb",
            "physical_bin_100kb",
            "physical_bin_250kb",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for i in range(10_000):
            bins = {}
            for window, identifiers in bin_sets.items():
                number = (int(data["pos"][i]) - 1) // window + 1
                identifier = f"{data['chrom'][i]}:{number}"
                bins[window] = identifier
                identifiers.add(identifier)

            writer.writerow(
                {
                    "row_index": i,
                    "selection_rank": int(data["ranks"][i]),
                    "chrom": data["chrom"][i],
                    "fasta_chrom": data["rows"][i]["fasta_chrom"],
                    "pos": int(data["pos"][i]),
                    "ref": data["ref"][i],
                    "alt": data["alt"][i],
                    "variant_id": variant_id(data, i),
                    "physical_bin_50kb": bins[50_000],
                    "physical_bin_100kb": bins[100_000],
                    "physical_bin_250kb": bins[250_000],
                }
            )
    os.replace(partial, final)

    threshold_counts = {}
    for window in (50_000, 100_000, 250_000):
        eligible = [result for result in pair_results if result[2] <= window and result[5] == "ok"]
        threshold_counts[str(window)] = {
            "valid_pairs": len(eligible),
            "r2_ge_0.10": sum(result[4] >= 0.10 for result in eligible),
            "r2_ge_0.20": sum(result[4] >= 0.20 for result in eligible),
            "r2_ge_0.50": sum(result[4] >= 0.50 for result in eligible),
        }

    summary = {
        "phase": "H5B",
        "status": "PASS",
        "variants": 10_000,
        "samples": 1_135,
        "maximum_distance_bp": 250_000,
        "minimum_pairwise_complete_samples": 200,
        "candidate_pair_counts": {str(key): value for key, value in candidate_counts.items()},
        "pairs_250kb_by_chromosome": counts_by_chrom,
        "pair_status_counts": dict(sorted(status_counts.items())),
        "threshold_counts": threshold_counts,
        "thinning": thinning_summary,
        "physical_bin_counts": {str(window): len(values) for window, values in bin_sets.items()},
        "primary_thinning_profile": "w100kb_r2ge0.20",
        "hypothesis_testing_performed": False,
    }

    summary_path = args.output_dir / "summary.json"
    write_json_atomic(summary_path, summary)

    provenance = {
        "phase": "H5B",
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_commit": args.repo_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "inputs": {
            "dosage": {
                "path": str(args.dosage.resolve()),
                "sha256": sha256(args.dosage),
            },
            "variant_index": {
                "path": str(args.variant_index.resolve()),
                "sha256": sha256(args.variant_index),
            },
        },
        "method": {
            "ld": "pairwise-complete Pearson dosage correlation squared",
            "dosage_encoding": "-1 missing; 0/1/2 ALT copies",
            "minimum_complete_samples": 200,
            "pair_scope": "same chromosome and distance <=250000 bp",
            "thinning": "deterministic genomic-order greedy removal; "
            "scores are never used to choose retained variants",
            "physical_bins": "fixed chromosome-specific bins anchored at position 1",
        },
        "outputs": {
            "nearby_ld_pairs_250kb.tsv": sha256(pair_path),
            "ld_thinning_membership.tsv": sha256(thinning_path),
            "physical_bins.tsv": sha256(bins_path),
            "summary.json": sha256(summary_path),
        },
    }

    provenance_path = args.output_dir / "provenance.json"
    write_json_atomic(provenance_path, provenance)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("H5B_LD_STRUCTURE_PASS")


if __name__ == "__main__":
    main()
