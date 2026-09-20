#!/usr/bin/env python3
"""Descriptive model/conservation concordance with cluster bootstrap."""

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

try:
    import scipy
    from scipy.stats import rankdata
except ImportError:
    scipy = None
    rankdata = None


EXPECTED = {
    "annotated": "428f6948180f6aa95366d931a49b1a86ccfb7ea98663875271af8bbb2f9e8b47",
    "thinning": "3795a593a633695e1f65dd7bfa90953de37f51dc4266d3b2600cf3aed00d45c3",
    "bins": "0b7fb319eb90cd8967e07f340e90dc1473eed2be1170c80a1032684bb97f244c",
    "repo_commit": "da430ff6ce3890e6a1a09f08fdcfd5fdfccc139b",
}

BOOTSTRAP_REPLICATES = 2000
BASE_SEED = 20260920

COMPARISONS = (
    ("gpn_vs_plantcad_ref_alt", "gpn_ref_alt", "plantcad_ref_alt"),
    ("gpn_vs_plantcad_minor_major", "gpn_minor_major", "plantcad_minor_major"),
    ("gpn_ref_alt_vs_phylop", "gpn_ref_alt", "phylop"),
    ("gpn_minor_major_vs_phylop", "gpn_minor_major", "phylop"),
    ("plantcad_ref_alt_vs_phylop", "plantcad_ref_alt", "phylop"),
    ("plantcad_minor_major_vs_phylop", "plantcad_minor_major", "phylop"),
)

PROFILE_ORDER = (
    "w050kb_r2ge0.20",
    "w100kb_r2ge0.20",
    "w250kb_r2ge0.20",
    "w100kb_r2ge0.10",
    "w100kb_r2ge0.50",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames or [], list(reader)


def write_tsv_atomic(path, rows):
    if not rows:
        raise ValueError(f"no output rows for {path}")
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    with partial.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


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


def numeric(rows, column):
    return np.asarray(
        [float(row[column]) if row[column] != "" else np.nan for row in rows], dtype=np.float64
    )


def correlation(x, y, mask, method):
    keep = mask & np.isfinite(x) & np.isfinite(y)
    xv = x[keep]
    yv = y[keep]

    if len(xv) < 2 or xv.min() == xv.max() or yv.min() == yv.max():
        return len(xv), None

    if method == "spearman":
        xv = rankdata(xv, method="average")
        yv = rankdata(yv, method="average")
    elif method != "pearson":
        raise ValueError(method)

    value = float(np.corrcoef(xv, yv)[0, 1])
    if not math.isfinite(value):
        raise ValueError("non-finite correlation")
    return len(xv), value


def weighted_correlations(stats):
    n, sx, sy, sxx, syy, sxy = stats.T
    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    denominator = np.sqrt(variance_x * variance_y)

    result = np.full(len(stats), np.nan, dtype=np.float64)
    valid = (n >= 2) & (denominator > 0)
    result[valid] = covariance[valid] / denominator[valid]
    return result


def bootstrap_ci(x, y, mask, chrom, bins, method, label):
    keep = mask & np.isfinite(x) & np.isfinite(y)
    xv = x[keep].astype(np.float64)
    yv = y[keep].astype(np.float64)
    cv = chrom[keep]
    bv = bins[keep]

    if method == "spearman":
        xv = rankdata(xv, method="average")
        yv = rankdata(yv, method="average")

    keys = np.asarray(
        [f"{c}|{b}" for c, b in zip(cv, bv)],
        dtype=object,
    )
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    groups = len(unique_keys)

    stats = np.column_stack(
        [
            np.bincount(inverse),
            np.bincount(inverse, weights=xv),
            np.bincount(inverse, weights=yv),
            np.bincount(inverse, weights=xv * xv),
            np.bincount(inverse, weights=yv * yv),
            np.bincount(inverse, weights=xv * yv),
        ]
    ).astype(np.float64)

    group_chrom = np.asarray([key.split("|", 1)[0] for key in unique_keys])

    seed_material = f"{BASE_SEED}|{label}".encode()
    seed = int.from_bytes(
        hashlib.sha256(seed_material).digest()[:8],
        "big",
    ) % (2**32)
    rng = np.random.default_rng(seed)

    estimates = []
    batch_size = 200

    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        batch = min(
            batch_size,
            BOOTSTRAP_REPLICATES - start,
        )
        totals = np.zeros((batch, 6), dtype=np.float64)

        for chromosome in ("1", "2", "3", "4", "5"):
            cluster_indices = np.flatnonzero(group_chrom == chromosome)
            if len(cluster_indices) == 0:
                continue

            draws = rng.integers(
                0,
                len(cluster_indices),
                size=(batch, len(cluster_indices)),
            )
            selected = cluster_indices[draws]
            totals += stats[selected].sum(axis=1)

        estimates.extend(weighted_correlations(totals).tolist())

    estimates = np.asarray(estimates, dtype=np.float64)
    estimates = estimates[np.isfinite(estimates)]

    if len(estimates) != BOOTSTRAP_REPLICATES:
        raise ValueError("non-finite bootstrap replicate")

    return {
        "clusters": groups,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_mean": float(estimates.mean()),
        "bootstrap_se": float(estimates.std(ddof=1)),
        "ci_lower_2.5": float(np.quantile(estimates, 0.025)),
        "ci_upper_97.5": float(np.quantile(estimates, 0.975)),
        "seed": seed,
    }


def substitution_class(row):
    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
    ref, alt = row["ref"], row["alt"]
    if ref in {"A", "G"}:
        ref, alt = complement[ref], complement[alt]
    return f"{ref}>{alt}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", required=True, type=Path)
    parser.add_argument("--thinning", required=True, type=Path)
    parser.add_argument("--bins", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--allow-unpinned-inputs", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    for name, path in (
        ("annotated", args.annotated),
        ("thinning", args.thinning),
        ("bins", args.bins),
    ):
        if not args.allow_unpinned_inputs and digest(path) != EXPECTED[name]:
            raise ValueError(f"{name} SHA-256 mismatch")

    if not args.allow_unpinned_inputs and args.repo_commit != EXPECTED["repo_commit"]:
        raise ValueError("repository commit mismatch")

    _, rows = read_tsv(args.annotated)
    _, thinning_rows = read_tsv(args.thinning)
    _, bin_rows = read_tsv(args.bins)

    if len(rows) != 10000:
        raise ValueError("annotated row count mismatch")
    if len(thinning_rows) != 50000:
        raise ValueError("thinning row count mismatch")
    if len(bin_rows) != 10000:
        raise ValueError("physical-bin row count mismatch")

    chrom = np.asarray([row["chrom"] for row in rows])

    masks = {"unpruned": np.ones(10000, dtype=bool)}
    for profile in PROFILE_ORDER:
        masks[profile] = np.zeros(10000, dtype=bool)

    for row in thinning_rows:
        profile = row["profile"]
        index = int(row["row_index"])
        source = rows[index]
        if (
            row["chrom"] != source["chrom"]
            or row["pos"] != source["pos"]
            or row["ref"] != source["ref"]
            or row["alt"] != source["alt"]
        ):
            raise ValueError("thinning alignment mismatch")
        masks[profile][index] = row["retained"] == "True"

    bin_arrays = {
        "50000": np.empty(10000, dtype=object),
        "100000": np.empty(10000, dtype=object),
        "250000": np.empty(10000, dtype=object),
    }

    for index, row in enumerate(bin_rows):
        source = rows[index]
        if (
            int(row["row_index"]) != index
            or row["chrom"] != source["chrom"]
            or row["pos"] != source["pos"]
            or row["ref"] != source["ref"]
            or row["alt"] != source["alt"]
        ):
            raise ValueError("physical-bin alignment mismatch")
        bin_arrays["50000"][index] = row["physical_bin_50kb"]
        bin_arrays["100000"][index] = row["physical_bin_100kb"]
        bin_arrays["250000"][index] = row["physical_bin_250kb"]

    values = {
        "gpn_ref_alt": numeric(rows, "gpn_score_ref_alt"),
        "gpn_minor_major": numeric(rows, "gpn_score_minor_vs_major"),
        "plantcad_ref_alt": numeric(rows, "plantcad_score_ref_alt"),
        "plantcad_minor_major": numeric(rows, "plantcad_score_minor_vs_major"),
        "phylop": numeric(rows, "phylop"),
    }

    contexts = np.asarray([row["primary_context"] for row in rows])
    frequencies = np.asarray([row["frequency_class"] for row in rows])
    substitutions = np.asarray([substitution_class(row) for row in rows])

    preflight = {
        "status": "PREFLIGHT_PASS",
        "rows": len(rows),
        "phylop_covered": int(np.isfinite(values["phylop"]).sum()),
        "retained_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "context_counts": dict(sorted(Counter(contexts).items())),
        "frequency_counts": dict(sorted(Counter(frequencies).items())),
        "substitution_counts": dict(sorted(Counter(substitutions).items())),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_bin_widths": [50000, 100000, 250000],
        "hypothesis_tests": False,
    }

    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    for scope, scope_mask in masks.items():
        for comparison, left, right in COMPARISONS:
            for method in ("pearson", "spearman"):
                n, estimate = correlation(
                    values[left],
                    values[right],
                    scope_mask,
                    method,
                )
                overall_rows.append(
                    {
                        "scope": scope,
                        "comparison": comparison,
                        "left_score": left,
                        "right_score": right,
                        "method": method,
                        "n": n,
                        "estimate": format(estimate, ".17g"),
                    }
                )

    overall_path = args.output_dir / "overall_ld_sensitivity_correlations.tsv"
    write_tsv_atomic(overall_path, overall_rows)

    stratified_rows = []
    strata = (
        ("primary_context", contexts),
        ("frequency_class", frequencies),
        ("substitution_class", substitutions),
    )

    for dimension, labels in strata:
        for stratum in sorted(set(labels)):
            stratum_mask = labels == stratum
            for comparison, left, right in COMPARISONS:
                for method in ("pearson", "spearman"):
                    n, estimate = correlation(
                        values[left],
                        values[right],
                        stratum_mask,
                        method,
                    )
                    stratified_rows.append(
                        {
                            "dimension": dimension,
                            "stratum": stratum,
                            "comparison": comparison,
                            "left_score": left,
                            "right_score": right,
                            "method": method,
                            "n": n,
                            "estimate": "" if estimate is None else format(estimate, ".17g"),
                        }
                    )

    stratified_path = args.output_dir / "stratified_correlations.tsv"
    write_tsv_atomic(stratified_path, stratified_rows)

    bootstrap_rows = []
    unpruned = masks["unpruned"]

    for width in ("50000", "100000", "250000"):
        for comparison, left, right in COMPARISONS:
            for method in ("pearson", "spearman"):
                n, estimate = correlation(
                    values[left],
                    values[right],
                    unpruned,
                    method,
                )
                label = f"{width}|{comparison}|{method}"
                result = bootstrap_ci(
                    values[left],
                    values[right],
                    unpruned,
                    chrom,
                    bin_arrays[width],
                    method,
                    label,
                )
                bootstrap_rows.append(
                    {
                        "scope": "unpruned",
                        "bin_width_bp": width,
                        "comparison": comparison,
                        "left_score": left,
                        "right_score": right,
                        "method": method,
                        "n": n,
                        "estimate": format(estimate, ".17g"),
                        "clusters": result["clusters"],
                        "bootstrap_replicates": result["bootstrap_replicates"],
                        "bootstrap_mean": format(result["bootstrap_mean"], ".17g"),
                        "bootstrap_se": format(result["bootstrap_se"], ".17g"),
                        "ci_lower_2.5": format(result["ci_lower_2.5"], ".17g"),
                        "ci_upper_97.5": format(result["ci_upper_97.5"], ".17g"),
                        "seed": result["seed"],
                    }
                )

    bootstrap_path = args.output_dir / "physical_bin_bootstrap_ci.tsv"
    write_tsv_atomic(bootstrap_path, bootstrap_rows)

    key_results = {}
    for row in overall_rows:
        if row["scope"] == "unpruned":
            key = f"{row['comparison']}|{row['method']}"
            key_results[key] = {
                "n": int(row["n"]),
                "estimate": float(row["estimate"]),
            }

    summary = {
        "phase": "H5C",
        "status": "PASS",
        "rows": 10000,
        "phylop_covered": 9453,
        "overall_correlation_rows": len(overall_rows),
        "stratified_correlation_rows": len(stratified_rows),
        "bootstrap_ci_rows": len(bootstrap_rows),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_design": "chromosome-stratified fixed physical-bin cluster bootstrap",
        "primary_ld_profile": "w100kb_r2ge0.20",
        "key_unpruned_results": key_results,
        "hypothesis_tests_performed": False,
        "p_values_calculated": False,
    }

    summary_path = args.output_dir / "summary.json"
    write_json_atomic(summary_path, summary)

    provenance = {
        "phase": "H5C",
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_commit": args.repo_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": None if scipy is None else scipy.__version__,
        "inputs": {
            "annotated": {
                "path": str(args.annotated.resolve()),
                "sha256": digest(args.annotated),
            },
            "thinning": {
                "path": str(args.thinning.resolve()),
                "sha256": digest(args.thinning),
            },
            "bins": {
                "path": str(args.bins.resolve()),
                "sha256": digest(args.bins),
            },
        },
        "methods": {
            "correlations": ["Pearson", "Spearman"],
            "missing_phylop": "complete-case exclusion",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed_base": BASE_SEED,
            "bootstrap": "resample physical bins within each chromosome; "
            "Spearman uses average ranks computed within the "
            "analysis subset before cluster resampling",
            "confidence_interval": "percentile 2.5% to 97.5%",
            "hypothesis_tests": False,
        },
        "interpretation": {
            "model_raw": "log P(ALT) - log P(REF)",
            "model_population": "log P(minor allele) - log P(major allele)",
            "phylop": "signed site-oriented conservation score; not allele-oriented",
        },
        "outputs": {
            overall_path.name: digest(overall_path),
            stratified_path.name: digest(stratified_path),
            bootstrap_path.name: digest(bootstrap_path),
            summary_path.name: digest(summary_path),
        },
    }

    provenance_path = args.output_dir / "provenance.json"
    write_json_atomic(provenance_path, provenance)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("H5C_MODEL_CONCORDANCE_PASS")


if __name__ == "__main__":
    main()
