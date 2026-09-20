#!/usr/bin/env python3
"""Deterministic genotype PCA/UMAP for the PopGenLM v0.3 cohort."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

if importlib.util.find_spec("matplotlib"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
    plt = None
import numpy as np

try:
    import scipy
    import sklearn
    import umap
    from scipy.spatial import procrustes
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    scipy = sklearn = umap = None
    procrustes = PCA = NearestNeighbors = None


EXPECTED = {
    "metadata": "ff14c461f6087b54d97ce130998a63a659793f6ac9c121a7dc82ff92828ea291",
    "annotated": "428f6948180f6aa95366d931a49b1a86ccfb7ea98663875271af8bbb2f9e8b47",
    "dosage": "e84eb449d8fc883c156ecb2d37706d3bbe8c757dc783641f3393d426e397e1c1",
    "thinning": "3795a593a633695e1f65dd7bfa90953de37f51dc4266d3b2600cf3aed00d45c3",
    "repo_commit": "da430ff6ce3890e6a1a09f08fdcfd5fdfccc139b",
}

SEED = 20260920
PRIMARY_CALL_RATE = 0.80
STRICT_CALL_RATE = 0.90
PCA_COMPONENTS = 50
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_float(value):
    value = value.strip()
    if value in {"", "—", "-", "NA", "NaN", "nan"}:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def write_tsv_atomic(path, rows):
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


def load_inputs(args):
    for name, path in (
        ("metadata", args.metadata),
        ("annotated", args.annotated),
        ("dosage", args.dosage),
        ("thinning", args.thinning),
    ):
        if not args.allow_unpinned_inputs and digest(path) != EXPECTED[name]:
            raise ValueError(f"{name} SHA-256 mismatch")

    if not args.allow_unpinned_inputs and args.repo_commit != EXPECTED["repo_commit"]:
        raise ValueError("repository commit mismatch")

    with args.metadata.open(newline="", encoding="utf-8-sig") as handle:
        metadata = {row["ID"].strip(): row for row in csv.DictReader(handle)}

    with args.annotated.open(newline="") as handle:
        table = list(csv.DictReader(handle, delimiter="\t"))

    with args.thinning.open(newline="") as handle:
        thinning = list(csv.DictReader(handle, delimiter="\t"))

    with np.load(args.dosage, allow_pickle=False) as archive:
        dosage = archive["dosage"].copy()
        sample_ids = archive["sample_ids"].astype(str)
        chrom = archive["chrom"].astype(str)
        pos = archive["pos"].astype(np.int64)
        ref = archive["ref"].astype(str)
        alt = archive["alt"].astype(str)

    if dosage.shape != (10000, 1135):
        raise ValueError("dosage shape mismatch")
    if len(table) != 10000 or len(metadata) != 1135:
        raise ValueError("input row-count mismatch")
    if set(sample_ids) != set(metadata):
        raise ValueError("metadata/sample-ID mismatch")

    for index, row in enumerate(table):
        if (
            row["chrom"] != chrom[index]
            or int(row["pos"]) != int(pos[index])
            or row["ref"] != ref[index]
            or row["alt"] != alt[index]
        ):
            raise ValueError(f"variant alignment mismatch at {index}")

    retained = np.zeros(10000, dtype=bool)
    primary_rows = [row for row in thinning if row["profile"] == "w100kb_r2ge0.20"]
    if len(primary_rows) != 10000:
        raise ValueError("primary thinning profile incomplete")

    for row in primary_rows:
        retained[int(row["row_index"])] = row["retained"] == "True"

    common = np.asarray([row["frequency_class"] == "common" for row in table])
    selected = common & retained

    if int(selected.sum()) != 1132:
        raise ValueError("selected variant count mismatch")

    matrix = dosage[selected].T
    call_rate = (matrix >= 0).mean(axis=1)

    latitude = np.asarray([parse_float(metadata[sample]["Latitude"]) for sample in sample_ids])
    longitude = np.asarray([parse_float(metadata[sample]["Longitude"]) for sample in sample_ids])

    return {
        "metadata": metadata,
        "sample_ids": sample_ids,
        "matrix": matrix,
        "call_rate": call_rate,
        "latitude": latitude,
        "longitude": longitude,
        "selected": selected,
        "selected_chrom": chrom[selected],
        "selected_pos": pos[selected],
    }


def preprocess(matrix):
    matrix = matrix.astype(np.float64, copy=True)
    missing = matrix < 0

    observed = np.where(missing, np.nan, matrix)
    means = np.nanmean(observed, axis=0)

    if not np.isfinite(means).all():
        raise ValueError("variant with no called genotypes")

    row_indices, column_indices = np.where(missing)
    matrix[row_indices, column_indices] = means[column_indices]

    means_after = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)

    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("non-variable UMAP input variant")

    matrix = (matrix - means_after) / std

    if not np.isfinite(matrix).all():
        raise ValueError("non-finite standardized genotype")

    return matrix


def fit_embedding(matrix):
    if PCA is None or umap is None:
        raise SystemExit("build_genotype_umap.py requires optional scikit-learn and umap-learn")
    standardized = preprocess(matrix)

    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=SEED,
    )
    pcs = pca.fit_transform(standardized)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=SEED,
        transform_seed=SEED,
        n_jobs=1,
        low_memory=True,
    )
    embedding = reducer.fit_transform(pcs)

    if embedding.shape != (len(matrix), 2):
        raise ValueError("UMAP shape mismatch")
    if not np.isfinite(embedding).all():
        raise ValueError("non-finite UMAP coordinates")

    return pcs, embedding, pca.explained_variance_ratio_


def neighborhood_jaccard(first, second, neighbors=15):
    first_knn = (
        NearestNeighbors(n_neighbors=neighbors + 1)
        .fit(first)
        .kneighbors(return_distance=False)[:, 1:]
    )

    second_knn = (
        NearestNeighbors(n_neighbors=neighbors + 1)
        .fit(second)
        .kneighbors(return_distance=False)[:, 1:]
    )

    values = []
    for left, right in zip(first_knn, second_knn):
        left_set, right_set = set(left), set(right)
        values.append(len(left_set & right_set) / len(left_set | right_set))
    return float(np.mean(values))


def save_embedding(path, indices, data, pcs, embedding, inclusion_label):
    rows = []
    for local, source in enumerate(indices):
        sample = data["sample_ids"][source]
        metadata = data["metadata"][sample]
        row = {
            "sample_index": int(source),
            "sample_id": sample,
            "name": metadata["Name"],
            "latitude": ""
            if not np.isfinite(data["latitude"][source])
            else format(data["latitude"][source], ".10g"),
            "longitude": ""
            if not np.isfinite(data["longitude"][source])
            else format(data["longitude"][source], ".10g"),
            "call_rate": format(data["call_rate"][source], ".17g"),
            "analysis_set": inclusion_label,
            "umap_1": format(float(embedding[local, 0]), ".17g"),
            "umap_2": format(float(embedding[local, 1]), ".17g"),
        }
        for component in range(10):
            row[f"pc_{component + 1}"] = format(float(pcs[local, component]), ".17g")
        rows.append(row)

    write_tsv_atomic(path, rows)


def plot_embedding(path_base, indices, data, embedding):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "popgenlm-v03-h6a-repaired",
        }
    )

    separated = embedding[:, 0] < 0
    main_cloud = ~separated

    if int(separated.sum()) != 53:
        raise ValueError("descriptive separated-cloud count changed")
    if int(main_cloud.sum()) != 1040:
        raise ValueError("descriptive main-cloud count changed")

    longitude = data["longitude"][indices]
    latitude = data["latitude"][indices]
    call_rate = data["call_rate"][indices]

    longitude_norm = matplotlib.colors.TwoSlopeNorm(
        vmin=float(np.nanmin(longitude)),
        vcenter=0.0,
        vmax=float(np.nanmax(longitude)),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 8.0),
        constrained_layout=True,
    )

    def panel(
        axis,
        subset,
        colour,
        title,
        colourbar_label,
        cmap,
        norm=None,
        annotate_separated=False,
    ):
        valid = subset & np.isfinite(colour)
        missing = subset & ~np.isfinite(colour)

        if np.any(missing):
            axis.scatter(
                embedding[missing, 0],
                embedding[missing, 1],
                s=18,
                c="#bdbdbd",
                marker="x",
                alpha=0.85,
                linewidths=0.8,
                label="Metadata unavailable",
            )

        points = axis.scatter(
            embedding[valid, 0],
            embedding[valid, 1],
            c=colour[valid],
            cmap=cmap,
            norm=norm,
            s=17,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )
        figure.colorbar(
            points,
            ax=axis,
            label=colourbar_label,
            shrink=0.88,
        )

        axis.set_title(title)
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_aspect("equal", adjustable="box")

        if annotate_separated:
            centre = embedding[separated].mean(axis=0)
            axis.annotate(
                "Separated cloud\n(n = 53)",
                xy=centre,
                xytext=(-7, 5),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=8.5,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#4d4d4d",
                    "lw": 0.9,
                },
            )

    all_samples = np.ones(len(indices), dtype=bool)

    panel(
        axes[0, 0],
        all_samples,
        longitude,
        "Full UMAP: longitude",
        "Longitude",
        "coolwarm",
        norm=longitude_norm,
        annotate_separated=True,
    )
    panel(
        axes[0, 1],
        all_samples,
        call_rate,
        "Full UMAP: genotype call rate",
        "Call rate",
        "magma",
    )
    panel(
        axes[1, 0],
        main_cloud,
        longitude,
        "Connected main cloud enlarged: longitude",
        "Longitude",
        "coolwarm",
        norm=longitude_norm,
    )
    panel(
        axes[1, 1],
        main_cloud,
        latitude,
        "Connected main cloud enlarged: latitude",
        "Latitude",
        "viridis",
    )

    for label, axis in zip("ABCD", axes.flat):
        current_title = axis.get_title()
        axis.set_title("")
        axis.set_title(
            f"{label}. {current_title}",
            loc="left",
            pad=8,
            fontsize=10.5,
        )

    figure.suptitle(
        "A. thaliana genotype structure from 1,132 common LD-thinned variants",
        fontsize=12.5,
    )

    figure.savefig(
        str(path_base) + ".png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "PopGenLM Bench v0.3"},
    )
    figure.savefig(
        str(path_base) + ".svg",
        bbox_inches="tight",
        metadata={
            "Date": None,
            "Creator": "PopGenLM Bench v0.3",
        },
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--annotated", required=True, type=Path)
    parser.add_argument("--dosage", required=True, type=Path)
    parser.add_argument("--thinning", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--allow-unpinned-inputs", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    data = load_inputs(args)

    primary = data["call_rate"] >= PRIMARY_CALL_RATE
    strict = data["call_rate"] >= STRICT_CALL_RATE
    valid_geo = np.isfinite(data["latitude"]) & np.isfinite(data["longitude"])

    preflight = {
        "status": "PREFLIGHT_PASS",
        "samples_total": 1135,
        "selected_variants": 1132,
        "primary_call_rate_threshold": PRIMARY_CALL_RATE,
        "primary_samples": int(primary.sum()),
        "strict_call_rate_threshold": STRICT_CALL_RATE,
        "strict_samples": int(strict.sum()),
        "primary_geographic_metadata": int((primary & valid_geo).sum()),
        "primary_missing_geographic_metadata": int((primary & ~valid_geo).sum()),
        "coordinates_used_as_model_input": False,
        "pca_components": PCA_COMPONENTS,
        "umap_neighbors": UMAP_NEIGHBORS,
        "umap_min_dist": UMAP_MIN_DIST,
        "random_seed": SEED,
    }

    if int(primary.sum()) != 1093 or int(strict.sum()) != 996:
        raise ValueError("sample threshold count mismatch")

    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary_indices = np.flatnonzero(primary)
    strict_indices = np.flatnonzero(strict)

    primary_pcs, primary_umap, primary_variance = fit_embedding(data["matrix"][primary_indices])
    strict_pcs, strict_umap, strict_variance = fit_embedding(data["matrix"][strict_indices])

    primary_lookup = {source: local for local, source in enumerate(primary_indices)}
    shared_primary = np.asarray([primary_lookup[source] for source in strict_indices])

    _transformed_primary, _transformed_strict, disparity = procrustes(
        primary_umap[shared_primary],
        strict_umap,
    )
    jaccard = neighborhood_jaccard(
        primary_umap[shared_primary],
        strict_umap,
        neighbors=15,
    )

    quality_rows = []
    for index, sample in enumerate(data["sample_ids"]):
        metadata = data["metadata"][sample]
        quality_rows.append(
            {
                "sample_index": index,
                "sample_id": sample,
                "name": metadata["Name"],
                "call_rate": format(data["call_rate"][index], ".17g"),
                "primary_include": str(bool(primary[index])),
                "strict_include": str(bool(strict[index])),
                "latitude": ""
                if not np.isfinite(data["latitude"][index])
                else format(data["latitude"][index], ".10g"),
                "longitude": ""
                if not np.isfinite(data["longitude"][index])
                else format(data["longitude"][index], ".10g"),
                "geographic_metadata_available": str(bool(valid_geo[index])),
            }
        )

    quality_path = args.output_dir / "sample_quality.tsv"
    write_tsv_atomic(quality_path, quality_rows)

    primary_path = args.output_dir / "sample_umap_primary.tsv"
    strict_path = args.output_dir / "sample_umap_strict.tsv"

    save_embedding(
        primary_path,
        primary_indices,
        data,
        primary_pcs,
        primary_umap,
        "call_rate_ge_0.80",
    )
    save_embedding(
        strict_path,
        strict_indices,
        data,
        strict_pcs,
        strict_umap,
        "call_rate_ge_0.90",
    )

    figure_base = args.output_dir / "genotype_umap_geography"
    plot_embedding(
        figure_base,
        primary_indices,
        data,
        primary_umap,
    )

    summary = {
        **preflight,
        "phase": "H6A",
        "status": "PASS",
        "primary_pca_explained_variance_first_10": primary_variance[:10].tolist(),
        "primary_pca_explained_variance_50_sum": float(primary_variance.sum()),
        "strict_pca_explained_variance_50_sum": float(strict_variance.sum()),
        "strict_sensitivity_shared_samples": len(strict_indices),
        "strict_sensitivity_procrustes_disparity": float(disparity),
        "strict_sensitivity_mean_15nn_jaccard": jaccard,
        "descriptive_separated_cloud_rule": "primary UMAP1 < 0; visualization only",
        "descriptive_separated_cloud_samples": 53,
        "descriptive_main_cloud_samples": 1040,
    }

    summary_path = args.output_dir / "summary.json"
    write_json_atomic(summary_path, summary)

    outputs = [
        quality_path,
        primary_path,
        strict_path,
        Path(str(figure_base) + ".png"),
        Path(str(figure_base) + ".svg"),
        summary_path,
    ]

    provenance = {
        "phase": "H6A",
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_commit": args.repo_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "umap_learn": umap.__version__,
        "inputs": {
            "metadata": {
                "path": str(args.metadata.resolve()),
                "sha256": digest(args.metadata),
            },
            "annotated": {
                "path": str(args.annotated.resolve()),
                "sha256": digest(args.annotated),
            },
            "dosage": {
                "path": str(args.dosage.resolve()),
                "sha256": digest(args.dosage),
            },
            "thinning": {
                "path": str(args.thinning.resolve()),
                "sha256": digest(args.thinning),
            },
        },
        "method": {
            "variant_subset": "common frequency class and primary 100-kb/r2>=0.20 "
            "deterministic LD thinning",
            "missing_genotypes": "variant-mean imputation",
            "standardization": "variant-wise zero mean and unit population variance",
            "pca": "50-component randomized PCA with fixed seed",
            "umap": "Euclidean; 30 neighbors; min_dist=0.10; fixed seed; single-threaded",
            "geography": "latitude/longitude used only for figure colouring",
            "figure_zoom": "UMAP1 >= 0 main-cloud enlargement for visualization "
            "only; no population labels inferred",
            "primary_sample_threshold": "selected-genotype call rate >=0.80",
            "strict_sensitivity_threshold": "selected-genotype call rate >=0.90",
        },
        "outputs": {output.name: digest(output) for output in outputs},
    }

    provenance_path = args.output_dir / "provenance.json"
    write_json_atomic(provenance_path, provenance)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("H6A_GENOTYPE_UMAP_PASS")


if __name__ == "__main__":
    main()
