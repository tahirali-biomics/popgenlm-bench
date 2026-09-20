#!/usr/bin/env python3
"""Build the validated PopGenLM v0.3 model-evidence figure."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

if importlib.util.find_spec("matplotlib"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
    plt = None
try:
    from matplotlib.lines import Line2D
except ImportError:
    Line2D = None
import numpy as np

EXPECTED_HASHES = {
    "annotated": "428f6948180f6aa95366d931a49b1a86ccfb7ea98663875271af8bbb2f9e8b47",
    "overall": "0aa9fe41296f77dbad43f650ea4ee6d4d0d5edea85ab2a6a91d7e93593d84df8",
    "stratified": "366b1cf37e39d0859b127a90ad0087a7219c611ac91569e461165c0736097ca3",
    "bootstrap": "339eb912e2869e70c425bb7a953e63cdb132483414b6471b3bec435dbb1cc76f",
}

COMPARISONS = (
    "gpn_vs_plantcad_minor_major",
    "gpn_minor_major_vs_phylop",
    "plantcad_minor_major_vs_phylop",
)

LABELS = {
    "gpn_vs_plantcad_minor_major": "GPN–PlantCAD",
    "gpn_minor_major_vs_phylop": "GPN–PhyloP",
    "plantcad_minor_major_vs_phylop": "PlantCAD–PhyloP",
}

COLORS = {
    "gpn_vs_plantcad_minor_major": "#009E73",
    "gpn_minor_major_vs_phylop": "#0072B2",
    "plantcad_minor_major_vs_phylop": "#D55E00",
}

CONTEXT_ORDER = (
    "coding_cds",
    "exonic_non_cds",
    "other_genic",
    "intergenic",
)

CONTEXT_LABELS = (
    "Coding\nCDS",
    "Exonic\nnon-CDS",
    "Other\ngenic",
    "Intergenic",
)

SCOPE_ORDER = (
    "unpruned",
    "w050kb_r2ge0.20",
    "w100kb_r2ge0.10",
    "w100kb_r2ge0.20",
    "w100kb_r2ge0.50",
    "w250kb_r2ge0.20",
)

SCOPE_LABELS = (
    "All\nvariants",
    "50 kb\nr²≥0.20",
    "100 kb\nr²≥0.10",
    "100 kb\nr²≥0.20",
    "100 kb\nr²≥0.50",
    "250 kb\nr²≥0.20",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def select_data(args):
    paths = {
        "annotated": Path(args.annotated),
        "overall": Path(args.overall),
        "stratified": Path(args.stratified),
        "bootstrap": Path(args.bootstrap),
    }

    hashes = {name: digest(path) for name, path in paths.items()}
    if not args.allow_unpinned_inputs and hashes != EXPECTED_HASHES:
        raise ValueError(f"input hash mismatch: observed={hashes}")

    annotated = read_tsv(paths["annotated"])
    overall = read_tsv(paths["overall"])
    stratified = read_tsv(paths["stratified"])
    bootstrap = read_tsv(paths["bootstrap"])

    panel_a = [
        row
        for row in overall
        if row["scope"] == "unpruned" and row["comparison"] == "gpn_vs_plantcad_minor_major"
    ]
    panel_b = [
        row
        for row in bootstrap
        if row["bin_width_bp"] == "100000" and row["comparison"] in COMPARISONS
    ]
    panel_c = [
        row
        for row in stratified
        if row["dimension"] == "primary_context"
        and row["comparison"] in COMPARISONS
        and row["method"] == "spearman"
    ]
    panel_d = [
        row for row in overall if row["comparison"] in COMPARISONS and row["method"] == "spearman"
    ]

    if len(annotated) != 10000:
        raise ValueError("annotated table must contain 10,000 rows")
    if [len(panel_a), len(panel_b), len(panel_c), len(panel_d)] != [2, 6, 12, 18]:
        raise ValueError("unexpected H6B panel-record counts")

    keys = {
        (
            row["fasta_chrom"],
            int(row["pos"]),
            row["ref"],
            row["alt"],
        )
        for row in annotated
    }
    if len(keys) != 10000:
        raise ValueError("annotated variant keys are not unique")

    status = Counter(row["phylop_status"] for row in annotated)
    if status != Counter({"covered": 9453, "missing_coverage": 547}):
        raise ValueError(f"unexpected PhyloP status counts: {status}")

    for row in annotated:
        for field in (
            "gpn_score_minor_vs_major",
            "plantcad_score_minor_vs_major",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite {field}")
        if row["phylop_status"] == "covered":
            if not math.isfinite(float(row["phylop"])):
                raise ValueError("non-finite covered PhyloP")
        elif row["phylop"] != "":
            raise ValueError("missing-coverage PhyloP must be empty")

    for row in panel_a + panel_b + panel_c + panel_d:
        if not math.isfinite(float(row["estimate"])):
            raise ValueError("non-finite correlation estimate")

    for row in panel_b:
        estimate = float(row["estimate"])
        lower = float(row["ci_lower_2.5"])
        upper = float(row["ci_upper_97.5"])
        if not lower <= estimate <= upper:
            raise ValueError("observed estimate outside bootstrap interval")
        if int(row["bootstrap_replicates"]) != 2000:
            raise ValueError("unexpected bootstrap replicate count")

        expected_clusters = 1146 if row["comparison"] == "gpn_vs_plantcad_minor_major" else 1141
        if int(row["clusters"]) != expected_clusters:
            raise ValueError("unexpected physical-bin cluster count")

    for comparison in COMPARISONS:
        contexts = {row["stratum"] for row in panel_c if row["comparison"] == comparison}
        scopes = {row["scope"] for row in panel_d if row["comparison"] == comparison}
        if contexts != set(CONTEXT_ORDER):
            raise ValueError("incomplete genomic-context records")
        if scopes != set(SCOPE_ORDER):
            raise ValueError("incomplete LD-sensitivity records")

    return {
        "paths": paths,
        "hashes": hashes,
        "annotated": annotated,
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_c": panel_c,
        "panel_d": panel_d,
    }


def record(rows, **criteria):
    selected = [
        row for row in rows if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one record for {criteria}; found {len(selected)}")
    return selected[0]


def write_figure_statistics(path, data):
    fields = [
        "panel",
        "source",
        "scope",
        "dimension",
        "stratum",
        "bin_width_bp",
        "comparison",
        "method",
        "n",
        "estimate",
        "ci_lower_2.5",
        "ci_upper_97.5",
        "clusters",
        "bootstrap_replicates",
    ]

    rows = []
    for panel, source, selected in (
        ("A", "overall", data["panel_a"]),
        ("B", "bootstrap", data["panel_b"]),
        ("C", "stratified", data["panel_c"]),
        ("D", "overall", data["panel_d"]),
    ):
        for row in selected:
            rows.append(
                {
                    field: (
                        panel
                        if field == "panel"
                        else source
                        if field == "source"
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )

    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    if len(rows) != 38:
        raise ValueError("figure-statistics table must contain 38 rows")


def build_figure(path_base, data):
    if plt is None or Line2D is None:
        raise SystemExit("build_model_evidence_figure.py requires optional matplotlib")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "popgenlm-v03-h6b",
        }
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.5),
        constrained_layout=True,
    )

    # Panel A: all variants, population-oriented model scores.
    axis = axes[0, 0]
    x = np.asarray([float(row["gpn_score_minor_vs_major"]) for row in data["annotated"]])
    y = np.asarray([float(row["plantcad_score_minor_vs_major"]) for row in data["annotated"]])

    density = axis.hexbin(
        x,
        y,
        gridsize=55,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    figure.colorbar(
        density,
        ax=axis,
        label="Variants per hexagon (log scale)",
        shrink=0.88,
    )
    axis.axhline(0, color="#808080", lw=0.7, ls="--")
    axis.axvline(0, color="#808080", lw=0.7, ls="--")
    axis.set_xlabel("GPN log P(minor) − log P(major)")
    axis.set_ylabel("PlantCAD log P(minor) − log P(major)")
    axis.set_title(
        "A. Population-oriented model concordance",
        loc="left",
        pad=8,
    )

    pearson = record(data["panel_a"], method="pearson")
    spearman = record(data["panel_a"], method="spearman")
    axis.text(
        0.03,
        0.97,
        (
            f"n = {int(pearson['n']):,}\n"
            f"Pearson r = {float(pearson['estimate']):.3f}\n"
            f"Spearman ρ = {float(spearman['estimate']):.3f}"
        ),
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#bdbdbd",
            "alpha": 0.92,
        },
    )

    # Panel B: primary 100-kb physical-bin bootstrap.
    axis = axes[0, 1]
    y_positions = {comparison: 2 - index for index, comparison in enumerate(COMPARISONS)}
    method_offsets = {"pearson": 0.12, "spearman": -0.12}
    method_markers = {"pearson": "o", "spearman": "s"}

    for comparison in COMPARISONS:
        for method in ("pearson", "spearman"):
            row = record(
                data["panel_b"],
                comparison=comparison,
                method=method,
            )
            estimate = float(row["estimate"])
            lower = float(row["ci_lower_2.5"])
            upper = float(row["ci_upper_97.5"])
            axis.errorbar(
                estimate,
                y_positions[comparison] + method_offsets[method],
                xerr=np.asarray(
                    [
                        [estimate - lower],
                        [upper - estimate],
                    ]
                ),
                fmt=method_markers[method],
                color=COLORS[comparison],
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=6.5,
                capsize=3,
                lw=1.4,
            )

    axis.axvline(0, color="#808080", lw=0.8, ls="--")
    axis.set_yticks([2, 1, 0])
    axis.set_yticklabels(
        [
            "GPN–PlantCAD\nn = 10,000",
            "GPN–PhyloP\nn = 9,453",
            "PlantCAD–PhyloP\nn = 9,453",
        ]
    )
    axis.set_xlim(-0.42, 0.72)
    axis.set_xlabel("Correlation estimate")
    axis.set_title(
        "B. 100-kb physical-bin bootstrap intervals",
        loc="left",
        pad=8,
    )
    axis.text(
        0.02,
        0.02,
        "95% percentile intervals; 2,000 replicates",
        transform=axis.transAxes,
        fontsize=8,
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="#555555",
                linestyle="none",
                label="Pearson r",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="#555555",
                linestyle="none",
                label="Spearman ρ",
            ),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )

    # Panel C: descriptive genomic-context correlations.
    axis = axes[1, 0]
    x_positions = np.arange(len(CONTEXT_ORDER), dtype=float)
    offsets = {
        COMPARISONS[0]: -0.20,
        COMPARISONS[1]: 0.0,
        COMPARISONS[2]: 0.20,
    }

    for comparison in COMPARISONS:
        estimates = [
            float(
                record(
                    data["panel_c"],
                    comparison=comparison,
                    stratum=context,
                )["estimate"]
            )
            for context in CONTEXT_ORDER
        ]
        axis.scatter(
            x_positions + offsets[comparison],
            estimates,
            s=42,
            color=COLORS[comparison],
            edgecolor="white",
            linewidth=0.5,
            label=LABELS[comparison],
            zorder=3,
        )

    axis.axhline(0, color="#808080", lw=0.8, ls="--")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(CONTEXT_LABELS)
    axis.set_ylim(-0.48, 0.75)
    axis.set_ylabel("Spearman ρ")
    axis.set_title(
        "C. Concordance across genomic contexts",
        loc="left",
        pad=8,
    )
    axis.text(
        0.02,
        0.02,
        "Descriptive estimates; no hypothesis tests",
        transform=axis.transAxes,
        fontsize=8,
    )
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="upper right",
    )

    # Panel D: LD-thinning sensitivity.
    axis = axes[1, 1]
    x_positions = np.arange(len(SCOPE_ORDER), dtype=float)

    for comparison in COMPARISONS:
        estimates = [
            float(
                record(
                    data["panel_d"],
                    comparison=comparison,
                    scope=scope,
                )["estimate"]
            )
            for scope in SCOPE_ORDER
        ]
        axis.plot(
            x_positions,
            estimates,
            marker="o",
            markersize=4.5,
            lw=1.4,
            color=COLORS[comparison],
            label=LABELS[comparison],
        )

    axis.axhline(0, color="#808080", lw=0.8, ls="--")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(SCOPE_LABELS, fontsize=8)
    axis.get_xticklabels()[3].set_fontweight("bold")
    axis.set_ylim(-0.48, 0.75)
    axis.set_ylabel("Spearman ρ")
    axis.set_title(
        "D. LD-thinning sensitivity",
        loc="left",
        pad=8,
    )
    axis.text(
        0.02,
        0.02,
        "Bold label = primary profile",
        transform=axis.transAxes,
        fontsize=8,
    )

    figure.suptitle(
        "Model agreement and evolutionary evidence across 10,000 A. thaliana variants",
        fontsize=13,
    )

    png_path = Path(str(path_base) + ".png")
    svg_path = Path(str(path_base) + ".svg")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)

    return png_path, svg_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", required=True)
    parser.add_argument("--overall", required=True)
    parser.add_argument("--stratified", required=True)
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--allow-unpinned-inputs", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    data = select_data(args)

    preflight = {
        "status": "PREFLIGHT_PASS",
        "annotated_rows": len(data["annotated"]),
        "phylop_covered_rows": 9453,
        "panel_record_counts": {
            "A": len(data["panel_a"]),
            "B": len(data["panel_b"]),
            "C": len(data["panel_c"]),
            "D": len(data["panel_d"]),
        },
        "score_orientation": "log P(minor allele) - log P(major allele)",
        "hypothesis_tests": False,
        "p_values": False,
        "input_sha256": data["hashes"],
    }

    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        args.output_dir / "model_evidence.png",
        args.output_dir / "model_evidence.svg",
        args.output_dir / "figure_statistics.tsv",
        args.output_dir / "summary.json",
        args.output_dir / "provenance.json",
    ]
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    statistics_path = args.output_dir / "figure_statistics.tsv"
    write_figure_statistics(statistics_path, data)

    png_path, svg_path = build_figure(
        args.output_dir / "model_evidence",
        data,
    )

    summary = {
        **preflight,
        "phase": "H6B",
        "status": "PASS",
        "figure_statistics_rows": 38,
        "primary_bootstrap_bin_width_bp": 100000,
        "bootstrap_replicates": 2000,
        "figure_png": str(png_path),
        "figure_png_sha256": digest(png_path),
        "figure_svg": str(svg_path),
        "figure_svg_sha256": digest(svg_path),
        "figure_statistics": str(statistics_path),
        "figure_statistics_sha256": digest(statistics_path),
        "interpretation": "Higher signed PhyloP values correlate with lower "
        "minor-allele model scores; PhyloP is site-oriented.",
    }

    provenance = {
        "phase": "H6B",
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_commit": args.repo_commit,
        "script": str(Path(__file__).resolve()),
        "script_sha256": digest(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "inputs": {
            name: {
                "path": str(data["paths"][name]),
                "sha256": data["hashes"][name],
            }
            for name in data["paths"]
        },
        "analysis_policy": {
            "new_statistics_calculated": False,
            "hypothesis_tests": False,
            "p_values": False,
            "primary_score_orientation": "log P(minor allele) - log P(major allele)",
            "phylop_semantics": "signed site-oriented conservation score",
            "panel_c": "descriptive genomic-context estimates without intervals",
            "panel_d": "descriptive LD-thinning sensitivity estimates",
        },
        "outputs": {
            "png": {
                "path": str(png_path),
                "sha256": digest(png_path),
            },
            "svg": {
                "path": str(svg_path),
                "sha256": digest(svg_path),
            },
            "figure_statistics": {
                "path": str(statistics_path),
                "sha256": digest(statistics_path),
            },
        },
    }

    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "provenance.json", provenance)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("H6B_MODEL_EVIDENCE_FIGURE_PASS")


if __name__ == "__main__":
    main()
