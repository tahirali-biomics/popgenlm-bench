"""Generate PopGenLM Bench v0.2 population-GPN figures."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "benchmarks" / "v0.2" / "1001g_population_10000_gpn.tsv"

REPORT_DIR = ROOT / "reports" / "v0.2"

ANALYSIS_PATH = REPORT_DIR / "population_gpn_analysis.json"
CLASS_PATH = REPORT_DIR / "frequency_class_summary.tsv"
PAIRWISE_PATH = REPORT_DIR / "frequency_class_pairwise.tsv"
SENSITIVITY_PATH = REPORT_DIR / "sensitivity_summary.tsv"
CHROMOSOME_PATH = REPORT_DIR / "chromosome_robustness.tsv"

FIGURE_DIR = REPORT_DIR / "figures"

OVERVIEW_PNG = FIGURE_DIR / "v02_population_gpn_overview.png"
OVERVIEW_PDF = FIGURE_DIR / "v02_population_gpn_overview.pdf"

MAF_PNG = FIGURE_DIR / "maf-spectrum.png"
MAF_SCORE_PNG = FIGURE_DIR / "score-vs-maf.png"
CLASS_PNG = FIGURE_DIR / "score-by-frequency-class.png"
ORIENTATION_PNG = FIGURE_DIR / "orientation-check.png"

METADATA_PATH = FIGURE_DIR / "figure_metadata.json"

EXPECTED_HASHES = {
    "joined_data": ("ef188e83d0e598d016c158b4daeec08be63e19723840fc42b494c76443fca469"),
    "analysis": ("c41e2a4e59c5a9fc25f12803ae30149c502096167328874b0890e1bf4b5f11e9"),
    "class_summary": ("a7dfd82ee5f0850a68dbfcdfc21b47a0df0a694d22e4ecfc462d75afc3785c3e"),
    "pairwise": ("3779fe5578ca4b33b5d66c4a34d4eefe61d85417982a4e5f45e31b3571f80739"),
    "sensitivity": ("d00b1a016f87102d45cbf276b08b0fa0fea3379ab08194c14a3f4e8091e2a3fa"),
    "chromosome": ("5f0791cf974a35524393d99cdc248dede8fac0bc25978aca7dbefe1dd2e74dc3"),
}

SCORE = "gpn_score_minor_vs_major"

CLASS_ORDER = [
    "rare",
    "low_frequency",
    "common",
]

CLASS_LABELS = [
    "Rare",
    "Low-frequency",
    "Common",
]

# Color-blind-friendly palette.
CLASS_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
]

FLIPPED_COLOR = "#D55E00"
NONFLIPPED_COLOR = "#7A7A7A"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_inputs() -> None:
    paths = {
        "joined_data": DATA_PATH,
        "analysis": ANALYSIS_PATH,
        "class_summary": CLASS_PATH,
        "pairwise": PAIRWISE_PATH,
        "sensitivity": SENSITIVITY_PATH,
        "chromosome": CHROMOSOME_PATH,
    }

    observed = {name: sha256_file(path) for name, path in paths.items()}

    if observed != EXPECTED_HASHES:
        raise ValueError(
            "Frozen analysis input checksum mismatch:\n"
            + json.dumps(
                observed,
                indent=2,
                sort_keys=True,
            )
        )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(
    ax: plt.Axes,
    label: str,
) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_maf_spectrum(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    maf = df["maf"].to_numpy(dtype=float)

    bins = np.geomspace(
        maf.min(),
        0.5,
        45,
    )

    ax.hist(
        maf,
        bins=bins,
        color="#777777",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.35,
    )

    ax.set_xscale("log")

    ax.axvline(
        0.01,
        linestyle="--",
        linewidth=1.0,
        color=CLASS_COLORS[1],
    )

    ax.axvline(
        0.05,
        linestyle="--",
        linewidth=1.0,
        color=CLASS_COLORS[2],
    )

    ax.set_xlabel("Minor-allele frequency (MAF)")
    ax.set_ylabel("Variants")
    ax.set_title("Population-frequency spectrum")

    ax.text(
        0.01,
        0.97,
        "Rare\n<0.01",
        transform=ax.transAxes,
        va="top",
        color=CLASS_COLORS[0],
    )

    ax.text(
        0.40,
        0.97,
        "Low-frequency\n0.01–0.05",
        transform=ax.transAxes,
        va="top",
        color=CLASS_COLORS[1],
    )

    ax.text(
        0.75,
        0.97,
        "Common\n≥0.05",
        transform=ax.transAxes,
        va="top",
        color=CLASS_COLORS[2],
    )


def plot_maf_score(
    ax: plt.Axes,
    df: pd.DataFrame,
    analysis: dict,
    *,
    add_colorbar: bool,
) -> None:
    maf = df["maf"].to_numpy(dtype=float)
    score = df[SCORE].to_numpy(dtype=float)

    log_maf = np.log10(maf)

    hexbin = ax.hexbin(
        log_maf,
        score,
        gridsize=58,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )

    ax.axhline(
        0,
        color="#777777",
        linewidth=0.8,
        linestyle=":",
    )

    ax.axvline(
        np.log10(0.01),
        color="#AAAAAA",
        linewidth=0.8,
        linestyle="--",
    )

    ax.axvline(
        np.log10(0.05),
        color="#AAAAAA",
        linewidth=0.8,
        linestyle="--",
    )

    ticks = np.array(
        [
            0.001,
            0.01,
            0.1,
            0.5,
        ]
    )

    ax.set_xticks(
        np.log10(ticks),
        labels=[
            "0.001",
            "0.01",
            "0.1",
            "0.5",
        ],
    )

    ax.set_xlabel("Minor-allele frequency (MAF)")
    ax.set_ylabel("GPN minor-vs-major score\nlog P(minor) − log P(major)")

    ax.set_title("GPN score versus population frequency")

    association = analysis["continuous_association"]

    rho = association["spearman_maf_minor_vs_major"]

    permutation_p = association["chromosome_stratified_permutation_p"]

    permutation_extreme = association["permutation_extreme_count"]

    n_permutations = analysis["settings"]["permutations"]

    ax.text(
        0.03,
        0.97,
        (
            f"Spearman ρ = {rho:.3f}\n"
            "Monte Carlo p = "
            f"{permutation_extreme + 1:,}/"
            f"{n_permutations + 1:,}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.92,
        },
    )

    if not np.isclose(
        permutation_p,
        (permutation_extreme + 1) / (n_permutations + 1),
    ):
        raise ValueError("Permutation p-value does not match Monte-Carlo correction")

    if add_colorbar:
        colorbar = ax.figure.colorbar(
            hexbin,
            ax=ax,
            fraction=0.045,
            pad=0.025,
        )

        colorbar.set_label("Variants per hexagon (log scale)")


def plot_frequency_classes(
    ax: plt.Axes,
    df: pd.DataFrame,
    class_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    groups = [
        df.loc[
            df["frequency_class"] == group,
            SCORE,
        ].to_numpy(dtype=float)
        for group in CLASS_ORDER
    ]

    parts = ax.violinplot(
        groups,
        positions=[
            1,
            2,
            3,
        ],
        widths=0.78,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        points=200,
    )

    for body, color in zip(
        parts["bodies"],
        CLASS_COLORS,
        strict=True,
    ):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.65)

    parts["cmedians"].set_color("#222222")
    parts["cmedians"].set_linewidth(1.5)

    ax.axhline(
        0,
        color="#777777",
        linewidth=0.8,
        linestyle=":",
    )

    counts = dict(
        zip(
            class_summary["frequency_class"],
            class_summary["n"],
            strict=True,
        )
    )

    labels = [
        (f"{label}\nn={int(counts[group]):,}")
        for label, group in zip(
            CLASS_LABELS,
            CLASS_ORDER,
            strict=True,
        )
    ]

    ax.set_xticks(
        [
            1,
            2,
            3,
        ],
        labels=labels,
    )

    ax.set_ylabel("GPN minor-vs-major score\nlog P(minor) − log P(major)")

    ax.set_title("Score distributions by frequency class")

    rare_common = pairwise.loc[pairwise["contrast"] == "rare_vs_common"].iloc[0]

    effect = rare_common["rank_biserial_a_vs_b"]

    ci_low = rare_common["rank_biserial_ci_low"]

    ci_high = rare_common["rank_biserial_ci_high"]

    ax.text(
        0.03,
        0.97,
        (
            "Rare vs common\n"
            f"rank-biserial = {effect:.3f}\n"
            f"bootstrap 95% CI [{ci_low:.3f}, {ci_high:.3f}]"
        ),
        transform=ax.transAxes,
        va="top",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.92,
        },
    )


def plot_orientation(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    raw = df["gpn_score_ref_alt"].to_numpy(dtype=float)

    oriented = df[SCORE].to_numpy(dtype=float)

    flipped = (
        df["flipped"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
        .to_numpy(dtype=bool)
    )

    not_flipped = ~flipped

    ax.scatter(
        raw[not_flipped],
        oriented[not_flipped],
        s=8,
        alpha=0.16,
        color=NONFLIPPED_COLOR,
        linewidths=0,
        label=(f"Not flipped (n={int(not_flipped.sum()):,})"),
        zorder=2,
    )

    ax.scatter(
        raw[flipped],
        oriented[flipped],
        s=17,
        alpha=0.75,
        color=FLIPPED_COLOR,
        linewidths=0,
        label=(f"ALT-major; sign flipped (n={int(flipped.sum()):,})"),
        zorder=3,
    )

    limit = float(
        np.ceil(
            max(
                np.max(np.abs(raw)),
                np.max(np.abs(oriented)),
            )
        )
    )

    reference = np.array(
        [
            -limit,
            limit,
        ]
    )

    ax.plot(
        reference,
        reference,
        color="#444444",
        linewidth=0.9,
        alpha=0.65,
        label="y = x",
        zorder=1,
    )

    ax.plot(
        reference,
        -reference,
        color=FLIPPED_COLOR,
        linewidth=0.9,
        linestyle="--",
        alpha=0.65,
        label="y = −x",
        zorder=1,
    )

    ax.axhline(
        0,
        color="#BBBBBB",
        linewidth=0.6,
    )

    ax.axvline(
        0,
        color="#BBBBBB",
        linewidth=0.6,
    )

    ax.set_xlim(
        -limit,
        limit,
    )

    ax.set_ylim(
        -limit,
        limit,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("Raw GPN score\nlog P(ALT) − log P(REF)")

    ax.set_ylabel("Population-oriented score\nlog P(minor) − log P(major)")

    ax.set_title("Population-oriented GPN scores")

    ax.legend(
        loc="upper center",
        ncol=2,
        frameon=False,
    )


def save_standalone(
    filename: Path,
    plotting_function,
    *args,
    **kwargs,
) -> None:
    fig, ax = plt.subplots(
        figsize=(
            6.4,
            4.8,
        ),
        layout="constrained",
    )

    plotting_function(
        ax,
        *args,
        **kwargs,
    )

    fig.savefig(
        filename,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    verify_inputs()
    configure_matplotlib()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        dtype={
            "chrom": str,
            "fasta_chrom": str,
        },
    )

    analysis = json.loads(ANALYSIS_PATH.read_text())

    class_summary = pd.read_csv(
        CLASS_PATH,
        sep="\t",
    )

    pairwise = pd.read_csv(
        PAIRWISE_PATH,
        sep="\t",
    )

    #
    # Standalone figures.
    #
    save_standalone(
        MAF_PNG,
        plot_maf_spectrum,
        df,
    )

    save_standalone(
        MAF_SCORE_PNG,
        plot_maf_score,
        df,
        analysis,
        add_colorbar=True,
    )

    save_standalone(
        CLASS_PNG,
        plot_frequency_classes,
        df,
        class_summary,
        pairwise,
    )

    save_standalone(
        ORIENTATION_PNG,
        plot_orientation,
        df,
    )

    #
    # Combined 2 × 2 overview figure.
    #
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.0,
            9.2,
        ),
        layout="constrained",
    )

    plot_maf_spectrum(
        axes[0, 0],
        df,
    )

    plot_maf_score(
        axes[0, 1],
        df,
        analysis,
        add_colorbar=True,
    )

    plot_frequency_classes(
        axes[1, 0],
        df,
        class_summary,
        pairwise,
    )

    plot_orientation(
        axes[1, 1],
        df,
    )

    for ax, label in zip(
        axes.flat,
        [
            "A",
            "B",
            "C",
            "D",
        ],
        strict=True,
    ):
        panel_label(
            ax,
            label,
        )

    fig.savefig(
        OVERVIEW_PNG,
        bbox_inches="tight",
    )

    fig.savefig(
        OVERVIEW_PDF,
        bbox_inches="tight",
        metadata={
            "Title": ("PopGenLM Bench v0.2 population-GPN overview"),
            "Author": "Tahir Ali",
            "Creator": ("PopGenLM Bench plot_v02_population_gpn.py"),
            "CreationDate": None,
            "ModDate": None,
        },
    )

    plt.close(fig)

    outputs = [
        MAF_PNG,
        MAF_SCORE_PNG,
        CLASS_PNG,
        ORIENTATION_PNG,
        OVERVIEW_PNG,
        OVERVIEW_PDF,
    ]

    figure_hashes = {path.name: sha256_file(path) for path in outputs}

    metadata = {
        "figure_set": ("popgenlm_v0.2_population_gpn"),
        "input_hashes": EXPECTED_HASHES,
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "panels": {
            "A": ("MAF spectrum with predefined 0.01 and 0.05 frequency thresholds"),
            "B": (
                "MAF versus GPN minor-vs-major score; "
                "hexbin density with frozen Spearman "
                "and permutation statistics"
            ),
            "C": (
                "GPN score distributions across rare, "
                "low-frequency, and common variants; "
                "rare-vs-common frozen effect size"
            ),
            "D": ("Raw REF-to-ALT versus population-oriented score showing ALT-major sign flips"),
        },
        "interpretation": (
            "Figures show weak but reproducible concordance "
            "between population frequency and GPN scores. "
            "They are not evidence by themselves of natural "
            "selection."
        ),
        "outputs": figure_hashes,
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("PopGenLM Bench v0.2 figures")
    print("=" * 35)

    for path in outputs:
        print(
            path.name,
            sha256_file(path),
        )

    print(
        METADATA_PATH.name,
        sha256_file(METADATA_PATH),
    )

    print()
    print("Matplotlib:", matplotlib.__version__)
    print("NumPy:", np.__version__)
    print("pandas:", pd.__version__)

    print()
    print("V0.2 FIGURE GENERATION PASSED")


if __name__ == "__main__":
    main()
