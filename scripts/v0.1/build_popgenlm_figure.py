from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets" / "popgenlm" / "gpn-scores-100.tsv"
SUMMARY = ROOT / "assets" / "popgenlm" / "summary.json"
OUTPUT = ROOT / "assets" / "popgenlm" / "benchmark-overview.png"

NAVY = "#0D2238"
BLUE = "#1972BD"
TEAL = "#14827A"
GOLD = "#D79B2E"
SLATE = "#5D6F80"
GRID = "#DDE6ED"
PALE_BLUE = "#E9F3FB"
PALE_TEAL = "#E8F6F3"
WHITE = "#FFFFFF"


def style_axis(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_facecolor(WHITE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BFCBD6")
    ax.tick_params(colors=SLATE, labelsize=9)
    ax.text(
        0.0,
        1.125,
        label,
        transform=ax.transAxes,
        color=TEAL,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )
    ax.set_title(title, loc="left", color=NAVY, fontsize=13, fontweight="bold", pad=4, y=1.015)


def main() -> None:
    scores = pd.read_csv(DATA, sep="\t")
    summary = json.loads(SUMMARY.read_text())
    values = scores["score"].to_numpy(float)
    positions = scores["pos"].to_numpy(int)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": SLATE,
            "axes.titlecolor": NAVY,
            "text.color": NAVY,
        }
    )

    fig = plt.figure(figsize=(14, 9), dpi=170, facecolor="#F7F9FC", constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 1], hspace=0.28, wspace=0.18)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.115, hspace=0.68, wspace=0.30)
    fig.suptitle(
        "PopGenLM Bench · verified v0.1 engineering fixture",
        x=0.02,
        ha="left",
        color=NAVY,
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.918,
        "100 deterministic SNVs · GPN Brassicales · alternate-minus-reference log-likelihood ratio",
        color=SLATE,
        fontsize=10.5,
        ha="left",
    )

    # A · distribution
    style_axis(ax_a, "A · SCORE DISTRIBUTION", "Most fixture variants received scores below zero")
    bins = np.linspace(values.min() - 0.15, values.max() + 0.15, 18)
    counts, edges, patches = ax_a.hist(values, bins=bins, edgecolor=WHITE, linewidth=1.1)
    for patch, left, right in zip(patches, edges[:-1], edges[1:]):
        patch.set_facecolor(TEAL if (left + right) / 2 < 0 else GOLD)
        patch.set_alpha(0.92)
    median = float(summary["median"])
    ax_a.axvspan(summary["q1"], summary["q3"], color=PALE_BLUE, alpha=0.9, zorder=0)
    ax_a.axvline(0, color=SLATE, linewidth=1.1, linestyle=(0, (4, 3)))
    ax_a.axvline(median, color=NAVY, linewidth=2.2)
    ax_a.set_xlabel("GPN score")
    ax_a.set_ylabel("Variant count")
    ax_a.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_a.text(
        0.97,
        0.92,
        f"median  {median:.3f}\nnegative  {summary['proportion_negative']:.0%}",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=NAVY,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": WHITE, "edgecolor": GRID},
    )

    # B · positional profile
    style_axis(ax_b, "B · POSITIONAL PROFILE", "Scores vary across the deterministic test locus")
    point_colors = np.where(values < 0, TEAL, GOLD)
    ax_b.plot(positions, values, color="#C7D3DD", linewidth=0.8, zorder=1)
    ax_b.scatter(positions, values, c=point_colors, s=25, edgecolors=WHITE, linewidths=0.55, zorder=2)
    ax_b.axhline(0, color=SLATE, linewidth=1.0, linestyle=(0, (4, 3)))
    lowest = np.argsort(values)[:2]
    offsets = [(12, -10), (-12, -10)]
    for idx, offset in zip(lowest, offsets):
        ax_b.annotate(
            f"{values[idx]:.2f}",
            (positions[idx], values[idx]),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color=NAVY,
        )
    ax_b.set_xlabel("Position within 700-bp fixture")
    ax_b.set_ylabel("GPN score")
    ax_b.margins(x=0.025, y=0.12)

    # C · ranked profile
    style_axis(ax_c, "C · RANKED PROFILE", "The lower tail is separated from a smaller positive-score set")
    ranked = np.sort(values)
    ranks = np.arange(1, len(ranked) + 1)
    neg = ranked < 0
    ax_c.fill_between(ranks, ranked, 0, where=neg, color=PALE_TEAL, alpha=1)
    ax_c.fill_between(ranks, ranked, 0, where=~neg, color="#FFF2D2", alpha=1)
    ax_c.plot(ranks[neg], ranked[neg], color=TEAL, linewidth=2.2)
    ax_c.plot(ranks[~neg], ranked[~neg], color=GOLD, linewidth=2.2)
    ax_c.axhline(0, color=SLATE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax_c.axvline(int(summary["proportion_negative"] * len(ranked)), color=NAVY, linewidth=1.0, linestyle=(0, (2, 3)))
    ax_c.text(
        0.04,
        0.92,
        "77 scores below zero",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=NAVY,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": WHITE, "edgecolor": GRID},
    )
    ax_c.set_xlabel("Variant rank")
    ax_c.set_ylabel("GPN score")
    ax_c.set_xlim(1, len(ranked))

    # D · interval summary
    style_axis(ax_d, "D · UNCERTAINTY + RANGE", "Bootstrap interval is reported separately from score spread")
    ax_d.grid(False)
    y = [3, 2, 1]
    ax_d.hlines(y[0], summary["minimum"], summary["maximum"], color="#AEBCC8", linewidth=7, capstyle="round")
    ax_d.scatter([summary["minimum"], summary["maximum"]], [y[0], y[0]], color=SLATE, s=28, zorder=3)
    ax_d.hlines(y[1], summary["q1"], summary["q3"], color=BLUE, linewidth=9, capstyle="round")
    ci_low, ci_high = summary["median_bootstrap_95pct_ci"]
    ax_d.hlines(y[2], ci_low, ci_high, color=TEAL, linewidth=9, capstyle="round")
    ax_d.scatter([median], [y[1]], color=NAVY, s=58, marker="D", zorder=4)
    ax_d.scatter([median], [y[2]], color=NAVY, s=58, marker="D", zorder=4)
    ax_d.axvline(0, color=SLATE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax_d.set_yticks(y, ["Observed range", "Interquartile range", "Median 95% bootstrap CI"])
    ax_d.tick_params(axis="y", labelsize=9.5, colors=NAVY)
    ax_d.set_xlabel("GPN score")
    ax_d.set_xlim(summary["minimum"] - 0.7, summary["maximum"] + 0.7)
    ax_d.text(
        0.98,
        0.08,
        f"median {median:.3f}\n95% CI {ci_low:.3f} to {ci_high:.3f}",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        color=NAVY,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": WHITE, "edgecolor": GRID},
    )

    fig.text(
        0.02,
        0.025,
        "Engineering fixture only · not a population sample and not evidence for selection or functional constraint",
        color=SLATE,
        fontsize=9,
        ha="left",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


if __name__ == "__main__":
    main()
