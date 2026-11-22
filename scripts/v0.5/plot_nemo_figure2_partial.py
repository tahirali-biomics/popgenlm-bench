"""Recreate only the valid Panel D PopGenLM development adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from popgenlm.expression import metrics

MODEL = "nemo_native_reimplementation"
REQUIRED = {"gene", "family", "chrom", "split", "observed_log10_tpm",
            "predicted_log10_tpm", "model"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_panel(input_path: str | Path, output_dir: str | Path) -> dict:
    source, out = Path(input_path), Path(output_dir)
    data = pd.read_csv(source, sep="\t")
    if not REQUIRED.issubset(data.columns):
        raise ValueError("missing_required_expression_columns")
    subset = data[(data["model"] == MODEL) & (data["split"] == "test")].copy()
    if subset.empty:
        raise ValueError("no_native_nemo_test_predictions")
    if subset["gene"].isna().any() or subset["gene"].duplicated().any():
        raise ValueError("duplicate_or_missing_test_gene")
    vals = subset[["observed_log10_tpm", "predicted_log10_tpm"]].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("nonfinite_expression_pair")
    if subset["family"].isna().any() or subset["chrom"].isna().any():
        raise ValueError("missing_family_or_chromosome_partition")

    out.mkdir(parents=True, exist_ok=True)
    subset.sort_values("gene").to_csv(out / "panel_d_plotting_data.tsv", sep="\t", index=False)
    observed = subset["observed_log10_tpm"].to_numpy()
    predicted = subset["predicted_log10_tpm"].to_numpy()
    summary = metrics(observed, predicted)

    fig = plt.figure(figsize=(7.4, 7.1))
    grid = fig.add_gridspec(4, 4, hspace=0.06, wspace=0.06)
    ax = fig.add_subplot(grid[1:, :3])
    top = fig.add_subplot(grid[0, :3], sharex=ax)
    right = fig.add_subplot(grid[1:, 3], sharey=ax)
    bins = 24
    hb = ax.hexbin(observed, predicted, gridsize=bins, mincnt=1, cmap="viridis")
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.9,
            label="identity")
    top.hist(observed, bins=25, color="#92acb3", edgecolor="white", linewidth=0.2)
    right.hist(predicted, bins=25, orientation="horizontal", color="#92acb3",
               edgecolor="white", linewidth=0.2)
    top.tick_params(labelbottom=False)
    right.tick_params(labelleft=False)
    ax.set_xlabel("Observed log10(median TPM + 0.1)")
    ax.set_ylabel("Predicted log10(median TPM + 0.1)")
    ax.legend(loc="lower right", frameon=False)
    corr = "undefined: constant prediction" if summary["pearson_r"] is None else (
        f"Pearson r={summary['pearson_r']:.3f}\n"
        f"r²={summary['pearson_r_squared']:.3f}\n")
    ax.text(.03, .97,
            f"N={summary['n']}\n{corr}"
            f"predictive R²={summary['predictive_r_squared']:.3f}\n"
            f"RMSE={summary['rmse']:.3f}; MAE={summary['mae']:.3f}",
            va="top", transform=ax.transAxes, fontsize=8.5)
    fig.suptitle("Figure 2D adaptation — native nemo reimplementation\n"
                 "Arabidopsis development set; one training run", fontsize=11)
    fig.colorbar(hb, ax=right, label="Genes per hexbin", fraction=.5, pad=.2)
    fig.savefig(out / "nemo_figure2_panel_d.svg", bbox_inches="tight")
    fig.savefig(out / "nemo_figure2_panel_d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Partial PopGenLM adaptation of Rockenbach et al. (2026) Figure 2D; "
        "not an exact reproduction. Each point is one of the 300 inspected, "
        "family/chromosome-partitioned Arabidopsis development genes and the "
        "single validation-selected native nemo reimplementation checkpoint. "
        "Observed target is log10(median TPM + 0.1) from the PopGenLM frozen "
        "expression preparation; exact correspondence to the paper's 54-of-56 "
        "condition selection is unresolved. Hexbin color is gene count; top/right "
        "histograms use the same paired genes. Dashed line is identity. Pearson r, "
        "squared Pearson correlation, predictive R², RMSE and MAE are distinct. "
        "This is development performance, not independent evaluation; one fit is "
        "not a repeated-training distribution. Panels A-C and paper comparators "
        "remain incomplete."
    )
    (out / "caption.md").write_text(caption + "\n")
    metric_row = {"model": MODEL, "split": "test", **summary,
                  "scope": "development-only partial Figure 2D adaptation",
                  "training_repetitions": 1}
    pd.DataFrame([metric_row]).to_csv(out / "panel_d_metrics.tsv", sep="\t", index=False)
    manifest = {
        "status": "PASS_PARTIAL_PANEL_D",
        "figure2_complete": False,
        "panel": "D",
        "model": MODEL,
        "input": str(source),
        "input_sha256": sha256(source),
        "plotting_data_sha256": sha256(out / "panel_d_plotting_data.tsv"),
        "code_sha256": sha256(Path(__file__).resolve()),
        "paired_gene_count": len(subset),
        "training_repetitions": 1,
        "reserve_accessed": False,
        "metrics": summary,
        "limitations": [
            "development genes only; not confirmatory",
            "not exact source panel; 54-of-56 condition mapping unresolved",
            "no training-repeat uncertainty",
            "panels A-C remain incomplete",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output_names = ["panel_d_plotting_data.tsv", "panel_d_metrics.tsv",
                    "caption.md", "nemo_figure2_panel_d.svg",
                    "nemo_figure2_panel_d.png"]
    manifest["outputs_sha256"] = {
        name: sha256(out / name) for name in output_names
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_panel(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
