"""CPU-only reanalysis of frozen inputs and precomputed scores; no inference."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd
from popgenlm.regulatory import paired_recovery, rank_candidates

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
a = p.parse_args()
run = Path(a.run)
frozen = run / "inputs/frozen"
out = run / "results/analysis"
out.mkdir(parents=True, exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


freeze = json.loads((frozen / "freeze.json").read_text())
for n, v in freeze["files"].items():
    assert sha(frozen / n) == v["sha256"], n
x = pd.read_csv(frozen / "candidates.tsv", sep="\t")
loci = pd.read_csv(frozen / "loci.tsv", sep="\t")
assert not x.duplicated(["gene", "variant_id"]).any()
scores = {}
coverage = []
for model in ["plantcad2_s", "plantcad_v1", "gpn_brassicales"]:
    folder = run / "results" / model
    for length in [512, 4096] if model == "plantcad2_s" else [512]:
        paths = sorted(folder.glob(f"chr*_{length}.tsv"))
        if not paths:
            continue
        table = pd.concat([pd.read_csv(q, sep="\t") for q in paths], ignore_index=True)
        assert table.variant_id.is_unique and np.isfinite(table.raw_alt_minus_ref).all()
        key = model + "_" + str(length)
        scores[key] = dict(zip(table.variant_id, table.raw_alt_minus_ref))
folder = run / "results/nemo_bnapus_transfer"
paths = sorted(folder.glob("AT*.tsv"))
if paths:
    table = pd.concat([pd.read_csv(q, sep="\t") for q in paths], ignore_index=True)
    scores["nemo_bnapus_transfer"] = {
        (r.gene, r.variant_id): r.delta_log10_tpm
        for r in table.itertuples()
        if r.status == "covered" and np.isfinite(r.delta_log10_tpm)
    }
for name, mapping in scores.items():
    x[name] = [
        mapping.get((r.gene, r.variant_id), np.nan)
        if name == "nemo_bnapus_transfer"
        else mapping.get(r.variant_id, np.nan)
        for r in x.itertuples()
    ]
    coverage.append(
        {
            "model": name,
            "candidate_pairs_covered": int(x[name].notna().sum()),
            "candidate_pairs_total": len(x),
            "loci_complete": int(x.groupby("gene")[name].apply(lambda a: a.notna().all()).sum()),
            "loci_total": len(loci),
            "primary_positive_pairs_covered": int(
                (x[name].notna() & x.independent_nucleotide_support).sum()
            ),
        }
    )
pd.DataFrame(coverage).to_csv(out / "coverage.tsv", sep="\t", index=False)
metrics = []
per_locus = []
rankings = []
shared = x[list(scores)].notna().all(axis=1) if scores else np.zeros(len(x), bool)
for scope in ["model_covered", "shared_all_runnable"]:
    for model in ["association_ld", "annotations"] + list(scores):
        for gene, full in x.groupby("gene", sort=True):
            subset = (
                full.loc[shared[full.index]]
                if scope == "shared_all_runnable"
                else full[full[model].notna()]
                if model in scores
                else full
            )
            if len(subset) < 6:
                continue
            ranked = rank_candidates(subset, model)
            baseline = rank_candidates(subset, "annotations")
            assoc = rank_candidates(subset, "association_ld")
            top = set(ranked.head(5).variant_id)
            base = set(baseline.head(5).variant_id)
            positives = set(full.loc[full.independent_nucleotide_support, "variant_id"])
            h = full[np.isfinite(full.heldout_p) & (full.heldout_p > 0)].sort_values(
                ["heldout_p", "variant_id"]
            )
            heldout = str(h.iloc[0].variant_id) if len(h) else None
            replicated = bool(len(h) and h.iloc[0].heldout_p < 0.05 / len(full))
            endpoints = []
            if positives and positives.intersection(subset.variant_id):
                endpoints.append(("independent_nucleotide", positives))
            if replicated and heldout in set(subset.variant_id):
                endpoints.append(("heldout_mapping_signal", {heldout}))
            for endpoint, truth in endpoints:
                record = {
                    "scope": scope,
                    "model": model,
                    "gene": gene,
                    "block": "chr" + str(int(full.iloc[0].chrom)),
                    "endpoint": endpoint,
                    "model_hit": int(bool(top & truth)),
                    "baseline_hit": int(bool(base & truth)),
                    "association_ld_hit": int(bool(set(assoc.head(5).variant_id) & truth)),
                    "candidates": len(subset),
                    "full_candidates": len(full),
                    "mean_ld_r2": float(full.ld_r2_lead.mean()),
                }
                per_locus.append(record)
            ranked = ranked.copy()
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            ranked["ranking_model"] = model
            ranked["scope"] = scope
            rankings.append(ranked)
        for endpoint in ["independent_nucleotide", "heldout_mapping_signal"]:
            records = [
                r
                for r in per_locus
                if r["scope"] == scope and r["model"] == model and r["endpoint"] == endpoint
            ]
            for stratum in ["all", "mean_ld_below_0.9", "mean_ld_at_least_0.9"]:
                selected = [
                    r
                    for r in records
                    if stratum == "all"
                    or (
                        r["mean_ld_r2"] < 0.9
                        if stratum == "mean_ld_below_0.9"
                        else r["mean_ld_r2"] >= 0.9
                    )
                ]
                metrics.append(
                    dict(
                        scope=scope,
                        model=model,
                        endpoint=endpoint,
                        ld_stratum=stratum,
                        **paired_recovery(selected),
                    )
                )
pd.DataFrame(per_locus).to_csv(out / "per_locus_recovery.tsv", sep="\t", index=False)
rank = pd.concat(rankings, ignore_index=True)
rank.to_csv(out / "rankings.tsv", sep="\t", index=False)
(out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
# Worked locus fixed by gene identifier, not improvement magnitude.
worked = min(x.gene.unique())
rank[(rank.gene == worked) & (rank.scope == "model_covered") & (rank["rank"] <= 5)].to_csv(
    out / "worked_locus_top5.tsv", sep="\t", index=False
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, endpoint, title in zip(
    axes,
    ["independent_nucleotide", "heldout_mapping_signal"],
    ["Independent reporter nucleotide recovery", "Held-out mapping signal recovery"],
):
    records = [
        r
        for r in metrics
        if r["scope"] == "model_covered" and r["endpoint"] == endpoint and r["ld_stratum"] == "all"
    ]
    usable = [r for r in records if r["delta"] is not None]
    ax.axvline(0, color="grey", lw=1)
    if usable:
        for i, r in enumerate(usable):
            lo, hi = r["ci95"]
            ax.errorbar(
                100 * r["delta"],
                i,
                xerr=[[100 * (r["delta"] - lo)], [100 * (hi - r["delta"])]],
                fmt="o",
                color="#245879",
                capsize=3,
            )
        ax.set_yticks(
            range(len(usable)), [r["model"] + " (n=" + str(r["loci"]) + ")" for r in usable]
        )
        ax.invert_yaxis()
    else:
        ax.text(
            0.5,
            0.5,
            "Not estimable\n0 eligible loci",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel(
        "Paired change in top-five recovery\npercentage points versus annotations baseline"
    )
fig.suptitle("PopGenLM v0.4 pilot — descriptive evidence; chromosomes resampled intact")
fig.tight_layout()
fig.savefig(out / "top5_recovery.png", dpi=180)
fig.savefig(out / "top5_recovery.svg")
plt.close(fig)
summary = {
    "loci": len(loci),
    "candidate_pairs": len(x),
    "unique_variants": x.variant_id.nunique(),
    "primary_loci": int(x.groupby("gene").independent_nucleotide_support.any().sum()),
    "worked_locus": worked,
    "models": list(scores),
    "coverage": coverage,
    "metrics": metrics,
    "baseline": "association + LD + standard annotations; model and baseline evaluated on identical covered candidates",
    "limitations": [
        "Reporter labels are isolated oligo substitutions, not endogenous genome-editing validation",
        "Held-out samples share the source study and may be related; LOCO LMM controls relatedness, but this is not independent-study replication",
        "Voichek tested phenotype inventory seeds recomputed GEO associations; original GEMMA batch results are not reproduced",
        "Fixed heuristic coefficients were frozen before scores; no supervised ranking weights fitted",
        "Only five chromosome blocks; bootstrap intervals are descriptive and coarse",
        "nemo is the publicly bundled Bnapus model transferred to Arabidopsis; family overlap unresolved",
        "Botanic1-S gated without existing authorization; AlphaGenome API key unavailable",
    ],
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k not in ["metrics"]}, indent=2))
