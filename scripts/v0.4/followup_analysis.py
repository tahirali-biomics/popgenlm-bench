"""Offline follow-up: verify original ranks, then audit exploratory score rules."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd

from popgenlm.regulatory import paired_recovery, percentile, rank_candidates


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outcomes(records):
    result = paired_recovery(records)
    result.update(
        {
            "into_top5": sum(r["model_hit"] == 1 and r["baseline_hit"] == 0 for r in records),
            "out_of_top5": sum(r["model_hit"] == 0 and r["baseline_hit"] == 1 for r in records),
            "stayed_in": sum(r["model_hit"] == 1 and r["baseline_hit"] == 1 for r in records),
            "stayed_out": sum(r["model_hit"] == 0 and r["baseline_hit"] == 0 for r in records),
            "model_hits": sum(r["model_hit"] for r in records),
            "baseline_hits": sum(r["baseline_hit"] for r in records),
        }
    )
    return result


def analyze(pilot, run):
    pilot, run = Path(pilot), Path(run)
    protocol = json.loads((run / "inputs/followup_protocol.json").read_text())
    f = pilot / "inputs/frozen"
    manifest = {}
    for name, meta in json.loads((f / "freeze.json").read_text())["files"].items():
        path = f / name
        assert digest(path) == meta["sha256"], name
        manifest[str(path)] = digest(path)
    x = pd.read_csv(f / "candidates.tsv", sep="\t")
    assert len(x) == 1205 and x.gene.nunique() == 40 and x.variant_id.is_unique
    oldrank = pd.read_csv(pilot / "results/analysis/rankings.tsv", sep="\t")
    oldlocus = pd.read_csv(pilot / "results/analysis/per_locus_recovery.tsv", sep="\t")
    original = []
    for (scope, model, endpoint), g in oldlocus.groupby(["scope", "model", "endpoint"]):
        original.append(
            dict(scope=scope, model=model, endpoint=endpoint, **outcomes(g.to_dict("records")))
        )
        for r in g.itertuples():
            z = oldrank[
                (oldrank.scope == scope)
                & (oldrank.ranking_model == model)
                & (oldrank.gene == r.gene)
            ]
            h = x[x.gene == r.gene].sort_values(["heldout_p", "variant_id"]).iloc[0].variant_id
            target = z[z.variant_id == h].iloc[0]
            assert int(target["rank"] <= 5) == r.model_hit
            assert int(target.paired_baseline_rank <= 5) == r.baseline_hit
    scores = {}
    for length in [512, 4096]:
        paths = sorted((pilot / "results/plantcad2_s").glob(f"chr*_{length}.tsv"))
        tables = []
        for p in paths:
            meta = json.loads(p.with_suffix(".complete.json").read_text())
            assert digest(p) == meta["sha256"]
            manifest[str(p)] = digest(p)
            tables.append(pd.read_csv(p, sep="\t"))
        z = pd.concat(tables).set_index("variant_id").loc[x.variant_id]
        assert np.allclose(z.magnitude, abs(z.raw_alt_minus_ref))
        assert np.allclose(z.raw_alt_minus_ref, (z.llr_forward + z.llr_rc) / 2)
        assert np.allclose(
            z.minor_minus_major,
            np.where(x.af_alt.to_numpy() > 0.5, -z.raw_alt_minus_ref, z.raw_alt_minus_ref),
        )
        scores[length] = z.raw_alt_minus_ref.to_numpy()
        x[f"raw_{length}"] = scores[length]
    (run / "results/reused_raw_scores.tsv").write_text(
        x[["gene", "variant_id", "raw_512", "raw_4096"]].to_csv(sep="\t", index=False)
    )
    covered = set()
    for path in sorted((pilot / "results/nemo_bnapus_transfer").glob("AT*.tsv")):
        table = pd.read_csv(path, sep="\t")
        covered.update(
            zip(
                table.loc[table.status == "covered", "gene"],
                table.loc[table.status == "covered", "variant_id"],
            )
        )
        manifest[str(path)] = digest(path)
    records, rankings = [], []
    for scope in ["model_covered", "shared_all_runnable"]:
        for gene, full in x.groupby("gene", sort=True):
            g = (
                full.copy()
                if scope == "model_covered"
                else full.loc[[(gene, v) in covered for v in full.variant_id]].copy()
            )
            if len(g) < 6:
                continue
            ann = (
                rank_candidates(g, "annotations")
                .set_index("variant_id")
                .priority.reindex(g.variant_id)
                .to_numpy()
            )
            truth = full.sort_values(["heldout_p", "variant_id"]).iloc[0]
            eligible = bool(
                np.isfinite(truth.heldout_p)
                and truth.heldout_p < 0.05 / len(full)
                and truth.variant_id in set(g.variant_id)
            )
            base = rank_candidates(g, "annotations")
            base_ranks = {v: i + 1 for i, v in enumerate(base.variant_id)}
            transforms = {
                "abs_512": percentile(abs(g.raw_512)),
                "abs_4096": percentile(abs(g.raw_4096)),
                "signed_512": percentile(g.raw_512),
                "negative_512": percentile(-g.raw_512),
                "context_mean": (percentile(abs(g.raw_512)) + percentile(abs(g.raw_4096))) / 2,
            }
            rules = [("baseline", ann)]
            for name, value in transforms.items():
                for weight in protocol["exploratory_weights"] if name.startswith("abs_") else [0.2]:
                    rules.append((f"{name}_w{weight:g}", (1 - weight) * ann + weight * value))
            for rule, priority in rules:
                z = g.assign(priority=priority).sort_values(
                    ["priority", "variant_id"], ascending=[False, True]
                )
                ranks = {v: i + 1 for i, v in enumerate(z.variant_id)}
                for v in z.variant_id:
                    rankings.append(
                        {
                            "scope": scope,
                            "gene": gene,
                            "variant_id": v,
                            "rule": rule,
                            "rank": ranks[v],
                            "baseline_rank": base_ranks[v],
                            "target": v == truth.variant_id,
                            "eligible": eligible,
                        }
                    )
                if rule in ["abs_512_w0.2", "abs_4096_w0.2"]:
                    length = 512 if rule.startswith("abs_512_") else 4096
                    old = oldrank[
                        (oldrank.scope == scope)
                        & (oldrank.ranking_model == f"plantcad2_s_{length}")
                        & (oldrank.gene == gene)
                    ]
                    assert dict(zip(old.variant_id, old["rank"])) == ranks, (gene, rule)
                if eligible:
                    records.append(
                        {
                            "scope": scope,
                            "gene": gene,
                            "block": "chr" + str(int(g.chrom.iloc[0])),
                            "rule": rule,
                            "target": truth.variant_id,
                            "target_rank": ranks[truth.variant_id],
                            "baseline_rank": base_ranks[truth.variant_id],
                            "model_hit": int(ranks[truth.variant_id] <= 5),
                            "baseline_hit": int(base_ranks[truth.variant_id] <= 5),
                            "mean_ld_r2": float(full.ld_r2_lead.mean()),
                            "candidates": len(g),
                        }
                    )
    records = pd.DataFrame(records)
    metrics = []
    for (scope, rule), g in records.groupby(["scope", "rule"], sort=True):
        for stratum in ["all", "LD<0.9", "LD>=0.9"]:
            subset = (
                g
                if stratum == "all"
                else g[g.mean_ld_r2 < 0.9]
                if stratum == "LD<0.9"
                else g[g.mean_ld_r2 >= 0.9]
            )
            metrics.append(
                dict(scope=scope, rule=rule, stratum=stratum, **outcomes(subset.to_dict("records")))
            )
    out = run / "results"
    pd.DataFrame(original).to_json(out / "reproduction.json", orient="records", indent=2)
    pd.DataFrame(rankings).to_csv(out / "rankings.tsv", sep="\t", index=False)
    records.to_csv(out / "paired_locus_outcomes.tsv", sep="\t", index=False)
    pd.DataFrame(metrics).to_csv(out / "paired_metrics.tsv", sep="\t", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    worked = min(x.gene)
    pd.DataFrame(rankings).loc[lambda z: z.gene == worked].to_csv(
        out / "worked_locus.tsv", sep="\t", index=False
    )
    context = []
    from scipy.stats import spearmanr

    for gene, g in x.groupby("gene", sort=True):
        context.append(
            {
                "gene": gene,
                "candidates": len(g),
                "raw_spearman": spearmanr(g.raw_512, g.raw_4096).statistic,
                "magnitude_spearman": spearmanr(abs(g.raw_512), abs(g.raw_4096)).statistic,
                "maximum_strand_difference": max(
                    float(
                        abs(
                            pd.concat(
                                [
                                    pd.read_csv(p, sep="\t")
                                    for p in sorted(
                                        (pilot / "results/plantcad2_s").glob(f"chr*_{n}.tsv")
                                    )
                                ]
                            )
                            .set_index("variant_id")
                            .loc[g.variant_id]
                            .eval("llr_forward-llr_rc")
                        ).max()
                    )
                    for n in [512, 4096]
                ),
            }
        )
    pd.DataFrame(context).to_csv(out / "context_audit.tsv", sep="\t", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    shown = [m for m in metrics if m["stratum"] == "all" and m["scope"] == "model_covered"]
    for i, m in enumerate(shown):
        lo, hi = m["ci95"]
        d = m["delta"]
        ax.errorbar(
            100 * d,
            i,
            xerr=[[100 * (d - lo)], [100 * (hi - d)]],
            fmt="o",
            capsize=3,
            color="#245879",
        )
    ax.set_yticks(
        range(len(shown)),
        [
            m["rule"] + f" ({m['model_hits']}/{m['loci']}; +{m['into_top5']}/−{m['out_of_top5']})"
            for m in shown
        ],
    )
    ax.axvline(0, color="grey", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Paired change versus annotation baseline (percentage points)\n95% descriptive bootstrap; 5 chromosome blocks"
    )
    ax.set_title(
        "Exploratory rules on original 40 loci\nHeld-out association signals: 34 loci; baseline 17/34\nIndependent nucleotide endpoint: not estimable"
    )
    fig.tight_layout()
    fig.savefig(out / "paired_results.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "paired_results.svg", bbox_inches="tight")
    plt.close(fig)
    # Record every source consumed, including original saved ranks and protocol.
    for p in [
        pilot / "results/analysis/rankings.tsv",
        pilot / "results/analysis/per_locus_recovery.tsv",
        run / "inputs/followup_protocol.json",
    ]:
        manifest[str(p)] = digest(p)
    (out / "reanalysis_manifest.json").write_text(
        json.dumps(
            {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "python": sys.executable,
                "inputs": manifest,
                "assertions": [
                    "all frozen hashes",
                    "40 loci/1205 SNPs",
                    "raw orientation/magnitude",
                    "saved paired outcomes independently recovered from ranks",
                    "both original PlantCAD2 ranks reproduced exactly",
                ],
            },
            indent=2,
        )
    )
    print(pd.DataFrame(metrics).query("stratum == 'all'").to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", required=True)
    p.add_argument("--run", required=True)
    a = p.parse_args()
    analyze(a.pilot, a.run)
