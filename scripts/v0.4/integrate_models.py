"""CPU-only candidate-matched integration of pinned regulatory model scores."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd

from popgenlm.regulatory import paired_recovery, percentile, rank_candidates


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize(records):
    m = paired_recovery(records)
    m.update(
        {
            "model_hits": sum(r["model_hit"] for r in records),
            "baseline_hits": sum(r["baseline_hit"] for r in records),
            "into_top5": sum(r["model_hit"] and not r["baseline_hit"] for r in records),
            "out_of_top5": sum(r["baseline_hit"] and not r["model_hit"] for r in records),
            "stayed_in": sum(r["model_hit"] and r["baseline_hit"] for r in records),
            "stayed_out": sum(not r["model_hit"] and not r["baseline_hit"] for r in records),
        }
    )
    return m


def integrate(run, pilot):
    run, pilot = Path(run), Path(pilot)
    out = run / "results/analysis"
    out.mkdir(parents=True, exist_ok=True)
    frozen = run / "inputs/frozen"
    inputs = {}
    for name, m in json.loads((frozen / "freeze.json").read_text())["files"].items():
        assert sha(frozen / name) == m["sha256"]
        inputs[str(frozen / name)] = sha(frozen / name)
    x = pd.read_csv(frozen / "candidates.tsv", sep="\t")
    assert len(x) == 1205 and x.gene.nunique() == 40 and x.variant_id.is_unique
    source_columns = []

    def load(folder, pattern, column, name, kind="variant"):
        paths = sorted(folder.glob(pattern))
        if not paths:
            return False
        tables = []
        for path in paths:
            done = path.with_suffix(".complete.json")
            assert done.exists(), str(done)
            assert sha(path) == json.loads(done.read_text())["sha256"], str(path)
            inputs[str(path)] = sha(path)
            tables.append(pd.read_csv(path, sep="\t"))
        table = pd.concat(tables, ignore_index=True)
        if kind == "nemo":
            table = table[table.status == "covered"]
        assert not table.duplicated(
            ["gene", "variant_id"] if kind == "nemo" else ["variant_id"]
        ).any()
        mapping = dict(
            zip(
                zip(table.gene, table.variant_id) if kind == "nemo" else table.variant_id,
                table[column],
            )
        )
        x[name] = [
            mapping.get((r.gene, r.variant_id) if kind == "nemo" else r.variant_id, np.nan)
            for r in x.itertuples()
        ]
        assert np.isfinite(table[column]).all()
        source_columns.append(name)
        return True

    for model in ["plantcad2_s", "plantcad_v1", "gpn_brassicales"]:
        for length in [512, 4096] if model == "plantcad2_s" else [512]:
            load(
                pilot / "results" / model,
                f"chr*_{length}.tsv",
                "raw_alt_minus_ref",
                f"{model}_{length}",
            )
    load(
        pilot / "results/nemo_bnapus_transfer",
        "AT*.tsv",
        "delta_log10_tpm",
        "nemo_bnapus_transfer",
        "nemo",
    )
    for length in [512, 4096]:
        load(
            run / "results/botanic1_s",
            f"chr*_{length}.tsv",
            "raw_alt_minus_ref",
            f"botanic1_s_{length}",
        )
    load(run / "results/alphagenome", "AT*.tsv", "score", "alphagenome")
    for model in ["botanic1_s_512", "botanic1_s_4096"]:
        if model in x:
            assert x[model].notna().sum() == 1205, "partial Botanic production panel"
    coverage = []
    for name in source_columns:
        coverage.append(
            {
                "model": name,
                "SNPs": int(x[name].notna().sum()),
                "total_SNPs": len(x),
                "loci_with_any": int(x[x[name].notna()].gene.nunique()),
                "complete_loci": int(
                    x.groupby("gene")[name].apply(lambda v: v.notna().all()).sum()
                ),
                "total_loci": 40,
                "primary_nucleotide_loci": 0,
            }
        )
    pd.DataFrame(coverage).to_csv(out / "coverage.tsv", sep="\t", index=False)
    x[["gene", "variant_id"] + source_columns].to_csv(
        out / "frozen_scores.tsv", sep="\t", index=False
    )
    rules = {"association_ld": (None, None), "annotations": (None, None)}
    for name in source_columns:
        rules[name + "_abs"] = (name, "abs")
    for model in ["plantcad2_s_512", "botanic1_s_512"]:
        if model in x:
            rules[model + "_signed_exploratory"] = (model, "signed")
    masks = {"model_covered": np.ones(len(x), bool)}
    mlm = [c for c in source_columns if c not in ["alphagenome", "nemo_bnapus_transfer"]]
    masks["matched_MLM"] = x[mlm].notna().all(axis=1)
    if "nemo_bnapus_transfer" in x:
        masks["legacy_nemo_shared"] = x[mlm + ["nemo_bnapus_transfer"]].notna().all(axis=1)
    if "alphagenome" in x:
        masks["alpha_MLM_shared"] = x[mlm + ["alphagenome"]].notna().all(axis=1)
    masks["all_models_shared"] = x[source_columns].notna().all(axis=1)
    records = []
    ranks = []
    for scope, mask in masks.items():
        for method, (column, transform) in rules.items():
            if scope == "matched_MLM" and column and column not in mlm:
                continue
            if scope == "alpha_MLM_shared" and column and column not in mlm + ["alphagenome"]:
                continue
            if scope == "legacy_nemo_shared" and column == "alphagenome":
                continue
            for gene, full in x.groupby("gene", sort=True):
                subset = full[mask[full.index]]
                if column:
                    subset = subset[subset[column].notna()]
                if len(subset) < 6:
                    continue
                baseline = rank_candidates(subset, "annotations")
                base_ranks = {v: i + 1 for i, v in enumerate(baseline.variant_id)}
                ann = (
                    baseline.set_index("variant_id").priority.reindex(subset.variant_id).to_numpy()
                )
                if column:
                    values = abs(subset[column]) if transform == "abs" else subset[column]
                    ranked = subset.assign(
                        priority=0.8 * ann + 0.2 * percentile(values)
                    ).sort_values(["priority", "variant_id"], ascending=[False, True])
                else:
                    ranked = rank_candidates(subset, method)
                new_ranks = {v: i + 1 for i, v in enumerate(ranked.variant_id)}
                valid = full[np.isfinite(full.heldout_p) & (full.heldout_p > 0)].sort_values(
                    ["heldout_p", "variant_id"]
                )
                target = valid.iloc[0]
                eligible = bool(
                    target.heldout_p < 0.05 / len(full) and target.variant_id in new_ranks
                )
                for v in ranked.variant_id:
                    ranks.append(
                        {
                            "scope": scope,
                            "method": method,
                            "gene": gene,
                            "variant_id": v,
                            "rank": new_ranks[v],
                            "baseline_rank": base_ranks[v],
                            "target": v == target.variant_id,
                            "eligible": eligible,
                        }
                    )
                if eligible:
                    records.append(
                        {
                            "scope": scope,
                            "method": method,
                            "gene": gene,
                            "block": "chr" + str(int(full.chrom.iloc[0])),
                            "target": target.variant_id,
                            "target_rank": new_ranks[target.variant_id],
                            "baseline_rank": base_ranks[target.variant_id],
                            "model_hit": int(new_ranks[target.variant_id] <= 5),
                            "baseline_hit": int(base_ranks[target.variant_id] <= 5),
                            "candidates": len(subset),
                            "full_candidates": len(full),
                            "mean_ld_r2": float(full.ld_r2_lead.mean()),
                        }
                    )
    rec = pd.DataFrame(records)
    metrics = []
    for (scope, method), g in rec.groupby(["scope", "method"], sort=True):
        for stratum in ["all", "mean_LD<0.9", "mean_LD>=0.9"]:
            selected = (
                g
                if stratum == "all"
                else g[g.mean_ld_r2 < 0.9]
                if stratum == "mean_LD<0.9"
                else g[g.mean_ld_r2 >= 0.9]
            )
            metrics.append(
                dict(
                    scope=scope,
                    method=method,
                    LD_stratum=stratum,
                    **summarize(selected.to_dict("records")),
                )
            )
    rec.to_csv(out / "per_locus_outcomes.tsv", sep="\t", index=False)
    rank = pd.DataFrame(ranks)
    rank.to_csv(out / "rankings.tsv", sep="\t", index=False)
    pd.DataFrame(metrics).to_csv(out / "paired_metrics.tsv", sep="\t", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Direct paired comparisons use explicit identical-candidate scopes.
    direct, direct_loci = [], []
    pairs = [
        ("matched_MLM", "botanic1_s_512_abs", "plantcad2_s_512_abs"),
        ("matched_MLM", "botanic1_s_4096_abs", "plantcad2_s_4096_abs"),
        ("matched_MLM", "botanic1_s_512_signed_exploratory", "plantcad2_s_512_signed_exploratory"),
        ("alpha_MLM_shared", "alphagenome_abs", "plantcad2_s_512_abs"),
        ("alpha_MLM_shared", "alphagenome_abs", "plantcad2_s_4096_abs"),
    ]
    for scope, challenger, comparator in pairs:
        left = rec[(rec.scope == scope) & (rec.method == challenger)].set_index("gene")
        right = rec[(rec.scope == scope) & (rec.method == comparator)].set_index("gene")
        assert set(left.index) == set(right.index)
        rows = []
        for gene in sorted(left.index):
            assert left.loc[gene, "candidates"] == right.loc[gene, "candidates"]
            row = {
                "gene": gene,
                "block": left.loc[gene, "block"],
                "model_hit": int(left.loc[gene, "model_hit"]),
                "baseline_hit": int(right.loc[gene, "model_hit"]),
            }
            rows.append(row)
            direct_loci.append(
                dict(scope=scope, challenger=challenger, comparator=comparator, **row)
            )
        direct.append(
            dict(scope=scope, challenger=challenger, comparator=comparator, **summarize(rows))
        )
    pd.DataFrame(direct).to_csv(out / "direct_comparisons.tsv", sep="\t", index=False)
    pd.DataFrame(direct_loci).to_csv(out / "direct_per_locus.tsv", sep="\t", index=False)
    from scipy.stats import spearmanr

    concordance = []
    for length in [512, 4096]:
        a, b = f"botanic1_s_{length}", f"plantcad2_s_{length}"
        if a not in x:
            continue
        for label, frame in [("all_SNPs", x)] + list(x.groupby("gene", sort=True)):
            finite = frame[[a, b]].dropna()
            concordance.append(
                {
                    "gene": label,
                    "context_bp": length,
                    "SNPs": len(finite),
                    "absolute_LLR_spearman": float(
                        spearmanr(abs(finite[a]), abs(finite[b])).statistic
                    ),
                }
            )
    pd.DataFrame(concordance).to_csv(out / "score_concordance.tsv", sep="\t", index=False)
    # Confirm original results survive extension; never substitute partial coverage.
    full_metrics = (
        pd.DataFrame(metrics)
        .query("scope=='model_covered' and LD_stratum=='all'")
        .set_index("method")
    )
    assert (
        full_metrics.loc[
            "plantcad2_s_512_signed_exploratory", ["model_hits", "baseline_hits", "loci"]
        ]
        .astype(int)
        .tolist()
    ) == [22, 17, 34]
    assert (
        full_metrics.loc["nemo_bnapus_transfer_abs", ["model_hits", "baseline_hits", "loci"]]
        .astype(int)
        .tolist()
    ) == [8, 8, 14]
    old = pd.read_csv(pilot / "results/analysis/rankings.tsv", sep="\t")
    for model in ["plantcad2_s_512", "plantcad2_s_4096"]:
        new = (
            rank[(rank.scope == "model_covered") & (rank.method == model + "_abs")]
            .set_index(["gene", "variant_id"])["rank"]
            .sort_index()
        )
        original = (
            old[(old.scope == "model_covered") & (old.ranking_model == model)]
            .set_index(["gene", "variant_id"])["rank"]
            .sort_index()
        )
        pd.testing.assert_series_equal(new, original, check_names=False)
    # Retain the worked original locus; also report all predefined Alpha loci.
    alpha_genes = json.loads((run / "inputs/model_protocol.json").read_text())["alpha_subset"]
    worked = rank[(rank.gene.isin([min(x.gene)] + alpha_genes)) & (rank["rank"] <= 5)]
    worked.to_csv(out / "worked_locus_top5.tsv", sep="\t", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, scope, title in zip(
        axes,
        ["model_covered", "alpha_MLM_shared"],
        [
            "Each model with its coverage-matched baseline",
            "Predefined AlphaGenome subset: matched candidates",
        ],
    ):
        items = [
            m for m in metrics if m["scope"] == scope and m["LD_stratum"] == "all" and m["loci"]
        ]
        ax.axvline(0, color="grey", lw=1)
        for i, m in enumerate(items):
            lo, hi = m["ci95"]
            d = m["delta"]
            ax.errorbar(100 * d, i, xerr=[[100 * (d - lo)], [100 * (hi - d)]], fmt="o", capsize=3)
        ax.set_yticks(
            range(len(items)),
            [
                m["method"]
                + f" ({m['model_hits']}/{m['loci']}; +{m['into_top5']}/−{m['out_of_top5']})"
                for m in items
            ],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(
            "Paired top-five change (percentage points)\n95% chromosome-block bootstrap; descriptive"
        )
    fig.suptitle(
        "Held-out association-signal recovery — causal-SNP endpoint unestimable", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "model_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    (out / "integration_manifest.json").write_text(
        json.dumps(
            {
                "inputs": inputs,
                "model_columns": source_columns,
                "paired_records": len(rec),
                "primary_loci": 0,
            },
            indent=2,
        )
    )
    print(pd.DataFrame(metrics).query("LD_stratum=='all'").to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--pilot", required=True)
    a = p.parse_args()
    integrate(a.run, a.pilot)
