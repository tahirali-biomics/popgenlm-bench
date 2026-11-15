"""Portable cached-evidence review. No inference, network, credentials or GPU.

Inputs must pass manifest verification before parsing. Derived outputs are written
only under the caller's output directory; the frozen pilot is never modified.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "plantcad2_s_512_abs": "PlantCAD2-S · 512 bp",
    "plantcad2_s_4096_abs": "PlantCAD2-S · 4 kb",
    "botanic1_s_512_abs": "Botanic1-S · 512 bp",
    "botanic1_s_4096_abs": "Botanic1-S · 4 kb",
    "gpn_brassicales_512_abs": "GPN Brassicales · 512 bp",
    "plantcad_v1_512_abs": "Original PlantCAD · l32 · 512 bp",
    "nemo_bnapus_transfer_abs": "nemo · B. napus transfer",
    "alphagenome_abs": "AlphaGenome · mammal → plant",
}
COLORS = ["#176b87", "#308fa3", "#b56828", "#d89c56", "#647749", "#76608a"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    for name, entry in m["files"].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"Unsafe manifest path: {name}")
        if sha(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    return m


def load(root):
    root = Path(root)
    verify(root)
    return {p.stem: pd.read_csv(p, sep="\t") for p in sorted((root / "data").glob("*.tsv"))}


def save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (name + ".svg"), bbox_inches="tight")
    fig.savefig(out / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def audit(d):
    """Reconstruct eligible outcomes from candidate ranks, not report prose."""
    ranks, outcomes = d["rankings"], d["per_locus_outcomes"]
    from frozen_regulatory import percentile, rank_candidates

    x = d["candidates"].merge(d["frozen_scores"], on=["gene", "variant_id"], validate="one_to_one")
    for (scope, method, gene), g in ranks.groupby(["scope", "method", "gene"]):
        candidate = x[(x.gene == gene) & x.variant_id.isin(g.variant_id)].copy()
        assert len(candidate) == len(g)
        baseline = rank_candidates(candidate, "annotations")
        base = dict(zip(baseline.variant_id, range(1, len(baseline) + 1)))
        assert all(base[row.variant_id] == row.baseline_rank for row in g.itertuples())
        if method in ["annotations", "association_ld"]:
            ranked = rank_candidates(candidate, method)
        else:
            exploratory = method.endswith("_signed_exploratory")
            column = (
                method.removesuffix("_signed_exploratory")
                if exploratory
                else method.removesuffix("_abs")
            )
            if column not in candidate:
                assert column.startswith("botanic1_s_"), f"Unexpected missing raw input: {column}"
                continue  # Botanic publication ranks verified below, raw reconstruction explicitly unavailable.
            values = candidate[column] if exploratory else abs(candidate[column])
            annotation = (
                baseline.set_index("variant_id").priority.reindex(candidate.variant_id).to_numpy()
            )
            ranked = candidate.assign(
                priority=0.8 * annotation + 0.2 * percentile(values)
            ).sort_values(["priority", "variant_id"], ascending=[False, True])
        assert list(ranked.variant_id) == list(g.sort_values("rank").variant_id), (
            scope,
            method,
            gene,
        )
        full = x[x.gene == gene]
        target = full.sort_values(["heldout_p", "variant_id"]).iloc[0]
        expected_eligible = target.heldout_p < 0.05 / len(full) and target.variant_id in set(
            g.variant_id
        )
        assert (g.eligible == expected_eligible).all()
        assert set(g.loc[g.target, "variant_id"]) == (
            {target.variant_id} if target.variant_id in set(g.variant_id) else set()
        )
    reconstructed = []
    for (scope, method, gene), g in ranks.groupby(["scope", "method", "gene"]):
        assert sorted(g["rank"]) == list(range(1, len(g) + 1))
        assert g.variant_id.is_unique
        if not g.eligible.all():
            continue
        target = g[g.target]
        assert len(target) == 1
        row = target.iloc[0]
        reconstructed.append(
            {
                "scope": scope,
                "method": method,
                "gene": gene,
                "target_rank": int(row["rank"]),
                "baseline_rank": int(row.baseline_rank),
            }
        )
    reconstructed = pd.DataFrame(reconstructed)
    z = reconstructed.merge(
        outcomes, on=["scope", "method", "gene"], suffixes=("_new", ""), validate="one_to_one"
    )
    assert len(z) == len(outcomes) == len(reconstructed)
    assert (z.target_rank_new == z.target_rank).all()
    assert (z.baseline_rank_new == z.baseline_rank).all()
    assert ((z.target_rank <= 5).astype(int) == z.model_hit).all()
    assert ((z.baseline_rank <= 5).astype(int) == z.baseline_hit).all()
    # Matching means identical SNP identities, not merely identical denominators.
    for scope in ["matched_MLM", "alpha_MLM_shared", "legacy_nemo_shared", "all_models_shared"]:
        for gene, g in ranks[ranks.scope == scope].groupby("gene"):
            universes = g.groupby("method").variant_id.apply(lambda s: frozenset(s))
            assert len(set(universes)) == 1, (scope, gene)
    counts = []
    for (scope, method), g in outcomes.groupby(["scope", "method"]):
        counts.append(
            {
                "scope": scope,
                "method": method,
                "loci": len(g),
                "model_hits": int(g.model_hit.sum()),
                "baseline_hits": int(g.baseline_hit.sum()),
                "gains": int(((g.model_hit == 1) & (g.baseline_hit == 0)).sum()),
                "losses": int(((g.model_hit == 0) & (g.baseline_hit == 1)).sum()),
            }
        )
    counts = pd.DataFrame(counts)
    expected = d["paired_metrics"].query('LD_stratum == "all"')
    z = counts.merge(expected, on=["scope", "method"], suffixes=("_new", ""), validate="one_to_one")
    for c in ["loci", "model_hits", "baseline_hits"]:
        assert (z[c + "_new"] == z[c]).all()
    assert (z.gains == z.into_top5).all() and (z.losses == z.out_of_top5).all()
    return counts


def overview(d, out):
    o = d["per_locus_outcomes"]
    main = o[(o.scope == "matched_MLM") & (o.method == "plantcad2_s_512_abs")].sort_values(
        ["baseline_rank", "gene"]
    )
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axs[0, 0]
    for i, r in enumerate(main.itertuples()):
        color = (
            "#257957"
            if r.model_hit > r.baseline_hit
            else "#ba4141"
            if r.model_hit < r.baseline_hit
            else "#9aa5af"
        )
        ax.plot([i, i], [r.baseline_rank, r.target_rank], color=color, lw=2)
    ax.scatter(
        range(len(main)),
        main.baseline_rank,
        color="#444444",
        s=23,
        label="Annotation baseline",
        zorder=3,
    )
    ax.scatter(
        range(len(main)),
        main.target_rank,
        color=COLORS[0],
        s=23,
        label="PlantCAD2-S added",
        zorder=3,
    )
    ax.axhline(5.5, color="#ba4141", ls="--", label="Top-five cutoff")
    ax.set(
        xticks=range(len(main)),
        xticklabels=main.gene,
        ylabel="Held-out target rank (lower is better)",
        title="A  PlantCAD2-S: 18/34 vs 17/34 · 2 gains, 1 loss",
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    methods = list(NAMES)[:6]
    for i, method in enumerate(methods):
        g = o[(o.scope == "matched_MLM") & (o.method == method)]
        gains = ((g.model_hit == 1) & (g.baseline_hit == 0)).sum()
        losses = ((g.model_hit == 0) & (g.baseline_hit == 1)).sum()
        ax.barh(i, gains, color="#257957")
        ax.barh(i, -losses, color="#ba4141")
        ax.text(
            gains + 0.12,
            i,
            f"{int(g.model_hit.sum())}/{len(g)}; +{gains}/−{losses}",
            va="center",
            fontsize=9,
        )
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set(
        yticks=range(len(methods)),
        yticklabels=[NAMES[m] for m in methods],
        xlabel="Signals leaving ← | → entering top five",
        title="B  Identical candidates and 34 loci",
    )
    ax.set_xlim(-3, 6)
    ax.invert_yaxis()
    ax = axs[1, 0]
    k = np.arange(1, 21)
    ax.plot(
        k,
        [(main.baseline_rank <= n).mean() for n in k],
        "--",
        color="#444444",
        label="Annotation baseline",
    )
    for color, method in zip(COLORS, methods):
        g = o[(o.scope == "matched_MLM") & (o.method == method)]
        ax.plot(k, [(g.target_rank <= n).mean() for n in k], color=color, label=NAMES[method])
    ax.axvline(5, ls=":", color="#ba4141")
    ax.set(
        xlabel="Shortlist size k",
        ylabel="Held-out signal recovery (n=34)",
        ylim=(0, 1),
        title="C  Recovery across shortlist sizes",
    )
    ax.legend(fontsize=7)
    ax = axs[1, 1]
    g = o[(o.scope == "alpha_MLM_shared") & (o.method == "alphagenome_abs")].sort_values("gene")
    for i, r in enumerate(g.itertuples()):
        ax.plot([i, i], [r.baseline_rank, r.target_rank], color="#9aa5af", lw=3)
    ax.scatter(range(len(g)), g.baseline_rank, color="#444444", label="Matched baseline")
    ax.scatter(range(len(g)), g.target_rank, color="#a05b91", label="AlphaGenome added")
    ax.axhline(5.5, color="#ba4141", ls="--")
    ax.set(
        xticks=range(len(g)),
        xticklabels=g.gene,
        ylabel="Held-out target rank",
        title="D  AlphaGenome: 2/4 vs 1/4 · separate subset",
    )
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.suptitle(
        "PopGenLM v0.4 · held-out association-signal recovery\nPopGenLM adaptation from frozen ranks; independent causal-SNP endpoint unestimable",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "benchmark")


def shortlist_changes(d, gene, method="plantcad2_s_512_abs"):
    r = d["rankings"]
    r = r[(r.scope == "matched_MLM") & (r.gene == gene) & (r.method == method)].copy()
    r["shortlist_change"] = np.select(
        [
            (r["rank"] <= 5) & (r.baseline_rank > 5),
            (r["rank"] > 5) & (r.baseline_rank <= 5),
            (r["rank"] <= 5) & (r.baseline_rank <= 5),
        ],
        ["entered", "left", "retained"],
        default="outside",
    )
    return r.sort_values("rank")


def locus(d, out, gene):
    c = d["candidates"].query("gene == @gene").sort_values("pos")
    features = d["gene_features"].query("gene == @gene")
    l = d["loci"].query("gene == @gene").iloc[0]
    changes = shortlist_changes(d, gene)
    changes.to_csv(Path(out) / (gene + "_shortlist_changes.tsv"), sep="\t", index=False)
    s = c.merge(d["frozen_scores"], on=["gene", "variant_id"], validate="one_to_one")
    fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True)
    for ax in axes:
        ax.grid(axis="x", alpha=0.15)
    for f in features.itertuples():
        if f.kind in ["gene", "exon", "CDS"]:
            axes[0].plot(
                [f.start, f.end],
                [0, 0],
                color="#2f4858",
                lw={"gene": 1, "exon": 5, "CDS": 9}[f.kind],
                solid_capstyle="butt",
            )
    axes[0].annotate("5′ → 3′" if l.strand == 1 else "3′ ← 5′", (l.tss, 0.2))
    axes[0].set(ylim=(-0.5, 0.7), yticks=[], ylabel=gene)
    axes[1].scatter(c.pos, c.discovery_neglogp, s=15, label="Discovery (469)")
    axes[1].scatter(c.pos, -np.log10(c.heldout_p), s=15, label="Held out (196)")
    axes[1].set_ylabel("−log10 P")
    axes[1].legend(fontsize=8)
    axes[2].scatter(c.pos, c.ld_r2_lead, s=17, color="#647749")
    axes[2].set(ylabel="Lead LD r²", ylim=(-0.05, 1.05))
    axes[3].scatter(c.pos, c.phylop, s=17, color="#76608a")
    axes[3].axhline(0, color="grey", lw=0.6)
    axes[3].set_ylabel("PhyloP\nsite score")
    for col, label, color in [
        ("plantcad2_s_512", "PlantCAD2-S", COLORS[0]),
        ("botanic1_s_512", "Botanic1-S", COLORS[2]),
        ("gpn_brassicales_512", "GPN", COLORS[4]),
        ("plantcad_v1_512", "Original PlantCAD", COLORS[5]),
    ]:
        if col in s:
            axes[4].scatter(s.pos, s[col], s=16, label=label, color=color)
    axes[4].set_ylabel("Signed LLR\nALT − REF (nats)")
    axes[4].legend(fontsize=7, ncol=2)
    axes[5].scatter(s.pos, s.nemo_bnapus_transfer, s=16, color="#ad618a")
    axes[5].set_ylabel("nemo transfer\nΔ log10(TPM+.1)")
    v = c.merge(
        changes[["variant_id", "rank", "baseline_rank", "shortlist_change", "target"]],
        on="variant_id",
    )
    axes[6].scatter(
        v.pos,
        v.baseline_rank,
        marker="o",
        facecolors="none",
        edgecolors="#444444",
        label="Baseline",
    )
    for key, color in [
        ("entered", "#257957"),
        ("left", "#ba4141"),
        ("retained", COLORS[0]),
        ("outside", "#bbbbbb"),
    ]:
        z = v[v.shortlist_change == key]
        axes[6].scatter(z.pos, z["rank"], s=24, color=color, label=key)
        for row in z.itertuples():
            if key in ["entered", "left"]:
                axes[6].annotate(
                    row.variant_id,
                    (row.pos, row.rank),
                    xytext=(5, 3),
                    textcoords="offset points",
                    fontsize=7,
                    rotation=25,
                )
    target = v[v.target].iloc[0]
    for ax in axes:
        ax.axvline(target.pos, color="#b08224", lw=1, ls="--")
    axes[6].axhline(5.5, color="#ba4141", ls=":")
    axes[6].invert_yaxis()
    axes[6].set(
        ylabel="Target/candidate rank",
        xlabel=f"TAIR10.1 chromosome {int(l.chrom)} · 1-based genomic position",
    )
    axes[6].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[6].legend(fontsize=7, ncol=5)
    fig.suptitle(
        f"{gene} · PlantCAD2-S shortlist · target {target.variant_id}: {int(target.baseline_rank)} → {int(target['rank'])}\nPopGenLM adaptation · association lead is a proxy, not causal truth",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "locus_" + gene)


def model_figure(d, out, model):
    if model == "botanic" and "botanic1_s_512" not in d["frozen_scores"]:
        import shutil

        for suffix in ["svg", "png"]:
            shutil.copy2(
                Path(d["_root"]) / "source_figures" / ("botanic." + suffix),
                Path(out) / ("botanic." + suffix),
            )
        return  # Frozen scholarly figure; raw Botanic outputs are not in the public dataset.
    if model == "alphagenome":
        a = d["alpha_tracks"]
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(a.pos, a.ref, label="REF", color=COLORS[0])
        axes[0].plot(a.pos, a.alt, label="ALT", color=COLORS[2], alpha=0.8)
        axes[0].set_ylabel("Predicted RNA-seq\nmodel units")
        axes[0].legend()
        axes[1].plot(a.pos, a.alt - a.ref, color="#a05b91")
        axes[1].set_ylabel("ALT − REF")
        axes[2].plot(a.pos, np.log1p(a.alt) - np.log1p(a.ref), color="#a05b91")
        axes[2].set_ylabel("Δ log1p prediction")
        meta = d["alpha_display"].iloc[0]
        gene = meta.gene  # noqa: F841 -- referenced by pandas query
        for f in d["gene_features"].query("gene == @gene").itertuples():
            if f.kind in ["gene", "exon", "CDS"]:
                axes[3].plot(
                    [f.start, f.end],
                    [0, 0],
                    color="#2f4858",
                    lw={"gene": 1, "exon": 5, "CDS": 9}[f.kind],
                )
        for ax in axes:
            ax.axvline(meta.pos, color="#b08224", ls="--")
            ax.set_xlim(a.pos.min(), a.pos.max())
        axes[3].set(yticks=[], ylabel=meta.gene, xlabel="TAIR10.1 · 1-based genomic position")
        axes[3].ticklabel_format(axis="x", style="plain", useOffset=False)
        fig.suptitle(
            f"AlphaGenome · mammalian-to-plant transfer · {meta.variant_id}\nPopGenLM adaptation of Avsec et al. (2026), Fig. 4b / 6b\nAPI v1 ALL_FOLDS · human RNA_SEQ · track 0: {meta.track_name}",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        save(fig, out, model)
        return
    if model == "nemo":
        n = d["nemo_positions"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, region in zip(axes[:2], ["promoter", "terminator"]):
            z = n[n.region == region]
            ax.scatter(z.offset_bp, z.delta_log10_tpm, s=12, alpha=0.65)
            ax.axhline(0, color="grey", lw=0.7)
            ax.axvline(0, color="grey", ls=":")
            ax.set(
                xlabel=f"Position relative to {'TSS' if region == 'promoter' else 'TTS'} (bp; transcript strand)",
                ylabel="Predicted ALT − REF log10(TPM+.1)",
                title=region,
            )
        axes[2].bar(["Matched baseline", "nemo transfer"], [8, 8], color=["#888888", "#a05b91"])
        axes[2].set(
            ylim=(0, 14),
            ylabel="Held-out signals recovered / 14",
            title="B. napus → A. thaliana; 8/14 vs 8/14",
        )
        fig.suptitle(
            "PopGenLM adaptation · position-dependent expression effects\nRockenbach et al. (2026), Fig. 3a motivates the contrast; no compatible native allele-aligned validation",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        save(fig, out, model)
        return
    c = d["candidates"].merge(d["frozen_scores"], on=["gene", "variant_id"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if model == "phylop":
        axes[0].hist(c.phylop.dropna(), bins=40, color=COLORS[5])
        axes[0].set(xlabel="Signed site-level PhyloP", ylabel="Candidate sites")
        axes[1].scatter(c.phylop, c.plantcad2_s_512, s=5, alpha=0.4)
        axes[1].set(xlabel="PhyloP", ylabel="PlantCAD2-S signed LLR")
        old = d["legacy_v03"]
        axes[2].scatter(old.phylop, old.gpn_minor, s=3, alpha=0.2)
        axes[2].set(xlabel="v0.3 PhyloP (9,453 covered)", ylabel="GPN minor − major LLR")
        title = "PhyloP · site-level conservation reference; never allele-oriented"
    else:
        column = {
            "gpn": "gpn_brassicales_512",
            "plantcad": "plantcad_v1_512",
            "plantcad2": "plantcad2_s_512",
            "botanic": "botanic1_s_512",
        }[model]
        axes[0].hist(c[column], bins=40, color=COLORS[0])
        axes[0].set(
            xlabel="Masked log P(ALT) − log P(REF), nats", ylabel="Candidates (1,205 covered)"
        )
        if model in ["plantcad2", "botanic"]:
            long = column.replace("512", "4096")
            axes[1].scatter(abs(c[column]), abs(c[long]), s=6, alpha=0.4)
            axes[1].set(xlabel="|LLR| · 512 bp", ylabel="|LLR| · 4 kb")
        else:
            old = d["legacy_v03"]
            axes[1].scatter(old.gpn_minor, old.plantcad_minor, s=3, alpha=0.2)
            axes[1].set(
                xlabel="v0.3 GPN minor − major", ylabel="v0.3 original PlantCAD minor − major"
            )
        method = column + "_abs"  # noqa: F841 -- referenced by pandas query
        o = d["per_locus_outcomes"].query('scope == "matched_MLM" and method == @method')
        axes[2].scatter(o.baseline_rank, o.target_rank, s=18)
        axes[2].axhline(5.5, ls="--", color="#ba4141")
        axes[2].axvline(5.5, ls="--", color="#ba4141")
        axes[2].set(
            xlabel="Baseline target rank",
            ylabel="Model-added target rank",
            title=f"{int(o.model_hit.sum())}/34 vs 17/34",
        )
        title = {
            "gpn": "GPN Brassicales · sequence likelihood",
            "plantcad": "Original PlantCAD · exact checkpoint l32 (legacy alias PlantCAD-S)",
            "plantcad2": "PlantCAD2-S · context and rank effects",
            "botanic": "Botanic1-S · 512 bp versus 4 kb zero-shot; ATAC reproduction separate/blocked",
        }[model]
    fig.suptitle("PopGenLM adaptation · " + title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, out, model)


def run(root, out, focus="overview"):
    d = load(root)
    d["_root"] = str(Path(root))
    counts = audit(d)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / "verified_counts.tsv", sep="\t", index=False)
    if focus == "overview":
        overview(d, out)
        for gene in ["AT1G01690", "AT4G02405"]:
            locus(d, out, gene)
    else:
        model_figure(d, out, focus)
    if focus == "alphagenome":
        observed_expression(d, out)
    return counts


def observed_expression(d, out):
    z = d["observed_expression"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, partition in zip(axes, ["discovery", "heldout"]):
        g = z[z.partition == partition]
        groups = [g.loc[g.genotype == n, "processed_expression"].to_numpy() for n in [0, 1, 2]]
        nonempty = [(n, a) for n, a in zip([0, 1, 2], groups) if len(a)]
        ax.boxplot(
            [a for _, a in nonempty],
            positions=[n for n, _ in nonempty],
            widths=0.5,
            showfliers=False,
        )
        ax.set(
            xticks=[0, 1, 2],
            xticklabels=[f"{n}\n(n={len(a)})" for n, a in zip([0, 1, 2], groups)],
            xlabel="ALT dosage (nearest integer; within .1)",
            title=partition,
        )
    axes[0].set_ylabel("GSE80744 processed gene expression\nsource units (not coverage or TPM)")
    fig.suptitle(
        f"Observed Arabidopsis expression · {z.gene.iloc[0]} · {z.variant_id.iloc[0]}\nSeparate observational panel; unadjusted genotype groups; population/LD confounding remains",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, out, "observed_arabidopsis")


def load_botanic(root, external_table):
    """Load separately lawfully obtained raw scores only if their frozen digest matches."""
    root, external_table = Path(root), Path(external_table)
    expected = json.loads((root / "restricted_inputs.json").read_text())["botanic_scores.tsv"]
    if sha(external_table) != expected["sha256"]:
        raise ValueError("External Botanic scores differ from the frozen table")
    d = load(root)
    extra = pd.read_csv(external_table, sep="\t")
    assert list(extra.columns) == expected["columns"]
    d["frozen_scores"] = d["frozen_scores"].merge(
        extra, on=["gene", "variant_id"], validate="one_to_one"
    )
    return d
