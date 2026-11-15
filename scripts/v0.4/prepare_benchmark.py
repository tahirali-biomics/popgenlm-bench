"""Deterministic pilot discovery, held-out mapping, LD and pre-score freeze."""

import gzip
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd
import pysam
from popgenlm.regulatory import ALIASES, alpha_subset, ld_r2, lmm_scan

ROOT = Path("/projects/ag-demeaux/tali/PopGenLM")
RUN = ROOT / "runs/v0.4/regulatory_pilot_20261114"
OUT = RUN / "inputs/frozen"
OUT.mkdir(exist_ok=True)
assert json.loads((RUN / "manifests/input_verification.json").read_text())["status"] == "PASS"
assert not (OUT / "freeze.json").exists(), "Frozen benchmark already exists; do not overwrite"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + "\n")


raw = ROOT / "data/raw_public/v0.4"
expression = pd.read_csv(raw / "GSE80744_expression.tsv.gz", sep="\t", index_col=0)
assert expression.index.is_unique
vcf = pysam.VariantFile(
    str(ROOT / "data/incoming/1001g_v3.1_desktop/1001genomes_snp-short-indel_only_ACGTN.vcf.gz")
)
vcfsamples = list(vcf.header.samples)
exprcols = list(expression.columns)
ids = [re.sub("^X", "", x) for x in exprcols]
rows = []
for column, acc in zip(exprcols, ids):
    rows.append(
        {
            "expression_column": column,
            "accession": acc,
            "vcf_sample": acc if acc in vcfsamples else "",
            "status": "duplicate_excluded"
            if ids.count(acc) > 1
            else ("matched" if acc in vcfsamples else "expression_only_unmatched"),
        }
    )
for acc in vcfsamples:
    if acc not in ids:
        rows.append(
            {
                "expression_column": "",
                "accession": acc,
                "vcf_sample": acc,
                "status": "excluded_normalization_Cod0" if acc == "9836" else "vcf_only_unmatched",
            }
        )
cross = pd.DataFrame(rows)
cross.to_csv(OUT / "accession_crosswalk.tsv", sep="\t", index=False)
matched = sorted(cross.loc[cross.status == "matched", "accession"], key=int)
columns = ["X" + x for x in matched]
expression = expression[columns]
vcfindices = np.array([vcfsamples.index(x) for x in matched])
# Stable accession-based hold-out, frozen before any discovery scan.
isdisc = np.array(
    [
        int(hashlib.sha256(("mapping_v04:" + s).encode()).hexdigest()[:8], 16) % 10 < 7
        for s in matched
    ]
)
pd.DataFrame({"accession": matched, "partition": np.where(isdisc, "discovery", "heldout")}).to_csv(
    OUT / "mapping_samples.tsv", sep="\t", index=False
)
print("samples", len(matched), "discovery", isdisc.sum(), "heldout", (~isdisc).sum(), flush=True)
# LOCO relatedness from the hash-verified v0.3 dosage matrix, same accessions.
gdir = ROOT / "runs/v0.3/statistics/ld_genotypes_20260919T215120Z"
assert (
    sha(gdir / "dosages_10000x1135.npz")
    == "e84eb449d8fc883c156ecb2d37706d3bbe8c757dc783641f3393d426e397e1c1"
)
z = np.load(gdir / "dosages_10000x1135.npz")
print("dosage_keys", z.files, flush=True)
dos = z[z.files[0]].astype(float)[:, vcfindices]
dos[dos < 0] = np.nan
idx = pd.read_csv(gdir / "variant_index.tsv", sep="\t", dtype={"chrom": str})
mean = np.nanmean(dos, axis=1)
call = np.isfinite(dos).mean(axis=1)
keep = (mean / 2 >= 0.05) & (mean / 2 <= 0.95) & (call >= 0.8)
# Genomic-spacing thinning, independent of model scores and expression.
last = {}
thin = []
for i in np.where(keep)[0]:
    c = idx.iloc[i]["chrom"]
    p = int(idx.iloc[i]["pos"])
    if p - last.get(c, -100000) >= 20000:
        thin.append(i)
        last[c] = p
markers = dos[thin]
mi = idx.iloc[thin].reset_index(drop=True)
means = np.nanmean(markers, axis=1)
markers = np.where(np.isfinite(markers), markers, means[:, None])
markers = (markers - means[:, None]) / np.std(markers, axis=1)[:, None]
kinships = {}
marker_counts = {}
print("KINSHIP_START", markers.shape, flush=True)
for c in ALIASES:
    print("KINSHIP_CHR", c, flush=True)
    m = markers[mi.chrom.to_numpy() != c]
    marker_counts[c] = len(m)
    for label, mask in [("discovery", isdisc), ("heldout", ~isdisc)]:
        x = m[:, mask]
        kinships[c, label] = x.T @ x / len(x)
save(
    "kinship.json",
    {
        "source": str(gdir / "dosages_10000x1135.npz"),
        "policy": "LOCO; MAF>=0.05; call>=0.8; genomic spacing >=20kb; mean imputation; standardized dosage",
        "markers_per_chrom_excluded": marker_counts,
    },
)
# Parse pinned annotation, preserving exact nuclear aliases and strand.
genes = {}
features = {c: [] for c in ALIASES}
reverse = {v: k for k, v in ALIASES.items()}
with gzip.open(
    ROOT / "data/incoming/GCF_000001735.4_TAIR10.1/GCF_000001735.4_TAIR10.1_genomic.gff.gz", "rt"
) as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.rstrip().split("\t")
        if len(fields) != 9 or fields[0] not in reverse:
            continue
        chrom = reverse[fields[0]]
        kind = fields[2]
        start, end = int(fields[3]), int(fields[4])
        attrs = fields[8]
        if kind in ["gene", "pseudogene"]:
            match = re.search(r"(?:locus_tag|Name)=(AT[1-5]G\d{5})", attrs)
            if match:
                gene = match[1]
                strand = 1 if fields[6] == "+" else -1
                genes[gene] = {
                    "gene": gene,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "tss": start if strand == 1 else end,
                }
        if kind in ["CDS", "exon", "gene", "pseudogene"]:
            features[chrom].append((start, end, kind))
print("genes", len(genes), flush=True)
archive = zipfile.ZipFile(raw / "voichek_code.zip")
seeds = sorted(
    {
        x.split("_", 1)[1]
        for x in archive.read("001_eQTL_analysis/000_run_eqtl_1001g/phenotypes2run_kawakatsu.txt")
        .decode()
        .splitlines()
    }
)
seeds = [s for s in seeds if s in genes and s in expression.index]
# These are published tested phenotypes, not a list of published significant eQTLs.
# Recompute association; do not claim reproduction of their GEMMA batch analysis.
assays = pd.read_csv(RUN / "inputs/assay_nucleotide_inventory.tsv", sep="\t")
positive = assays[assays.independent_nucleotide_support == True]  # noqa: E712 -- preserve audited production execution/order
support = set(zip(positive.gene, positive.variant_id))
ordered = []
for c in ALIASES:
    cur = sorted(
        [s for s in seeds if genes[s]["chrom"] == c],
        key=lambda s: hashlib.sha256(("loci_v04:" + s).encode()).hexdigest(),
    )[:200]
    for s in sorted(set(positive.gene)) + ["AT1G64150", "AT4G16800"]:
        if s in seeds and genes[s]["chrom"] == c:
            cur = [s] + [x for x in cur if x != s]
    ordered += cur
pd.DataFrame([genes[s] for s in ordered]).to_csv(
    OUT / "screen_inventory.tsv", sep="\t", index=False
)
# Pre-score policy: familywise threshold across <=1002 screened genes and local SNPs.
selected = []
screens = []
allc = []
savedg = []
counts = {c: 0 for c in ALIASES}
for s in ordered:
    info = genes[s]
    c = info["chrom"]
    if counts[c] >= 8:
        continue
    # Avoid overlapping windows; all loci on one chromosome remain a single evaluation block.
    if any(t["chrom"] == c and abs(t["tss"] - info["tss"]) < 50000 for t in selected):
        screens.append({"gene": s, "status": "overlapping_window_excluded"})
        continue
    lo = max(1, info["start"] - 10000)
    hi = info["end"] + 10000
    records = []
    gs = []
    for r in vcf.fetch(c, lo - 1, hi):
        if (
            len(r.ref) != 1
            or r.ref not in "ACGT"
            or not r.alts
            or len(r.alts) != 1
            or len(r.alts[0]) != 1
            or r.alts[0] not in "ACGT"
        ):
            continue
        gt = []
        for sample in matched:
            t = r.samples[sample]["GT"]
            gt.append(
                sum(t) if t and len(t) == 2 and None not in t and set(t) <= {0, 1} else np.nan
            )
        gt = np.array(gt, float)
        d = gt[isdisc]
        af = np.nanmean(d) / 2
        if np.isfinite(d).mean() < 0.9 or not 0.05 <= af <= 0.95:
            continue
        records.append(
            {
                "variant_id": f"{c}:{r.pos}:{r.ref}:{r.alts[0]}",
                "chrom": c,
                "pos": r.pos,
                "ref": r.ref,
                "alt": r.alts[0],
                "af_alt": af,
                "call_rate": float(np.isfinite(d).mean()),
            }
        )
        gs.append(gt)
    if len(gs) < 6:
        screens.append({"gene": s, "status": "too_few_common_snps"})
        continue
    g = np.array(gs)
    y = expression.loc[s].to_numpy(float)
    if not np.isfinite(y).all():
        screens.append({"gene": s, "status": "expression_nonfinite"})
        continue
    beta, se, p, delta = lmm_scan(y[isdisc], g[:, isdisc], kinships[c, "discovery"])
    good = np.isfinite(p) & (p > 0)
    if not good.any():
        screens.append({"gene": s, "status": "invalid_association"})
        continue
    lead = int(np.nanargmin(np.where(good, p, np.nan)))
    threshold = 0.05 / (len(ordered) * len(gs))
    if p[lead] > threshold:
        screens.append(
            {
                "gene": s,
                "status": "no_discovery_cis_signal",
                "min_p": float(p[lead]),
                "threshold": threshold,
            }
        )
        continue
    ld = ld_r2(g[:, isdisc], g[lead, isdisc])
    candidate = good & (ld >= 0.6)
    if candidate.sum() < 6:
        screens.append(
            {"gene": s, "status": "fewer_than_six_linked_candidates", "count": int(candidate.sum())}
        )
        continue
    # All candidates are frozen from discovery only. Held-out evidence never selects SNPs.
    b2, se2, p2, delta2 = lmm_scan(y[~isdisc], g[candidate][:, ~isdisc], kinships[c, "heldout"])
    cf = pd.DataFrame(records).loc[candidate].reset_index(drop=True)
    cf["gene"] = s
    cf["block"] = "chr" + c
    cf["tss_distance"] = (cf.pos - info["tss"]) * info["strand"]
    cf["discovery_beta_alt"] = beta[candidate]
    cf["discovery_se"] = se[candidate]
    cf["discovery_p"] = p[candidate]
    cf["discovery_neglogp"] = -np.log10(p[candidate])
    cf["heldout_beta_alt"] = b2
    cf["heldout_se"] = se2
    cf["heldout_p"] = p2
    cf["ld_r2_lead"] = ld[candidate]

    def annotate(pos, c=c):
        present = {kind for a, b, kind in features[c] if a <= pos <= b}
        return (
            "CDS"
            if "CDS" in present
            else "exon_nonCDS"
            if "exon" in present
            else "genic"
            if "gene" in present or "pseudogene" in present
            else "intergenic"
        )

    cf["context"] = [annotate(p) for p in cf.pos]
    cf["independent_nucleotide_support"] = [(s, v) in support for v in cf.variant_id]
    cf["evidence_resolution"] = np.where(
        cf.independent_nucleotide_support, "nucleotide_reporter", "association_signal"
    )
    allc.append(cf)
    savedg.append(g[candidate])
    info = info.copy()
    info.update(
        candidates=len(cf),
        lead_variant=records[lead]["variant_id"],
        discovery_min_p=float(p[lead]),
        discovery_threshold=threshold,
        delta_discovery=delta,
        delta_heldout=delta2,
        block="chr" + c,
        evidence_resolution="association_signal",
        primary_eligible=bool(cf.independent_nucleotide_support.any()),
    )
    selected.append(info)
    counts[c] += 1
    screens.append({"gene": s, "status": "selected", "candidates": len(cf)})
    print("SELECT", s, len(cf), counts, flush=True)
assert selected, "No eligible loci; no model scoring allowed"
candidates = pd.concat(allc, ignore_index=True)
# Stream the exact v0.3 track once; never treat missing coverage as observed zero.
positions = {(str(r.chrom), int(r.pos) - 1): [] for r in candidates.itertuples()}
for i, r in enumerate(candidates.itertuples()):
    positions[str(r.chrom), int(r.pos) - 1].append(i)
phylop = np.full(len(candidates), np.nan)
with gzip.open(ROOT / "data/raw_public/plantregmap_tair10/Ath_PhyloP.bedGraph.gz", "rt") as f:
    for line in f:
        if line.startswith(("#", "track", "browser")):
            continue
        chrom, a, b, score = line.split()
        c = chrom.replace("Chr", "")
        a = int(a)
        b = int(b)
        # The validated source has one-base nuclear intervals.
        if (c, a) in positions:
            assert b == a + 1
            phylop[positions[c, a]] = float(score)
candidates["phylop"] = phylop
candidates["phylop_status"] = np.where(np.isfinite(phylop), "covered", "missing_coverage")
candidates.to_csv(OUT / "candidates.tsv", sep="\t", index=False)
pd.DataFrame(selected).to_csv(OUT / "loci.tsv", sep="\t", index=False)
pd.DataFrame(screens).to_csv(OUT / "screen_exclusions.tsv", sep="\t", index=False)
np.savez_compressed(
    OUT / "candidate_dosages.npz",
    dosages=np.concatenate(savedg),
    samples=np.array(matched),
    variant_id=candidates.variant_id.to_numpy(dtype=str),
    gene=candidates.gene.to_numpy(dtype=str),
)
subset = alpha_subset([s["gene"] for s in selected])
candidates[candidates.gene.isin(subset)].to_csv(
    OUT / "alphagenome_subset.tsv", sep="\t", index=False
)
save(
    "protocol.json",
    {
        "seed": 20261114,
        "target_loci": 40,
        "actual_loci": len(selected),
        "candidate_rule": "biallelic ACGT SNP; discovery MAF>=0.05 and call>=0.9; gene +/-10kb; r2>=0.6 to discovery lead; at least six; no score-based exclusions",
        "association": "ALT dosage null-REML LMM; rank-normalized expression; LOCO kinship; accession-hash 70/30 holdout; GEO processed expression, not exact Voichek batch pipeline",
        "evidence": "Exact isolated single substitutions in Supplementary Table S4 were mapped through S3 oligos to reference and full VCF. Consistent two-promoter absolute activity differences >=0.5 qualify as reporter nucleotide support; region-level MPRA and fine-mapping PIP alone do not qualify. Assays have no per-SNP uncertainty; report surrogate context.",
        "primary_endpoint": "paired change in top-five recovery of independently supported nucleotides; assayed natural SNPs only; report zero denominator as not estimable",
        "secondary_endpoint": "held-out lead recovery among identical frozen candidate sets; descriptive mapping replication, not causal recovery",
        "evaluation_blocks": "whole chromosomes keep nearby/linked loci together; no weights fitted",
        "ranking": "fixed pre-score coefficients: association+LD=.9 association-percentile+.1 lead-r2; annotations=.8 previous+.1 exp(-abs(TSS)/2000)+.05 phyloP-percentile (NA->0 explicit missing flag)+.05 nonCDS; model=.8 annotations+.2 absolute-LLR-percentile",
        "alpha_subset": subset,
        "alpha_score": "paired REF/ALT custom 131072-base sequences; prespecified human leaf-unavailable RNA_SEQ tracks, mean absolute log1p ALT-minus-REF over central 2048bp across all RNA_SEQ tracks; exploratory only; access unavailable",
        "shared_contexts": [512, 4096],
        "orientation": "raw ALT-minus-REF; minor-minus-major sign flip AF_ALT>0.5; ranking uses magnitude",
        "primary_nucleotide_loci": int(
            candidates.groupby("gene").independent_nucleotide_support.any().sum()
        ),
    },
)
manifest = {
    p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
    for p in sorted(OUT.iterdir())
    if p.is_file()
}
save(
    "freeze.json",
    {
        "created_unix": time.time(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "files": manifest,
        "model_scores_inspected": False,
        "primary_nucleotide_loci": int(
            candidates.groupby("gene").independent_nucleotide_support.any().sum()
        ),
    },
)
print("FROZEN", len(selected), len(candidates), flush=True)
