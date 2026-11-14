"""Explain why independently assayed natural SNPs do not enter frozen candidates."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd
import pysam
from popgenlm.regulatory import ld_r2

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
a = p.parse_args()
run = Path(a.run)
f = run / "inputs/frozen"
x = pd.read_csv(run / "inputs/assay_nucleotide_inventory.tsv", sep="\t")
x = x[x.independent_nucleotide_support].copy()
candidates = pd.read_csv(f / "candidates.tsv", sep="\t")
samples = pd.read_csv(f / "mapping_samples.tsv", sep="\t", dtype={"accession": str})
samples = samples.loc[samples.partition == "discovery", "accession"].tolist()
loci = pd.read_csv(f / "loci.tsv", sep="\t").set_index("gene")
screens = pd.read_csv(f / "screen_exclusions.tsv", sep="\t").set_index("gene")
root = Path("/projects/ag-demeaux/tali/PopGenLM")
v = pysam.VariantFile(
    str(root / "data/incoming/1001g_v3.1_desktop/1001genomes_snp-short-indel_only_ACGTN.vcf.gz")
)


def dosage(variant_id):
    chrom, pos, ref, alt = variant_id.split(":")
    record = next(
        t
        for t in v.fetch(chrom, int(pos) - 1, int(pos))
        if t.pos == int(pos) and t.ref == ref and t.alts == (alt,)
    )
    out = []
    for sample in samples:
        gt = record.samples[sample]["GT"]
        out.append(sum(gt) if gt and len(gt) == 2 and None not in gt else np.nan)
    return np.array(out, float)


rows = []
for r in x.itertuples():
    d = dosage(r.variant_id)
    af = np.nanmean(d) / 2
    call = np.isfinite(d).mean()
    reasons = []
    ld = np.nan
    if r.gene not in loci.index:
        reasons.append(
            "locus:" + str(screens.loc[r.gene, "status"])
            if r.gene in screens.index
            else "locus:not_eligible_in_published_phenotype_inventory"
        )
    if not 0.05 <= af <= 0.95:
        reasons.append("MAF_below_0.05")
    if call < 0.9:
        reasons.append("call_rate_below_0.9")
    if r.gene in loci.index:
        lead = dosage(loci.loc[r.gene, "lead_variant"])
        ld = ld_r2(d[None, :], lead)[0]
        if not np.isfinite(ld) or ld < 0.6:
            reasons.append("LD_r2_below_0.6")
    included = bool(((candidates.gene == r.gene) & (candidates.variant_id == r.variant_id)).any())
    rows.append(
        {
            "gene": r.gene,
            "variant_id": r.variant_id,
            "af_alt": af,
            "call_rate": call,
            "ld_r2_lead": ld,
            "included": included,
            "exclusion": ";".join(reasons),
            "effect_35S": r.effect_35S,
            "effect_pTRP1": r.effect_pTRP1,
        }
    )
pd.DataFrame(rows).to_csv(run / "results/assay_exclusions.tsv", sep="\t", index=False)
print(pd.DataFrame(rows).to_string(index=False))
