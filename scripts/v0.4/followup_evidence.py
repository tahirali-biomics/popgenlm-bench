"""Score-blind eligibility funnel, raw reporter inventory and genotype audit."""

import argparse
import gzip
import json
import re
import sys
import xml.etree.ElementTree as E
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd
import pysam

from popgenlm.regulatory import ALIASES, reverse_complement

p = argparse.ArgumentParser()
p.add_argument("--pilot", required=True)
p.add_argument("--run", required=True)
a = p.parse_args()
pilot = Path(a.pilot)
run = Path(a.run)
root = pilot.parents[2]
protocol = json.loads((run / "inputs/followup_protocol.json").read_text())
assert "at least 10" in protocol["revised_evidence_eligibility"]
z = zipfile.ZipFile(root / "data/raw_public/v0.4/voichek_supplement.xlsx")
ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ss = [
    "".join(t.itertext()) for t in E.fromstring(z.read("xl/sharedStrings.xml")).findall("x:si", ns)
]


def sheet(n):
    rows = []
    for row in E.fromstring(z.read(f"xl/worksheets/sheet{n}.xml")).findall("x:sheetData/x:row", ns):
        cells = {}
        for c in row:
            v = c.find("x:v", ns)
            if v is not None:
                cells[re.sub("[0-9]", "", c.get("r"))] = (
                    ss[int(v.text)] if c.get("t") == "s" else v.text
                )
        if cells:
            rows.append(cells)
    return rows


op1 = {r["A"]: r for r in sheet(4)[1:] if all(k in r for k in ["A", "B", "C"])}
seq = {}
name = None
for line in (root / "data/reference/TAIR10.1_genomic.fna").open():
    if line.startswith(">"):
        name = line[1:].split()[0]
        seq[name] = []
    else:
        seq[name].append(line.strip().upper())
seq = {k: "".join(v) for k, v in seq.items()}
genes = {}
with gzip.open(
    root / "data/incoming/GCF_000001735.4_TAIR10.1/GCF_000001735.4_TAIR10.1_genomic.gff.gz", "rt"
) as f:
    for line in f:
        if "\tgene\t" not in line:
            continue
        fields = line.rstrip().split("\t")
        m = re.search("locus_tag=(AT[1-5]G[0-9]{5})", fields[8])
        if m:
            genes[m[1]] = (fields[0], int(fields[3]), int(fields[4]))
rows = []
for r in sheet(5)[1:]:
    row = {
        "modification": r.get("A"),
        "parent": r.get("D"),
        "reason": "eligible_single_substitution",
    }
    if not all(k in r for k in ["A", "C", "D", "E", "F", "G", "H"]):
        row["reason"] = "missing_assay_fields"
    elif r["D"] not in op1:
        row["reason"] = "missing_parent_oligo"
    else:
        original = op1[r["D"]]["B"].upper()
        modified = r["C"].upper()
        gene = op1[r["D"]]["C"]
        row["gene"] = gene
        if len(original) != len(modified):
            row["reason"] = "length_change"
        else:
            diff = [i for i, (x, y) in enumerate(zip(original, modified)) if x != y]
            if len(diff) != 1:
                row["reason"] = "not_exactly_one_substitution"
            elif gene not in genes:
                row["reason"] = "gene_not_in_annotation"
            else:
                chrom, lo, hi = genes[gene]
                lo = max(0, lo - 10000)
                hi = min(len(seq[chrom]), hi + 10000)
                window = seq[chrom][lo:hi]
                hits = []
                for strand, s in [("+", original), ("-", reverse_complement(original))]:
                    starts = [m.start() for m in re.finditer("(?=" + s + ")", window)]
                    hits.extend((strand, lo + j) for j in starts)
                if not hits:
                    row["reason"] = "no_exact_reference_oligo_match"
                elif len(hits) != 1:
                    row["reason"] = "ambiguous_reference_oligo_match"
                else:
                    strand, start = hits[0]
                    i = diff[0]
                    pos = start + i + 1 if strand == "+" else start + len(original) - i
                    ref = original[i] if strand == "+" else reverse_complement(original[i])
                    alt = modified[i] if strand == "+" else reverse_complement(modified[i])
                    c = next(k for k, v in ALIASES.items() if v == chrom)
                    assert seq[chrom][pos - 1] == ref
                    row["variant_id"] = f"{c}:{pos}:{ref}:{alt}"
    rows.append(row)
pd.DataFrame(rows).to_csv(run / "results/reporter_mapping_funnel_rows.tsv", sep="\t", index=False)
counts = pd.Series([r["reason"] for r in rows]).value_counts().to_dict()
prior = pd.read_csv(pilot / "inputs/assay_nucleotide_inventory.tsv", sep="\t")
# A stricter uniqueness audit must agree with the original mapped oligos before reuse.
new_ids = {
    (r["modification"], r["variant_id"])
    for r in rows
    if r["reason"] == "eligible_single_substitution"
}
old_ids = set(zip(prior.modification.astype(str), prior.variant_id))
assert new_ids == old_ids, (len(new_ids), len(old_ids))
positive = prior[prior.independent_nucleotide_support].copy()
old_ex = pd.read_csv(pilot / "results/assay_exclusions.tsv", sep="\t")
loci = pd.read_csv(pilot / "inputs/frozen/loci.tsv", sep="\t")
samples = pd.read_csv(
    pilot / "inputs/frozen/mapping_samples.tsv", sep="\t", dtype={"accession": str}
)
discovery = samples.loc[samples.partition == "discovery", "accession"].tolist()
v = pysam.VariantFile(
    str(root / "data/incoming/1001g_v3.1_desktop/1001genomes_snp-short-indel_only_ACGTN.vcf.gz")
)
vcf_audit = []
for r in prior.itertuples():
    chrom, position, ref, alt = r.variant_id.split(":")
    position = int(position)
    at_position = [t for t in v.fetch(chrom, position - 1, position) if t.pos == position]
    exact = [t for t in at_position if t.ref == ref and t.alts == (alt,)]
    reason = (
        "exact_biallelic_match"
        if exact
        else "no_VCF_record_at_position"
        if not at_position
        else "reference_allele_mismatch"
        if not any(t.ref == ref for t in at_position)
        else "ALT_or_biallelic_definition_mismatch"
    )
    assert bool(exact) == bool(r.natural_vcf_match)
    vcf_audit.append(
        {
            "gene": r.gene,
            "variant_id": r.variant_id,
            "modification": r.modification,
            "status": reason,
        }
    )
pd.DataFrame(vcf_audit).to_csv(run / "results/vcf_allele_audit.tsv", sep="\t", index=False)
records = []
for r in positive.itertuples():
    chrom, pos, ref, alt = r.variant_id.split(":")
    pos = int(pos)
    record = next(
        t
        for t in v.fetch(chrom, pos - 1, pos)
        if t.pos == pos and t.ref == ref and t.alts == (alt,)
    )
    gt = []
    for s in discovery:
        g = record.samples[s]["GT"]
        gt.append(sum(g) if g and len(g) == 2 and None not in g else np.nan)
    d = np.array(gt)
    call = np.isfinite(d).mean()
    refcar = int((d == 0).sum())
    altcar = int((d > 0).sum())
    overlap = any(
        str(int(l.chrom)) == chrom and pos >= l.start - 60000 and pos <= l.end + 60000
        for l in loci.itertuples()
    )
    ex = old_ex[old_ex.variant_id == r.variant_id].iloc[0]
    reasons = []
    if call < 0.9:
        reasons.append("call_rate_below_0.9")
    if min(refcar, altcar) < 10:
        reasons.append("fewer_than_10_carriers_per_allele")
    if overlap:
        reasons.append("overlap_original_locus_50kb_blocks")
    if "locus:" in ex.exclusion:
        reasons.append("no_eligible_discovery_cis_locus")
    if "LD_" in ex.exclusion:
        reasons.append("not_linked_to_discovery_lead")
    records.append(
        {
            "gene": r.gene,
            "variant_id": r.variant_id,
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "build": "TAIR10.1 GCF_000001735.4",
            "source": "https://doi.org/10.1038/s41588-024-01907-3",
            "assay": "isolated 160bp oligo single substitution; two minimal promoters",
            "tissue": "Arabidopsis leaf protoplasts; three replicates, downstream oligo insertion; surrogate for endogenous rosette expression",
            "support_level": "individual_nucleotide_reporter",
            "effect_35S": r.effect_35S,
            "effect_pTRP1": r.effect_pTRP1,
            "call_rate": call,
            "discovery_ref_carriers": refcar,
            "discovery_alt_carriers": altcar,
            "overlap_original": overlap,
            "eligible": not reasons,
            "exclusions": ";".join(reasons),
            "training_overlap": "nemo paper benchmarked Voichek regulatory effects; pretraining overlap in genomic MLMs expected; sample overlap unresolved",
        }
    )
inventory = pd.DataFrame(records)
inventory.to_csv(run / "results/independent_nucleotide_cases.tsv", sep="\t", index=False)
# Ordered old-policy attrition, with overlapping reason counts separately.
steps = [("positive_natural_reporter_substitutions", len(old_ex))]
keep = np.ones(len(old_ex), bool)
for name, mask in [
    ("MAF>=0.05", old_ex.af_alt.between(0.05, 0.95)),
    ("call>=0.9", old_ex.call_rate >= 0.9),
    ("eligible_discovery_locus", ~old_ex.exclusion.str.contains("locus:")),
    ("LD>=0.6", old_ex.ld_r2_lead >= 0.6),
]:
    keep &= np.asarray(mask)
    steps.append((name, int(keep.sum())))
funnel = {
    "modified_rows": len(rows),
    "mapping_reasons": counts,
    "mapped_rows": len(prior),
    "vcf_exact_matches": int(prior.natural_vcf_match.sum()),
    "mapped_not_vcf": int((~prior.natural_vcf_match).sum()),
    "vcf_mismatch_categories": pd.Series([r["status"] for r in vcf_audit]).value_counts().to_dict(),
    "positive_natural_rows": len(positive),
    "positive_genes": positive.gene.nunique(),
    "old_ordered_funnel": steps,
    "old_overlapping_exclusions": {
        k: int(old_ex.exclusion.str.contains(k).sum())
        for k in ["MAF_", "call_rate_", "locus:", "LD_"]
    },
    "new_policy_eligible": int(inventory.eligible.sum()),
    "new_policy_nonoverlap_cases": int((~inventory.overlap_original).sum()),
    "new_policy_no_alt_carriers": int((inventory.discovery_alt_carriers == 0).sum()),
    "primary_endpoint": "not_estimable"
    if not inventory.eligible.any()
    else "requires_candidate_freeze_before_scoring",
}
(run / "results/evidence_funnel.json").write_text(json.dumps(funnel, indent=2))
print(json.dumps(funnel, indent=2))
