"""Map isolated reporter substitutions to exact natural SNP alleles; no model scores."""

import gzip
import json
import re
import sys
import xml.etree.ElementTree as E
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pandas as pd
import pysam
from popgenlm.regulatory import ALIASES, reverse_complement

root = Path("/projects/ag-demeaux/tali/PopGenLM")
run = root / "runs/v0.4/regulatory_pilot_20261114"
z = zipfile.ZipFile(root / "data/raw_public/v0.4/voichek_supplement.xlsx")
ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ss = [
    "".join(t.itertext()) for t in E.fromstring(z.read("xl/sharedStrings.xml")).findall("x:si", ns)
]


def sheet(n):
    rows = []
    for r in E.fromstring(z.read("xl/worksheets/sheet" + str(n) + ".xml")).findall(
        "x:sheetData/x:row", ns
    ):
        cells = {}
        for c in r:
            v = c.find("x:v", ns)
            if v is not None:
                cells[re.sub("[0-9]", "", c.get("r"))] = (
                    ss[int(v.text)] if c.get("t") == "s" else v.text
                )
        if cells:
            rows.append(cells)
    return rows


op1 = {r["A"]: r for r in sheet(4)[1:] if "A" in r and "B" in r and "C" in r}
mutations = sheet(5)[1:]
seq = {}
name = None
with open(root / "data/reference/TAIR10.1_genomic.fna") as f:
    for line in f:
        if line.startswith(">"):
            name = line[1:].split()[0]
            seq[name] = []
        else:
            seq[name].append(line.strip().upper())
seq = {k: "".join(s) for k, s in seq.items()}
genes = {}
with gzip.open(
    root / "data/incoming/GCF_000001735.4_TAIR10.1/GCF_000001735.4_TAIR10.1_genomic.gff.gz", "rt"
) as f:
    for l in f:
        if "\tgene\t" not in l:
            continue
        a = l.rstrip().split("\t")
        m = re.search("locus_tag=(AT[1-5]G[0-9]{5})", a[8])
        if m:
            genes[m[1]] = (a[0], int(a[3]), int(a[4]))
v = pysam.VariantFile(
    str(root / "data/incoming/1001g_v3.1_desktop/1001genomes_snp-short-indel_only_ACGTN.vcf.gz")
)
records = []
stats = {
    "modified_rows": len(mutations),
    "single_substitutions": 0,
    "mapped_single_substitutions": 0,
    "natural_vcf_matches": 0,
    "positive_consistent_assays": 0,
}
for r in mutations:
    if not all(k in r for k in ["A", "C", "D", "E", "F", "G", "H"]):
        continue
    parent = op1.get(r["D"])
    if parent is None:
        continue
    original = parent["B"].upper()
    modified = r["C"].upper()
    if len(original) != len(modified):
        continue
    differences = [i for i, (a, b) in enumerate(zip(original, modified)) if a != b]
    if len(differences) != 1:
        continue
    i = differences[0]
    stats["single_substitutions"] += 1
    gene = parent["C"]
    info = genes.get(gene)
    if info is None:
        continue
    chrom, lo, hi = info
    lo = max(0, lo - 10000)
    hi = min(len(seq[chrom]), hi + 10000)
    window = seq[chrom][lo:hi]
    hits = []
    for strand, s in [("+", original), ("-", reverse_complement(original))]:
        start = window.find(s)
        if start >= 0 and window.find(s, start + 1) < 0:
            hits.append((strand, lo + start))
    if len(hits) != 1:
        continue
    strand, start = hits[0]
    pos = start + i + 1 if strand == "+" else start + len(original) - i
    ref = original[i] if strand == "+" else reverse_complement(original[i])
    alt = modified[i] if strand == "+" else reverse_complement(modified[i])
    c = next(k for k, a in ALIASES.items() if a == chrom)
    assert seq[chrom][pos - 1] == ref
    stats["mapped_single_substitutions"] += 1
    match = any(
        t.pos == pos and t.ref == ref and t.alts == (alt,) for t in v.fetch(c, pos - 1, pos)
    )
    d35 = float(r["G"]) - float(r["E"])
    dtr = float(r["H"]) - float(r["F"])
    positive = match and abs(d35) >= 0.5 and abs(dtr) >= 0.5 and d35 * dtr > 0
    stats["natural_vcf_matches"] += int(match)
    stats["positive_consistent_assays"] += int(positive)
    records.append(
        {
            "gene": gene,
            "variant_id": f"{c}:{pos}:{ref}:{alt}",
            "chrom": c,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "op1_index": r["D"],
            "modification": r["A"],
            "effect_35S": d35,
            "effect_pTRP1": dtr,
            "natural_vcf_match": match,
            "independent_nucleotide_support": positive,
            "evidence_resolution": "nucleotide_reporter"
            if positive
            else "assayed_substitution_not_qualified",
            "assay_context": "isolated substitution in 160bp oligo; heterologous minimal promoters; two-promoter consistent absolute difference>=0.5; no per-SNP uncertainty available",
        }
    )
pd.DataFrame(records).to_csv(run / "inputs/assay_nucleotide_inventory.tsv", sep="\t", index=False)
(run / "manifests/assay_audit.json").write_text(json.dumps(stats, indent=2) + "\n")
print(stats, flush=True)
