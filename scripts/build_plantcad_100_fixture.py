#!/usr/bin/env python3
"""Build deterministic 100-real-SNV PlantCAD inference fixture."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
table = ROOT / "data/benchmarks/v0.2/1001g_population_10000.tsv"
out = ROOT.parent / "data/fixtures/plantcad_real_snvs_100_20260913T190300Z"
out.mkdir(parents=True, exist_ok=False)
# FASTA derivative is outside repo.
fasta = ROOT.parent / "data/reference/TAIR10.1_genomic.fna"
seqs = {}
name = None
chunks = []
for line in fasta.open():
    line = line.rstrip()
    if line.startswith(">"):
        if name:
            seqs[name] = "".join(chunks).upper()
        name = line[1:].split()[0]
        chunks = []
    else:
        chunks.append(line)
if name:
    seqs[name] = "".join(chunks).upper()
rows = list(csv.DictReader(table.open(), delimiter="\t"))
chosen = []
for chrom in ["1", "2", "3", "4", "5"]:
    group = [r for r in rows if r["chrom"] == chrom]
    required = [
        r
        for r in group
        if (r["chrom"], r["pos"], r["ref"], r["alt"])
        in {
            ("1", "3755", "T", "A"),
            ("2", "73614", "C", "T"),
            ("3", "9391", "G", "A"),
            ("4", "7345", "T", "C"),
            ("5", "672", "C", "T"),
        }
    ]
    rest = [r for r in group if r not in required]
    rest.sort(key=lambda r: (r["selection_hash"], int(r["pos"]), r["ref"], r["alt"]))
    chosen.extend(required + rest[:19])
chosen.sort(key=lambda r: (int(r["chrom"]), int(r["pos"]), r["ref"], r["alt"]))
for r in chosen:
    r["variant_id"] = f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']}"
fields = ["chrom", "pos", "ref", "alt", "fasta_chrom", "selection_hash", "variant_id"]
with (out / "real_snvs_100.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    for r in chosen:
        if seqs[r["fasta_chrom"]][int(r["pos"]) - 1] != r["ref"]:
            raise SystemExit("REF mismatch")
        r = dict(r)
        w.writerow({k: r[k] for k in fields})
with (out / "real_snvs_100.vcf").open("w") as f:
    f.write(
        "##fileformat=VCFv4.3\n##source=PopGenLM-PlantCAD-inference-fixture\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    for r in chosen:
        f.write(
            f"{r['fasta_chrom']}\t{r['pos']}\t{r['variant_id']}\t{r['ref']}\t{r['alt']}\t.\tPASS\t.\n"
        )
meta = {
    "selection_rule": "five required SNVs retained; 19 additional per chromosome sorted by selection_hash, then numeric position, REF, ALT",
    "source": str(table),
    "source_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
    "count": len(chosen),
    "per_chromosome": {c: sum(r["chrom"] == c for r in chosen) for c in "12345"},
}
(out / "fixture_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
print(out)
print(json.dumps(meta, indent=2))
