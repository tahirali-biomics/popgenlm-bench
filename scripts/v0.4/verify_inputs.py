"""Reverify historical inputs without changing them; run under Slurm."""

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path("/projects/ag-demeaux/tali/PopGenLM")
OUT = ROOT / "runs/v0.4/regulatory_pilot_20261114/manifests/input_verification.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


history = json.loads(
    (
        ROOT
        / "provenance/inputs/scoring_preparation_20260913T153854528991Z/input_verification.json"
    ).read_text()
)
records = []
for e in history["records"]:
    if e.get("hash_basis") or not e.get("observed_sha256"):
        continue
    p = Path(e["path"])
    observed = sha(p)
    records.append(
        {
            "path": str(p),
            "expected": e["observed_sha256"],
            "observed": observed,
            "passed": observed == e["observed_sha256"],
        }
    )
extra = [
    (
        "data/reference/TAIR10.1_genomic.fna",
        "de99316d1d5597e9a5d40d943ff919c17d59b77116354a5d388b183b6f21075f",
    ),
    (
        "data/raw_public/plantregmap_tair10/Ath_PhyloP.bedGraph.gz",
        "8944714faaf8942cecdfe4a7b496a4d0948531dff8a8f721c390d53a775cbab7",
    ),
    (
        "runs/v0.3/annotation/tair10_context_20260919T213842Z/popgenlm_v03_master_10000_annotated.tsv",
        "428f6948180f6aa95366d931a49b1a86ccfb7ea98663875271af8bbb2f9e8b47",
    ),
]
for rel, expected in extra:
    p = ROOT / rel
    observed = sha(p)
    records.append(
        {"path": str(p), "expected": expected, "observed": observed, "passed": expected == observed}
    )
fixture = "data/fixtures/1001g_v3.1_population_gpn_fixture.tsv"
tagbytes = subprocess.check_output(["git", "-C", str(ROOT / "repo"), "show", "v0.3.0:" + fixture])
actual = sha(ROOT / "repo" / fixture)
resolution = {
    "path": str(ROOT / "repo" / fixture),
    "release_sha256": hashlib.sha256(tagbytes).hexdigest(),
    "current_sha256": actual,
    "historical_g3b_sha256": "bd7d909b50c148ad0d95b305c6f56cfcb3289f095bd86f3b83bda123d51f8d9b",
    "used": "exact v0.3.0 fixture; historical G3B JSON contains a mistyped digest",
    "release_matches_current": hashlib.sha256(tagbytes).hexdigest() == actual,
}
import pysam

v = pysam.VariantFile(
    str(ROOT / "data/incoming/1001g_v3.1_desktop/1001genomes_snp-short-indel_only_ACGTN.vcf.gz")
)
samples = list(v.header.samples)
assert (
    samples
    == (ROOT / "data/incoming/1001g_v3.1_desktop/1001g_v3.1_samples.txt").read_text().splitlines()
)
aliases = {
    "1": "NC_003070.9",
    "2": "NC_003071.7",
    "3": "NC_003074.8",
    "4": "NC_003075.7",
    "5": "NC_003076.8",
}
seq = {}
name = None
with open(ROOT / "data/reference/TAIR10.1_genomic.fna") as f:
    for line in f:
        if line.startswith(">"):
            name = line[1:].split()[0]
            seq[name] = []
        else:
            seq[name].append(line.strip().upper())
seq = {k: "".join(x) for k, x in seq.items()}
checks = {}
for chrom, acc in aliases.items():
    r = next(v.fetch(chrom, 0, 100000))
    assert seq[acc][r.pos - 1 : r.pos - 1 + len(r.ref)] == r.ref
    checks[chrom] = {"alias": acc, "length": len(seq[acc]), "indexed_first_pos": r.pos}
report = {
    "status": "PASS"
    if all(r["passed"] for r in records) and resolution["release_matches_current"]
    else "FAIL",
    "records": records,
    "fixture_resolution": resolution,
    "chromosome_alias_checks": checks,
    "samples": len(samples),
    "python": platform.python_version(),
    "job_id": os.environ.get("SLURM_JOB_ID"),
}
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
assert report["status"] == "PASS"
