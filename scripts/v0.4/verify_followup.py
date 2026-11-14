"""Independent audit of saved completion hashes, fixture contracts and paired ranks."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd

from popgenlm.regulatory import ALIASES, context, reverse_complement

p = argparse.ArgumentParser()
p.add_argument("--pilot", required=True)
p.add_argument("--run", required=True)
a = p.parse_args()
pilot = Path(a.pilot)
run = Path(a.run)
root = pilot.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


checked = {}
for path in sorted((pilot / "results").glob("*/*.complete.json")):
    target = path.with_name(path.name.replace(".complete.json", ".tsv"))
    meta = json.loads(path.read_text())
    assert sha(target) == meta["sha256"], str(target)
    checked[str(target)] = sha(target)
smoke = json.loads((pilot / "results/plantcad2_s/smoke_manifest.json").read_text())
for r in smoke["runs"]:
    path = pilot / "results/plantcad2_s" / f"smoke_{r['context_bp']}.tsv"
    assert sha(path) == r["sha256"]
    checked[str(path)] = sha(path)
x = pd.read_csv(pilot / "inputs/frozen/candidates.tsv", sep="\t")
seq = {}
name = None
for line in (root / "data/reference/TAIR10.1_genomic.fna").open():
    if line.startswith(">"):
        name = line[1:].split()[0]
        seq[name] = []
    else:
        seq[name].append(line.strip().upper())
seq = {k: "".join(v) for k, v in seq.items()}
for row in x.itertuples():
    for length in [512, 4096]:
        s, i = context(seq[ALIASES[str(row.chrom)]], row.pos, row.ref, length)
        assert s[i] == row.ref and reverse_complement(s)[length - 1 - i] == reverse_complement(
            row.ref
        )
        assert reverse_complement(reverse_complement(s)) == s
ranks = pd.read_csv(run / "results/rankings.tsv", sep="\t")
records = pd.read_csv(run / "results/paired_locus_outcomes.tsv", sep="\t")
metrics = pd.read_csv(run / "results/paired_metrics.tsv", sep="\t")
for r in records.itertuples():
    z = ranks[(ranks.scope == r.scope) & (ranks.rule == r.rule) & (ranks.gene == r.gene)]
    t = z[z.variant_id == r.target].iloc[0]
    assert t["rank"] == r.target_rank and t.baseline_rank == r.baseline_rank
    assert int(t["rank"] <= 5) == r.model_hit and int(t.baseline_rank <= 5) == r.baseline_hit
for r in metrics[metrics.stratum == "all"].itertuples():
    z = records[(records.scope == r.scope) & (records.rule == r.rule)]
    assert (
        len(z) == r.loci
        and int(z.model_hit.sum()) == r.model_hits
        and int(z.baseline_hit.sum()) == r.baseline_hits
    )
    assert r.into_top5 - r.out_of_top5 == r.model_hits - r.baseline_hits
    assert r.stayed_in + r.stayed_out + r.into_top5 + r.out_of_top5 == r.loci
assert len(x) == 1205 and np.isfinite(x.discovery_neglogp).all()
(run / "manifests/followup_verification.json").write_text(
    json.dumps(
        {
            "status": "PASS",
            "completion_files": len(checked),
            "candidate_context_ref_rc_checks": 2410,
            "paired_records_checked": len(records),
            "score_sha256": checked,
        },
        indent=2,
    )
)
print("PASS hashes/REF/contexts/strand/paired ranks:", len(checked), len(records))
