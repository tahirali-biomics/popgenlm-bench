"""Independent structural and legacy-overlap checks after all model jobs finish."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
a = p.parse_args()
run = Path(a.run)
root = Path("/projects/ag-demeaux/tali/PopGenLM")
x = pd.read_csv(run / "inputs/frozen/candidates.tsv", sep="\t")
report = {}
for model, lengths in [
    ("plantcad2_s", [512, 4096]),
    ("plantcad_v1", [512]),
    ("gpn_brassicales", [512]),
]:
    for length in lengths:
        tables = []
        for chrom in range(1, 6):
            dest = run / "results" / model / f"chr{chrom}_{length}.tsv"
            done = json.loads(dest.with_suffix(".complete.json").read_text())
            assert hashlib.sha256(dest.read_bytes()).hexdigest() == done["sha256"]
            t = pd.read_csv(dest, sep="\t")
            assert len(t) == done["rows"]
            assert t.variant_id.is_unique
            tables.append(t)
        s = pd.concat(tables, ignore_index=True)
        assert set(s.variant_id) == set(x.variant_id)
        assert s.variant_id.is_unique
        merged = s.merge(x[["variant_id", "af_alt"]], on="variant_id", validate="one_to_one")
        assert np.allclose(merged.raw_alt_minus_ref, (merged.llr_forward + merged.llr_rc) / 2)
        assert np.allclose(
            merged.minor_minus_major,
            np.where(merged.af_alt > 0.5, -merged.raw_alt_minus_ref, merged.raw_alt_minus_ref),
        )
        assert np.isfinite(merged.raw_alt_minus_ref).all()
        report[f"{model}_{length}"] = {
            "rows": len(s),
            "complete_chromosomes": 5,
            "finite_scores": True,
            "orientation_checks": "PASS",
        }
        if model == "plantcad_v1":
            old = pd.read_csv(
                root
                / "runs/v0.3/plantcad/matched_baseline_continuation_20260915T192031613054773Z/scores/plantcad_10000.tsv",
                sep="\t",
            )
            overlap = s.merge(old, on="variant_id")
            diff = (overlap.raw_alt_minus_ref - overlap.plantcad_score_ref_alt).abs()
            report[f"{model}_{length}"]["legacy_overlap"] = len(overlap)
            report[f"{model}_{length}"]["legacy_max_abs_difference"] = (
                float(diff.max()) if len(overlap) else None
            )
            if len(overlap):
                assert np.allclose(
                    overlap.raw_alt_minus_ref, overlap.plantcad_score_ref_alt, atol=1e-4, rtol=1e-4
                )
        if model == "gpn_brassicales":
            old = pd.read_csv(
                root / "repo/data/benchmarks/v0.2/gpn/1001g_population_10000_gpn_scores.tsv",
                sep="\t",
            )
            aliases = {
                "NC_003070.9": "1",
                "NC_003071.7": "2",
                "NC_003074.8": "3",
                "NC_003075.7": "4",
                "NC_003076.8": "5",
            }
            old["variant_id"] = (
                old.chrom.map(aliases) + ":" + old.pos.astype(str) + ":" + old.ref + ":" + old.alt
            )
            overlap = s.merge(old, on="variant_id")
            diff = (overlap.raw_alt_minus_ref - overlap.gpn_score_ref_alt).abs()
            report[f"{model}_{length}"]["legacy_overlap"] = len(overlap)
            report[f"{model}_{length}"]["legacy_max_abs_difference"] = (
                float(diff.max()) if len(overlap) else None
            )
            # Historical GPN environment/dtype provenance differs; report comparisons without claiming bitwise reproduction.
nemo = []
for dest in (run / "results/nemo_bnapus_transfer").glob("AT*.tsv"):
    done = json.loads(dest.with_suffix(".complete.json").read_text())
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == done["sha256"]
    nemo.append(pd.read_csv(dest, sep="\t"))
n = pd.concat(nemo, ignore_index=True)
assert len(n) == len(x)
assert not n.duplicated(["gene", "variant_id"]).any()
covered = n[n.status == "covered"]
assert np.isfinite(covered.delta_log10_tpm).all()
assert np.allclose(
    covered.delta_log10_tpm, (covered.alt_scaled - covered.ref_scaled) * 1.0498368135592189
)
report["nemo"] = {
    "total": len(n),
    "coverage": n.status.value_counts().to_dict(),
    "finite_covered_scores": True,
}

master_path = (
    root
    / "runs/v0.3/annotation/tair10_context_20260919T213842Z/popgenlm_v03_master_10000_annotated.tsv"
)
assert (
    hashlib.sha256(master_path.read_bytes()).hexdigest()
    == "428f6948180f6aa95366d931a49b1a86ccfb7ea98663875271af8bbb2f9e8b47"
)
master = pd.read_csv(master_path, sep="\t")
master["variant_id"] = (
    master.chrom.astype(str) + ":" + master.pos.astype(str) + ":" + master.ref + ":" + master.alt
)
assert len(master) == 10000 and master.variant_id.is_unique
assert master.phylop.notna().sum() == 9453
for model in ["gpn", "plantcad"]:
    assert np.allclose(
        master[model + "_score_minor_vs_major"],
        np.where(
            master.af_alt > 0.5, -master[model + "_score_ref_alt"], master[model + "_score_ref_alt"]
        ),
    )
overlap = x.merge(master, on="variant_id", suffixes=("_pilot", "_v03"))
if len(overlap):
    assert np.allclose(overlap.phylop_pilot, overlap.phylop_v03, equal_nan=True)
report["v03_master_regression"] = {
    "rows": 10000,
    "unique_keys": 10000,
    "phylop_covered": 9453,
    "orientation_equations": "PASS",
    "pilot_key_overlap": len(overlap),
    "shared_phylop": "PASS" if len(overlap) else "no_shared_keys",
}

report["status"] = "PASS"
(run / "manifests/output_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
