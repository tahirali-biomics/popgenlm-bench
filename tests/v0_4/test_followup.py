"""Paired cutoff counts must retain gains and losses, including net-zero changes."""

import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "scripts/v0.4/followup_analysis.py"
spec = importlib.util.spec_from_file_location("followup", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_paired_cutoff_counts():
    rows = [
        {"block": "chr1", "model_hit": a, "baseline_hit": b}
        for a, b in [(1, 0), (0, 1), (1, 1), (0, 0)]
    ]
    x = m.outcomes(rows)
    assert (x["into_top5"], x["out_of_top5"], x["stayed_in"], x["stayed_out"]) == (1, 1, 1, 1)
    assert x["delta"] == 0 and x["ci95"] == [0, 0] and x["model_hits"] == x["baseline_hits"] == 2


def test_absent_truth_not_zero_recovery():
    x = m.outcomes([])
    assert x["loci"] == 0 and x["delta"] is None and x["ci95"] is None
