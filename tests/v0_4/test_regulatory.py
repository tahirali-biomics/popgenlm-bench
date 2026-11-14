import numpy as np
import pandas as pd
import pytest
from popgenlm.regulatory import (
    alpha_subset,
    context,
    ld_r2,
    lmm_scan,
    paired_recovery,
    rank_candidates,
    reverse_complement,
)


def test_context_coordinate_and_rc():
    seq = "ACGT" * 200
    s, i = context(seq, 301, "A", 512)
    assert s[i] == "A" and i == 255
    assert reverse_complement(s)[511 - i] == "T"
    with pytest.raises(ValueError):
        context(seq, 301, "C", 512)
    with pytest.raises(ValueError):
        context(seq, 1, "A", 512)


def test_ld_missing_and_monomorphic():
    x = np.array([[0, 0, 2, 2, np.nan], [2, 2, 0, 0, 0], [0, 0, 0, 0, 0]], float)
    r = ld_r2(x, x[0])
    assert np.allclose(r[:2], 1) and np.isnan(r[2])


def test_lmm_alt_sign_and_null():
    rng = np.random.default_rng(4)
    g = rng.binomial(2, 0.4, (6, 100)).astype(float)
    beta, se, p, _delta = lmm_scan(g[0] * 2 + rng.normal(0, 0.2, 100), g, np.eye(100))
    assert beta[0] > 0 and p[0] < 1e-10 and np.isfinite(se).all()
    b2, _, p2, _ = lmm_scan(g[0] * 2 + rng.normal(0, 0.01, 100), 2 - g, np.eye(100))
    assert b2[0] < 0 and p2[0] < 1e-10


def test_zero_endpoint_and_paired_blocks():
    assert paired_recovery([])["delta"] is None
    x = paired_recovery([{"block": "1", "model_hit": 1, "baseline_hit": 0}] * 3, 100)
    assert x["model_recovery"] == 1
    assert x["delta"] == 1 and x["ci95"] == [1, 1] and x["blocks"] == 1


def test_rank_ties_and_coverage_contract():
    x = pd.DataFrame(
        {
            "variant_id": ["b", "a"],
            "discovery_neglogp": [2, 2],
            "ld_r2_lead": [1, 1],
            "tss_distance": [2, 2],
            "phylop": [0, 0],
            "context": ["exon", "exon"],
            "model": [1, 1],
        }
    )
    assert rank_candidates(x, "model").variant_id.tolist() == ["a", "b"]
    assert alpha_subset(["a", "b", "c"], 2) == alpha_subset(["c", "b", "a"], 2)
