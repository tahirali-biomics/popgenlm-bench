"""Independent output checks; no GPU, API calls, model loading or credentials."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd

from popgenlm.regulatory import percentile, rank_candidates
from popgenlm.regulatory_transfer import transfer_magnitude


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(run):
    run = Path(run)
    checked = {}
    out = run / "results/analysis"
    for p in (run / "results").glob("*/*.complete.json"):
        target = p.with_name(p.name.replace(".complete.json", ".tsv"))
        assert sha(target) == json.loads(p.read_text())["sha256"]
        checked[str(target)] = sha(target)
    # Scores must reconstruct from paired cached predictions, without API access.
    count = 0
    cache = run / "results/alphagenome/cache"
    for p in sorted((run / "results/alphagenome").glob("AT*.tsv")):
        for r in pd.read_csv(p, sep="\t").itertuples():
            pair = []
            for key in [r.ref_cache, r.alt_cache]:
                path = cache / (key + ".npz")
                m = json.loads(path.with_suffix(".json").read_text())
                assert sha(path) == m["sha256"]
                with np.load(path) as data:
                    pair.append(data["central"])
            assert np.isclose(r.score, transfer_magnitude(*pair), atol=1e-14, rtol=1e-12)
            count += 1
    if count:
        assert count == 194
    for p in sorted((run / "results/botanic1_s").glob("chr*.tsv")):
        x = pd.read_csv(p, sep="\t")
        assert np.isfinite(x.raw_alt_minus_ref).all()
        assert np.allclose(x.raw_alt_minus_ref, (x.llr_forward + x.llr_rc) / 2)
        assert np.allclose(x.magnitude, abs(x.raw_alt_minus_ref))
    scores = pd.read_csv(out / "frozen_scores.tsv", sep="\t").set_index(["gene", "variant_id"])
    candidates = pd.read_csv(run / "inputs/frozen/candidates.tsv", sep="\t")
    ranks = pd.read_csv(out / "rankings.tsv", sep="\t")
    rec = pd.read_csv(out / "per_locus_outcomes.tsv", sep="\t")
    for (scope, method, gene), group in ranks.groupby(["scope", "method", "gene"]):
        assert sorted(group["rank"]) == list(range(1, len(group) + 1))
        g = candidates[
            (candidates.gene == gene) & (candidates.variant_id.isin(group.variant_id))
        ].copy()
        base = rank_candidates(g, "annotations")
        expected_base = dict(zip(base.variant_id, range(1, len(base) + 1)))
        assert dict(zip(group.variant_id, group.baseline_rank)) == expected_base
        if method not in ["association_ld", "annotations"]:
            column = method.removesuffix("_abs").removesuffix("_signed_exploratory")
            values = scores.loc[[(gene, v) for v in g.variant_id], column].to_numpy()
            if method.endswith("_abs"):
                values = abs(values)
            ann = base.set_index("variant_id").priority.reindex(g.variant_id).to_numpy()
            ranked = g.assign(priority=0.8 * ann + 0.2 * percentile(values)).sort_values(
                ["priority", "variant_id"], ascending=[False, True]
            )
        else:
            ranked = rank_candidates(g, method)
        assert dict(zip(group.variant_id, group["rank"])) == dict(
            zip(ranked.variant_id, range(1, len(g) + 1))
        )
    for r in rec.itertuples():
        t = ranks[
            (ranks.scope == r.scope)
            & (ranks.method == r.method)
            & (ranks.gene == r.gene)
            & (ranks.variant_id == r.target)
        ].iloc[0]
        assert int(t["rank"] <= 5) == r.model_hit and int(t.baseline_rank <= 5) == r.baseline_hit
    metrics = pd.read_csv(out / "paired_metrics.tsv", sep="\t")
    for r in metrics[metrics.LD_stratum == "all"].itertuples():
        g = rec[(rec.scope == r.scope) & (rec.method == r.method)]
        assert (
            len(g) == r.loci
            and g.model_hit.sum() == r.model_hits
            and g.baseline_hit.sum() == r.baseline_hits
        )
        assert r.model_hits - r.baseline_hits == r.into_top5 - r.out_of_top5
        assert r.stayed_in + r.stayed_out + r.into_top5 + r.out_of_top5 == r.loci
    result = {
        "status": "PASS",
        "completion_hashes": checked,
        "alpha_pairs_reconstructed": count,
        "paired_outcomes": len(rec),
        "all_ranks_independently_reconstructed": True,
    }
    (run / "manifests/output_verification.json").write_text(json.dumps(result, indent=2))
    print("PASS independent output verification", count, len(rec))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    verify(a.run)
