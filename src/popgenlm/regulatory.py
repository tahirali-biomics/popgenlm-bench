"""CPU contracts for the v0.4 regulatory prioritization pilot.

All ranks use explicit deterministic ties. Evidence resolution is never inferred
from association, LD, conservation, motif overlap, or model scores.
"""

import hashlib

import numpy as np

ALIASES = dict(
    zip("12345", ["NC_003070.9", "NC_003071.7", "NC_003074.8", "NC_003075.7", "NC_003076.8"])
)
COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


def context(sequence, pos, ref, length):
    """1-based POS; even context puts REF at length//2-1, no boundary padding."""
    i = length // 2 - 1
    start = pos - 1 - i
    if start < 0 or start + length > len(sequence):
        raise ValueError("context exceeds chromosome")
    result = sequence[start : start + length].upper()
    if len(ref) != 1 or result[i] != ref:
        raise ValueError("REF/context mismatch")
    return result, i


def ld_r2(dosages, lead):
    """Pairwise-complete dosage LD; NaN for nonvariable pairs."""
    out = []
    for x in dosages:
        valid = np.isfinite(x) & np.isfinite(lead)
        a, b = x[valid], lead[valid]
        out.append(
            np.corrcoef(a, b)[0, 1] ** 2 if len(a) > 2 and a.std() > 0 and b.std() > 0 else np.nan
        )
    return np.array(out)


def residualize(y, covariates):
    return y - covariates @ np.linalg.lstsq(covariates, y, rcond=None)[0]


def lmm_scan(y, genotypes, kinship):
    """Null-REML GLS scan, ALT dosage effects, intercept, no per-SNP refit.

    LOCO kinship and separate sample subsets must be supplied by caller.
    Missing genotypes are mean-imputed within this mapping subset.
    """
    from scipy import optimize, stats

    y = stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y))
    g = genotypes.copy()
    means = np.nanmean(g, axis=1)
    bad = np.where(~np.isfinite(g))
    g[bad] = means[bad[0]]
    values, vectors = np.linalg.eigh(kinship)
    values = np.maximum(values, 0)
    uy = vectors.T @ y
    uc = vectors.T @ np.ones(len(y))

    def null(log_delta):
        d = values + np.exp(log_delta)
        w = 1 / d
        r = uy - uc * (np.sum(w * uc * uy) / np.sum(w * uc * uc))
        rss = np.sum(w * r * r)
        return (
            np.log(d).sum()
            + np.log(np.sum(w * uc * uc))
            + (len(y) - 1) * np.log(rss / (len(y) - 1))
        )

    fit = optimize.minimize_scalar(null, bounds=(-8, 8), method="bounded")
    scale = 1 / np.sqrt(values + np.exp(fit.x))
    wy = uy * scale
    wc = (uc * scale)[:, None]
    wg = (vectors.T @ g.T) * scale[:, None]
    yr = residualize(wy, wc)
    gr = residualize(wg, wc)
    xx = np.sum(gr * gr, axis=0)
    xy = gr.T @ yr
    beta = np.divide(xy, xx, out=np.full(len(xx), np.nan), where=xx > 1e-12)
    rss = np.maximum(np.sum(yr * yr) - beta * xy, 0)
    se = np.sqrt(rss / (len(y) - 2) / xx)
    p = 2 * stats.t.sf(np.abs(beta / se), df=len(y) - 2)
    return beta, se, p, float(np.exp(fit.x))


def percentile(values):
    from scipy import stats

    a = np.asarray(values, float)
    if not np.isfinite(a).all():
        raise ValueError("rank requires finite scores")
    return (stats.rankdata(a, method="average") - 0.5) / len(a)


def rank_candidates(frame, model=None):
    """Fixed coefficients chosen before scores; no label-based fitting."""
    x = frame.copy()
    assoc = percentile(x["discovery_neglogp"])
    ld = np.nan_to_num(x["ld_r2_lead"].to_numpy(float), nan=0.0)
    baseline = 0.9 * assoc + 0.1 * ld
    ann = (
        0.8 * baseline
        + 0.1 * np.exp(-np.abs(x["tss_distance"]) / 2000)
        + 0.05 * percentile(x["phylop"].fillna(0))
        + 0.05 * (x["context"] != "CDS").astype(float)
    )
    score = baseline if model == "association_ld" else ann
    if model not in [None, "association_ld", "annotations"]:
        score = 0.8 * ann + 0.2 * percentile(np.abs(x[model]))
    x["priority"] = score
    return x.sort_values(["priority", "variant_id"], ascending=[False, True])


def paired_recovery(records, replicates=2000, seed=20261114):
    """Paired locus hit change; resample evaluation blocks intact."""
    if len(records) == 0:
        return {"loci": 0, "blocks": 0, "delta": None, "ci95": None, "status": "not_estimable"}
    blocks = sorted({r["block"] for r in records})
    rng = np.random.default_rng(seed)
    differences = np.array([r["model_hit"] - r["baseline_hit"] for r in records], float)
    estimates = []
    for _ in range(replicates):
        chosen = rng.choice(blocks, len(blocks), replace=True)
        sample = [differences[i] for b in chosen for i, r in enumerate(records) if r["block"] == b]
        estimates.append(np.mean(sample))
    return {
        "loci": len(records),
        "blocks": len(blocks),
        "baseline": float(np.mean([r["baseline_hit"] for r in records])),
        "model": float(np.mean([r["model_hit"] for r in records])),
        "delta": float(differences.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "status": "descriptive_block_bootstrap",
    }


def alpha_subset(loci, number=5):
    return sorted(loci, key=lambda x: hashlib.sha256(("alpha_v04:" + x).encode()).hexdigest())[
        :number
    ]
