"""Gene-level expression contracts and CPU baselines, separate from variant scores."""

import hashlib
import json

import numpy as np
import pandas as pd


def validate_gene_splits(table):
    required = {"gene", "family", "split", "chrom"}
    if not required.issubset(table.columns) or not table.gene.is_unique:
        raise ValueError("duplicate_gene_or_missing_split_columns")
    if not set(table.split) <= {"train", "validation", "test"}:
        raise ValueError("invalid_split")
    if (table.groupby("family").split.nunique() > 1).any():
        raise ValueError("family_split_leakage")
    if (table.groupby("chrom").split.nunique() > 1).any():
        raise ValueError("chromosome_split_leakage")


def fit_scaler(y, training_identity, checkpoint_identity):
    y = np.asarray(y, float)
    if not len(y) or not np.isfinite(y).all() or y.std() <= 0:
        raise ValueError("invalid_training_labels")
    return {
        "mean": float(y.mean()),
        "scale": float(y.std()),
        "training_identity": training_identity,
        "checkpoint_identity": checkpoint_identity,
        "units": "log10(TPM+0.1)",
    }


def inverse_scale(z, scaler, checkpoint_identity):
    if scaler["checkpoint_identity"] != checkpoint_identity or scaler["scale"] <= 0:
        raise ValueError("checkpoint_scaler_mismatch")
    return np.asarray(z, float) * scaler["scale"] + scaler["mean"]


def metrics(actual, predicted):
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if len(y) < 2 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("invalid_paired_expression_values")
    # std() can be nonzero for a constant floating-point array because its
    # computed mean rounds. Exact range detects a constant predictor reliably.
    r = float(np.corrcoef(y, p)[0, 1]) if np.ptp(y) > 0 and np.ptp(p) > 0 else None
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(y),
        "pearson_r": r,
        "pearson_r_squared": None if r is None else r * r,
        "correlation_status": ("defined" if r is not None else
                            "undefined: constant prediction" if np.ptp(p) == 0 else
                            "undefined: constant observation"),
        "predictive_r_squared": float(1 - np.sum((y - p) ** 2) / sst) if sst else None,
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(abs(y - p))),
    }


def condition_filter(frame):
    """Published algorithm, explicit zero-fill assumption. New export != historical export."""
    # R read.table infers an entirely NA column as logical; select_if(is.numeric)
    # in the published code drops it before replacing numeric NAs with zero.
    missing = frame.isna().all()
    x = frame.loc[:, ~missing].astype(float).fillna(0)
    if not x.index.is_unique or not x.columns.is_unique:
        raise ValueError("duplicate_gene_or_condition")
    top = x.loc[x.sum(axis=1).sort_values(ascending=False, kind="stable").index[:5000]]
    corr = top.corr(method="spearman").mean(axis=1)
    if not np.isfinite(corr).all():
        raise ValueError("undefined_condition_correlation")
    result = pd.DataFrame(
        {
            "condition": x.columns,
            "mean_spearman": corr.to_numpy(),
            "included": (corr > 0.1).to_numpy(),
            "reason": np.where(corr > 0.1, "mean_spearman_gt_0.1", "mean_spearman_le_0.1"),
        }
    )
    omitted = pd.DataFrame(
        {
            "condition": frame.columns[missing],
            "mean_spearman": np.nan,
            "included": False,
            "reason": "all_missing_R_logical_column_dropped",
        }
    )
    return (
        pd.concat([result, omitted], ignore_index=True)
        .set_index("condition")
        .loc[frame.columns]
        .rename_axis("condition")
        .reset_index()
    )


def sequence_features(promoter, terminator, k=3):
    """Fixed DNA kmer representation; no expression fit or model-score substitution."""
    import itertools

    vocabulary = ["".join(x) for x in itertools.product("ACGT", repeat=k)]
    result = []
    for s in [promoter, terminator]:
        counts = dict.fromkeys(vocabulary, 0)
        for i in range(len(s) - k + 1):
            word = s[i : i + k]
            if word in counts:
                counts[word] += 1
        denominator = max(1, sum(counts.values()))
        result.extend(counts[w] / denominator for w in vocabulary)
        result.extend([s.count("N") / len(s), (s.count("G") + s.count("C")) / len(s)])
    return np.asarray(result, np.float32)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fit_ridge(features, labels, alpha):
    """Training-only scaling and L2 ridge, with an unpenalized intercept."""
    x, y = np.asarray(features, float), np.asarray(labels, float)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0:
        raise ValueError("invalid_ridge_training_data")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = (x-mean)/scale
    intercept = float(y.mean())
    centered = y - intercept
    # Use the dual identity when the feature width exceeds the number of
    # training genes. This avoids an unnecessary p x p solve for wide heads.
    if z.shape[1] > z.shape[0]:
        dual = np.linalg.solve(z @ z.T + alpha * np.eye(z.shape[0]), centered)
        coefficients = z.T @ dual
    else:
        coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ centered)
    return {"alpha": alpha, "standardizer_mean": mean.tolist(), "standardizer_scale": scale.tolist(),
                "coefficients": coefficients.tolist(), "intercept": intercept}


def fit_ridge_grid(features, labels, alphas):
    """Fit a validation-selected ridge grid with one symmetric eigendecomposition.

    This is algebraically equivalent to the dual normal-equation ridge used by
    :func:`fit_ridge`, while avoiding five repeated dense factorizations for a
    wide representation and a fixed training set.
    """
    from scipy.linalg import eigh

    x, y = np.asarray(features, float), np.asarray(labels, float)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid_ridge_training_data")
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(a <= 0 for a in alphas):
        raise ValueError("invalid_ridge_alphas")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = (x - mean) / scale
    intercept = float(y.mean())
    centered = y - intercept
    if z.shape[1] > z.shape[0]:
        gram = z @ z.T
        eigenvalues, eigenvectors = eigh(gram, lower=True, check_finite=False, driver="evd")
        projected = eigenvectors.T @ centered
        heads = {}
        for alpha in alphas:
            dual = eigenvectors @ (projected / (eigenvalues + alpha))
            coefficients = z.T @ dual
            heads[alpha] = {"alpha": alpha, "standardizer_mean": mean.tolist(),
                                "standardizer_scale": scale.tolist(), "coefficients": coefficients.tolist(),
                                "intercept": intercept}
    else:
        gram = z.T @ z
        eigenvalues, eigenvectors = eigh(gram, lower=True, check_finite=False, driver="evd")
        projected = eigenvectors.T @ (z.T @ centered)
        heads = {}
        for alpha in alphas:
            coefficients = eigenvectors @ (projected / (eigenvalues + alpha))
            heads[alpha] = {"alpha": alpha, "standardizer_mean": mean.tolist(),
                                "standardizer_scale": scale.tolist(), "coefficients": coefficients.tolist(),
                                "intercept": intercept}
    return heads


def predict_ridge(features, head):
    x = np.asarray(features, float)
    return ((x-np.asarray(head["standardizer_mean"]))/np.asarray(head["standardizer_scale"])) @ np.asarray(head["coefficients"]) + head["intercept"]
