"""CPU-only contracts for exploratory custom-sequence transfer predictions."""

import hashlib
import json

import numpy as np


def allele_sequences(reference, index, ref, alt):
    if len(ref) != 1 or len(alt) != 1 or ref == alt or reference[index] != ref:
        raise ValueError("REF/ALT or focal reference mismatch")
    alternate = reference[:index] + alt + reference[index + 1 :]
    assert sum(a != b for a, b in zip(reference, alternate)) == 1
    return reference, alternate


def cache_key(sequence, specification):
    return hashlib.sha256(
        sequence.encode() + json.dumps(specification, sort_keys=True).encode()
    ).hexdigest()


def central_values(values, resolution, length=131072, width=2048):
    values = np.asarray(values)
    if resolution <= 0 or len(values) * resolution != length or width % resolution:
        raise ValueError("output resolution or width mismatch")
    start = length // 2 - width // 2
    end = start + width
    if start % resolution or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("nonfinite/negative predictions or misaligned bins")
    return values[start // resolution : end // resolution].copy()


def transfer_magnitude(ref, alt):
    ref, alt = np.asarray(ref), np.asarray(alt)
    if (
        ref.shape != alt.shape
        or not np.isfinite(ref).all()
        or not np.isfinite(alt).all()
        or (ref < 0).any()
        or (alt < 0).any()
    ):
        raise ValueError("invalid paired tracks")
    return float(np.mean(np.abs(np.log1p(alt.astype(float)) - np.log1p(ref.astype(float)))))
