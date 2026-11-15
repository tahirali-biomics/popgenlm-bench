import numpy as np
import pytest

from popgenlm.regulatory_transfer import (
    allele_sequences,
    cache_key,
    central_values,
    transfer_magnitude,
)


def test_allele_pair_differs_only_at_focal_base():
    ref, alt = allele_sequences("ACGTA", 2, "G", "T")
    assert ref == "ACGTA" and alt == "ACTTA"
    with pytest.raises(ValueError):
        allele_sequences(ref, 2, "C", "T")


def test_cache_key_includes_flanks_and_model():
    assert cache_key("ACGT", {"version": 1}) != cache_key("ACGA", {"version": 1})
    assert cache_key("ACGT", {"version": 1}) != cache_key("ACGT", {"version": 2})


def test_central_crop_and_aggregation():
    a = np.zeros((1024, 2))
    a[504:520] = 3
    cropped = central_values(a, 128)
    assert cropped.shape == (16, 2) and np.all(cropped == 3)
    assert transfer_magnitude(np.zeros_like(cropped), cropped) == pytest.approx(np.log(4))
    assert transfer_magnitude(cropped, cropped) == 0
    with pytest.raises(ValueError):
        central_values(a, 127)
    with pytest.raises(ValueError):
        transfer_magnitude(cropped, -cropped)
