from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE = (
    Path(__file__).resolve().parents[1] / "data" / "fixtures" / "1001g_v3.1_population_fixture.tsv"
)


def load_fixture() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, sep="\t", dtype={"chrom": str})


def test_fixture_structure():
    frame = load_fixture()

    assert len(frame) == 20
    assert frame["n_genotypes"].eq(1135).all()
    assert frame["filter"].eq("PASS").all()

    counts = frame["selection_class"].value_counts().to_dict()

    assert counts == {
        "rare": 5,
        "alt_major": 5,
        "missing": 5,
        "common": 5,
    }

    assert frame["ref"].str.fullmatch("[ACGT]").all()
    assert frame["alt"].str.fullmatch("[ACGT]").all()
    assert (frame["ref"] != frame["alt"]).all()


def test_fixture_population_arithmetic():
    frame = load_fixture()

    assert (frame["ac_ref"] + frame["ac_alt"] == frame["an"]).all()
    assert (frame["an"] <= 2 * frame["n_genotypes"]).all()

    expected_af = frame["ac_alt"] / frame["an"]
    expected_maf = np.minimum(expected_af, 1.0 - expected_af)
    expected_mac = np.minimum(frame["ac_ref"], frame["ac_alt"])
    expected_call_rate = frame["an"] / (2 * frame["n_genotypes"])

    assert np.allclose(frame["af_alt"], expected_af)
    assert np.allclose(frame["maf"], expected_maf)
    assert np.array_equal(frame["mac"], expected_mac)
    assert np.allclose(frame["allele_call_rate"], expected_call_rate)

    assert frame["maf"].gt(0).all()
    assert frame["maf"].le(0.5).all()


def test_fixture_selection_classes():
    frame = load_fixture()

    rare = frame[frame["selection_class"] == "rare"]
    alt_major = frame[frame["selection_class"] == "alt_major"]
    missing = frame[frame["selection_class"] == "missing"]
    common = frame[frame["selection_class"] == "common"]

    assert rare["maf"].lt(0.01).all()
    assert rare["allele_call_rate"].ge(0.90).all()

    assert alt_major["af_alt"].gt(0.5).all()
    assert alt_major["allele_call_rate"].ge(0.90).all()

    assert missing["allele_call_rate"].lt(0.80).all()

    assert common["maf"].ge(0.05).all()
    assert common["af_alt"].le(0.5).all()
    assert common["allele_call_rate"].ge(0.90).all()
