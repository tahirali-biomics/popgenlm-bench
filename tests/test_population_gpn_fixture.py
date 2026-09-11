from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

JOINED_PATH = ROOT / "data/fixtures/1001g_v3.1_population_gpn_fixture.tsv"
GPN_PATH = ROOT / "data/fixtures/1001g_v3.1_gpn_scores.tsv"
META_PATH = ROOT / "data/fixtures/1001g_v3.1_gpn_scores.meta.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def test_population_gpn_fixture_orientation_and_edge_padding():
    df = pd.read_csv(
        JOINED_PATH,
        sep="\t",
        dtype={"chrom": str},
    )

    assert len(df) == 20

    assert df["selection_class"].value_counts().to_dict() == {
        "missing": 5,
        "rare": 5,
        "common": 5,
        "alt_major": 5,
    }

    assert df["gpn_score_ref_alt"].notna().all()
    assert df["gpn_score_minor_major"].notna().all()

    assert int(df["flipped"].sum()) == 5
    assert int(df["edge_padded"].sum()) == 5

    for row in df.itertuples():
        assert bool(row.edge_padded) == (row.pos <= 256)

        if row.af_alt < 0.5:
            assert row.orientation == "alt_minor"
            assert row.minor_allele == row.alt
            assert row.major_allele == row.ref
            assert bool(row.flipped) is False
            assert row.gpn_score_minor_major == pytest.approx(row.gpn_score_ref_alt)

        elif row.af_alt > 0.5:
            assert row.orientation == "ref_minor"
            assert row.minor_allele == row.ref
            assert row.major_allele == row.alt
            assert bool(row.flipped) is True
            assert row.gpn_score_minor_major == pytest.approx(-row.gpn_score_ref_alt)

        else:
            pytest.fail("Fixture unexpectedly contains AF_ALT == 0.5")


def test_joined_fixture_uses_canonical_gpn_scores():
    joined = pd.read_csv(
        JOINED_PATH,
        sep="\t",
        dtype={"chrom": str},
    )

    scores = pd.read_csv(
        GPN_PATH,
        sep="\t",
        dtype={"chrom": str},
    )

    fasta_to_vcf = {
        "NC_003070.9": "1",
        "NC_003071.7": "2",
        "NC_003074.8": "3",
        "NC_003075.7": "4",
        "NC_003076.8": "5",
    }

    scores["vcf_chrom"] = scores["chrom"].map(fasta_to_vcf)

    check = joined.merge(
        scores,
        left_on=["chrom", "pos", "ref", "alt"],
        right_on=["vcf_chrom", "pos", "ref", "alt"],
        how="inner",
        validate="one_to_one",
        suffixes=("_joined", "_source"),
    )

    assert len(check) == 20

    for row in check.itertuples():
        assert row.gpn_score_ref_alt_joined == pytest.approx(row.gpn_score_ref_alt_source)


def test_gpn_score_fixture_provenance():
    metadata = json.loads(META_PATH.read_text())
    scores = pd.read_csv(GPN_PATH, sep="\t")

    assert len(scores) == 20

    assert metadata["dataset_id"] == "1001g_1135_v3.1"

    assert metadata["reference"]["assembly"] == "TAIR10.1"
    assert metadata["reference"]["ncbi_assembly"] == "GCF_000001735.4"
    assert metadata["reference"]["ref_matches_fixture"] == "20/20"

    assert metadata["reference"]["sha256"] == (
        "de99316d1d5597e9a5d40d943ff919c17d59b77116354a5d388b183b6f21075f"
    )

    assert metadata["gpn"]["model_id"] == "songlab/gpn-brassicales"
    assert metadata["gpn"]["model_revision"] == ("eb9c35d0d18571abe84390d22e74f2b21d319ce3")
    assert metadata["gpn"]["window_size"] == 512

    assert metadata["reproducibility"]["canonical_revision_rerun_bit_identical"] is True

    assert metadata["reproducibility"]["canonical_revision_rerun_max_abs_difference"] == 0.0

    assert metadata["output"]["n_variants"] == 20
    assert metadata["output"]["sha256"] == sha256(GPN_PATH)
