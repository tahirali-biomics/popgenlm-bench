import pytest

from popgenlm.orientation import orient_gpn_score


def test_alt_minor_keeps_score():
    result = orient_gpn_score("A", "G", 0.10, -2.0)

    assert result["minor_allele"] == "G"
    assert result["major_allele"] == "A"
    assert result["orientation"] == "alt_minor"
    assert result["flipped"] is False
    assert result["maf"] == pytest.approx(0.10)
    assert result["gpn_score_paper_oriented"] == pytest.approx(-2.0)
    assert result["gpn_score_minor_major"] == pytest.approx(-2.0)


def test_alt_major_flips_score():
    result = orient_gpn_score("A", "G", 0.90, -2.0)

    assert result["minor_allele"] == "A"
    assert result["major_allele"] == "G"
    assert result["orientation"] == "ref_minor"
    assert result["flipped"] is True
    assert result["maf"] == pytest.approx(0.10)
    assert result["gpn_score_paper_oriented"] == pytest.approx(2.0)
    assert result["gpn_score_minor_major"] == pytest.approx(2.0)


def test_frequency_tie_has_no_unique_minor_allele():
    result = orient_gpn_score("C", "T", 0.50, -1.5)

    assert result["maf"] == pytest.approx(0.50)
    assert result["minor_allele"] is None
    assert result["major_allele"] is None
    assert result["orientation"] == "frequency_tie"
    assert result["flipped"] is False

    # Reproduce the published >0.5 sign-flip convention.
    assert result["gpn_score_paper_oriented"] == pytest.approx(-1.5)

    # But do not falsely label either allele as minor at exactly 50:50.
    assert result["gpn_score_minor_major"] is None


def test_invalid_frequency_fails():
    with pytest.raises(ValueError, match="ALT allele frequency"):
        orient_gpn_score("A", "G", 1.1, -1.0)


def test_identical_ref_alt_fails():
    with pytest.raises(ValueError, match="REF and ALT must differ"):
        orient_gpn_score("A", "A", 0.1, -1.0)
