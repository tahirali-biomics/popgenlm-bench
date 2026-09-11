import pytest

from popgenlm.population import summarize_biallelic_genotypes


def test_population_counts_homozygotes_and_missing():
    result = summarize_biallelic_genotypes(["0|0", "0|0", "1|1", "./."])

    assert result["n_genotypes"] == 4
    assert result["n_missing"] == 1
    assert result["n_heterozygous"] == 0
    assert result["ac_ref"] == 4
    assert result["ac_alt"] == 2
    assert result["an"] == 6
    assert result["af_alt"] == pytest.approx(2 / 6)
    assert result["mac"] == 2
    assert result["maf"] == pytest.approx(2 / 6)
    assert result["allele_call_rate"] == pytest.approx(6 / 8)


def test_population_counts_heterozygote():
    result = summarize_biallelic_genotypes(["0|0", "0|1", "1|1"])

    assert result["ac_ref"] == 3
    assert result["ac_alt"] == 3
    assert result["an"] == 6
    assert result["af_alt"] == pytest.approx(0.5)
    assert result["maf"] == pytest.approx(0.5)
    assert result["n_heterozygous"] == 1


def test_unphased_genotypes_are_supported():
    result = summarize_biallelic_genotypes(["0/0", "0/1", "1/1"])

    assert result["ac_alt"] == 3
    assert result["an"] == 6


def test_partial_missing_alleles_do_not_enter_an():
    result = summarize_biallelic_genotypes(["0|.", "1|.", "1|1"])

    assert result["ac_ref"] == 1
    assert result["ac_alt"] == 3
    assert result["an"] == 4


def test_multiallelic_genotype_fails():
    with pytest.raises(ValueError, match="Expected biallelic genotype"):
        summarize_biallelic_genotypes(["0|2"])


def test_all_missing_fails():
    with pytest.raises(ValueError, match="no called alleles"):
        summarize_biallelic_genotypes(["./.", "./."])
