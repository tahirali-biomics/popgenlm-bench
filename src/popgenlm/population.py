from __future__ import annotations

import re
from collections.abc import Iterable


def summarize_biallelic_genotypes(genotypes: Iterable[str]) -> dict[str, float | int]:
    """Calculate allele counts and frequencies from diploid biallelic GT values."""

    ac_alt = 0
    an = 0
    n_genotypes = 0
    n_missing = 0
    n_heterozygous = 0

    for genotype in genotypes:
        n_genotypes += 1
        gt = str(genotype).split(":", maxsplit=1)[0]

        alleles = re.split(r"[|/]", gt)

        if len(alleles) != 2:
            raise ValueError(f"Expected diploid genotype, found: {gt}")

        called = [allele for allele in alleles if allele != "."]

        if not called:
            n_missing += 1
            continue

        if any(allele not in {"0", "1"} for allele in called):
            raise ValueError(
                f"Expected biallelic genotype containing only 0, 1 or '.', found: {gt}"
            )

        an += len(called)
        ac_alt += sum(allele == "1" for allele in called)

        if len(called) == 2 and called[0] != called[1]:
            n_heterozygous += 1

    if an == 0:
        raise ValueError("Cannot calculate allele frequency: no called alleles")

    ac_ref = an - ac_alt
    af_alt = ac_alt / an
    mac = min(ac_ref, ac_alt)
    maf = mac / an
    allele_call_rate = an / (2 * n_genotypes)

    return {
        "n_genotypes": n_genotypes,
        "n_missing": n_missing,
        "n_heterozygous": n_heterozygous,
        "ac_ref": ac_ref,
        "ac_alt": ac_alt,
        "an": an,
        "af_alt": af_alt,
        "mac": mac,
        "maf": maf,
        "allele_call_rate": allele_call_rate,
    }
