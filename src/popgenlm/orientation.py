from __future__ import annotations

import math


def orient_gpn_score(
    ref: str,
    alt: str,
    af_alt: float,
    score_ref_alt: float,
) -> dict[str, str | float | bool | None]:
    """Orient a REF→ALT GPN score using population allele frequency.

    The paper-oriented score follows the published GPN Arabidopsis benchmark:
    flip the sign when ALT frequency is greater than 0.5.

    A true minor→major interpretation is undefined when AF_ALT == 0.5, so
    minor/major alleles and the minor-major score are returned as None there.
    """

    ref = str(ref)
    alt = str(alt)

    if ref not in {"A", "C", "G", "T"}:
        raise ValueError("REF must be a canonical single nucleotide")

    if alt not in {"A", "C", "G", "T"}:
        raise ValueError("ALT must be a canonical single nucleotide")

    if ref == alt:
        raise ValueError("REF and ALT must differ")

    af_alt = float(af_alt)
    score_ref_alt = float(score_ref_alt)

    if not math.isfinite(af_alt) or not 0.0 <= af_alt <= 1.0:
        raise ValueError("ALT allele frequency must be finite and between 0 and 1")

    if not math.isfinite(score_ref_alt):
        raise ValueError("GPN score must be finite")

    maf = min(af_alt, 1.0 - af_alt)

    if af_alt < 0.5:
        return {
            "maf": maf,
            "minor_allele": alt,
            "major_allele": ref,
            "orientation": "alt_minor",
            "flipped": False,
            "gpn_score_paper_oriented": score_ref_alt,
            "gpn_score_minor_major": score_ref_alt,
        }

    if af_alt > 0.5:
        return {
            "maf": maf,
            "minor_allele": ref,
            "major_allele": alt,
            "orientation": "ref_minor",
            "flipped": True,
            "gpn_score_paper_oriented": -score_ref_alt,
            "gpn_score_minor_major": -score_ref_alt,
        }

    return {
        "maf": 0.5,
        "minor_allele": None,
        "major_allele": None,
        "orientation": "frequency_tie",
        "flipped": False,
        "gpn_score_paper_oriented": score_ref_alt,
        "gpn_score_minor_major": None,
    }
