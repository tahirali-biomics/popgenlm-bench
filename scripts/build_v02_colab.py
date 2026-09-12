"""Build the PopGenLM Bench v0.2 Colab notebook deterministically."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "popgenlm-bench-v0.2.ipynb"

RELEASE_REF = "v0.2.0"
DEVELOPMENT_REF = "012975df95c69648f01eb7c19ee94f8d04e7782e"

EXPECTED_DATA_SHA256 = "ef188e83d0e598d016c158b4daeec08be63e19723840fc42b494c76443fca469"

EXPECTED_ANALYSIS_SHA256 = "c41e2a4e59c5a9fc25f12803ae30149c502096167328874b0890e1bf4b5f11e9"

EXPECTED_PAIRWISE_SHA256 = "3779fe5578ca4b33b5d66c4a34d4eefe61d85417982a4e5f45e31b3571f80739"

EXPECTED_SENSITIVITY_SHA256 = "d00b1a016f87102d45cbf276b08b0fa0fea3379ab08194c14a3f4e8091e2a3fa"

EXPECTED_CHROMOSOME_SHA256 = "5f0791cf974a35524393d99cdc248dede8fac0bc25978aca7dbefe1dd2e74dc3"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# PopGenLM Bench v0.2 — population-aware validation of GPN scores

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tahirali-biomics/popgenlm-bench/blob/v0.2.0/notebooks/popgenlm-bench-v0.2.ipynb)

This notebook reproduces the main public result from **PopGenLM Bench v0.2**
using the frozen 10,000-variant *Arabidopsis thaliana* population benchmark.

It does **not** rerun genomic-language-model inference. Instead, it starts from
the released population + GPN table, verifies its checksum and schema, checks
the population orientation of GPN scores, recomputes the principal descriptive
statistics and Spearman association, and visualizes the result.

The primary score is:

`gpn_score_minor_vs_major = log P(minor) - log P(major)`

Negative values mean that GPN assigns lower likelihood to the population minor
allele than to the population major allele.

> **Interpretation boundary:** this benchmark measures concordance between GPN
> sequence plausibility and population allele-frequency patterns. It is not a
> test of natural selection. Demography, population structure, linkage,
> ascertainment, allele age, and other evolutionary processes may contribute.
"""
    ),
    markdown(
        """## 1. Load the frozen v0.2 benchmark

The notebook first attempts to load the immutable `v0.2.0` release. Before that tag exists, the notebook falls back to a pinned
development commit containing the same frozen benchmark artifacts.

The table checksum is verified before analysis.
"""
    ),
    code(
        f"""from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY = "tahirali-biomics/popgenlm-bench"
RELEASE_REF = "{RELEASE_REF}"
DEVELOPMENT_REF = "{DEVELOPMENT_REF}"

DATA_PATH = "data/benchmarks/v0.2/1001g_population_10000_gpn.tsv"
ANALYSIS_PATH = "reports/v0.2/population_gpn_analysis.json"

EXPECTED_DATA_SHA256 = "{EXPECTED_DATA_SHA256}"
EXPECTED_ANALYSIS_SHA256 = "{EXPECTED_ANALYSIS_SHA256}"
EXPECTED_PAIRWISE_SHA256 = "{EXPECTED_PAIRWISE_SHA256}"
EXPECTED_SENSITIVITY_SHA256 = "{EXPECTED_SENSITIVITY_SHA256}"
EXPECTED_CHROMOSOME_SHA256 = "{EXPECTED_CHROMOSOME_SHA256}"


def raw_url(ref: str, path: str) -> str:
    return (
        "https://raw.githubusercontent.com/"
        f"{{REPOSITORY}}/{{ref}}/{{path}}"
    )


def fetch_bytes(path: str) -> tuple[bytes, str]:
    errors = []

    for ref in [RELEASE_REF, DEVELOPMENT_REF]:
        url = raw_url(ref, path)

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read(), ref
        except (urllib.error.URLError, TimeoutError) as error:
            errors.append(f"{{ref}}: {{error}}")

    raise RuntimeError(
        "Could not download benchmark artifact:\\n"
        + "\\n".join(errors)
    )


data_bytes, data_ref = fetch_bytes(DATA_PATH)
analysis_bytes, analysis_ref = fetch_bytes(ANALYSIS_PATH)

data_sha256 = hashlib.sha256(data_bytes).hexdigest()
analysis_sha256 = hashlib.sha256(analysis_bytes).hexdigest()

assert data_sha256 == EXPECTED_DATA_SHA256, (
    "Benchmark checksum mismatch: "
    f"{{data_sha256}}"
)

assert analysis_sha256 == EXPECTED_ANALYSIS_SHA256, (
    "Analysis checksum mismatch: "
    f"{{analysis_sha256}}"
)

df = pd.read_csv(
    io.BytesIO(data_bytes),
    sep="\\t",
    dtype={{
        "chrom": str,
        "fasta_chrom": str,
    }},
)

analysis = json.loads(analysis_bytes)

print(f"Loaded {{len(df):,}} variants")
print(f"Benchmark ref: {{data_ref}}")
print(f"Analysis ref: {{analysis_ref}}")
print(f"Benchmark SHA256: {{data_sha256}}")
"""
    ),
    markdown(
        """## 2. Validate schema and population orientation

The released table should contain exactly 10,000 unique biallelic SNPs.

For variants where ALT is the population minor allele, the raw GPN score
already equals `log P(minor) - log P(major)`. When ALT is the major allele,
the sign must be reversed.
"""
    ),
    code(
        """required = {
    "chrom",
    "pos",
    "ref",
    "alt",
    "maf",
    "af_alt",
    "frequency_class",
    "alt_major",
    "flipped",
    "gpn_score_ref_alt",
    "gpn_score_minor_vs_major",
}

missing = required - set(df.columns)
assert not missing, f"Missing columns: {sorted(missing)}"

key = ["chrom", "pos", "ref", "alt"]

assert len(df) == 10_000
assert not df.duplicated(key).any()

assert df["ref"].isin(list("ACGT")).all()
assert df["alt"].isin(list("ACGT")).all()
assert (df["ref"] != df["alt"]).all()

alt_major = (
    df["alt_major"]
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
)

flipped = (
    df["flipped"]
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
)

assert alt_major.notna().all()
assert flipped.notna().all()

raw = df["gpn_score_ref_alt"].to_numpy(float)
oriented = df["gpn_score_minor_vs_major"].to_numpy(float)

expected_oriented = np.where(
    alt_major.to_numpy(),
    -raw,
    raw,
)

np.testing.assert_allclose(
    oriented,
    expected_oriented,
    rtol=0.0,
    atol=1e-12,
)

assert (flipped == alt_major).all()

print("✓ Schema and variant-key checks passed")
print("✓ Population score orientation verified")
print(f"ALT-major sign flips: {int(flipped.sum()):,}")
"""
    ),
    markdown(
        """## 3. Population-frequency spectrum

The benchmark was selected deterministically from real 1001 Genomes variation;
it was not balanced across frequency classes. Consequently, rare variants
dominate the dataset.
"""
    ),
    code(
        """counts = (
    df["frequency_class"]
    .value_counts()
    .reindex(
        ["rare", "low_frequency", "common"]
    )
)

display(
    counts.rename("n_variants").to_frame()
)

assert counts.to_dict() == {
    "rare": 7209,
    "low_frequency": 1561,
    "common": 1230,
}

maf = df["maf"].to_numpy(float)

bins = np.geomspace(
    maf.min(),
    0.5,
    45,
)

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.hist(
    maf,
    bins=bins,
    color="0.45",
    edgecolor="white",
    linewidth=0.4,
)

ax.set_xscale("log")
ax.axvline(0.01, linestyle="--", linewidth=1)
ax.axvline(0.05, linestyle="--", linewidth=1)

ax.set_xlabel("Minor-allele frequency (MAF)")
ax.set_ylabel("Variants")
ax.set_title("Population-frequency spectrum")

plt.show()
"""
    ),
    markdown(
        """## 4. Reproduce the MAF–GPN association

We recompute Spearman's rank correlation directly from the released benchmark.

The full release analysis also uses a chromosome-stratified permutation test.
Its frozen result is read from the released analysis record rather than
rerunning 10,000 permutations in this lightweight notebook.
"""
    ),
    code(
        """def spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    rank_x = (
        pd.Series(x)
        .rank(method="average")
        .to_numpy(float)
    )

    rank_y = (
        pd.Series(y)
        .rank(method="average")
        .to_numpy(float)
    )

    return float(
        np.corrcoef(rank_x, rank_y)[0, 1]
    )


rho = spearman_rho(
    df["maf"].to_numpy(float),
    df["gpn_score_minor_vs_major"].to_numpy(float),
)

frozen = analysis["continuous_association"]
frozen_rho = frozen["spearman_maf_minor_vs_major"]

assert np.isclose(
    rho,
    frozen_rho,
    rtol=0.0,
    atol=1e-12,
)

print(f"Recomputed Spearman rho: {rho:.6f}")
print(
    "Frozen chromosome-stratified Monte Carlo p:",
    frozen["chromosome_stratified_permutation_p"],
)
print(
    "Extreme permutations:",
    f'{frozen["permutation_extreme_count"]}/'
    f'{analysis["settings"]["permutations"]}',
)
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(7.2, 4.8))

log_maf = np.log10(
    df["maf"].to_numpy(float)
)

hb = ax.hexbin(
    log_maf,
    df["gpn_score_minor_vs_major"],
    gridsize=58,
    mincnt=1,
    bins="log",
    cmap="viridis",
    linewidths=0,
)

ticks = np.array(
    [0.001, 0.01, 0.1, 0.5]
)

ax.set_xticks(
    np.log10(ticks),
    labels=["0.001", "0.01", "0.1", "0.5"],
)

ax.axhline(
    0,
    color="0.5",
    linestyle=":",
    linewidth=0.8,
)

ax.set_xlabel("Minor-allele frequency (MAF)")
ax.set_ylabel(
    "GPN minor-vs-major score\\n"
    "log P(minor) − log P(major)"
)

ax.set_title(
    f"Weak positive population concordance: "
    f"Spearman ρ = {rho:.3f}"
)

colorbar = fig.colorbar(
    hb,
    ax=ax,
)

colorbar.set_label(
    "Variants per hexagon (log scale)"
)

plt.show()
"""
    ),
    markdown(
        """## 5. Compare frequency classes

Rare variants are shifted toward more negative scores than common variants,
but the distributions overlap substantially. The release therefore emphasizes
effect size rather than statistical-significance stars.
"""
    ),
    code(
        """summary = (
    df.groupby(
        "frequency_class",
        observed=True,
    )["gpn_score_minor_vs_major"]
    .agg(
        n="size",
        median="median",
        mean="mean",
    )
    .reindex(
        ["rare", "low_frequency", "common"]
    )
)

display(summary)

expected_medians = {
    "rare": -0.34853882,
    "low_frequency": -0.27562457,
    "common": -0.15618226,
}

for group, expected in expected_medians.items():
    assert np.isclose(
        summary.loc[group, "median"],
        expected,
        rtol=0.0,
        atol=1e-8,
    )

pairwise_bytes, pairwise_ref = fetch_bytes(
    "reports/v0.2/frequency_class_pairwise.tsv"
)

pairwise_sha256 = hashlib.sha256(
    pairwise_bytes
).hexdigest()

assert pairwise_sha256 == EXPECTED_PAIRWISE_SHA256, (
    "Pairwise-results checksum mismatch: "
    f"{pairwise_sha256}"
)

pairwise = pd.read_csv(
    io.BytesIO(pairwise_bytes),
    sep="\\t",
)

rare_common = (
    pairwise
    .set_index("contrast")
    .loc["rare_vs_common"]
)

print(
    f"Pairwise results verified from {pairwise_ref}"
)

print()
print("Rare versus common:")
print(
    "median difference:",
    f'{rare_common["median_difference_a_minus_b"]:.4f}',
)
print(
    "rank-biserial:",
    f'{rare_common["rank_biserial_a_vs_b"]:.4f}',
)
print(
    "bootstrap 95% CI:",
    (
        f'[{rare_common["rank_biserial_ci_low"]:.4f}, '
        f'{rare_common["rank_biserial_ci_high"]:.4f}]'
    ),
)
"""
    ),
    code(
        """groups = [
    df.loc[
        df["frequency_class"] == group,
        "gpn_score_minor_vs_major",
    ].to_numpy(float)
    for group in [
        "rare",
        "low_frequency",
        "common",
    ]
]

fig, ax = plt.subplots(figsize=(7, 4.8))

parts = ax.violinplot(
    groups,
    positions=[1, 2, 3],
    widths=0.8,
    showmeans=False,
    showmedians=True,
    showextrema=False,
)

for body in parts["bodies"]:
    body.set_alpha(0.65)

ax.axhline(
    0,
    color="0.5",
    linestyle=":",
    linewidth=0.8,
)

ax.set_xticks(
    [1, 2, 3],
    labels=[
        "Rare\\n(n=7,209)",
        "Low-frequency\\n(n=1,561)",
        "Common\\n(n=1,230)",
    ],
)

ax.set_ylabel(
    "GPN minor-vs-major score\\n"
    "log P(minor) − log P(major)"
)

ax.set_title(
    "Score distributions overlap substantially"
)

plt.show()
"""
    ),
    markdown(
        """## 6. Inspect score orientation

Most variants have ALT as the population minor allele, so the raw REF-to-ALT
score is already population-oriented. For 246 ALT-major variants, the score
sign is reversed.
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(6.5, 6.5))

raw = df["gpn_score_ref_alt"].to_numpy(float)
oriented = df["gpn_score_minor_vs_major"].to_numpy(float)

flip_mask = flipped.to_numpy(bool)

ax.scatter(
    raw[~flip_mask],
    oriented[~flip_mask],
    s=8,
    alpha=0.15,
    label=f"Not flipped (n={(~flip_mask).sum():,})",
)

ax.scatter(
    raw[flip_mask],
    oriented[flip_mask],
    s=18,
    alpha=0.75,
    label=f"ALT-major; sign flipped (n={flip_mask.sum():,})",
)

limit = np.ceil(
    max(
        np.max(np.abs(raw)),
        np.max(np.abs(oriented)),
    )
)

reference = np.array([-limit, limit])

ax.plot(
    reference,
    reference,
    linewidth=0.9,
    label="y = x",
)

ax.plot(
    reference,
    -reference,
    linewidth=0.9,
    linestyle="--",
    label="y = −x",
)

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel(
    "Raw GPN score\\n"
    "log P(ALT) − log P(REF)"
)

ax.set_ylabel(
    "Population-oriented score\\n"
    "log P(minor) − log P(major)"
)

ax.set_title("Population orientation of GPN scores")
ax.legend(frameon=False)

plt.show()
"""
    ),
    markdown(
        """## 7. Robustness

The full v0.2 analysis tested stricter genotype-call and minor-allele-count
filters and chromosome-level robustness.
"""
    ),
    code(
        """sensitivity_bytes, sensitivity_ref = fetch_bytes(
    "reports/v0.2/sensitivity_summary.tsv"
)

chromosome_bytes, chromosome_ref = fetch_bytes(
    "reports/v0.2/chromosome_robustness.tsv"
)

sensitivity_sha256 = hashlib.sha256(
    sensitivity_bytes
).hexdigest()

chromosome_sha256 = hashlib.sha256(
    chromosome_bytes
).hexdigest()

assert sensitivity_sha256 == EXPECTED_SENSITIVITY_SHA256, (
    "Sensitivity-results checksum mismatch: "
    f"{sensitivity_sha256}"
)

assert chromosome_sha256 == EXPECTED_CHROMOSOME_SHA256, (
    "Chromosome-results checksum mismatch: "
    f"{chromosome_sha256}"
)

print(
    f"Sensitivity results verified from {sensitivity_ref}"
)
print(
    f"Chromosome results verified from {chromosome_ref}"
)

sensitivity = pd.read_csv(
    io.BytesIO(sensitivity_bytes),
    sep="\\t",
)

chromosome = pd.read_csv(
    io.BytesIO(chromosome_bytes),
    sep="\\t",
)

display(sensitivity)

display(
    chromosome.rename(
        columns={
            "chromosome": "chromosome_or_excluded",
        }
    )
)

assert (
    sensitivity["spearman_rho"] > 0
).all()

assert (
    chromosome["spearman_rho"] > 0
).all()

print(
    "✓ Positive association retained in all "
    "sensitivity and chromosome analyses"
)
"""
    ),
    markdown(
        """## 8. What does v0.2 show?

The benchmark supports a deliberately narrow conclusion:

> **GPN sequence plausibility shows weak but reproducible concordance with
> population allele frequency in real *Arabidopsis thaliana* variants.**

Higher-frequency minor alleles tend to receive slightly less negative
minor-vs-major GPN scores. Rare variants are shifted toward more negative
scores relative to common variants, but the distributions overlap
substantially.

The effect is therefore **detectable but small**.

This result should **not** be interpreted as direct evidence that GPN detects
natural selection. Population demography, population structure, linkage
disequilibrium, ascertainment, allele age, and other processes may contribute.

### Reproduce the complete release analysis

For the full deterministic analysis, bootstrap effect-size intervals,
chromosome-stratified permutation test, sensitivity analyses, and publication
figures, see:

- `scripts/analyze_v02_population_gpn.py`
- `scripts/plot_v02_population_gpn.py`
- `reports/v0.2/README.md`

in the PopGenLM Bench repository.
"""
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
        "colab": {
            "name": ("PopGenLM Bench v0.2 — population-aware validation"),
            "provenance": [],
        },
    },
    "cells": cells,
}

OUTPUT.write_text(
    json.dumps(
        notebook,
        indent=1,
        ensure_ascii=False,
    )
    + "\n"
)

digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

print("Notebook:", OUTPUT)
print("Cells:", len(cells))
print("SHA256:", digest)
print("V0.2 COLAB BUILD PASSED")


if __name__ == "__main__":
    pass
