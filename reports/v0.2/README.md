# PopGenLM Bench v0.2 — Population-aware GPN benchmark

## Overview

PopGenLM Bench v0.2 evaluates whether genomic language-model variant scores
show concordance with population allele-frequency patterns in real
*Arabidopsis thaliana* variation.

The benchmark contains 10,000 deterministically selected biallelic SNPs from
the 1001 Genomes dataset. Reference alleles were validated against TAIR10.1,
GPN scores were generated using a fixed 512-bp sequence context, and raw
REF-to-ALT scores were re-oriented relative to the population minor and major
alleles.

The primary population-oriented score is:

`gpn_score_minor_vs_major = log P(minor) - log P(major)`

Negative values indicate that GPN assigns lower likelihood to the minor allele
than to the major allele.

## Benchmark dataset

- Variants: 10,000
- Rare, MAF < 0.01: 7,209
- Low-frequency, 0.01 <= MAF < 0.05: 1,561
- Common, MAF >= 0.05: 1,230
- ALT-major variants requiring score sign reversal: 246
- Frequency ties at AF = 0.5: 0
- Edge-padded GPN contexts: 0
- Duplicate variant keys: 0

Canonical joined dataset:

`data/benchmarks/v0.2/1001g_population_10000_gpn.tsv`

SHA256:

`ef188e83d0e598d016c158b4daeec08be63e19723840fc42b494c76443fca469`

## Main result

Minor-allele frequency showed a weak positive association with the
population-oriented GPN score:

- Spearman rho = 0.0611
- chromosome-stratified Monte Carlo permutation p = 1 / 10,001
- 0 of 10,000 permuted statistics were at least as extreme as observed

Thus, as minor alleles become more frequent, their GPN scores tend to become
slightly less negative relative to the major allele.

The association is reproducible but small in magnitude.

## Frequency-class comparison

Median GPN minor-vs-major scores were:

| Frequency class | n | Median score |
|---|---:|---:|
| Rare | 7,209 | -0.3485 |
| Low-frequency | 1,561 | -0.2756 |
| Common | 1,230 | -0.1562 |

For the rare-versus-common comparison:

- median difference = -0.1924
- rank-biserial correlation = -0.1322
- bootstrap 95% CI = [-0.1665, -0.0973]

The distributions overlap extensively. The result therefore represents a
small population-level shift rather than strong separation between frequency
classes.

## Robustness

The positive MAF-score relationship persisted under stricter filters:

| Analysis | n | Spearman rho |
|---|---:|---:|
| All variants | 10,000 | 0.0611 |
| Call rate >= 0.95 | 6,761 | 0.0581 |
| MAC >= 5 | 5,057 | 0.0876 |
| Call rate >= 0.95 and MAC >= 5 | 3,279 | 0.1002 |

The association was also positive separately on chromosomes 1–5 and remained
positive when each chromosome was excluded in turn.

The genome-wide direction is therefore not attributable to a single
chromosome.

## Score orientation

GPN produces the raw score:

`log P(ALT) - log P(REF)`

For variants where ALT is the population minor allele, this already equals:

`log P(minor) - log P(major)`

For the 246 variants where ALT is the population major allele, the sign is
reversed.

This orientation is required for a consistent population interpretation. It
does not create the observed genome-wide MAF association.

## Interpretation

The benchmark supports a deliberately narrow conclusion:

> GPN sequence plausibility shows weak but reproducible concordance with
> population allele frequency in real *Arabidopsis thaliana* variants.

Rare variants tend to receive more negative minor-vs-major GPN scores than
common variants, but the effect is modest and the score distributions overlap
substantially.

This analysis is not a test of natural selection.

Population demography, population structure, linkage disequilibrium,
ascertainment, allele age, and other evolutionary processes may contribute to
the observed relationship. The chromosome-stratified permutation preserves
chromosome membership but does not explicitly model local linkage
disequilibrium.

## Figure

Main overview figure:

`reports/v0.2/figures/v02_population_gpn_overview.png`

Panels:

- **A** — population minor-allele-frequency spectrum
- **B** — MAF versus GPN minor-vs-major score
- **C** — score distributions across frequency classes
- **D** — raw REF-to-ALT versus population-oriented GPN scores

PDF version:

`reports/v0.2/figures/v02_population_gpn_overview.pdf`

## Reproducibility

Run the statistical analysis:

    python scripts/analyze_v02_population_gpn.py

Generate the figures:

    python scripts/plot_v02_population_gpn.py

Run validation:

    ruff check .
    pytest -q
    git diff --check

The v0.2.0 release is validated by a full test suite containing 58 passing
tests.
