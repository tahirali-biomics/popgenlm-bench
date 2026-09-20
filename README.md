# PopGenLM Bench

PopGenLM Bench is an open-source framework for evaluating genomic
language-model variant scores using reproducible engineering,
population-genetic evidence, and evolutionary information.

The project asks a simple but important question:

> When a genomic language model assigns a score to a variant, what biological
> evidence supports the interpretation of that score?

## v0.2.0 — population-aware GPN benchmark

PopGenLM Bench v0.2.0 extends the project from an engineering fixture to a
population-aware benchmark using real *Arabidopsis thaliana* variation from
the 1001 Genomes Project.

The benchmark contains 10,000 deterministically selected biallelic SNPs with:

- exact TAIR10.1 reference-allele validation;
- population allele counts, allele frequencies, MAF, MAC, and call rate;
- reproducible GPN Brassicales variant scores;
- explicit REF-to-ALT and population minor-vs-major score orientation;
- deterministic statistical analysis and uncertainty estimates;
- sensitivity and chromosome-level robustness analyses;
- publication-quality figures;
- a reproducible Google Colab notebook;
- automated tests and continuous integration.

The primary population-oriented score is:

`gpn_score_minor_vs_major = log P(minor) - log P(major)`

Negative values indicate that GPN assigns lower likelihood to the population
minor allele than to the major allele.

### Main result

Across 10,000 real variants:

- Spearman correlation between MAF and the GPN minor-vs-major score:
  **rho = 0.0611**
- chromosome-stratified Monte Carlo permutation:
  **p = 1 / 10,001**
- rare variants:
  **n = 7,209**, median score **-0.3485**
- common variants:
  **n = 1,230**, median score **-0.1562**
- rare-versus-common rank-biserial effect:
  **-0.1322**
- bootstrap 95% CI:
  **[-0.1665, -0.0973]**

The signal is therefore **weak but reproducible**.

Rare variants tend to receive more negative GPN minor-vs-major scores than
common variants, but the distributions overlap substantially.

This is not evidence that GPN directly predicts natural selection.

Population demography, population structure, linkage disequilibrium,
ascertainment, allele age, and other evolutionary processes may contribute to
the observed relationship.

## v0.2 overview figure

![PopGenLM Bench v0.2 population-GPN overview](reports/v0.2/figures/v02_population_gpn_overview.png)

The four panels show:

- **A** — the population minor-allele-frequency spectrum;
- **B** — MAF versus GPN minor-vs-major score;
- **C** — score distributions across rare, low-frequency, and common variants;
- **D** — raw REF-to-ALT versus population-oriented GPN scores.

## Reproduce the v0.2 result

### Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tahirali-biomics/popgenlm-bench/blob/v0.2.0/notebooks/popgenlm-bench-v0.2.ipynb)

The v0.2 notebook verifies the frozen benchmark artifacts, reproduces the
principal descriptive results and Spearman association, checks score
orientation, and visualizes the population result.

It does not rerun the expensive GPN inference step.

### Full report

See:

`reports/v0.2/README.md`

### Statistical analysis

    python scripts/analyze_v02_population_gpn.py

### Figures

Install the visualization dependency:

    pip install -e ".[viz]"

Then run:

    python scripts/plot_v02_population_gpn.py

## Installation and validation

Install the package and development dependencies:

    pip install -e ".[dev]"

Run the test suite:

    pytest -q

Run linting:

    ruff check src tests scripts

The v0.2.0 release is validated by 58 automated tests and GitHub Actions
on Python 3.11 and 3.12.

## v0.3 CPU workflows

The portable H3–H6 command-line workflows are under `scripts/v0.3/`. Install their optional scientific and plotting dependencies with:

    pip install -e ".[v03]"

The `plantcad` extra contains only the optional PyTorch/Transformers tokenizer layer; the validated PlantCAD model stack remains environment-specific and is not installed by ordinary CI.

See `docs/v0.3_production_provenance.md` for validated-source mappings and pinned scientific contracts.

## v0.1 — reproducible analysis foundation

Version 0.1 preserves the first verified public PopGenLM Bench engineering
fixture:

- 100 deterministic SNVs scored with GPN Brassicales;
- explicit variant coordinates, reference and alternate alleles;
- inference provenance and run metadata;
- reproducible score-table validation;
- descriptive statistics and bootstrap uncertainty;
- benchmark figures;
- a CPU-compatible Google Colab analysis notebook.

The v0.1 fixture is an engineering validation dataset. It is not a sampled
population and should not be interpreted as evidence for natural selection or
functional constraint.

The frozen v0.1 release is available under tag `v0.1.0`.

## Repository structure

- `data/fixtures/` — small deterministic integration fixtures
- `data/benchmarks/v0.2/` — canonical population benchmark artifacts
- `notebooks/` — reproducible public demonstrations
- `reports/v0.1/` — verified v0.1 outputs
- `reports/v0.2/` — v0.2 statistical results, figures, and report
- `scripts/` — benchmark construction, validation, analysis, and plotting
- `scripts/v0.1/` — preserved v0.1 analysis scripts
- `src/popgenlm/` — installable PopGenLM Bench Python package
- `tests/` — automated release and package tests

Large source genomic datasets, reference genomes, and model weights are not
stored in this repository.

## Scope

PopGenLM Bench is an evaluation framework, not a claim that genomic
language-model scores are direct measurements of fitness, selection, or
pathogenicity.

Its purpose is to make those claims testable through explicit data,
orientation rules, provenance, statistical checks, and reproducible
benchmarks.

## License

Apache-2.0
