# PopGenLM Bench

PopGenLM Bench is an open-source project for evaluating genomic language-model
variant scores using reproducible engineering, population-genetic evidence,
and evolutionary information.

## v0.1 — reproducible analysis foundation

Version 0.1 preserves the first verified public PopGenLM Bench fixture:

- 100 deterministic SNVs scored with GPN Brassicales;
- explicit variant coordinates, reference and alternate alleles;
- inference provenance and run metadata;
- reproducible score-table validation;
- descriptive statistics and bootstrap uncertainty;
- benchmark figures;
- a CPU-compatible Google Colab analysis notebook.

The v0.1 fixture is an engineering validation dataset. It is not a population
sample and should not be interpreted as evidence for natural selection or
functional constraint.

## v0.2 — population benchmark

Development of v0.2 extends the framework to public Arabidopsis thaliana
population data from the 1001 Genomes Project. It will introduce the installable
Python package, automated tests and CI, population-data ingestion,
allele-frequency evaluation, reproducible GPN inference, and population-genetic
benchmarking.

## Repository structure

- `data/fixtures/` — small public deterministic fixtures
- `notebooks/` — reproducible demonstrations
- `reports/v0.1/` — verified v0.1 outputs
- `reports/v0.2/` — population-benchmark outputs under development
- `scripts/v0.1/` — preserved v0.1 analysis/figure scripts
- `src/popgenlm/` — package source introduced during v0.2 development
- `tests/` — automated package tests introduced during v0.2 development

Large genomic datasets and model weights are not stored in this repository.
