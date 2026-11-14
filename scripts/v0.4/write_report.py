"""Write the pilot handover from verified artifact tables, without inventing endpoints."""

import argparse
import json
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
a = p.parse_args()
run = Path(a.run)
analysis = run / "results/analysis"
s = json.loads((analysis / "summary.json").read_text())
coverage = pd.read_csv(analysis / "coverage.tsv", sep="\t")
worked = pd.read_csv(analysis / "worked_locus_top5.tsv", sep="\t")
loci = pd.read_csv(run / "inputs/frozen/loci.tsv", sep="\t")
x = pd.read_csv(run / "inputs/frozen/candidates.tsv", sep="\t")
exclusions = pd.read_csv(run / "results/assay_exclusions.tsv", sep="\t")
lines = [
    "# PopGenLM Bench v0.4 regulatory prioritization pilot — review handover",
    "",
    "Date: 2026-11-14 UTC. Base release: v0.3.0 (`4c4258f4a8014d4ee2b19e32329362baf34c44a0`). Isolated worktree: `/projects/ag-demeaux/tali/PopGenLM/worktrees/v0.4-regulatory-pilot`.",
    "",
    f"The frozen pilot contains **{s['loci']} loci and {s['candidate_pairs']} candidate SNPs**. The independent nucleotide endpoint has **{s['primary_loci']} eligible loci**. No causal prioritization improvement can be estimated from a zero denominator. The completed scores below support a descriptive same-study held-out mapping comparison.",
    "",
    "## Input and evidence audit",
    "",
    "Full VCF, index, reference, annotation, signed PhyloP track and the v0.3 annotated master passed recorded-hash checks in Slurm job 7609268. `manifests/input_verification.json` records exact absolute paths and hashes. Chromosome aliases and VCF indexing were checked; the legacy dosage sample order was independently matched to the VCF header. The apparent population fixture hash difference is a mistyped old G3B digest: current fixture bytes equal the v0.3.0 release tag.",
    "",
    "Expression data: [GSE80744](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE80744), processed normalized count matrix. Of 727 expression columns, 665 match VCF accessions and 62 do not. There are 469 other VCF samples and one explicit Cod-0 normalization exclusion. No duplicate accession IDs were merged. The crosswalk retains all records. Mapping uses 469 discovery and 196 held-out accessions, assigned before mapping by a deterministic accession hash.",
    "",
    "Voichek et al. [study](https://doi.org/10.1038/s41588-024-01907-3), [analysis archive](https://doi.org/10.5281/zenodo.13170729), and supplementary tables S3/S4 were audited. Of 17,929 modified reporter sequences, 9,317 are single substitutions, 7,110 map exactly to the reference, and 176 match natural VCF alleles. Thirteen substitutions across 11 genes have consistent activity differences of at least 0.5 in both promoter constructs. These are exploratory reporter nucleotide supports with no per-SNP uncertainty, not endogenous editing validation.",
    "",
    "All 13 are excluded by the frozen common-SNP, call-rate, association or LD rules. Exact reasons and assay effects are in `results/assay_exclusions.tsv`. No association lead, haplotype, conservation score or motif overlap was promoted to a nucleotide truth label.",
    "",
    "## Frozen design",
    "",
    "Eight eligible loci per chromosome were selected from a deterministic inventory seeded by the published tested-phenotype list, prioritizing the audited reporter genes. This is not a list of published significant eQTLs: cis signals were recomputed. Gene windows include 10 kb on either side. A rank-normalized null-REML LMM uses chromosome-excluded kinship from 1,010 quality-filtered, physically thinned v0.3 markers and the expression-matched accessions. Candidate discovery MAF >=0.05, call rate >=0.9 and lead r² >=0.6 are required, with at least six linked candidates. Each locus passes the prespecified discovery association threshold.",
    "",
    f"Frozen candidates include {(x.tss_distance < 0).sum()} upstream, {(x.tss_distance == 0).sum()} at-TSS and {(x.tss_distance > 0).sum()} downstream SNPs. Individual linked sets need not span both sides. The complete inventory, exclusions, genotype dosages, mapping partitions, annotations, outcome rules, fixed coefficients and AlphaGenome subset are in `inputs/frozen/`. `freeze.json` was written by completed job 7609387 before pilot scores.",
    "",
    "Association+LD, annotation baseline, and baseline+model rankings use fixed coefficients specified in `protocol.json`; no supervised weights are fitted. Whole chromosomes are evaluation blocks. Each model is paired with the same baseline on identical covered candidates; a shared intersection across all runnable models is additionally reported.",
    "",
    "## Model coverage",
    "",
    "| Model | Candidate SNPs covered | Complete loci |",
    "|---|---:|---:|",
]
for r in coverage.itertuples():
    lines.append(
        f"| {r.model} | {r.candidate_pairs_covered}/{r.candidate_pairs_total} | {r.loci_complete}/{r.loci_total} |"
    )
lines += [
    "",
    "PlantCAD2-S was run at 512 and 4096 bp; original PlantCAD and GPN at 512 bp. Checkpoints, source revisions, weight hashes, environment versions, orientation and shard completion hashes are retained under `results/<model>/` and `manifests/`. Raw masked ALT-minus-REF and discovery-cohort minor-minus-major LLRs are separate; magnitude is used for ranking. GPN follows its official center convention.",
    "",
    "nemo uses the released B. napus v1.0 prediction bundle, explicitly as transfer to Arabidopsis. Its CDS-masked 6200-base promoter/terminator inputs follow transcript orientation and deterministic longest-mRNA selection. Scores are unrounded paired ALT-minus-REF log10(TPM+0.1). Masked and outside-window SNPs remain uncovered. Ortholog/gene-family overlap and exact training partitions are unresolved, so no independent-family generalization claim is made.",
    "",
    "Botanic1-S (`813d2d46ca40ab5cd91a6acaa8852d636224992e`) is gated; existing-access check returned HTTP 401 and there was no saved Hugging Face credential. No terms were accepted on the user’s behalf. AlphaGenome has no configured authorized API key. Its frozen subset comprises "
    + ", ".join(json.loads((run / "inputs/frozen/protocol.json").read_text())["alpha_subset"])
    + ". Both unavailable models have no fabricated or substituted score columns.",
    "",
    "## Top-five results",
    "",
    "The primary nucleotide endpoint is not estimable (0 loci), so the figure explicitly displays that absence. It is not a zero-effect estimate. The secondary endpoint targets the lowest-p held-out mapping SNP within the discovery-frozen set, conditional on held-out Bonferroni association. It is same-study sample holdout, not independent-study replication and not a causal target.",
    "",
    "| Model / coverage | Loci | Annotation baseline | Model | Paired change, percentage points (95% interval) |",
    "|---|---:|---:|---:|---:|",
]
for r in s["metrics"]:
    if (
        r["endpoint"] == "heldout_mapping_signal"
        and r["ld_stratum"] == "all"
        and r["delta"] is not None
    ):
        lo, hi = r["ci95"]
        lines.append(
            f"| {r['model']} / {r['scope']} | {r['loci']} | {r['baseline']:.1%} | {r['model_recovery']:.1%} | {100 * r['delta']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |"
        )
lines += [
    "",
    "See `results/analysis/metrics.json`, `per_locus_recovery.tsv` and `coverage.tsv` for exact paired denominators and uncertainty, including LD strata and the shared subset. Intervals resample chromosomes intact (2,000 fixed-seed replicates); only five blocks are available and intervals are coarse. No causal-SNP hypothesis test is claimed.",
    "",
    "![Top-five recovery](results/analysis/top5_recovery.png)",
    "",
    "## Worked locus",
    "",
    f"The worked locus was fixed as the lexicographically first selected gene, **{s['worked_locus']}**, before inspecting model gains. It is an association-signal example, not a validated causal SNP case.",
    "",
    "| Ranking | Top five SNPs | Newly included versus annotation baseline |",
    "|---|---|---|",
]
base = set(worked.loc[worked.ranking_model == "annotations", "variant_id"])
for name, g in worked.groupby("ranking_model", sort=True):
    ordered = g.sort_values("rank").variant_id.tolist()
    lines.append(
        f"| {name} | "
        + ", ".join(ordered)
        + " | "
        + (", ".join(v for v in ordered if v not in base) or "none")
        + " |"
    )
lines += [
    "",
    "Coverage differs for nemo, so its shortlist comparison is valid only against its corresponding covered-candidate baseline, available in the rankings table.",
    "",
    "## Reproduction and handover",
    "",
    "Use the absolute commands in `manifests/production_submissions.json`, `mig_submissions.json` and `scripts/*.sbatch`. The initial full-GPU arrays were cancelled while pending; smoke-sized generic-GPU arrays replaced them. Completed model shards are hash-checked and reusable. Input verification and production logs remain intact.",
    "",
    "The first nemo technical attempt failed before model loading because of an unnecessary SciPy import; the import dependency was corrected. The initial GPN technical attempt failed before loading because the existing torchvision stack lacked `torchvision::nms`; an isolated compatible environment passed. Original environments were preserved. Exact commands/errors and all job IDs are in `manifests/run_overview.json` and Slurm logs.",
    "",
    "A portable bundle contains frozen inputs, scores, implementation, a manifest and `reanalyze.ipynb`. The notebook verifies every hash and pins the verified v0.3.0 base plus local implementation commit. It runs the analysis on CPU without model dependencies or network access. Publication and a v0.4 release tag are deferred.",
    "",
    "Required next scientific step: obtain sufficiently resolved independently supported regulatory SNP loci with usable matched genotypes and linked candidates. Review the 13 excluded reporter alleles before changing ascertainment. Any revised frequency/LD/candidate policy requires a new benchmark freeze and must not be optimized against these pilot model scores. Existing Botanic1-S authorization and an AlphaGenome API key are also required to complete those comparisons.",
    "",
]

lines += ["## Reporter-supported alleles outside the linked benchmark", ""]
for gene in ["AT1G32360", "AT2G19310"]:
    row = exclusions[exclusions.gene == gene].iloc[0]
    lines += [
        f"{gene}: `{row.variant_id}` has discovery ALT frequency {row.af_alt:.3f}, call rate {row.call_rate:.3f}, and r² {row.ld_r2_lead:.4f} with the discovery lead. Reporter activity differences are {row.effect_35S:+.2f} and {row.effect_pTRP1:+.2f}. It is excluded for `{row.exclusion}`. These assayed alleles cannot be counted as top-five failures or successes when absent from the frozen linked candidate sets.",
        "",
    ]
direction = pd.read_csv(analysis / "nemo_direction.tsv", sep="\t")
lines += [
    "## nemo expression direction",
    "",
    f"Among {len(direction)} covered held-out lead SNPs with locus-level held-out association, nemo transfer agrees in sign with the ALT-aligned expression coefficient at {int(direction.sign_agreement.sum()) if len(direction) else 0}. This descriptive direction check is separate from magnitude ranking; tagging associations are not isolated nucleotide effects. See `results/analysis/nemo_direction.tsv`.",
    "",
]

(run / "REPORT.md").write_text("\n".join(lines))
print(run / "REPORT.md")
