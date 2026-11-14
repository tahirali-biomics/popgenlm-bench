"""Write the local review report and portable offline CPU reanalysis bundle."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table(frame):
    # Avoid an optional tabulate dependency in the CPU bundle.
    return (
        "| "
        + " | ".join(frame.columns)
        + " |\n| "
        + " | ".join(["---"] * len(frame.columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(str(x) for x in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        )
    )


def package(pilot, run, worktree, bundle):
    pilot, run, worktree, bundle = map(Path, [pilot, run, worktree, bundle])
    root = run.parents[2]
    metrics = pd.read_csv(run / "results/paired_metrics.tsv", sep="\t")
    original = pd.read_json(run / "results/reproduction.json")
    funnel = json.loads((run / "results/evidence_funnel.json").read_text())
    native = json.loads((run / "manifests/nemo_native_feasibility.json").read_text())
    cases = pd.read_csv(run / "results/independent_nucleotide_cases.tsv", sep="\t")
    pending = pd.DataFrame(
        [
            {
                "gene": "AT5G10140",
                "source": "https://doi.org/10.1101/gad.258814.115",
                "published_variant": "FLC SNP+259, G/T",
                "assay": "reciprocal site-directed mutagenesis of transgenic FLC; qPCR expression and COOLAIR splicing",
                "tissue": "transgenic Arabidopsis plants; tissue conditions require supplement audit",
                "build": "not reconciled to TAIR10.1",
                "ref": "",
                "alt": "",
                "support_level": "individual_nucleotide_expression",
                "exclusions": "exact coordinate and genomic allele orientation not confirmed; supplements returned HTML rather than DOCX/XLSX; not scored",
                "training_overlap": "unknown; genomic pretraining may include locus",
            },
            {
                "gene": "AT5G10140",
                "source": "https://doi.org/10.1016/j.cub.2023.08.079",
                "published_variant": "FLC SNP−230, G/A",
                "assay": "promoter variant manipulation; altered upstream TSS usage and expression",
                "tissue": "Arabidopsis; conditions pending full methods audit",
                "build": "not reconciled to TAIR10.1",
                "ref": "",
                "alt": "",
                "support_level": "individual_nucleotide_expression_source_only",
                "exclusions": "relative TSS notation insufficient coordinate proof; exact alleles/genotypes not confirmed; not scored",
                "training_overlap": "unknown; genomic pretraining may include locus",
            },
            {
                "gene": "AT4G00650",
                "source": "https://doi.org/10.1111/tpj.14716",
                "published_variant": "natural FRI promoter haplotypes",
                "assay": "native-promoter natural-allele transgenes with multiple promoter differences",
                "tissue": "10-day aerial seedlings",
                "build": "not harmonized",
                "ref": "",
                "alt": "",
                "support_level": "haplotype_cis",
                "exclusions": "multiple variants per construct; does not isolate an individual expression SNP",
                "training_overlap": "unknown",
            },
        ]
    )
    pd.concat([cases, pending], ignore_index=True).to_csv(
        run / "results/published_evidence_inventory.tsv", sep="\t", index=False
    )
    # Freeze the empty new evaluation set explicitly, preserving all exclusion cases.
    frozen = run / "inputs/independent_validation"
    frozen.mkdir(exist_ok=True)
    pd.DataFrame(columns=["gene", "block", "evidence_resolution"]).to_csv(
        frozen / "loci.tsv", sep="\t", index=False
    )
    pd.DataFrame(columns=["gene", "variant_id", "ref", "alt"]).to_csv(
        frozen / "candidates.tsv", sep="\t", index=False
    )
    shutil.copy2(run / "results/published_evidence_inventory.tsv", frozen / "exclusions.tsv")
    decision = {
        "status": "no eligible new nonoverlapping evaluation loci; no new model scores inspected",
        "n_loci": 0,
        "n_candidates": 0,
        "original_rule": "0.8 annotations + 0.2 within-locus percentile(abs(512bp raw ALT-minus-REF LLR))",
        "exploratory_challenger": "0.8 annotations + 0.2 within-locus percentile(signed 512bp raw ALT-minus-REF LLR)",
        "selection": "challenger identified from original 40 loci; all further testing must use new frozen loci; no fitted weights",
        "files": {p.name: sha(p) for p in sorted(frozen.glob("*.tsv"))},
    }
    (frozen / "freeze.json").write_text(json.dumps(decision, indent=2))
    results = metrics[(metrics.scope == "model_covered") & (metrics.stratum == "all")].copy()
    display = results[
        ["rule", "model_hits", "loci", "into_top5", "out_of_top5", "stayed_in", "stayed_out"]
    ]
    coverage = pd.read_csv(pilot / "results/analysis/coverage.tsv", sep="\t")
    rows = []
    for r in coverage.itertuples():
        m = original[(original.scope == "model_covered") & (original.model == r.model)].iloc[0]
        rows.append(
            {
                "model": r.model,
                "status": "reused; ranks verified",
                "SNPs": f"{r.candidate_pairs_covered}/1205",
                "loci": f"{r.loci_complete}/40 complete",
                "endpoint": f"held-out signal {int(m.model_hits)}/{int(m.loci)} vs {int(m.baseline_hits)}/{int(m.loci)}",
            }
        )
    rows += [
        {
            "model": "Botanic1-S",
            "status": "blocked: HF 401; no token",
            "SNPs": "0",
            "loci": "0",
            "endpoint": "not evaluated",
        },
        {
            "model": "native Arabidopsis nemo",
            "status": "blocked: native weights/folds absent; faithful retraining inputs unresolved",
            "SNPs": "0",
            "loci": "0",
            "endpoint": "not evaluated",
        },
        {
            "model": "AlphaGenome",
            "status": "blocked: no API key",
            "SNPs": "0",
            "loci": "0/5 predefined",
            "endpoint": "not evaluated",
        },
    ]
    pd.DataFrame(rows).to_csv(run / "results/model_comparison.tsv", sep="\t", index=False)
    worked = pd.read_csv(run / "results/worked_locus.tsv", sep="\t")
    shortlists = []
    for rule in ["baseline", "abs_512_w0.2", "abs_4096_w0.2", "signed_512_w0.2"]:
        z = worked[(worked.scope == "model_covered") & (worked.rule == rule)].sort_values("rank")
        target = z[z.target].iloc[0]
        shortlists.append(
            {
                "rule": rule,
                "target": target.variant_id,
                "target_rank": int(target["rank"]),
                "top5": ", ".join(z.head(5).variant_id),
            }
        )
    pd.DataFrame(shortlists).to_csv(
        run / "results/worked_locus_shortlists.tsv", sep="\t", index=False
    )
    context = pd.read_csv(run / "results/context_audit.tsv", sep="\t")
    report = f"""# PopGenLM Bench v0.4 follow-up — local review, 2026-11-14

PlantCAD2-S has an exploratory signed-score result worth validating: **22/34 (64.7%)** held-out association signals recovered versus **17/34 (50.0%)** for the conventional annotation baseline. Five loci move into the top five, none move out; 17 stay in and 12 stay out. This rule was evaluated on the original 40 loci and is not independently validated. The primary nucleotide endpoint remains **not estimable**, with zero eligible new loci. No new model inference or training was run.

## Reproduction of the original pilot

The v0.3.0 base is `4c4258f4a8014d4ee2b19e32329362baf34c44a0`; the main `repo/` checkout remains unchanged. The review worktree is `{worktree}`. The original report and executed notebook remain at `{pilot}/REPORT.md` and `{root}/reports/v0.4/regulatory_pilot_20261114/cpu_bundle/reanalyze.executed.ipynb`.

All ten frozen input hashes pass: 40 loci, 1,205 unique SNPs, 665 matched accessions split into 469 discovery and 196 holdout. Candidate SHA-256: `b4318ae403e413242a76f5807b0c9d7662ed0ec4e94623146e5d60b6793fe4eb`; locus SHA-256: `79708f8782c9e25d8a602272ce3630edcb67e5b61433a4fda8185f664c821314`. The full original freeze, crosswalk, genotype and holdout hashes are retained in the bundle. No association or holdout definitions changed. Targets are the lowest held-out p-value among each full frozen candidate set, eligible at p<0.05/full candidate count. Only 34/40 loci qualify. This is same-study held-out mapping, with related accessions, not independent-study replication or causal SNP truth.

Saved rank tables independently reproduce every original paired outcome. The 512bp magnitude rule has **18/34 vs 17/34**, with 2 gains, 1 loss, 16 retained hits and 15 retained misses. The 4096bp magnitude rule has **18/34 vs 17/34**, with 3 gains, 2 losses, 15 retained hits and 14 retained misses. Thus equal aggregate recovery conceals different loci. All original runnable models on the identical shared candidate subset recover **8/14**, with 0 gains, 0 losses, 8 retained hits and 6 retained misses. `results/reproduction.json` contains exact counts and intervals for every model/scope.

## Exploratory scoring changes on original loci

The pilot already used |LLR|. Raw score = log P(ALT)−log P(REF) at the single masked focal nucleotide, averaged across forward and reverse-complement inputs. ALT is the VCF ALT on the reference genomic strand, not an inferred ancestral or minor allele. Population minor-minus-major sign flips are stored separately; prioritization originally discarded this sign. REF is checked against the pinned TAIR10.1 FASTA; the focal index is length/2−1 and the reverse complement uses its mirrored index. Saved smoke manifests validate one mask, allele-token identity, finite logits, logit/log-probability equivalence, batch parity and strand agreement on separate fixture SNPs. Their hashes and all original score hashes are checked. Both contexts use PlantCAD2-S revision `f756c255cb76e9f538c3acec04acf4214ed03fb3` on all 1,205 identical SNPs. No new inference is needed.

Baseline: association+LD = .9 association percentile + .1 lead r²; annotation baseline = .8 of that + .1 exp(−|TSS distance|/2000) + .05 PhyloP percentile (missing→0) + .05 non-CDS. Follow-up = (1−w) baseline + w score percentile. Mean-context rule averages the two within-locus magnitude percentiles. Deterministic variant-ID ties and coverage-matched baselines are retained. All coefficients and transformations were listed in `inputs/followup_protocol.json` before follow-up outcomes were computed; none was fitted. This is an exploratory sweep, not a development/evaluation split.

{table(display)}

Signed 512bp LLR at w=.2 improves descriptive recovery by **14.7 percentage points**; its five-chromosome paired bootstrap interval is **[13.5,16.1] pp**. This interval conditions on the chosen rule and these ascertained loci, and does not account for choosing a rule among exploratory alternatives. A positive REF→ALT likelihood score has no general causal direction interpretation and is sensitive to reference-allele coding. Freeze this challenger and the original magnitude rule for future independent testing; do not claim validated superiority. The 4kb magnitude rule at w=.1 gives 19/34, but is also exploratory. Per-locus target ranks, paired transitions and LD strata are in `paired_locus_outcomes.tsv`, `rankings.tsv` and `paired_metrics.tsv`. Mean within-locus context magnitude Spearman correlation is {context.magnitude_spearman.mean():.3f}; full per-locus correlations and strand differences are saved.

![Paired exploratory results](results/paired_results.png)

The shared-subset exploratory rules are reranked from scratch on the same nemo-covered candidates, with full-panel target/significance definitions unchanged. Their exact counts and denominators are in `paired_metrics.tsv` under `shared_all_runnable`; the original 8/14 comparison is kept distinct.

## Model coverage and access

{table(pd.DataFrame(rows))}

All evaluated model evidence is held-out association signals; primary independent-nucleotide coverage is zero. nemo transfer coverage is 507/1,205 SNPs, with 435 CDS-masked and 263 outside input windows; no whole locus is fully covered. Its candidate-matched 14-locus comparison is unchanged at 8/14; directional agreement remains separately 9/20 covered held-out leads, not evidence of nucleotide causality.

Botanic1-S metadata pins revision `813d2d46ca40ab5cd91a6acaa8852d636224992e`. Existing access was checked: no token, gated endpoint HTTP 401. Public metadata declares the `livingmodels-research` noncommercial license; the actual LICENSE text is gated and must be reviewed before public packaging of any derived outputs. No terms were accepted and no Botanic scores are packaged. [Checkpoint metadata](https://huggingface.co/living-models/Botanic1-S).

Native nemo source and both upstream release endpoints were checked. The source revision remains `a2d4894f166c2a7e1fd6c9553de1393d7d37cd41`; neither release endpoint has assets. No native weights or per-gene family folds are in the checked source tree or local model bundle. The [published protocol](https://doi.org/10.1093/plphys/kiag473) uses joint CDS-family partitions and a historical E-CURD-1 expression matrix. The currently advertised matrix was recovered (30,504 genes, {native["numeric_expression_conditions"]} expression conditions; all 40 evaluation genes present). GSE80744 is absent from its study headers; this does not prove sample independence. Exact retained samples, joint-family partitions, scalers and processed Express617 inputs are unresolved. Required filenames and tool availability are in `manifests/nemo_native_feasibility.json`; arbitrary gene splits or a different expression matrix would not establish held-out family provenance. Native training was not submitted on unverified inputs. The B. napus model is not presented as native nemo.

AlphaGenome has no existing API key. The original five-locus subset and paired-sequence aggregation are retained; no predictions were requested. A concrete API model version and output-track metadata must be pinned after authorized access is available and before any prediction. Version/track selection remains a dependency, not a claimed completed specification.

## Independent validation audit and new held-out performance

Original assay extraction was independently rerun from the published workbook with a stricter all-hit uniqueness check, reproducing every mapped substitution. {funnel["modified_rows"]:,} modified rows: {funnel["mapping_reasons"]}. Thus 9,317 single substitutions yield 7,110 exact unique oligo mappings; 2,203 lack an exact reference match and 4 are ambiguous. Of mapped rows, 176 match exact biallelic VCF alleles and 6,934 do not; `vcf_allele_audit.tsv` distinguishes absent VCF positions, reference mismatches and ALT/biallelic mismatches. Only 13 natural substitutions across 11 genes pass the frozen two-promoter support heuristic; no per-variant assay uncertainty was available. These assays are nucleotide-level leaf-protoplast reporter effects, not endogenous rosette editing. [Voichek et al.](https://doi.org/10.1038/s41588-024-01907-3).

Ordered original-policy filter counts: 13 → 2 after MAF≥.05 → 1 after call≥.9 → 1 after eligible discovery locus → 0 after LD≥.6. Overlapping reasons are separately counted: 11 frequency failures, 5 call-rate failures, 9 locus failures and 4 LD failures. Four of the 13 have zero ALT carriers in discovery despite a VCF site record. Nine cases lie outside the old locus blocks. The revised rule requires exact natural alleles, ≥10 called carriers of each allele, call≥.9, no overlap within 50kb of old candidate windows, and an eligible linked cis signal. It was frozen before new-case inspection and yields **0 new loci**. No new model scores were generated. `inputs/independent_validation/freeze.json` records empty locus/candidate tables, exclusion hashes and future fixed rules. No weights were fitted on evaluation blocks.

`published_evidence_inventory.tsv` includes all 13 verified reporter cases and targeted literature leads. [FLC +259](https://doi.org/10.1101/gad.258814.115) has reciprocal single-base mutagenesis evidence; [FLC −230](https://doi.org/10.1016/j.cub.2023.08.079) has promoter-variant expression evidence. Their exact TAIR10.1 coordinates/alleles were not established here: relative names are not enough, and retrieved supplements were HTML rather than valid DOCX/XLSX. They remain unscored pending reconciliation. Natural FRI promoter haplotypes are recorded separately as haplotype evidence, not individual SNP truth. The inventory is targeted, not an exhaustive systematic review. Voichek study-level overlap with nemo's upstream regulatory-effect benchmark is flagged; MLM genomic pretraining exposure is expected, while exact allele-level benchmark/training overlap remains unresolved.

Consequently **new held-out nucleotide-level recovery is unestimable**. The original 40-locus rules and results are unchanged. Verified reporter cases are documented examples only. For example AT2G19310 `2:8370090:G:T` has two reporter effects −0.93/−0.78, call rate .942 and ALT AF .319, but LD r²=.0168 to its discovery lead; it cannot serve as truth for that frozen linked-candidate set.

## Worked original locus: AT1G01690

{table(pd.DataFrame(shortlists))}

This locus was selected lexicographically, before observed improvements. The table shows the exact held-out target rank and changing shortlist for each rule on the same 38 candidates. It illustrates ranking behavior, not causal nucleotide validation.

## Verification and reproduction

New run: `{run}`. CPU environment: `/projects/ag-demeaux/tali/tools/gpn_env/bin/python`; tests: `/projects/ag-demeaux/tali/tools/conda_envs/popgenlm_release_py311/bin/python`. Source/environment/model/input hashes and job accounting are retained. Original smoke/inference jobs are reused; this follow-up submits bounded CPU analysis, evidence and notebook jobs only. All jobs and exit states are recorded in `manifests/job_accounting.tsv`. Fast tests require no GPUs, weights or downloads. No new installations or inference failures occurred.

Portable bundle: `{bundle}`. The notebook verifies immutable file hashes, reruns the CPU analysis from bundled frozen inputs/raw scores, and compares exact numerical outputs with saved expectations. `reanalyze.executed.ipynb` contains the executed result. CPU bundle regeneration needs the original run and release worktree; reanalysis itself needs neither the full VCF nor model weights/network. Commands:

```bash
python implementation/scripts/v0.4/followup_analysis.py --pilot pilot --run .
```

For the score-blind full-VCF evidence audit on RAMSES:

```bash
/projects/ag-demeaux/tali/tools/gpn_env/bin/python -I -B {worktree}/scripts/v0.4/followup_evidence.py --pilot {pilot} --run {run}
```

Review blockers: authorized Botanic/API access; native nemo family-safe weights or verified training inputs; exact coordinate validation of additional independent loci. Further evidence ascertainment must remain independent of model gains. Local review only: no push, publication or tag.
"""
    (run / "REPORT.md").write_text(report)
    if not bundle.exists():
        bundle.mkdir(parents=True)
        source = root / "reports/v0.4/regulatory_pilot_20261114/cpu_bundle"
        shutil.copytree(source, bundle / "pilot", ignore=shutil.ignore_patterns("*.ipynb"))
        shutil.copytree(run / "inputs", bundle / "inputs")
        shutil.copytree(run / "manifests", bundle / "manifests")
        shutil.copytree(run / "results", bundle / "results")
        shutil.copytree(
            worktree / "src",
            bundle / "implementation/src",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        dest = bundle / "implementation/scripts/v0.4"
        dest.mkdir(parents=True)
        shutil.copy2(worktree / "scripts/v0.4/followup_analysis.py", dest / "followup_analysis.py")
        shutil.copy2(worktree / "requirements-v04-cpu.txt", bundle / "requirements-v04-cpu.txt")
        shutil.copy2(run / "REPORT.md", bundle / "REPORT.md")
        (bundle / "expected").mkdir()
        for name in [
            "metrics.json",
            "reproduction.json",
            "rankings.tsv",
            "paired_locus_outcomes.tsv",
            "paired_metrics.tsv",
            "context_audit.tsv",
            "worked_locus.tsv",
        ]:
            shutil.copy2(run / "results" / name, bundle / "expected" / name)
        files = {
            str(p.relative_to(bundle)): sha(p)
            for p in sorted(bundle.rglob("*"))
            if p.is_file() and not str(p.relative_to(bundle)).startswith("results/")
        }
        (bundle / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "files": files,
                    "implementation_revision": subprocess.check_output(
                        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "analysis_source_sha256": sha(dest / "followup_analysis.py"),
                },
                indent=2,
            )
        )
        import nbformat as nbf

        nb = nbf.v4.new_notebook()
        nb.cells = [
            nbf.v4.new_markdown_cell(
                "# PopGenLM v0.4 follow-up: offline CPU reanalysis\nReproduction and exploratory scoring are separate from unestimable independent validation. No inference/downloads."
            ),
            nbf.v4.new_code_cell("""from pathlib import Path
import json, hashlib, subprocess, sys
bundle=Path.cwd()
manifest=json.loads((bundle/'bundle_manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==expected,name
print('PASS immutable inputs, raw scores, source and expectations:',len(manifest['files']))
subprocess.run([sys.executable,'-I','-B','implementation/scripts/v0.4/followup_analysis.py','--pilot','pilot','--run','.'],check=True)
for name in ['metrics.json','reproduction.json','rankings.tsv','paired_locus_outcomes.tsv','paired_metrics.tsv','context_audit.tsv','worked_locus.tsv']:
    assert (bundle/'results'/name).read_bytes()==(bundle/'expected'/name).read_bytes(),name
print('PASS exact numerical reproduction')"""),
            nbf.v4.new_code_cell("""import pandas as pd
from IPython.display import display,Image
m=pd.read_csv('results/paired_metrics.tsv',sep='\\t')
display(m[m.stratum=='all'])
display(pd.read_csv('results/model_comparison.tsv',sep='\\t'))
display(pd.read_csv('results/worked_locus_shortlists.tsv',sep='\\t'))
print('Independent nucleotide primary endpoint: not estimable; 0 eligible new loci')
display(Image(filename='results/paired_results.png'))"""),
        ]
        nb.metadata["kernelspec"] = {
            "display_name": "PopGenLM v0.4 CPU",
            "language": "python",
            "name": "popgenlm-v04",
        }
        nbf.write(nb, bundle / "reanalyze.ipynb")
    print("REPORT", run / "REPORT.md")
    print("BUNDLE", bundle)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ["pilot", "run", "worktree", "bundle"]:
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    package(a.pilot, a.run, a.worktree, a.bundle)
