"""Local restricted-use review report and offline raw-score CPU notebook."""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table(df):
    return (
        "| "
        + " | ".join(df.columns)
        + " |\n| "
        + " | ".join(["---"] * len(df.columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False, name=None)
        )
    )


def package(run, pilot, worktree, bundle):
    run, pilot, worktree, bundle = map(Path, [run, pilot, worktree, bundle])
    out = run / "results/analysis"
    m = pd.read_csv(out / "paired_metrics.tsv", sep="\t")
    cov = pd.read_csv(out / "coverage.tsv", sep="\t")
    ranks = pd.read_csv(out / "rankings.tsv", sep="\t")
    protocol = json.loads((run / "inputs/model_protocol.json").read_text())
    track = json.loads((run / "inputs/alphagenome_freeze.json").read_text())
    tables = []
    for scope in ["model_covered", "alpha_MLM_shared", "legacy_nemo_shared", "all_models_shared"]:
        z = m[(m.scope == scope) & (m.LD_stratum == "all")][
            [
                "method",
                "model_hits",
                "baseline_hits",
                "loci",
                "blocks",
                "into_top5",
                "out_of_top5",
                "stayed_in",
                "stayed_out",
                "delta",
                "ci95",
            ]
        ]
        z = z.copy()
        z["delta_pp"] = z.pop("delta").map(lambda value: f"{100 * value:+.1f}")
        z["CI95_pp"] = z.pop("ci95").map(
            lambda value: ", ".join(f"{100 * v:.1f}" for v in ast.literal_eval(value))
        )
        tables.append("### " + scope + "\n\n" + table(z))
    direct = pd.read_csv(out / "direct_comparisons.tsv", sep="\t")
    direct_display = direct[
        [
            "challenger",
            "comparator",
            "model_hits",
            "baseline_hits",
            "loci",
            "into_top5",
            "out_of_top5",
        ]
    ]
    concordance = pd.read_csv(out / "score_concordance.tsv", sep="\t")
    concordance_display = concordance[concordance.gene == "all_SNPs"].round(3)
    worked = []
    for gene in ["AT1G01690", min(protocol["alpha_subset"])]:
        for method in [
            "annotations",
            "plantcad2_s_512_abs",
            "plantcad2_s_4096_abs",
            "plantcad2_s_512_signed_exploratory",
            "botanic1_s_512_abs",
            "botanic1_s_4096_abs",
            "botanic1_s_512_signed_exploratory",
            "alphagenome_abs",
        ]:
            z = ranks[
                (ranks.scope == "model_covered") & (ranks.gene == gene) & (ranks.method == method)
            ].sort_values("rank")
            if len(z) == 0:
                continue
            target = z[z.target]
            worked.append(
                {
                    "gene": gene,
                    "method": method,
                    "target_rank": int(target["rank"].iloc[0]) if len(target) else "uncovered",
                    "top5": ", ".join(z.head(5).variant_id),
                }
            )
    pd.DataFrame(worked).to_csv(out / "worked_shortlists.tsv", sep="\t", index=False)
    restriction = """# Restricted-use local research bundle

Local RAMSES review only. Model-derived scores are NOT relicensed under the repository's Apache-2.0 code license.

Botanic1-S is developed by Darween Bioscience SAS / Living Models. This work uses the Botanic™ model developed by Living Models, available under the Botanic™ Model License. Full pinned terms are in licenses/Botanic1-LICENSE. Scores are modified only by forward/RC averaging, absolute value/percentile transformation and fixed baseline combination, as recorded in the protocol. The model is provided as-is, without warranties. Research-only output sharing requires attribution/citation and license/disclaimer notices; broad non-transferability language must be reconciled before public redistribution. No weights or vendor inference code are included in this bundle.

AlphaGenome direct API outputs are governed by AlphaGenome Terms of Service, not the SDK's Apache code license. Noncommercial research only; no use to train other machine learning models. Consult https://www.alphagenomedocs.com/faqs.html and the linked Terms before any public redistribution.

This bundle has not been approved or prepared as an unrestricted public release. No publication, push or tagging was performed.
"""
    report = f"""# PopGenLM v0.4: Botanic1-S and AlphaGenome extension

Local RAMSES review. Endpoint throughout is **held-out association-signal recovery**, not causal-SNP recovery. The independent nucleotide endpoint remains **unestimable (zero eligible loci)**. Original frozen candidates, holdout labels and conventional baseline are unchanged.

## Results

{chr(10).join(tables)}

`model_covered` compares every method with a baseline reranked on its identical covered candidates. `alpha_MLM_shared` uses the predefined AlphaGenome candidates and reranks each method/baseline on the same SNPs; nemo is excluded from this scope so all displayed methods use exactly the same candidates. `legacy_nemo_shared` preserves the original 507-SNP shared universe and 14 eligible loci. `all_models_shared` restricts every method to the intersection including nemo and AlphaGenome. Exact candidate IDs/counts, target ranks, gains, losses, retained hits/misses and LD strata are in `rankings.tsv`, `per_locus_outcomes.tsv` and `paired_metrics.tsv`. Uncertainty is a 2,000-replicate paired whole-chromosome bootstrap (seed 20261114), with only the stated number of chromosome blocks; small-subset intervals are descriptive and coarse. No weights were fitted, and no hypothesis test is claimed.

The carried-forward PlantCAD2-S signed 512bp result is asserted exactly at **22/34 versus 17/34**, five gains/zero losses. Its old shared-coverage result remains **8/14 versus 8/14**, as do saved B. napus nemo transfer scores. Native A. thaliana nemo remains unavailable because family-safe weights/folds and reconciled training inputs are missing. Signed-rule results on the original 40 loci remain exploratory; no independent validation is claimed.

![Paired results](results/analysis/model_comparison.png)

## Direct matched model comparison

{table(direct_display)}

In this table baseline_hits is the named PlantCAD2 comparator's recovery, not the annotation baseline. Exact paired intervals and per-locus outcomes are in direct_comparisons.tsv and direct_per_locus.tsv; every row uses the same SNPs and loci for both models.

{table(concordance_display)}

Absolute-LLR correlations are descriptive across the same SNPs at each context; per-locus correlations are also saved. They do not establish regulatory function.

## Coverage

{table(cov)}

All 40 loci and 1,205 original SNPs are preserved. AlphaGenome's subset was selected before pilot scores: {", ".join(protocol["alpha_subset"])}; it contains 194 SNPs. Coverage differences never redefine the full-panel held-out target or its p<.05/full candidate count eligibility threshold. At least six covered candidates are required for a top-five ranking comparison.

## Methods frozen before new scores

Conventional baseline: association+LD = .9 discovery association percentile + .1 lead r²; annotations = .8 association+LD + .1 exp(−|TSS distance|/2000) + .05 PhyloP percentile (missing→0) + .05 non-CDS. Each model adds .2 within-locus score percentile to .8 annotation baseline, with variant-ID tie breaking. Magnitude rules use |LLR| for genomic MLMs and |predicted expression delta| for nemo. Signed Botanic 512bp LLR uses the exact fixed .2 exploratory rule carried forward from PlantCAD2-S, prespecified in `inputs/model_protocol.json` before Botanic score inspection.

Botanic1-S: `living-models/Botanic1-S`, revision `813d2d46ca40ab5cd91a6acaa8852d636224992e`. All assets/weights are hash-recorded. Scores are masked logP(ALT)−logP(REF), averaged across forward/RC inputs; raw strand scores, signed REF→ALT score, population minor-minus-major orientation and |LLR| are saved. Same genomic bases/SNPs and 512/4096bp contexts as PlantCAD2-S; Botanic additionally requires its checkpoint's CLS token (513/4097 model tokens), with focal offset +1. No padding/truncation is used. Independent three-SNP TAIR10 fixtures test REF identity, exactly one mask, token offsets, finite scores, logit/log-probability identity, batch/single parity, official helper agreement and RC-order invariance. Pure-PyTorch inference avoids optional native-kernel dependencies. Botanic's two strands need not have equal raw logits; averaging gives a strand-invariant aggregate.

AlphaGenome: explicit **AlphaGenome 1 API v1 / ALL_FOLDS**, pinned SDK `0.11.0`, organism HOMO_SAPIENS, output RNA_SEQ, all **{track["tracks"]} tracks** with exact metadata SHA-256 `{track["tracks_sha256"]}` frozen before predictions. This retains the original all-track protocol instead of choosing tracks from observed effects. Each SNP uses paired 131,072bp public TAIR10 sequences with identical flanks and only the focal nucleotide changed. Score = mean absolute difference of log1p predicted RNA-seq values over the central 2,048bp and all tracks. Outputs have 1bp resolution. Requests cache exact central prediction arrays, sequence/specification hashes and response metadata per allele; complete requests are reused after interruptions. No whole-genome or personal data were transmitted. The service does not expose a cryptographic checkpoint-weight digest; the reproducible analysis uses frozen cached predictions and scores, alongside the explicit API version/enum and metadata snapshot. This is exploratory cross-species transfer from mammalian-trained outputs; magnitude has no plant expression-direction interpretation.

## Worked loci

The original AT1G01690 and lexicographically first predefined AlphaGenome gene were selected independently of new model gains. The AlphaGenome worked gene AT2G22240 does not meet the frozen held-out significance threshold, so its shortlist is illustrative and contributes no recovery outcome. The other four predefined genes supply the AlphaGenome endpoint denominator, across only two chromosome blocks.

{table(pd.DataFrame(worked))}

## Verification, environments and review

Independent verification reconstructs every rank and baseline from raw scores, recovers paired outcomes, validates all completion hashes, and recomputes all 194 AlphaGenome scores from paired cached arrays. Prior PlantCAD2-S ranks and original count assertions pass. Relevant tests require no GPUs, model downloads or credentials. Exact validation, code/environment/asset manifests and Slurm job IDs/exit states are retained under `manifests/` and `logs/`.

Botanic environment: `/projects/ag-demeaux/tali/PopGenLM/environments/v0.4/botanic1/bin/python`; isolated CUDA dependencies plus `runtime_paths.sh`. AlphaGenome environment: `/projects/ag-demeaux/tali/PopGenLM/environments/v0.4/alphagenome/bin/python`. CPU analysis: `/projects/ag-demeaux/tali/tools/gpn_env/bin/python`. Existing model environments/data were preserved. Early Botanic attempts stopped before checkpoint inference due CPU-only inherited Torch, a corrected source-edit syntax error, and CUDA runtime lookup/version conflicts. Exact commands/errors and resolution are retained in logs and `manifests/recovery.json`; successful smoke-sized jobs supersede them. No failed API prediction is silently discarded.

Private CPU bundle: `{bundle}`. It contains frozen candidates and scores, pinned implementation, exact expected counts/ranks, license notices and an executed notebook. The notebook verifies immutable hashes and reproduces numerical results without model weights, API access, credentials or GPU. Full paired API caches remain in `{run}/results/alphagenome/cache` outside Git. Reproduction:

```bash
python implementation/scripts/v0.4/integrate_models.py --run . --pilot pilot
```

[Botanic pinned license](https://huggingface.co/living-models/Botanic1-S/blob/813d2d46ca40ab5cd91a6acaa8852d636224992e/LICENSE), [AlphaGenome terms FAQ](https://www.alphagenomedocs.com/faqs.html). Both outputs carry research-use restrictions. Full Botanic terms, required attribution, modifications and disclaimer notice are in the private bundle; ambiguous redistribution conditions require review before any public bundle. No weights are packaged; no publication/tag/push was performed.
"""
    (run / "REPORT.md").write_text(report)
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        run / "inputs",
        bundle / "inputs",
        ignore=shutil.ignore_patterns("setup_models.py", "alphagenome_pypi.json"),
    )
    for folder in ["plantcad2_s", "plantcad_v1", "gpn_brassicales", "nemo_bnapus_transfer"]:
        shutil.copytree(pilot / "results" / folder, bundle / "pilot/results" / folder)
    shutil.copytree(pilot / "results/analysis", bundle / "pilot/results/analysis")
    shutil.copytree(run / "results/botanic1_s", bundle / "results/botanic1_s")
    shutil.copytree(
        run / "results/alphagenome",
        bundle / "results/alphagenome",
        ignore=shutil.ignore_patterns("cache", "*.jsonl"),
    )
    shutil.copytree(out, bundle / "results/analysis")
    shutil.copytree(
        worktree / "src",
        bundle / "implementation/src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    dest = bundle / "implementation/scripts/v0.4"
    dest.mkdir(parents=True)
    shutil.copy2(worktree / "scripts/v0.4/integrate_models.py", dest / "integrate_models.py")
    shutil.copytree(run / "manifests", bundle / "manifests")
    shutil.copy2(worktree / "requirements-v04-cpu.txt", bundle / "requirements-v04-cpu.txt")
    shutil.copy2(run / "REPORT.md", bundle / "REPORT.md")
    (bundle / "RESTRICTED_USE.md").write_text(restriction)
    (bundle / "licenses").mkdir()
    shutil.copy2(
        Path("/projects/ag-demeaux/tali/PopGenLM/models/botanic1")
        / protocol["botanic_revision"]
        / "LICENSE",
        bundle / "licenses/Botanic1-LICENSE",
    )
    (bundle / "expected").mkdir()
    outputs = [
        "metrics.json",
        "paired_metrics.tsv",
        "rankings.tsv",
        "per_locus_outcomes.tsv",
        "frozen_scores.tsv",
        "coverage.tsv",
        "worked_locus_top5.tsv",
        "direct_comparisons.tsv",
        "direct_per_locus.tsv",
        "score_concordance.tsv",
    ]
    for name in outputs:
        shutil.copy2(out / name, bundle / "expected" / name)
    files = {
        str(p.relative_to(bundle)): sha(p)
        for p in sorted(bundle.rglob("*"))
        if p.is_file() and not str(p.relative_to(bundle)).startswith("results/analysis/")
    }
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "files": files,
                "source_commit": subprocess.check_output(
                    ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
                ).strip(),
                "analysis_source_sha256": sha(dest / "integrate_models.py"),
                "restricted_local_review": True,
            },
            indent=2,
        )
    )
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# PopGenLM v0.4 model extension — local research review\nHeld-out association signals only; causal-SNP endpoint unestimable. See RESTRICTED_USE.md."
        ),
        nbf.v4.new_code_cell("""from pathlib import Path
import json,hashlib,subprocess,sys
b=Path.cwd();manifest=json.loads((b/'bundle_manifest.json').read_text())
for name,h in manifest['files'].items():
 assert hashlib.sha256((b/name).read_bytes()).hexdigest()==h,name
print('PASS immutable files:',len(manifest['files']))
subprocess.run([sys.executable,'-I','-B','implementation/scripts/v0.4/integrate_models.py','--run','.','--pilot','pilot'],check=True)
for name in ['metrics.json','paired_metrics.tsv','rankings.tsv','per_locus_outcomes.tsv','frozen_scores.tsv','coverage.tsv','worked_locus_top5.tsv','direct_comparisons.tsv','direct_per_locus.tsv','score_concordance.tsv']:
 assert (b/'results/analysis'/name).read_bytes()==(b/'expected'/name).read_bytes(),name
print('PASS exact numerical reproduction')"""),
        nbf.v4.new_code_cell("""import pandas as pd
from IPython.display import display,Image
m=pd.read_csv('results/analysis/paired_metrics.tsv',sep='\\t')
display(m[(m.LD_stratum=='all') & m.scope.isin(['model_covered','alpha_MLM_shared'])])
display(pd.read_csv('results/analysis/coverage.tsv',sep='\\t'))
display(Image(filename='results/analysis/model_comparison.png'))"""),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "PopGenLM CPU",
        "language": "python",
        "name": "popgenlm-v04",
    }
    nbf.write(nb, bundle / "reanalyze.ipynb")
    print("REPORT", run / "REPORT.md", "BUNDLE", bundle)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["run", "pilot", "worktree", "bundle"]:
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    package(a.run, a.pilot, a.worktree, a.bundle)
