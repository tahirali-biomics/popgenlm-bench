"""Create eight notebooks using one checksum-verified shared implementation."""

import argparse
import json
from pathlib import Path

MODELS = {
    "overview": (
        "Regulatory variant prioritization",
        "Does sequence evidence improve recovery of held-out association signals? Start with 34 paired loci; keep the four AlphaGenome outcomes and 14-locus nemo transfer separate. Inspect the gain AT1G01690 (rank 6 → 4) and loss AT4G02405 (3 → 11).",
    ),
    "phylop": (
        "PhyloP: site conservation",
        "Signed site-level conservation is not an allele effect. Positive values indicate conservation; negative values indicate acceleration relative to the source model. Inspect v0.4 site coverage and the retained v0.3 9,453-site reference. No standalone regulatory function or causal-SNP claim follows.",
    ),
    "gpn": (
        "GPN Brassicales",
        "songlab/gpn-brassicales @ eb9c35d0d18571abe84390d22e74f2b21d319ce3; A. thaliana plus seven other Brassicales. Inspect masked sequence likelihood at 512 bp, all 1,205 candidates, v0.3 population-oriented concordance, and paired 17/34 vs 17/34.",
    ),
    "plantcad": (
        "Original PlantCAD (legacy PlantCAD-S label)",
        "The verified checkpoint is kuleshov-group/PlantCaduceus_l32 @ e624c13c3d35415348b854c87a218893b23564f7, not a verified Small checkpoint. Preserve its 16-angiosperm sequence-likelihood evidence at 512 bp, 1,205-candidate coverage, v0.3 concordance and 17/34 vs 17/34 recovery.",
    ),
    "plantcad2": (
        "PlantCAD2-S",
        "kuleshov-group/PlantCAD2-Small-l24-d0768 @ f756c255cb76e9f538c3acec04acf4214ed03fb3. Inspect 512-bp and 4-kb masked LLRs, context sensitivity and per-locus rank changes. Both magnitude rules recover 18/34 vs 17/34. Signed 22/34 vs 17/34 on original loci is exploratory.",
    ),
    "botanic": (
        "Botanic1-S",
        "living-models/Botanic1-S @ 813d2d46ca40ab5cd91a6acaa8852d636224992e. Complete PopGenLM 512-bp versus 4-kb zero-shot comparison: 17/34 and 19/34 vs 17/34. This is a PopGenLM adaptation, not ATAC reproduction. Barozet et al. (2026), bioRxiv v3 Fig. 4b/4d (DOI 10.64898/2026.09.04.749355) remain a separate blocked ATAC task: obtain verified chromosome-4 peaks, accessibility measurements, split, fine-tuned ATAC head or official source predictions and ChromBPNet before redrawing. Full v3 caption could not be independently fetched (403); do not substitute v1/v2 numbering.",
    ),
    "nemo": (
        "nemo: B. napus transfer",
        "Rockenbach et al. (2026), Fig. 3a, DOI 10.1093/plphys/kiag473 motivates allele-aligned expression contrasts. This PopGenLM adaptation uses cached B. napus v1.0 weights (SHA256 85bec6231833d41a432e7be99e6cf3a109ac275453017062a64d990fbaad374c), source c8deaaf18a8bd174300bf40e11fa2c69c0d6bb4e. Show predicted ALT-minus-REF expression by promoter/terminator position and 8/14 vs 8/14 matched recovery. Native A. thaliana family-safe weights/folds, reconciled 54/56/57 condition inputs, processed Express617 joint-family splits and scalers remain missing. These predictions do not validate native expression direction.",
    ),
    "alphagenome": (
        "AlphaGenome: mammalian-to-plant transfer",
        "PopGenLM adaptation of Avsec et al. (2026), Fig. 4b (paired allele tracks) and Fig. 6b (ALT-minus-REF), DOI 10.1038/s41586-025-10014-0. Cached API v1 ALL_FOLDS; SDK 0.11.0; HOMO_SAPIENS RNA_SEQ; 131,072-bp inputs, central 2,048-bp 1-bp predictions. Display track index 0 by metadata order (neuronal stem cell, plus-strand polyA plus RNA-seq), selected without effect sizes. Ranking uses all 667 frozen tracks, not this illustrative track. Observed GSE80744 Arabidopsis gene expression by genotype is separate, unadjusted, and not RNA-seq coverage or calibrated mammalian-track validation. Four eligible loci recover 2/4 vs 1/4; AT2G22240 is illustrative but ineligible.",
    ),
}


def create(dest, archive_sha):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for focus, (title, description) in MODELS.items():
        cells = []

        def md(s, cells=cells):
            cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)})

        def code(s, cells=cells):
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": s.splitlines(True),
                }
            )

        md(
            f"# PopGenLM Bench v0.4 · {title}\n\n{description}\n\n**Endpoint: held-out association-signal recovery. Independent causal-SNP recovery is unestimable (zero eligible loci).** Signed analyses on the original loci remain exploratory. All figures here are **PopGenLM adaptations from cached inference**, not source-data redraws or new inference reruns.\n\nCPU only: Python 3.10+, NumPy, pandas, SciPy, Matplotlib. No credentials, model weights, or RAMSES paths. The same `review.py` verifies inputs and reconstructs ranks in every notebook. The package data retain their source terms; see `RESTRICTED_USE.md`. Public redistribution is pending review."
        )
        md(
            "## Input setup\n\nUse a lawfully obtained review archive, `popgenlm-v04-review-inputs.zip`. The public code does not grant rights to restricted model outputs. Put the extracted package beside this notebook, set `POPGENLM_REVIEW_INPUTS`, or upload the archive in Colab. The archive SHA-256 is pinned below; every internal file is then verified before use. No public download URL is claimed before release. For dependencies see `requirements-cpu.txt`; this notebook does not install packages automatically."
        )
        code(
            """from pathlib import Path
import os, sys, hashlib, zipfile
ROOT = Path(os.environ.get('POPGENLM_REVIEW_INPUTS', 'review_inputs')).resolve()
if not (ROOT / 'manifest.json').exists():
    archive = Path('popgenlm-v04-review-inputs.zip')
    if not archive.exists():
        from google.colab import files
        files.upload()  # user obtains permitted inputs under source terms
    expected = ARCHIVE_SHA
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected, 'Archive checksum mismatch'
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (ROOT / member.filename).resolve()
            assert ROOT in target.parents, 'Unsafe archive path'
        ROOT.mkdir(parents=True, exist_ok=False)
        z.extractall(ROOT)
sys.path.insert(0, str(ROOT))
import review
manifest = review.verify(ROOT)
print('Verified', len(manifest['files']), 'inputs; source revision', manifest['source_revision'])
""".replace("ARCHIVE_SHA", repr(archive_sha))
        )
        md(
            "## Recompute and inspect\n\nReconstruct each candidate rank from raw cached signed scores and fixed baseline coefficients, check the frozen targets and eligibility, reconstruct every gain/loss count, then export SVG and PNG. Missing PhyloP uses the frozen baseline convention `percentile(fillna(0))`; it is not treated as observed conservation. Candidate sets are checked by SNP identity for every shared scope."
        )
        code(
            f"OUT = Path('outputs_{focus}')\ncounts = review.run(ROOT, OUT, focus='{focus}')\nprint(counts[counts.scope == 'model_covered'].to_string(index=False))\nfrom IPython.display import display, Image\nfor figure in sorted(OUT.glob('*.png')):\n    display(Image(filename=str(figure)))\n"
        )
        if focus in ["overview", "plantcad2", "botanic", "alphagenome", "nemo"]:
            code(
                'data = review.load(ROOT)\nprint(data[\'per_locus_outcomes\'].query("scope == \'alpha_MLM_shared\' and method == \'alphagenome_abs\'")[["gene", "target", "baseline_rank", "target_rank", "baseline_hit", "model_hit"]].to_string(index=False))\nprint(\'nemo transfer matched result: 8/14 vs 8/14; no native family-heldout claim\')\n'
            )
        md(
            "## Optional model inference (not executed)\n\nThe CPU analysis is complete above. To rerun inference, obtain the exact checkpoint revision and reference inputs in the release evidence table, verify weight/reference hashes, and use a separate model environment. Run the corresponding `scripts/v0.4/score_*.py --help` from the pinned source. These production scripts currently include RAMSES source locations and require adaptation; this notebook makes no claim of portable GPU/API inference. Never replace raw scores with transformed ranks. Preserve REF/ALT, strand, context, request and SLURM metadata. No installation, weight download, inference request or job submission occurs here.\n\n## Interpretation\n\nThe held-out accessions come from the same study. LD, ascertainment, population structure and coverage affect recovery. Four AlphaGenome loci are not pooled with 34-locus model results. No test of causal-nucleotide recovery is possible with the current independent evidence inventory. Full-paper ATAC panels and native nemo validation remain blocked as documented in the release panel ledger."
        )
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3 (CPU)",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.10"},
                "colab": {"name": f"popgenlm-v04-{focus}.ipynb"},
            },
            "cells": cells,
        }
        for i, cell in enumerate(cells):
            cell["id"] = f"{focus}-{i:02d}"
        (dest / f"popgenlm-v04-{focus}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest", required=True)
    p.add_argument("--archive-sha", required=True)
    a = p.parse_args()
    create(a.dest, a.archive_sha)
