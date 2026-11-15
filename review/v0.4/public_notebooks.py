"""Build publication notebooks using the same evaluator and permitted input archive."""

import argparse
import json
from pathlib import Path

from make_notebooks import MODELS


def build(dest, archive_sha):
    dest = Path(dest)
    dest.mkdir(exist_ok=True, parents=True)
    for focus, (title, description) in MODELS.items():
        cells = []

        def md(text, cells=cells):
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

        def code(text, cells=cells):
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": text.splitlines(True),
                    "execution_count": None,
                    "outputs": [],
                }
            )

        md(
            f"# PopGenLM Bench v0.4.0 · {title}\n\n{description}\n\n**Held-out association-signal recovery. Causal-SNP recovery remains unestimable.** Signed original-locus rules are exploratory. All plots are PopGenLM adaptations from cached inference, not source-data redraws or inference reruns.\n\n[Release and citations](https://github.com/tahirali-biomics/popgenlm-bench/blob/v0.4.0/docs/v0.4_release.md) · [Data terms](https://github.com/tahirali-biomics/popgenlm-bench/blob/v0.4.0/docs/v0.4_data_terms.md). Code is Apache-2.0; model-derived data retain their provider terms."
        )
        md(
            "## CPU reproduction\n\nRun all cells with NumPy, pandas, SciPy and Matplotlib. Colab normally provides these; a clean local environment can use `requirements-v04-public-cpu.txt`. The notebook downloads the **public v0.4.0 input archive**, verifies its pinned digest and every internal file, and uses one shared evaluator. No model weights, credentials or RAMSES paths are needed.\n\n**Scope:** reconstruct non-Botanic candidate ranks from cached scores; verify every model’s published rank and paired-outcome arithmetic. Botanic raw scores are omitted under its transfer restrictions, so its frozen LLR/context figure is displayed rather than regenerated from omitted scores. AlphaGenome reproduces score-based ranks and the illustrative cached track, not all194 full667-track aggregation requests. Native nemo and Botanic ATAC reproduction remain blocked. Read `DATA_TERMS.md` before reusing model-derived data."
        )
        code(
            """from pathlib import Path
import os, sys, hashlib, urllib.request, zipfile
ROOT = Path('public_inputs').resolve()
ARCHIVE = Path(os.environ.get('POPGENLM_PUBLIC_ARCHIVE', 'popgenlm-v04-public-inputs.zip'))
URL = 'https://github.com/tahirali-biomics/popgenlm-bench/releases/download/v0.4.0/popgenlm-v04-public-inputs.zip'
EXPECTED = ARCHIVE_SHA
if not ARCHIVE.exists():
    urllib.request.urlretrieve(URL, ARCHIVE)
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == EXPECTED, 'Archive checksum mismatch'
if not ROOT.exists():
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            assert ROOT in (ROOT / member.filename).resolve().parents, 'Unsafe archive path'
        archive.extractall(ROOT)
sys.path.insert(0, str(ROOT))
import review
manifest = review.verify(ROOT)
print('Verified', len(manifest['files']), 'files for', manifest['release'])
print(manifest['raw_reconstruction'])
""".replace("ARCHIVE_SHA", repr(archive_sha))
        )
        code(
            f"OUT = Path('outputs_{focus}')\ncounts = review.run(ROOT, OUT, '{focus}')\nprint(counts.query(\"scope == 'model_covered'\").to_string(index=False))\nfrom IPython.display import display, Image\nfor figure in sorted(OUT.glob('*.png')):\n    display(Image(filename=str(figure)))\n"
        )
        code(
            'data = review.load(ROOT)\nprint(data[\'per_locus_outcomes\'].query("scope == \'alpha_MLM_shared\' and method == \'alphagenome_abs\'")[["gene","baseline_rank","target_rank","baseline_hit","model_hit"]].to_string(index=False))\nprint(\'nemo B. napus transfer: 8/14 versus 8/14 on matched coverage; not native A. thaliana validation\')\n'
        )
        md(
            "## Optional separate inputs and inference — not run\n\nThe public analysis is complete above. Exact Botanic checkpoint, revision, scoring recipe and expected separately obtained score-table digest are in `restricted_inputs.json`; `review.load_botanic(ROOT, external_table)` rejects mismatching bytes before full raw reconstruction. No public endpoint for the withheld original raw outputs is claimed. See the release acquisition record for original reference/track hashes. New API/GPU inference can differ from cached output and is not needed here.\n\n## Interpretation\n\nCandidate identities are checked within each paired scope. The same-study holdout does not supply causal nucleotide truth. AlphaGenome’s four-locus subset is separate from the34-locus panel. Expression-by-genotype summaries are observational and can reflect LD or population structure. The exact original PlantCAD checkpoint is l32, despite the historical PlantCAD-S alias.\n\nBotanic attribution: **This work uses the Botanic™ model developed by Living Models, available under the Botanic™ Model License.** Research use only, as-is; provider license/disclaimer in the bundle. AlphaGenome outputs/derivatives are subject to the **AlphaGenome Output Terms of Use**, linked in DATA_TERMS.md; research restrictions and prohibited-training rules apply. Figures are scholarly adaptations from cached model outputs; none copies source paper artwork."
        )
        for i, c in enumerate(cells):
            c["id"] = f"{focus}-{i:02}"
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3 (CPU)",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
                "colab": {"name": f"popgenlm-v04-{focus}.ipynb"},
            },
            "cells": cells,
        }
        (dest / f"popgenlm-v04-{focus}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest", required=True)
    p.add_argument("--archive-sha", required=True)
    a = p.parse_args()
    build(a.dest, a.archive_sha)
