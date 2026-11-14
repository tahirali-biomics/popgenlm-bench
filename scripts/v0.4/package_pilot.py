"""Build a portable CPU reanalysis bundle; no network, inference, or publishing."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
run = Path(a.run)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
work = Path(__file__).resolve().parents[2]
for rel in ["inputs", "results", "manifests"]:
    shutil.copytree(run / rel, out / rel, dirs_exist_ok=True)
shutil.copytree(work / "scripts/v0.4", out / "implementation/scripts/v0.4", dirs_exist_ok=True)
shutil.copytree(
    work / "src/popgenlm",
    out / "implementation/src/popgenlm",
    dirs_exist_ok=True,
    ignore=shutil.ignore_patterns("__pycache__"),
)
revision = subprocess.check_output(["git", "-C", str(work), "rev-parse", "HEAD"], text=True).strip()
manifest = {
    str(p.relative_to(out)): {
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    for p in sorted(out.rglob("*"))
    if p.is_file()
    and p.name not in ["bundle_manifest.json", "reanalyze.ipynb"]
    and "results/analysis/" not in str(p.relative_to(out))
}
(out / "bundle_manifest.json").write_text(
    json.dumps(
        {
            "base_release": "v0.3.0",
            "base_commit": "4c4258f4a8014d4ee2b19e32329362baf34c44a0",
            "implementation_commit": revision,
            "files": manifest,
        },
        indent=2,
    )
    + "\n"
)
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PopGenLM v0.4 pilot CPU reanalysis\n",
            "Review bundle; publication and v0.4 tag pending. Pins the verified v0.3.0 base, implementation commit and every input/score hash. No model dependencies, VCF processing, downloads or GPU inference.",
        ],
    }
]
code = """from pathlib import Path
import hashlib, json, subprocess, sys
bundle = Path.cwd()  # directory containing this notebook
manifest = json.loads((bundle / 'bundle_manifest.json').read_text())
assert manifest['base_commit'] == '4c4258f4a8014d4ee2b19e32329362baf34c44a0'
for name, metadata in manifest['files'].items():
    assert hashlib.sha256((bundle/name).read_bytes()).hexdigest() == metadata['sha256'], name
print('Verified implementation:', manifest['implementation_commit'])
subprocess.run([sys.executable, str(bundle/'implementation/scripts/v0.4/analyze_pilot.py'), '--run', str(bundle)], check=True)
import pandas as pd
from IPython.display import display, Image
display(pd.read_csv(bundle/'results/analysis/coverage.tsv', sep='\\t'))
display(pd.read_csv(bundle/'results/analysis/worked_locus_top5.tsv', sep='\\t'))
display(Image(filename=str(bundle/'results/analysis/top5_recovery.png')))
"""
cells.append(
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }
)
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(out / "reanalyze.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
print(out, len(manifest), "verified files")
