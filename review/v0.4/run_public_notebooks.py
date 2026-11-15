"""Execute proposed public notebooks against an isolated, checksum-verified archive."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(worktree, release):
    w, release = Path(worktree).resolve(), Path(release).resolve()
    archive = release / "popgenlm-v04-public-inputs.zip"
    dest = release / "executed"
    dest.mkdir(exist_ok=False)
    figures = release / "figures"
    figures.mkdir(exist_ok=False)
    expected = w.parents[1] / "reports/v0.4/review_20261115/figures/verified_counts.tsv"
    results = []
    for path in sorted((w / "notebooks/v0.4").glob("*.ipynb")):
        with tempfile.TemporaryDirectory(prefix="popgenlm-v04-public-", dir="/tmp") as temporary:
            nb = nbformat.read(path, as_version=4)
            km = KernelManager(kernel_name="python3")
            km.kernel_spec.argv = [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ]
            env = {
                key: value
                for key, value in os.environ.items()
                if not any(
                    x in key.upper()
                    for x in ["TOKEN", "SECRET", "API_KEY", "PASSWORD", "CREDENTIAL"]
                )
                and key not in ["PYTHONPATH", "POPGENLM_REVIEW_INPUTS"]
            }
            env["POPGENLM_PUBLIC_ARCHIVE"] = str(archive)
            env["PYTHONNOUSERSITE"] = "1"
            start = time.time()
            client = NotebookClient(
                nb, km=km, timeout=300, resources={"metadata": {"path": temporary}}
            )
            try:
                client.execute(env=env)
            finally:
                km.shutdown_kernel(now=True)
            focus = path.stem.removeprefix("popgenlm-v04-")
            out = Path(temporary) / ("outputs_" + focus)
            assert digest(out / "verified_counts.tsv") == digest(expected)
            assert not [
                o
                for c in nb.cells
                if c.cell_type == "code"
                for o in c.outputs
                if o.output_type == "error"
            ]
            nbformat.write(nb, dest / path.name)
            for f in out.iterdir():
                shutil.copy2(f, figures / f.name)
            results.append(
                {
                    "notebook": path.name,
                    "status": "PASS",
                    "seconds": round(time.time() - start, 2),
                    "counts_sha256": digest(out / "verified_counts.tsv"),
                    "source_sha256": digest(path),
                    "executed_sha256": digest(dest / path.name),
                }
            )
            print("PASS", path.name, flush=True)
    result = {
        "status": "PASS",
        "release": "v0.4.0",
        "local_execution": "new venv; system_site_packages=False; fresh CPU kernel and fresh temporary extraction per notebook",
        "python": sys.version,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "archive_sha256": digest(archive),
        "notebooks": results,
        "hosted_colab": {
            "executed": 0,
            "status": "Anonymous preview opens; Google sign-in required to connect runtime; not represented as hosted execution",
        },
        "raw_score_reconstruction": "non-Botanic only; Botanic published ranks/counts verified; frozen Botanic LLR figure displayed",
        "full_alpha_aggregation": "194 cached scalar scores reused; illustrative central track included; full per-SNP667-track array aggregation remains independently verified in original run, not recomputed from this reduced package",
    }
    (release / "cpu_verification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worktree", required=True)
    p.add_argument("--release", required=True)
    a = p.parse_args()
    execute(a.worktree, a.release)
