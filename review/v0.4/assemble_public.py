"""Create v0.4.0 public CPU views; preserve the private frozen review bundle."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(project, worktree, output):
    project, worktree, output = (Path(x).resolve() for x in [project, worktree, output])
    old = project / "reports/v0.4/review_20261115"
    output.mkdir(parents=True, exist_ok=False)
    for name, entry in json.loads((old / "manifest.json").read_text())["files"].items():
        assert sha(old / name) == entry["sha256"], name
    for folder in ["data", "provenance", "licenses"]:
        shutil.copytree(old / folder, output / folder)
    scores = pd.read_csv(old / "data/frozen_scores.tsv", sep="\t")
    botanic = ["gene", "variant_id", "botanic1_s_512", "botanic1_s_4096"]
    excluded = scores[botanic].to_csv(sep="\t", index=False).encode()
    scores.drop(columns=botanic[2:]).to_csv(
        output / "data/frozen_scores.tsv", sep="\t", index=False
    )
    (output / "restricted_inputs.json").write_text(
        json.dumps(
            {
                "botanic_scores.tsv": {
                    "sha256": hashlib.sha256(excluded).hexdigest(),
                    "columns": botanic,
                    "checkpoint": "living-models/Botanic1-S",
                    "revision": "813d2d46ca40ab5cd91a6acaa8852d636224992e",
                    "acquisition": "Obtain checkpoint from the pinned Hugging Face revision after accepting its terms. Use score_botanic.py with the pinned reference, protocol and frozen candidates. Convert shard raw_alt_minus_ref to the two named columns in original candidate order. Exact original cached bytes require licensor-authorized research sharing; no unrestricted download endpoint is claimed. Verify this digest before full raw-score reconstruction.",
                    "url": "https://huggingface.co/living-models/Botanic1-S/tree/813d2d46ca40ab5cd91a6acaa8852d636224992e",
                }
            },
            indent=2,
        )
        + "\n"
    )
    for file in ["review.py"]:
        shutil.copy2(worktree / "review/v0.4" / file, output / file)
    shutil.copy2(worktree / "src/popgenlm/regulatory.py", output / "frozen_regulatory.py")
    shutil.copy2(worktree / "requirements-v04-public-cpu.txt", output / "requirements-cpu.txt")
    (output / "source_figures").mkdir()
    for suffix in ["svg", "png"]:
        shutil.copy2(
            old / "figures" / ("botanic." + suffix),
            output / "source_figures" / ("botanic." + suffix),
        )
    (output / "DATA_TERMS.md").write_text((worktree / "docs/v0.4_data_terms.md").read_text())
    # Preserve provider/SLURM/reference information, replacing machine-specific roots only.
    for file in (output / "provenance").glob("*.json"):
        text = (
            file.read_text()
            .replace("/projects/ag-demeaux/tali/PopGenLM/", "PROJECT_ROOT/")
            .replace("/projects/ag-demeaux/tali/tools/", "ENVIRONMENTS/")
        )
        file.write_text(text)
    files = {
        str(f.relative_to(output)): {"sha256": sha(f), "bytes": f.stat().st_size}
        for f in sorted(output.rglob("*"))
        if f.is_file()
    }
    manifest = {
        "schema": 1,
        "release": "v0.4.0",
        "source_revision": "v0.4.0",
        "status": "Public research reproduction; code Apache-2.0, data retain upstream terms; Botanic raw score columns excluded",
        "raw_reconstruction": "All available non-Botanic score ranks reconstructed. Botanic outcomes checked from published rank records, not rerun from withheld raw scores.",
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output.parent / "popgenlm-v04-public-inputs.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(list(files) + ["manifest.json"]):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (output / name).read_bytes())
    print(json.dumps({"archive": str(archive), "sha256": sha(archive), "input_files": len(files)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--worktree", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assemble(a.project, a.worktree, a.output)
