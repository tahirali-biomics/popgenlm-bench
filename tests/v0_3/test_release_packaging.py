"""Release-contract tests for the PopGenLM Bench v0.3 package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "benchmarks" / "v0.3" / "manifest.json"
SUMS = ROOT / "data" / "benchmarks" / "v0.3" / "SHA256SUMS.txt"
NOTEBOOK = ROOT / "notebooks" / "popgenlm-bench-v0.3.ipynb"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def test_manifest_covers_exactly_promoted_assets() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assets = manifest["assets"]
    paths = [asset["path"] for asset in assets]
    expected = sorted(
        str(path.relative_to(ROOT))
        for directory in (ROOT / "data" / "benchmarks" / "v0.3", ROOT / "figures" / "v0.3")
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS.txt"}
    )
    assert paths == expected
    assert len(paths) == 14
    assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths)


def test_manifest_hashes_sizes_and_tsv_dimensions() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for asset in manifest["assets"]:
        path = ROOT / asset["path"]
        assert path.exists()
        assert digest(path) == asset["sha256"]
        assert path.stat().st_size == asset["bytes"]
        if path.suffix == ".tsv":
            table = pd.read_csv(path, sep="\t")
            assert table.shape == (asset["rows"], asset["columns"])
        else:
            assert "rows" not in asset and "columns" not in asset


def test_sha256sums_matches_manifest_and_is_portable() -> None:
    manifest = json.loads(MANIFEST.read_text())
    expected = [f"{asset['sha256']}  {asset['path']}" for asset in manifest["assets"]]
    assert SUMS.read_text().splitlines() == expected
    assert "manifest.json" not in SUMS.read_text()
    assert all("/projects/" not in line and "/home/" not in line for line in expected)


def test_v03_notebook_is_clear_and_release_facing() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert all(
        cell["execution_count"] is None for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert all(not cell["outputs"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "v0.3.0" in source
    assert "raw.githubusercontent.com/tahirali-biomics/popgenlm-bench/v0.3.0/" in source
    assert "10,000 rows and 39 columns" in source
    assert "no hypothesis-test p-values" in source.lower()
    assert "model inference" in source.lower()
    assert "umap fitting" in source.lower()
    assert "bootstrap resampling" in source.lower()
    assert "/projects/" not in source and "/home/" not in source


def test_readme_keeps_three_active_versions() -> None:
    text = (ROOT / "README.md").read_text()
    assert "v0.1, v0.2, and v0.3 are all active" in text
    assert "notebooks/popgenlm-bench-v0.3.ipynb" in text
    assert "docs/v0.3_release.md" in text
    assert "data/benchmarks/v0.3/manifest.json" in text
    assert "figures/v0.3/model_evidence.svg" in text
    assert "figures/v0.3/genotype_umap_geography.svg" in text
